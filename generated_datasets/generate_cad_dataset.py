"""
generate_cad_dataset.py - generate N diverse STEP parts with per-face
machining-feature labels.

Features: hole, slot, pocket, chamfer, fillet, boss, rib
Stocks  : block, L-bracket, U-channel, stepped block, T-section

Usage:
    python generate_cad_dataset.py --n 1000 --out data --workers 4 --seed 0

Output (in <out>/):
    step/part_00000.step      the CAD model
    meta/part_00000.json      stock + feature params + signature->label registry
    dataset_stats.json        class balance, failures, etc.
"""
import argparse
import json
import os
import random
import traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np
import cadquery as cq

from cad_utils import (LABELS, LABEL_NAMES, Registry, face_signature, label_faces,
                       edge_face_map, plane_sig, sig_match)

V = cq.Vector
MACHINED = ["hole", "slot", "pocket", "boss", "rib"]
EDGE_OPS = ["chamfer", "fillet"]
ALL_FEATURES = MACHINED + EDGE_OPS  # cycled so every class is covered evenly


class Fail(Exception):
    """Raised when a feature cannot be placed; caller just retries."""


# Stock generation (polygon profile in XZ, extruded along Y)

def make_stock(rng):
    kind = rng.choice(["block", "L", "U", "step", "T"])
    Wy = round(rng.uniform(30, 90), 1)
    if kind == "block":
        L, H = round(rng.uniform(40, 120), 1), round(rng.uniform(15, 50), 1)
        pts = [(0, 0), (L, 0), (L, H), (0, H)]
        dims = dict(L=L, H=H)
    elif kind == "L":
        L, H, t = (round(rng.uniform(50, 110), 1), round(rng.uniform(40, 90), 1),
                   round(rng.uniform(8, 16), 1))
        pts = [(0, 0), (L, 0), (L, t), (t, t), (t, H), (0, H)]
        dims = dict(L=L, H=H, t=t)
    elif kind == "U":
        L, H, t = (round(rng.uniform(60, 110), 1), round(rng.uniform(35, 70), 1),
                   round(rng.uniform(8, 14), 1))
        pts = [(0, 0), (L, 0), (L, H), (L - t, H), (L - t, t), (t, t), (t, H), (0, H)]
        dims = dict(L=L, H=H, t=t)
    elif kind == "step":
        L, H1 = round(rng.uniform(70, 120), 1), round(rng.uniform(15, 30), 1)
        L2, H2 = round(rng.uniform(0.3, 0.6) * L, 1), round(rng.uniform(15, 30), 1)
        pts = [(0, 0), (L, 0), (L, H1), (L2, H1), (L2, H1 + H2), (0, H1 + H2)]
        dims = dict(L=L, H1=H1, L2=L2, H2=H2)
    else:  # T
        Lf, tf = round(rng.uniform(70, 120), 1), round(rng.uniform(10, 18), 1)
        ts, Hs = round(rng.uniform(12, 20), 1), round(rng.uniform(30, 60), 1)
        x0 = (Lf - ts) / 2
        pts = [(0, 0), (Lf, 0), (Lf, tf), (x0 + ts, tf), (x0 + ts, tf + Hs),
               (x0, tf + Hs), (x0, tf), (0, tf)]
        dims = dict(Lf=Lf, tf=tf, ts=ts, Hs=Hs)
    dims["Wy"] = Wy
    wp = cq.Workplane("XZ").polyline(pts).close().extrude(-Wy)  # +Y direction
    solid = wp.val()
    return kind, dims, solid


# Candidate mounting faces (axis-aligned rectangles of the stock)

AXES = {0: (V(0, 1, 0), V(0, 0, 1)), 1: (V(1, 0, 0), V(0, 0, 1)), 2: (V(1, 0, 0), V(0, 1, 0))}


def candidate_faces(solid, min_size=14.0):
    out = []
    for f in solid.Faces():
        if f.geomType() != "PLANE":
            continue
        n = f.normalAt(f.Center())
        ax = int(np.argmax([abs(n.x), abs(n.y), abs(n.z)]))
        if abs([n.x, n.y, n.z][ax]) < 0.999:
            continue
        u, v = AXES[ax]
        bb = f.BoundingBox()
        ext = [bb.xlen, bb.ylen, bb.zlen]
        W, H = {0: (ext[1], ext[2]), 1: (ext[0], ext[2]), 2: (ext[0], ext[1])}[ax]
        if abs(f.Area() - W * H) > 0.01 * W * H:  # not a plain rectangle
            continue
        if min(W, H) < min_size:
            continue
        out.append(dict(face=f, n=V(*(round(c) for c in (n.x, n.y, n.z))), u=u, v=v,
                        c=f.Center(), W=W, H=H, occ=[]))
    return out


def thickness(solid, origin, n):
    """Material thickness of the stock below `origin` along -n (ray cast)."""
    ray = (cq.Workplane(cq.Plane(origin=origin + n * 0.01, xDir=V(1, 0, 0) if abs(n.x) < .5 else V(0, 1, 0),
                                 normal=n)).rect(0.02, 0.02).extrude(-2000).val())
    try:
        inter = solid.intersect(ray)
    except Exception:
        return None
    best = None
    axis = int(np.argmax([abs(n.x), abs(n.y), abs(n.z)]))
    o = [origin.x, origin.y, origin.z][axis]
    for s in inter.Solids():
        bb = s.BoundingBox()
        lo, hi = [(bb.xmin, bb.xmax), (bb.ymin, bb.ymax), (bb.zmin, bb.zmax)][axis]
        top = hi if n.toTuple()[axis] > 0 else lo
        if abs(top - o) < 0.2:
            best = hi - lo
    return best



# Feature builders  (each returns tool solid, kind 'cut'|'add', params,
#    footprint (a0,a1,b0,b1), special plane for coincidence check)

def plane_of(face, a, b):
    o = face["c"] + face["u"] * a + face["v"] * b
    return cq.Plane(origin=o, xDir=face["u"], normal=face["n"])


def fits(face, fp, margin=1.5, open_ended=False):
    a0, a1, b0, b1 = fp
    if open_ended:  # slot: runs edge to edge, so only the cross-direction is bounded
        full_u = abs(a1 - a0 - face["W"]) < 1e-6
        in_u = full_u or (a0 >= -face["W"] / 2 + margin and a1 <= face["W"] / 2 - margin)
        full_v = abs(b1 - b0 - face["H"]) < 1e-6
        in_v = full_v or (b0 >= -face["H"] / 2 + margin and b1 <= face["H"] / 2 - margin)
        if not (in_u and in_v): return False
    else:
        if a0 < -face["W"] / 2 + margin or a1 > face["W"] / 2 - margin: return False
        if b0 < -face["H"] / 2 + margin or b1 > face["H"] / 2 - margin: return False
    for (x0, x1, y0, y1) in face["occ"]:
        if a0 < x1 + margin and a1 > x0 - margin and b0 < y1 + margin and b1 > y0 - margin:
            return False
    return True


def build_hole(face, T, rng):
    r = round(rng.uniform(1.5, min(6.0, 0.3 * min(face["W"], face["H"]))), 1)
    a = rng.uniform(-face["W"] / 2 + r + 2, face["W"] / 2 - r - 2)
    b = rng.uniform(-face["H"] / 2 + r + 2, face["H"] / 2 - r - 2)
    through = rng.random() < 0.5
    depth = T + 1.0 if through else round(rng.uniform(0.35, 0.75) * T, 1)
    if depth < 2: raise Fail
    fp = (a - r, a + r, b - r, b + r)
    tool = (cq.Workplane(plane_of(face, a, b)).workplane(offset=0.5).circle(r)
            .extrude(-(depth + 0.5)).val())
    bottom = None if through else (face["n"], face["c"] + face["u"] * a + face["v"] * b - face["n"] * depth)
    return tool, "cut", dict(r=r, depth=depth, through=through, pos=(a, b)), fp, bottom


def build_pocket(face, T, rng):
    if T < 4: raise Fail
    l = round(rng.uniform(0.2, 0.5) * face["W"], 1); w = round(rng.uniform(0.2, 0.5) * face["H"], 1)
    l, w = max(l, 6), max(w, 6)
    if l > face["W"] - 4 or w > face["H"] - 4: raise Fail
    a = rng.uniform(-face["W"] / 2 + l / 2 + 2, face["W"] / 2 - l / 2 - 2)
    b = rng.uniform(-face["H"] / 2 + w / 2 + 2, face["H"] / 2 - w / 2 - 2)
    depth = round(max(2.0, rng.uniform(0.3, 0.7) * T), 1)
    fp = (a - l / 2, a + l / 2, b - w / 2, b + w / 2)
    tool = (cq.Workplane(plane_of(face, a, b)).workplane(offset=0.5).rect(l, w)
            .extrude(-(depth + 0.5)).val())
    return (tool, "cut", dict(l=l, w=w, depth=depth, pos=(a, b)), fp,
            (face["n"], face["c"] + face["u"] * a + face["v"] * b - face["n"] * depth))


def build_slot(face, T, rng):
    """Open-ended rectangular channel running across the whole face."""
    if T < 4: raise Fail
    along_u = rng.random() < 0.5
    across = face["H"] if along_u else face["W"]
    w = round(rng.uniform(4, min(12, 0.35 * across)), 1)
    pos = rng.uniform(-across / 2 + w / 2 + 2, across / 2 - w / 2 - 2)
    depth = round(max(2.0, rng.uniform(0.3, 0.7) * T), 1)
    if along_u:
        a, b, l, ww = 0, pos, face["W"] + 2, w
        fp = (-face["W"] / 2, face["W"] / 2, pos - w / 2, pos + w / 2)
    else:
        a, b, l, ww = pos, 0, w, face["H"] + 2
        fp = (pos - w / 2, pos + w / 2, -face["H"] / 2, face["H"] / 2)
    tool = (cq.Workplane(plane_of(face, a, b)).workplane(offset=0.5).rect(l, ww)
            .extrude(-(depth + 0.5)).val())
    return (tool, "cut", dict(width=w, depth=depth, along="u" if along_u else "v", pos=pos), fp,
            (face["n"], face["c"] + face["u"] * a + face["v"] * b - face["n"] * depth))


def build_boss(face, T, rng):
    r = round(rng.uniform(3, min(10, 0.3 * min(face["W"], face["H"]))), 1)
    h = round(rng.uniform(3, 12), 1)
    a = rng.uniform(-face["W"] / 2 + r + 2, face["W"] / 2 - r - 2)
    b = rng.uniform(-face["H"] / 2 + r + 2, face["H"] / 2 - r - 2)
    fp = (a - r, a + r, b - r, b + r)
    tool = cq.Workplane(plane_of(face, a, b)).workplane(offset=-0.5).circle(r).extrude(h + 0.5).val()
    return (tool, "add", dict(r=r, h=h, pos=(a, b)), fp,
            (face["n"], face["c"] + face["u"] * a + face["v"] * b + face["n"] * h))


def build_rib(face, T, rng):
    along_u = rng.random() < 0.5
    span = face["W"] if along_u else face["H"]
    other = face["H"] if along_u else face["W"]
    t = round(rng.uniform(2, 4), 1)
    l = round(rng.uniform(0.3, 0.7) * span, 1)
    h = round(rng.uniform(4, 14), 1)
    a = rng.uniform(-span / 2 + l / 2 + 2, span / 2 - l / 2 - 2)
    b = rng.uniform(-other / 2 + t / 2 + 2, other / 2 - t / 2 - 2)
    if along_u: a, b, fp, (rl, rw) = a, b, (a - l / 2, a + l / 2, b - t / 2, b + t / 2), (l, t)
    else:       a, b, fp, (rl, rw) = b, a, (b - t / 2, b + t / 2, a - l / 2, a + l / 2), (t, l)
    tool = cq.Workplane(plane_of(face, a, b)).workplane(offset=-0.5).rect(rl, rw).extrude(h + 0.5).val()
    return (tool, "add", dict(t=t, l=l, h=h, along="u" if along_u else "v"), fp,
            (face["n"], face["c"] + face["u"] * a + face["v"] * b + face["n"] * h))


BUILDERS = dict(hole=build_hole, pocket=build_pocket, slot=build_slot, boss=build_boss, rib=build_rib)



# Applying operations + label bookkeeping

def to_single_solid(shape):
    try:
        shape = shape.clean()
    except Exception:
        pass
    solids = shape.Solids()
    if len(solids) != 1 or not solids[0].isValid() or solids[0].Volume() < 100:
        raise Fail
    return solids[0]


def register_new_faces(solid, registry, label):
    new = 0
    for f in solid.Faces():
        sig = face_signature(f)
        if registry.find(sig) is None:
            registry.add(sig, label)
            new += 1
    if new == 0:
        raise Fail  # operation had no visible effect
    return new


def apply_machined(solid, registry, faces, ftype, rng, stock_solid, tries=40):
    for _ in range(tries):
        face = rng.choices(faces, weights=[f["W"] * f["H"] for f in faces])[0]
        try:
            ctr = face["c"]
            T = thickness(stock_solid, ctr, face["n"])
            if T is None: raise Fail
            tool, kind, params, fp, special = BUILDERS[ftype](face, T, rng)
            if not fits(face, fp, open_ended=(ftype == "slot")): raise Fail
            if special is not None and registry.has_plane(*special):
                raise Fail  # coplanar with an existing face -> label would be ambiguous
            new_solid = to_single_solid(solid.cut(tool) if kind == "cut" else solid.fuse(tool))
            trial = Registry(); trial.items = list(registry.items)
            register_new_faces(new_solid, trial, LABELS[ftype])
            registry.items = trial.items
            face["occ"].append(fp)
            params["face_normal"] = [int(face["n"].x), int(face["n"].y), int(face["n"].z)]
            return new_solid, dict(type=ftype, **params)
        except Fail:
            continue
        except (RuntimeError, ValueError, Exception) as e:  # OCC kernel errors -> retry
            if os.environ.get("CAD_DEBUG"): traceback.print_exc()
            continue
    return solid, None


def apply_edge_op(solid, registry, op, rng, tries=60):
    for _ in range(tries):
        faces, edges, pairs = edge_face_map(solid)
        cand = [e for e, p in zip(edges, pairs)
                if p is not None and e.Length() > 4.0 and e.geomType() in ("LINE", "CIRCLE")]
        if not cand: break
        e = rng.choice(cand)
        size = round(rng.uniform(0.6, 2.0), 1) if op == "chamfer" else round(rng.uniform(0.8, 3.0), 1)
        try:
            res = solid.chamfer(size, None, [e]) if op == "chamfer" else solid.fillet(size, [e])
            new_solid = to_single_solid(res)
            trial = Registry(); trial.items = list(registry.items)
            register_new_faces(new_solid, trial, LABELS[op])
            registry.items = trial.items
            return new_solid, dict(type=op, size=size, edge_type=e.geomType(),
                                   edge_len=round(e.Length(), 2))
        except Exception:
            continue
    return solid, None



# One part

def plan_features(idx, rng):
    """Every sample gets a guaranteed 'focus' feature (cycled -> balanced
    classes) plus random extras."""
    focus = ALL_FEATURES[idx % len(ALL_FEATURES)]
    extras = rng.sample(MACHINED, k=rng.randint(1, 3))
    n_edge = rng.randint(0, 2)
    extras += [rng.choice(EDGE_OPS) for _ in range(n_edge)]
    plan = [focus] + extras
    adds = [p for p in plan if p in ("boss", "rib")]
    cuts = [p for p in plan if p in ("hole", "slot", "pocket")]
    edges = [p for p in plan if p in EDGE_OPS]
    return adds + cuts + edges  # order: additive, subtractive, edge treatments


def build_part(idx, seed):
    rng = random.Random(seed * 100003 + idx)
    kind, dims, stock = make_stock(rng)
    registry = Registry()
    for f in stock.Faces():
        registry.add(face_signature(f), LABELS["stock"])
    cand = candidate_faces(stock)
    if not cand: raise Fail
    solid, applied = stock, []
    plan = plan_features(idx, rng)
    for ftype in plan:
        if ftype in MACHINED:
            solid, rec = apply_machined(solid, registry, cand, ftype, rng, stock)
        else:
            solid, rec = apply_edge_op(solid, registry, ftype, rng)
        if rec: applied.append(rec)
    focus = ALL_FEATURES[idx % len(ALL_FEATURES)]
    if not any(r["type"] == focus for r in applied):
        raise Fail  # guarantee the focus class is really present
    return kind, dims, solid, registry, applied


def generate_one(args):
    idx, seed, out = args
    for attempt in range(30):  # retry with different random draws
        try:
            kind, dims, solid, registry, applied = build_part(idx, seed + attempt * 7919)
            name = f"part_{idx:05d}"
            step_path = os.path.join(out, "step", name + ".step")
            cq.exporters.export(cq.Workplane("XY").add(solid), step_path)
            # integrity check: re-import and make sure every face has a label
            back = cq.importers.importStep(step_path).val()
            faces = back.Faces() if not isinstance(back, cq.Solid) else back.Faces()
            labels = label_faces(faces, registry)
            if -1 in labels or len(faces) != len(solid.Faces()):
                os.remove(step_path); continue
            meta = dict(id=name, stock=kind, stock_dims=dims, features=applied,
                        n_faces=len(faces), face_label_counts=dict(Counter(LABEL_NAMES[l] for l in labels)),
                        registry=registry.to_json())
            with open(os.path.join(out, "meta", name + ".json"), "w") as fh:
                json.dump(meta, fh, default=float)
            return idx, meta["face_label_counts"], kind, attempt
        except Fail:
            continue
        except Exception:
            continue
    return idx, None, None, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "step"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "meta"), exist_ok=True)
    jobs = [(i, a.seed, a.out) for i in range(a.n)]
    face_cnt, stocks, failed, done = Counter(), Counter(), [], 0
    with Pool(a.workers) as pool:
        for idx, counts, kind, att in pool.imap_unordered(generate_one, jobs, chunksize=4):
            if counts is None:
                failed.append(idx)
            else:
                face_cnt.update(counts); stocks[kind] += 1
            done += 1
            if done % 25 == 0:
                print(f"{done}/{a.n}  failed so far: {len(failed)}", flush=True)
    stats = dict(requested=a.n, generated=a.n - len(failed), failed_idx=sorted(failed),
                 faces_per_class=dict(face_cnt), stock_types=dict(stocks))
    with open(os.path.join(a.out, "dataset_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()