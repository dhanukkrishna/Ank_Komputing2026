"""
build_graphs.py - STEP files -> face-adjacency graphs (same 11 node / 5 edge
features as the 300-part L-bracket dataset) + per-face feature labels.

Usage:
    python build_graphs.py --data data --out data/graphs.pkl --workers 4

Each graph is a dict of numpy arrays:
    x          [N, 11]  is_cylinder,is_plane,other,area,radius,axis xyz,centroid xyz
    edge_index [2, E]   (both directions)
    edge_attr  [E, 5]   length,angle,is_convex,is_concave,is_smooth
    y          [N]      face label 0..7  (see cad_utils.LABELS)
    name       str
"""
import argparse
import glob
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import cadquery as cq

from cad_utils import Registry, face_signature, label_faces, edge_face_map

SMOOTH_DEG = 1.0
EPS = 0.05  # mm, probe distance for the convex/concave test


def node_features(face):
    sig = face_signature(face)
    t, v = sig
    c = face.Center()
    is_cyl, is_plane = float(t == "CYL"), float(t == "PLANE")
    other = float(not (is_cyl or is_plane))
    radius, axis = 0.0, np.zeros(3)
    if t == "CYL":
        radius, axis = float(v[6]), v[:3]
    elif t == "PLANE":
        n = face.normalAt(c)  # true outward normal (signature sign is canonical)
        axis = np.array([n.x, n.y, n.z])
    elif t == "CONE":
        axis = v[:3]
    elif t == "SPHERE":
        radius = float(v[3])
    elif t == "TORUS":
        axis, radius = v[:3], float(v[7])  # minor radius
    return np.r_[is_cyl, is_plane, other, face.Area(), radius, axis, c.x, c.y, c.z]


def edge_features(solid, edge, fa, fb):
    """Dihedral test: step from the edge midpoint into face B (perpendicular to
    the edge, tangent to B). If that point lies *below* face A's plane
    (against A's outward normal) the edge is convex, otherwise concave."""
    p = edge.positionAt(0.5)
    n1, n2 = fa.normalAt(p), fb.normalAt(p)
    angle = float(np.arccos(np.clip(n1.dot(n2), -1, 1)))  # 0 = tangent-continuous
    if np.degrees(angle) < SMOOTH_DEG:
        convex, concave, smooth = 0.0, 0.0, 1.0
    else:
        d = edge.tangentAt(0.5).cross(n2).normalized()
        cand = [d, d * -1]
        dist = [fb.distance(cq.Vertex.makeVertex(*(p + c * EPS).toTuple())) for c in cand]
        d_in = cand[int(np.argmin(dist))]  # direction pointing into face B
        is_convex = n1.dot(d_in) < 0
        convex, concave, smooth = (1.0, 0.0, 0.0) if is_convex else (0.0, 1.0, 0.0)
    return np.array([edge.Length(), angle, convex, concave, smooth])


def build_graph(args):
    step_path, meta_path = args
    try:
        solid = cq.importers.importStep(step_path).val()
        if not isinstance(solid, cq.Solid):
            solid = solid.Solids()[0]
        with open(meta_path) as fh:
            registry = Registry.from_json(json.load(fh)["registry"])
        faces, edges, pairs = edge_face_map(solid)
        y = np.array(label_faces(faces, registry))
        if (y < 0).any():
            return None
        x = np.stack([node_features(f) for f in faces])
        src, dst, attr = [], [], []
        for e, p in zip(edges, pairs):
            if p is None:
                continue
            a, b = p
            ef = edge_features(solid, e, faces[a], faces[b])
            src += [a, b]; dst += [b, a]; attr += [ef, ef]
        return dict(name=os.path.basename(step_path)[:-5], x=x.astype(np.float32),
                    edge_index=np.array([src, dst], dtype=np.int64),
                    edge_attr=np.stack(attr).astype(np.float32), y=y.astype(np.int64))
    except Exception as ex:
        print("FAILED", step_path, repr(ex))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/graphs.pkl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    steps = sorted(glob.glob(os.path.join(a.data, "step", "*.step")))
    jobs = [(s, os.path.join(a.data, "meta", os.path.basename(s)[:-5] + ".json")) for s in steps]
    with Pool(a.workers) as pool:
        graphs = [g for g in pool.imap(build_graph, jobs, chunksize=8) if g is not None]
    with open(a.out, "wb") as fh:
        pickle.dump(graphs, fh)
    nodes = sum(len(g["y"]) for g in graphs)
    print(f"{len(graphs)}/{len(steps)} graphs, {nodes} faces total -> {a.out}")


if __name__ == "__main__":
    main()