"""
cad_utils.py - shared helpers for the 1000-part feature-recognition dataset.

Core idea: every face gets a *geometric signature* (surface type + surface
parameters, NOT area/boundary). When a machining feature is added, the faces
whose signature did not exist before are the faces created by that feature,
so they receive that feature's label. Because the signature depends only on
the underlying surface, the labels can be recovered after the part is
exported to STEP and re-imported (face order may change, geometry does not).
"""
import numpy as np
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE

# Per-face class labels (node-level targets for the GNN)
LABELS = {"stock": 0, "hole": 1, "slot": 2, "pocket": 3,
          "chamfer": 4, "fillet": 5, "boss": 6, "rib": 7}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
SIG_TOL = 1e-2  # mm


# --------------------------------------------------------------------------
# Face signatures
# --------------------------------------------------------------------------
def _canon(d):
    """Unit vector with a canonical sign (first non-zero component > 0)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    for x in d:
        if abs(x) > 1e-6:
            return d if x > 0 else -d
    return d


def _xyz(o):
    return np.array([o.X(), o.Y(), o.Z()])


def _arr(v):
    return np.array(v.toTuple(), float) if hasattr(v, "toTuple") else np.asarray(v, float)


def plane_sig(normal, point):
    n = _canon(_arr(normal))
    return ("PLANE", np.r_[n, n @ _arr(point)])


def face_signature(face):
    s = BRepAdaptor_Surface(face.wrapped)
    t = s.GetType()
    if t == GeomAbs_Plane:
        pl = s.Plane()
        return plane_sig(_xyz(pl.Axis().Direction()), _xyz(pl.Location()))
    if t == GeomAbs_Cylinder:
        c = s.Cylinder()
        d = _canon(_xyz(c.Axis().Direction()))
        p = _xyz(c.Location())
        q = p - (p @ d) * d  # closest point on the axis to the origin
        return ("CYL", np.r_[d, q, c.Radius()])
    if t == GeomAbs_Cone:
        c = s.Cone()
        d = _canon(_xyz(c.Axis().Direction()))
        return ("CONE", np.r_[d, _xyz(c.Apex()), abs(c.SemiAngle())])
    if t == GeomAbs_Sphere:
        c = s.Sphere()
        return ("SPHERE", np.r_[_xyz(c.Location()), c.Radius()])
    if t == GeomAbs_Torus:
        c = s.Torus()
        d = _canon(_xyz(c.Axis().Direction()))
        return ("TORUS", np.r_[d, _xyz(c.Location()), c.MajorRadius(), c.MinorRadius()])
    ctr = face.Center()
    return ("OTHER", np.array([ctr.x, ctr.y, ctr.z, face.Area()]))


def sig_match(a, b):
    return a[0] == b[0] and np.allclose(a[1], b[1], atol=SIG_TOL)


class Registry:
    """signature -> label lookup (stock faces start as label 0)."""

    def __init__(self):
        self.items = []  # list of (sig, label)

    def find(self, sig):
        for s, lab in self.items:
            if sig_match(s, sig):
                return lab
        return None

    def add(self, sig, label):
        self.items.append((sig, label))

    def has_plane(self, normal, point):
        sig = plane_sig(normal, point)
        return self.find(sig) is not None

    def to_json(self):
        return [{"type": s[0], "values": [round(float(x), 5) for x in s[1]], "label": int(l)}
                for s, l in self.items]

    @staticmethod
    def from_json(rows):
        r = Registry()
        for row in rows:
            r.add((row["type"], np.array(row["values"])), row["label"])
        return r


def label_faces(faces, registry):
    """Return label per face (-1 if the face signature is unknown)."""
    out = []
    for f in faces:
        lab = registry.find(face_signature(f))
        out.append(-1 if lab is None else lab)
    return out


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------
def edge_face_map(solid):
    """Return (faces, edges, pairs) where pairs[k] = (face_idx_a, face_idx_b)
    for edge k, or None if the edge is not shared by two *different* faces
    (e.g. cylinder seam edges)."""
    faces = solid.Faces()
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(solid.wrapped, TopAbs_EDGE, TopAbs_FACE, m)

    def face_index(tface):
        for i, f in enumerate(faces):
            if f.wrapped.IsSame(tface):
                return i
        return -1

    edges, pairs = [], []
    for k in range(1, m.Extent() + 1):
        edges.append(cq.Edge(m.FindKey(k)))
        idx = sorted({face_index(f) for f in m.FindFromIndex(k)})
        pairs.append(tuple(idx) if len(idx) == 2 and -1 not in idx else None)
    return faces, edges, pairs
