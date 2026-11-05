from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus
)

# STEP file path
step_file = "D:\\ank_2026\\B_Rep_GNN\\Step Files\\block\\block_with_hole.step"

# Read STEP file
reader = STEPControl_Reader()
status = reader.ReadFile(step_file)

if status != IFSelect_RetDone:
    raise Exception("Could not read STEP file!")

# Transfer geometry into an OCCT shape
reader.TransferRoots()
shape = reader.OneShape()

# Map OCCT surface types to readable names
surface_names = {
    GeomAbs_Plane: "Plane",
    GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone",
    GeomAbs_Sphere: "Sphere",
    GeomAbs_Torus: "Torus"
}

# Explore all faces
explorer = TopExp_Explorer(shape, TopAbs_FACE)

face_id = 0
surface_counts = {}

print("\nB-REP FACE INFORMATION")

while explorer.More():

    face = TopoDS.Face_s(explorer.Current())

    # Identify the surface geometry
    surface = BRepAdaptor_Surface(face)
    surface_type = surface.GetType()

    name = surface_names.get(surface_type, "Other")

    face_id += 1

    surface_counts[name] = surface_counts.get(name, 0) + 1

    print(f"Face {face_id}: {name}")

    explorer.Next()

# Print summary
print("\nSURFACE SUMMARY")

for name, count in surface_counts.items():
    print(f"{name}: {count}")

print(f"\nTotal faces: {face_id}")