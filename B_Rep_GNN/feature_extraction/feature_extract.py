from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder

# Read STEP file
reader = STEPControl_Reader()
status = reader.ReadFile("D:\\ank_2026\\B_Rep_GNN\\Step Files\\block\\block_with_hole.step")

if status != IFSelect_RetDone:
    raise Exception("STEP file could not be read")

reader.TransferRoots()
shape = reader.OneShape()

explorer = TopExp_Explorer(shape, TopAbs_FACE)

face_id = 0

while explorer.More():

    face = TopoDS.Face_s(explorer.Current())

    surface = BRepAdaptor_Surface(face)
    surface_type = surface.GetType()

    # Calculate face surface area and centroid
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)

    area = props.Mass()
    center = props.CentreOfMass()

    print(f"\nFace {face_id + 1}")

    print("Area:", round(area, 3))
    print(
        "Centroid:",
        round(center.X(), 3),
        round(center.Y(), 3),
        round(center.Z(), 3)
    )

    if surface_type == GeomAbs_Plane:
        print("Type: Plane")

    elif surface_type == GeomAbs_Cylinder:
        cylinder = surface.Cylinder()

        print("Type: Cylinder")
        print("Radius:", round(cylinder.Radius(), 3))

    else:
        print("Type: Other")

    face_id += 1
    explorer.Next()