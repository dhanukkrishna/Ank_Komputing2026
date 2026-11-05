
import json

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps


# Read STEP file
reader = STEPControl_Reader()
status = reader.ReadFile("B_Rep_GNN/Step Files/block/block_with_hole.step")

if status != IFSelect_RetDone:
    raise Exception("STEP file could not be read")

reader.TransferRoots()
shape = reader.OneShape()


#Extract all faces
faces = []
explorer = TopExp_Explorer(shape, TopAbs_FACE)

while explorer.More():
    faces.append(TopoDS.Face_s(explorer.Current()))
    explorer.Next()

print("Total faces:", len(faces))


#Extract node features
nodes = []

for i, face in enumerate(faces):

    surface = BRepAdaptor_Surface(face)

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)

    center = props.CentreOfMass()

    node = {
        "face_id": i,
        "surface_type": str(surface.GetType()),
        "area": props.Mass(),
        "centroid": [
            center.X(),
            center.Y(),
            center.Z()
        ]
    }

    nodes.append(node)


#Build edge-to-face mapping
edge_to_faces = {}

for face_id, face in enumerate(faces):

    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)

    while edge_explorer.More():

        edge = TopoDS.Edge_s(edge_explorer.Current())

        # Find an existing matching edge
        edge_id = None

        for existing_id, existing_edge in edge_to_faces.items():
            if edge.IsSame(existing_edge["edge"]):
                edge_id = existing_id
                break

        if edge_id is None:
            edge_id = len(edge_to_faces)
            edge_to_faces[edge_id] = {
                "edge": edge,
                "faces": []
            }

        if face_id not in edge_to_faces[edge_id]["faces"]:
            edge_to_faces[edge_id]["faces"].append(face_id)

        edge_explorer.Next()


# Create graph connections
adjacency = []
edge_attributes = []

for edge_id, data in edge_to_faces.items():

    connected_faces = data["faces"]

    if len(connected_faces) == 2:

        face_a, face_b = connected_faces

        adjacency.append([face_a, face_b])
        adjacency.append([face_b, face_a])

        props = GProp_GProps()
        BRepGProp.LinearProperties_s(data["edge"], props)

        edge_length = props.Mass()

        edge_attributes.append({
            "source": face_a,
            "target": face_b,
            "length": edge_length
        })

        edge_attributes.append({
            "source": face_b,
            "target": face_a,
            "length": edge_length
        })


# Save graph
graph = {
    "nodes": nodes,
    "adjacency": adjacency,
    "edge_attributes": edge_attributes
}

with open("face_graph.json", "w") as file:
    json.dump(graph, file, indent=4)

print("\nGraph generated successfully!")
print("Number of nodes:", len(nodes))
print("Number of directed connections:", len(adjacency))
print("Saved as face_graph.json")
