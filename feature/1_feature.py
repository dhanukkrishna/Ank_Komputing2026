
import cadquery as cq
import json
import math

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder


# --------------------------------------------------
# 1. PARAMETERS
# --------------------------------------------------

length = 100
width = 50
height = 30

hole_diameter = 15
hole_depth = 18

radius = hole_diameter / 2

model_id = "block_blind_hole_001"
feature_id = "blind_hole_001"


# --------------------------------------------------
# 2. GENERATE GEOMETRY USING CADQUERY
# --------------------------------------------------

block = cq.Workplane("XY").box(length, width, height)

model = (
    block.faces(">Z")
    .workplane()
    .hole(hole_diameter, hole_depth)
)

# Check geometry validity
if not model.val().isValid():
    raise Exception("Invalid CAD geometry!")

print("Geometry generated successfully!")


# --------------------------------------------------
# 3. EXPORT STEP FILE
# --------------------------------------------------

step_file = "block_with_blind_hole_001.step"

cq.exporters.export(model, step_file)

print("STEP exported:", step_file)


# --------------------------------------------------
# 4. READ STEP AND EXTRACT B-REP FACES
# --------------------------------------------------

reader = STEPControl_Reader()

status = reader.ReadFile(step_file)

if status != IFSelect_RetDone:
    raise Exception("Could not read STEP file!")

reader.TransferRoots()
shape = reader.OneShape()

explorer = TopExp_Explorer(shape, TopAbs_FACE)

faces = []

while explorer.More():
    face = TopoDS.Face_s(explorer.Current())
    faces.append(face)
    explorer.Next()

print("Total B-Rep faces:", len(faces))


# --------------------------------------------------
# 5. EXTRACT FACE INFORMATION AND LABELS
# --------------------------------------------------

face_data = []

for i, face in enumerate(faces):

    surface = BRepAdaptor_Surface(face, True)
    surface_type = surface.GetType()

    # Calculate area and centroid
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, properties)

    area = properties.Mass()
    center = properties.CentreOfMass()

    x = center.X()
    y = center.Y()
    z = center.Z()

    # Default label
    label = "base"
    instance_id = None
    geometry_type = "Other"

    # Identify planar faces
    if surface_type == GeomAbs_Plane:
        geometry_type = "Plane"

    # Identify cylindrical faces
    elif surface_type == GeomAbs_Cylinder:
        geometry_type = "Cylinder"

        cylinder = surface.Cylinder()
        cylinder_radius = cylinder.Radius()

        axis = cylinder.Axis().Direction()
        axis_location = cylinder.Axis().Location()

        # Check whether this is our blind-hole wall
        if (
            abs(cylinder_radius - radius) < 0.01
            and abs(axis.Z()) > 0.99
            and abs(axis_location.X()) < 0.01
            and abs(axis_location.Y()) < 0.01
        ):
            label = "blind_hole"
            instance_id = feature_id

    # Identify the flat bottom of the blind hole
    if (
        surface_type == GeomAbs_Plane
        and abs(area - math.pi * radius**2) < 0.1
        and abs(z - (height / 2 - hole_depth)) < 0.01
        and abs(x) < 0.01
        and abs(y) < 0.01
    ):
        label = "blind_hole"
        instance_id = feature_id

    face_data.append({
        "face_id": i,
        "geometry_type": geometry_type,
        "area": round(area, 3),
        "centroid": [
            round(x, 3),
            round(y, 3),
            round(z, 3)
        ],
        "feature_label": label,
        "feature_instance_id": instance_id
    })


# --------------------------------------------------
# 6. SAVE GROUND-TRUTH LABELS
# --------------------------------------------------

output = {
    "model_id": model_id,

    "parameters": {
        "length": length,
        "width": width,
        "height": height,
        "hole_diameter": hole_diameter,
        "hole_depth": hole_depth
    },

    "feature_instances": [
        {
            "feature_id": feature_id,
            "feature_type": "blind_hole",
            "parameters": {
                "diameter": hole_diameter,
                "depth": hole_depth,
                "position": [0, 0]
            }
        }
    ],

    "faces": face_data
}

json_file = "blind_hole_labels_001.json"

with open(json_file, "w") as file:
    json.dump(output, file, indent=4)

print("Ground-truth labels saved:", json_file)

# --------------------------------------------------
# 6. BUILD FACE ADJACENCY GRAPH
# --------------------------------------------------

# Extract all edges belonging to each face
face_edges = []

for face in faces:

    edges = []
    explorer = TopExp_Explorer(face, TopAbs_EDGE)

    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        edges.append(edge)
        explorer.Next()

    face_edges.append(edges)


# Compare faces to find shared edges
graph_edges = []

for i in range(len(faces)):

    for j in range(i + 1, len(faces)):

        shared_count = 0

        for edge1 in face_edges[i]:
            for edge2 in face_edges[j]:

                if edge1.IsSame(edge2):
                    shared_count += 1

        # If two faces share an edge, connect them
        if shared_count > 0:

            graph_edges.append({
                "source": i,
                "target": j,
                "shared_edge_count": shared_count
            })

            # Add reverse connection
            graph_edges.append({
                "source": j,
                "target": i,
                "shared_edge_count": shared_count
            })


print("\nFace adjacency graph generated!")
print("Number of nodes:", len(face_data))
print("Number of directed edges:", len(graph_edges))


# --------------------------------------------------
# 7. PREPARE GRAPH NODES
# --------------------------------------------------

label_ids = {
    "base": 0,
    "blind_hole": 1
}

surface_ids = {
    "Plane": 0,
    "Cylinder": 1,
    "Other": 2
}

graph_nodes = []

for face in face_data:

    graph_nodes.append({
        "face_id": face["face_id"],

        "features": [
            surface_ids[face["geometry_type"]],
            face["area"],
            *face["centroid"]
        ],

        "feature_label": face["feature_label"],

        "label_id": label_ids[face["feature_label"]],

        "feature_instance_id": face["feature_instance_id"]
    })


# --------------------------------------------------
# 8. SAVE GRAPH AND LABELS TO JSON
# --------------------------------------------------

output = {
    "model_id": model_id,

    "parameters": {
        "length": length,
        "width": width,
        "height": height,
        "hole_diameter": hole_diameter,
        "hole_depth": hole_depth
    },

    "feature_instances": [
        {
            "feature_id": feature_id,
            "feature_type": "blind_hole",
            "parameters": {
                "diameter": hole_diameter,
                "depth": hole_depth,
                "position": [0, 0]
            }
        }
    ],

    "nodes": graph_nodes,

    "edges": graph_edges
}

with open("blind_hole_graph_001.json", "w") as file:
    json.dump(output, file, indent=4)

print("Graph and labels saved successfully!")



# --------------------------------------------------
# 7. VERIFY LABELS
# --------------------------------------------------

blind_hole_faces = [
    face for face in face_data
    if face["feature_label"] == "blind_hole"
]

base_faces = [
    face for face in face_data
    if face["feature_label"] == "base"
]

print("\n---------- VERIFICATION ----------")

for face in face_data:
    print(
        f"Face {face['face_id']} | "
        f"Geometry: {face['geometry_type']} | "
        f"Label: {face['feature_label']} | "
        f"Area: {face['area']}"
    )

print("\nTotal faces:", len(face_data))
print("Blind-hole faces:", len(blind_hole_faces))
print("Base faces:", len(base_faces))

# Expected: one cylindrical wall + one planar bottom
if len(blind_hole_faces) == 2:
    print("\nSUCCESS: Blind-hole labels verified!")
else:
    print("\nWARNING: Expected 2 blind-hole faces.")
    print("Inspect the extracted face information.")
