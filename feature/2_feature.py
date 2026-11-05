
import json
import torch
from torch_geometric.data import Data


# LOAD JSON GRAPH

with open("blind_hole_graph_001.json", "r") as file:
    graph = json.load(file)

nodes = graph["nodes"]
edges = graph["edges"]


# EXTRACT NODE FEATURES

# Each node contains:
# [surface_type, area, centroid_x, centroid_y, centroid_z]

x = torch.tensor(
    [node["features"] for node in nodes],
    dtype=torch.float
)


# EXTRACT EDGE CONNECTIONS

# PyG expects edge_index with shape [2, number_of_edges]

edge_index = torch.tensor(
    [
        [edge["source"] for edge in edges],
        [edge["target"] for edge in edges]
    ],
    dtype=torch.long
)


#  EXTRACT GROUND-TRUTH LABELS

# base = 0
# blind_hole = 1

y = torch.tensor(
    [node["label_id"] for node in nodes],
    dtype=torch.long
)


# CREATE PYTORCH GEOMETRIC DATA

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)


#  VERIFY GRAPH

print("\n---------- GRAPH INFORMATION ----------")

print("Node features (x):")
print(data.x)

print("\nEdge connections (edge_index):")
print(data.edge_index)

print("\nGround-truth labels (y):")
print(data.y)

print("\nNumber of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of node features:", data.num_node_features)

print("\nGraph valid:", data.validate())


# SAVE PYTORCH GRAPH


torch.save(data, "blind_hole_graph_001.pt")

print("\nPyTorch Geometric graph saved!")
