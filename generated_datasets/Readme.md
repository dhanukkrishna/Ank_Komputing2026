# 1000-part diverse CAD dataset (feature recognition)

    pip install cadquery numpy
    python generate_cad_dataset.py --n 1000 --out data --workers 4 --seed 0
    python build_graphs.py --data data --out data/graphs.pkl --workers 4

Files
- cad_utils.py            face signatures, label registry, face adjacency
- generate_cad_dataset.py stock + feature generator -> STEP + JSON metadata
- build_graphs.py         STEP -> graph (11 node / 5 edge feats) + per-face labels

Face labels: 0 stock, 1 hole, 2 slot, 3 pocket, 4 chamfer, 5 fillet, 6 boss, 7 rib
Stocks: block, L, U, stepped, T. Every part has 1 guaranteed "focus" feature
(cycled over the 7 types) plus random extras -> balanced class presence.
Slot = open-ended channel across a face; pocket = closed rectangle.
<br>
**[Datasets](https://drive.google.com/drive/u/2/folders/1vO5I-w18YlLybtYrGVnJ9pJ24LY-kJdg)**
<br>
# Project File Structure

| File | Role |
|------|------|
| `Readme.md` | Instructions and overview of the dataset |
| `cad_utils.py` | Shared geometry, signature and labeling functions |
| `generate_cad_dataset.py` | Generates CAD models and establishes face labels |
| `build_graphs.py` | Converts STEP models into graph data |
| `gnn_face_classification.ipynb` | GNN training and experimentation notebook |
| `results/` | Intended location for results |
