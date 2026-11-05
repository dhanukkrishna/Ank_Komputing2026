import cadquery as cq

length = 100
width = 50
height = 30

hole_diameter = 15

block = (
    cq.Workplane("XY")
    .box(length, width, height)
)

block_with_hole = (
    block.faces(">Z")
    .hole(hole_diameter)
)

cq.exporters.export(
    block_with_hole,
    "D:\\ank_2026\\B_Rep_GNN\\Step Files\\block\\block_with_hole.step"
)

print("Done!")