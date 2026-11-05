
import cadquery as cq
import os


def generate_block(
    length=100,
    width=50,
    height=30,
    output_path="D:\\ank_2026\\B_Rep_GNN\\Step Files\\block\\block_001.step"
):

    # Create a rectangular solid
    model = (
        cq.Workplane("XY")
        .box(
            length,
            width,
            height
        )
    )

    # Export as STEP
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cq.exporters.export(
        model,
        output_path
    )

    print("CAD model generated successfully!")
    print("Length:", length)
    print("Width:", width)
    print("Height:", height)
    print("Saved at:", output_path)

    return model


if __name__ == "__main__":

    generate_block()
