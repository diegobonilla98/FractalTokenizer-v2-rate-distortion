import json
from pathlib import Path

from PIL import Image

import gradio_fractal_decoder as decoder_app


output_root = Path("runs/segmentation_decoder_lab_smoke")
token_budget = 64
density_temperature = 0.7
density_blur = 0.5
image_edge_weight = 1.0
segmentation_edge_weight = 0.0
density_floor = 0.02
dct_bandwidth = 4
border_snap_radius = 3
border_compactness = 0.002
merge_tolerance = 0.02
postprocess_guide = "DCT reconstruction"
warmup_runs = 1


def main():
    output_root.mkdir(parents=True, exist_ok=True)
    decoder_app.model = decoder_app.load_model()
    decoder_app.coco_examples = decoder_app.CocoExamples(
        decoder_app.coco_annotation_path,
        decoder_app.coco_image_root,
        decoder_app.example_count,
    )
    decoder_app.build_app()
    example_label = decoder_app.coco_examples.labels[0]
    image, mask, status = decoder_app.load_example(example_label)
    parameters = (
        image,
        mask,
        token_budget,
        density_temperature,
        density_blur,
        image_edge_weight,
        segmentation_edge_weight,
        density_floor,
        dct_bandwidth,
        border_snap_radius,
        border_compactness,
        merge_tolerance,
        postprocess_guide,
    )
    for _ in range(warmup_runs):
        decoder_app.analyze(*parameters)
    outputs = decoder_app.analyze(*parameters)
    names = [
        "01_rectangular_tokens",
        "02_controlled_density",
        "03_dct_decode",
        "04_tile_upper_bound",
        "05_classifier_features",
        "06_refined_borders",
        "07_clustered_mean_decode",
        "08_dct_error",
        "09_ground_truth",
    ]
    for name, values in zip(names, outputs[:9]):
        if values is not None:
            Image.fromarray(values).save(output_root / f"{name}.png")
    report = {
        "example": example_label,
        "status": status,
        "summary": outputs[9],
        "metrics": outputs[10],
    }
    (output_root / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
