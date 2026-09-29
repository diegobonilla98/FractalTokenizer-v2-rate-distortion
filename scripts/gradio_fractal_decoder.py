from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask

from fractaltokenizer_v2.classification import ClassifierConfig, make_classifier
from fractaltokenizer_v2.interactive import (
    colorize_labels,
    controlled_density,
    density_heatmap,
    error_heatmap,
    image_metrics,
    labels_from_mask,
    mean_color_decode,
    merge_adjacent_regions,
    overlay_boundaries,
    pca_token_map,
    resize_center_crop,
    rgb_array,
    segmentation_metrics,
    watershed_refine,
)
from fractaltokenizer_v2.partition import QuantileGridPartitioner
from fractaltokenizer_v2.regions import LocalDCTBasis, render_token_tiles, sample_region_tiles


project_root = Path(__file__).resolve().parents[1]
checkpoint_path = project_root / "runs/imagenet100_comparison/fractal_128_seed42/best.pt"
coco_root = Path("/mnt/hdd/coco2017")
coco_annotation_path = coco_root / "annotations/instances_val2017.json"
coco_image_root = coco_root / "val2017"
image_size = 256
server_name = "0.0.0.0"
server_port = 7860
example_count = 16
warmup_enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
coco_examples = None
image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CocoExamples:
    def __init__(self, annotation_path: Path, image_root: Path, count: int):
        data = json.loads(annotation_path.read_text())
        self.image_root = image_root
        self.images = {int(record["id"]): record for record in data["images"]}
        self.categories = {int(record["id"]): record["name"] for record in data["categories"]}
        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations[int(annotation["image_id"])].append(annotation)
        candidates = []
        for image_id, annotations in self.annotations.items():
            image = self.images[image_id]
            image_area = float(image["width"] * image["height"])
            coverage = sum(float(annotation["area"]) for annotation in annotations) / image_area
            category_count = len({int(annotation["category_id"]) for annotation in annotations})
            object_count = len(annotations)
            if 2 <= object_count <= 20 and 0.08 <= coverage <= 1.25:
                score = min(object_count, 10) + 1.5 * category_count + min(coverage, 1.0)
                candidates.append((score, image_id))
        candidates.sort(reverse=True)
        pool = candidates[: max(count * 10, count)]
        positions = np.linspace(0, len(pool) - 1, count, dtype=int)
        self.selected_ids = [pool[position][1] for position in positions]
        self.labels = []
        self.label_to_id = {}
        for image_id in self.selected_ids:
            image = self.images[image_id]
            annotations = self.annotations[image_id]
            category_names = sorted({self.categories[int(annotation["category_id"])] for annotation in annotations})
            category_text = ", ".join(category_names[:4])
            label = f"{image['file_name']} · {len(annotations)} instances · {category_text}"
            self.labels.append(label)
            self.label_to_id[label] = image_id

    def load(self, label: str) -> tuple[np.ndarray, np.ndarray, str]:
        image_id = self.label_to_id[label]
        image_record = self.images[image_id]
        with Image.open(self.image_root / image_record["file_name"]) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        height = int(image_record["height"])
        width = int(image_record["width"])
        labels = np.zeros((height, width), dtype=np.int32)
        annotations = sorted(self.annotations[image_id], key=lambda annotation: float(annotation["area"]), reverse=True)
        for instance_index, annotation in enumerate(annotations, start=1):
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list):
                encoded = coco_mask.frPyObjects(segmentation, height, width)
                encoded = coco_mask.merge(encoded)
            else:
                encoded = segmentation
            decoded = coco_mask.decode(encoded).astype(bool)
            labels[decoded] = instance_index
        image = resize_center_crop(image, image_size)
        labels = resize_center_crop(labels, image_size, nearest=True).astype(np.int32)
        mask_image = colorize_labels(labels)
        categories = sorted({self.categories[int(annotation["category_id"])] for annotation in annotations})
        status = (
            f"Loaded COCO val image `{image_record['file_name']}` with {len(annotations)} instance masks "
            f"across {len(categories)} categories."
        )
        return image, mask_image, status


def synchronize() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_image(image: np.ndarray) -> torch.Tensor:
    values = torch.from_numpy(rgb_array(image)).permute(2, 0, 1).float() / 255.0
    return values.unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    values = tensor[0].detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (values * 255.0).round().astype(np.uint8)


def load_model():
    classifier = make_classifier(ClassifierConfig(variant="fractal_128")).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(checkpoint["model"])
    return classifier


def load_example(label: str):
    return coco_examples.load(label)


def analyze(
    image,
    segmentation_mask,
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
):
    if image is None:
        raise gr.Error("Upload an image or load a COCO example first.")
    source = resize_center_crop(rgb_array(image), image_size)
    target_labels = labels_from_mask(segmentation_mask, image_size)
    raw = tensor_image(source)
    normalized = (raw - image_mean.to(device)) / image_std.to(device)
    token_budget = int(token_budget)
    dct_bandwidth = int(dct_bandwidth)
    synchronize()
    density_start = perf_counter()
    with torch.inference_mode():
        base_density = model.token_embed.density(raw)
        density = controlled_density(
            base_density,
            raw,
            float(density_temperature),
            float(density_blur),
            float(image_edge_weight),
            target_labels,
            float(segmentation_edge_weight),
        )
    synchronize()
    density_ms = (perf_counter() - density_start) * 1000.0
    partition_start = perf_counter()
    with torch.inference_mode():
        partitioner = QuantileGridPartitioner(density_floor_ratio=float(density_floor)).to(device)
        partition = partitioner(density, target_tokens=token_budget)
    synchronize()
    partition_ms = (perf_counter() - partition_start) * 1000.0
    decode_start = perf_counter()
    with torch.inference_mode():
        tiles = sample_region_tiles(raw, partition.boxes, output_size=8)
        basis = LocalDCTBasis(tile_size=8, basis_size=dct_bandwidth).to(device)
        coefficients = basis.encode(tiles)
        dct_tiles = basis.decode(coefficients)
        dct_tensor = render_token_tiles(
            dct_tiles,
            boxes=partition.boxes,
            assignment=partition.assignment,
            image_shape=(image_size, image_size),
        ).clamp(0.0, 1.0)
        tile_tensor = render_token_tiles(
            tiles,
            boxes=partition.boxes,
            assignment=partition.assignment,
            image_shape=(image_size, image_size),
        ).clamp(0.0, 1.0)
        content, geometry, _, _ = model.token_embed.region_encoder(
            normalized,
            boxes=partition.boxes,
            depth=partition.depth,
            mask=partition.mask,
        )
        latents = model.token_embed.norm(content + geometry)
    synchronize()
    decode_ms = (perf_counter() - decode_start) * 1000.0
    dct_image = tensor_to_image(dct_tensor)
    tile_image = tensor_to_image(tile_tensor)
    assignment = F.interpolate(
        partition.assignment.unsqueeze(1).float(),
        size=(image_size, image_size),
        mode="nearest",
    )[0, 0].long().cpu().numpy()
    density_values = F.interpolate(
        density,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    feature_map = pca_token_map(latents, assignment)
    guide = dct_image if postprocess_guide == "DCT reconstruction" else source
    postprocess_start = perf_counter()
    refined = watershed_refine(
        assignment,
        guide,
        int(border_snap_radius),
        float(border_compactness),
    )
    clustered, effective_tokens = merge_adjacent_regions(refined, guide, float(merge_tolerance))
    clustered_decode = mean_color_decode(clustered, guide)
    postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
    dct_psnr, dct_ssim = image_metrics(source, dct_image)
    tile_psnr, tile_ssim = image_metrics(source, tile_image)
    cluster_psnr, cluster_ssim = image_metrics(source, clustered_decode)
    segmentation = segmentation_metrics(clustered, target_labels)
    boxes = partition.boxes[0].float().cpu()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    masses = partition.mass[0].float().cpu()
    coefficient_values = token_budget * 3 * dct_bandwidth * dct_bandwidth
    attention_proxy = (token_budget / 128.0) ** 2
    metrics = {
        "pipeline": {
            "density_temperature": float(density_temperature),
            "density_blur_sigma": float(density_blur),
            "image_edge_weight": float(image_edge_weight),
            "segmentation_edge_weight_oracle": float(segmentation_edge_weight),
            "density_floor_ratio": float(density_floor),
            "border_snap_radius": int(border_snap_radius),
            "border_compactness": float(border_compactness),
            "adjacent_merge_tolerance": float(merge_tolerance),
            "postprocess_guide": postprocess_guide,
        },
        "requested_tokens": token_budget,
        "effective_regions_after_merge": effective_tokens,
        "dct_bandwidth": dct_bandwidth,
        "dct_coefficient_values": coefficient_values,
        "dct_psnr_db": round(dct_psnr, 4),
        "dct_ssim": round(dct_ssim, 6),
        "source_assisted_tile_psnr_db": round(tile_psnr, 4),
        "source_assisted_tile_ssim": round(tile_ssim, 6),
        "clustered_mean_psnr_db": round(cluster_psnr, 4),
        "clustered_mean_ssim": round(cluster_ssim, 6),
        "token_area_cv": round(float(areas.std(unbiased=False) / areas.mean()), 6),
        "information_mass_cv": round(float(masses.std(unbiased=False) / masses.mean()), 6),
        "density_ms": round(density_ms, 3),
        "partition_ms": round(partition_ms, 3),
        "region_encode_and_decode_ms": round(decode_ms, 3),
        "cpu_border_postprocess_ms": round(postprocess_ms, 3),
        "generation_attention_proxy_vs_128_tokens": round(attention_proxy, 4),
        **{name: round(value, 6) for name, value in segmentation.items()},
    }
    segmentation_text = ""
    if segmentation:
        segmentation_text = (
            f" · GT boundary F1 **{segmentation['boundary_f1']:.3f}**"
            f" · region purity **{segmentation['region_purity']:.3f}**"
        )
    metrics_text = (
        f"**{token_budget} rectangular tokens → {effective_tokens} merged regions** · "
        f"DCT {dct_bandwidth}×{dct_bandwidth}: **{dct_psnr:.2f} dB / {dct_ssim:.3f} SSIM** · "
        f"encode+decode **{density_ms + partition_ms + decode_ms:.1f} ms** · "
        f"border postprocess **{postprocess_ms:.1f} ms**{segmentation_text}\n\n"
        "The DCT panel is a coefficient round trip. The 8×8 tile panel is source-assisted and is only an upper bound. "
        "The trained classifier checkpoint has no learned neural image decoder."
    )
    original_overlay = overlay_boundaries(source, assignment, color=(0, 255, 255), thickness=1)
    refined_overlay = overlay_boundaries(source, clustered, color=(0, 255, 128), thickness=1)
    feature_overlay = overlay_boundaries(feature_map, assignment, color=(255, 255, 255), thickness=1)
    target_overlay = None if target_labels is None else overlay_boundaries(source, target_labels, color=(255, 64, 64), thickness=2)
    return (
        original_overlay,
        density_heatmap(density_values, source),
        dct_image,
        tile_image,
        feature_overlay,
        refined_overlay,
        clustered_decode,
        error_heatmap(source, dct_image),
        target_overlay,
        metrics_text,
        metrics,
    )


def build_app():
    with gr.Blocks(title="FractalTokenizer segmentation and decoder lab") as demo:
        gr.Markdown(
            "# FractalTokenizer segmentation + decoding lab\n"
            "Explore token count, density temperature, edge allocation, DCT bandwidth, boundary snapping, and adjacent-region merging. "
            "COCO instance masks are optional oracle guidance/evaluation; leave segmentation edge weight at zero for a deployable image-only pipeline."
        )
        with gr.Row():
            with gr.Column(scale=1):
                example = gr.Dropdown(coco_examples.labels, value=coco_examples.labels[0], label="COCO 2017 validation example")
                load_button = gr.Button("Load COCO example", variant="secondary")
                example_status = gr.Markdown()
                image_input = gr.Image(type="numpy", image_mode="RGB", label="Image")
                mask_input = gr.Image(type="numpy", image_mode="RGB", label="Optional segmentation labels")
                analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                token_budget = gr.Slider(16, 256, value=128, step=8, label="Token budget")
                density_temperature = gr.Slider(0.2, 3.0, value=1.0, step=0.05, label="Density temperature (<1 is more aggressive)")
                density_blur = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Density blur sigma")
                image_edge_weight = gr.Slider(0.0, 4.0, value=0.0, step=0.1, label="Image edge allocation weight")
                segmentation_edge_weight = gr.Slider(0.0, 4.0, value=0.0, step=0.1, label="GT segmentation edge allocation (oracle)")
                density_floor = gr.Slider(0.0, 0.25, value=0.02, step=0.005, label="Uniform density floor")
                dct_bandwidth = gr.Slider(1, 8, value=4, step=1, label="DCT bandwidth per axis")
                border_snap_radius = gr.Slider(0, 12, value=3, step=1, label="Edge-snap radius")
                border_compactness = gr.Slider(0.0, 0.05, value=0.002, step=0.001, label="Watershed compactness")
                merge_tolerance = gr.Slider(0.0, 0.3, value=0.0, step=0.005, label="Adjacent-region merge tolerance")
                postprocess_guide = gr.Radio(
                    ["DCT reconstruction", "Original image (oracle)"],
                    value="DCT reconstruction",
                    label="Border/merge guide",
                )
        metrics_text = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("Core decoding"):
                with gr.Row():
                    original_overlay = gr.Image(label="Adaptive rectangular tokens")
                    density_output = gr.Image(label="Controlled information density")
                    dct_output = gr.Image(label="DCT coefficient decode")
                with gr.Row():
                    tile_output = gr.Image(label="Source-assisted 8×8 tile upper bound")
                    error_output = gr.Image(label="DCT absolute error")
                    feature_output = gr.Image(label="Trained classifier token features")
            with gr.Tab("Segmentation postprocess"):
                with gr.Row():
                    refined_overlay = gr.Image(label="Edge-snapped and merged borders")
                    clustered_decode = gr.Image(label="Postprocessed region-mean decode")
                    target_overlay = gr.Image(label="Ground-truth instance borders")
            with gr.Tab("Metrics"):
                metrics_json = gr.JSON(label="Measurements")
        inputs = [
            image_input,
            mask_input,
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
        ]
        outputs = [
            original_overlay,
            density_output,
            dct_output,
            tile_output,
            feature_output,
            refined_overlay,
            clustered_decode,
            error_output,
            target_overlay,
            metrics_text,
            metrics_json,
        ]
        load_button.click(load_example, inputs=[example], outputs=[image_input, mask_input, example_status])
        analyze_button.click(analyze, inputs=inputs, outputs=outputs, concurrency_limit=1, trigger_mode="always_last")
        for control in [
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
        ]:
            control.release(analyze, inputs=inputs, outputs=outputs, concurrency_limit=1, trigger_mode="always_last")
        postprocess_guide.change(analyze, inputs=inputs, outputs=outputs, concurrency_limit=1, trigger_mode="always_last")
    return demo


def main():
    global model, coco_examples
    model = load_model()
    coco_examples = CocoExamples(coco_annotation_path, coco_image_root, example_count)
    if warmup_enabled:
        warmup_image, warmup_mask, _ = coco_examples.load(coco_examples.labels[0])
        analyze(
            warmup_image,
            warmup_mask,
            64,
            1.0,
            0.0,
            0.0,
            0.0,
            0.02,
            4,
            2,
            0.002,
            0.0,
            "DCT reconstruction",
        )
    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=4)
    app.launch(server_name=server_name, server_port=server_port, share=False, show_error=True)


if __name__ == "__main__":
    main()
