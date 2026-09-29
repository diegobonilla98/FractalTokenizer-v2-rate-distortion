import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fractaltokenizer_v2.classification import ClassifierConfig, make_classifier
from fractaltokenizer_v2.regions import render_token_tiles, sample_region_tiles


validation_root = Path("/dev/shm/ImageNet100/val")
checkpoint_path = Path("runs/imagenet100_comparison/fractal_128_seed42/best.pt")
output_root = Path("runs/imagenet100_comparison/fractal_token_visualizations")
candidate_images_per_class = 2
selected_image_count = 8
image_size = 256
token_count = 128
random_seed = 42


resize_transform = transforms.Compose(
    [
        transforms.Resize(image_size + 32, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(path):
    with Image.open(path) as image:
        return resize_transform(image.convert("RGB"))


def candidate_paths():
    paths = []
    for class_root in sorted(path for path in validation_root.iterdir() if path.is_dir()):
        images = sorted(path for path in class_root.iterdir() if path.suffix.lower() in {".jpeg", ".jpg", ".png"})
        positions = np.linspace(0, len(images) - 1, candidate_images_per_class, dtype=int)
        paths.extend(images[position] for position in positions)
    return paths


def score_candidates(model, paths, device):
    records = []
    batch_size = 32
    edge_mask = torch.ones(32, 32, device=device, dtype=torch.bool)
    edge_mask[8:24, 8:24] = False
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        raw = torch.stack([load_image(path) for path in batch_paths]).to(device)
        normalized = (raw - mean.to(device)) / std.to(device)
        density = model.token_embed.density(raw)
        partition = model.token_embed.partitioner(density, target_tokens=token_count)
        probabilities = density[:, 0].flatten(1)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1) / math.log(probabilities.shape[1])
        density_cv = density[:, 0].flatten(1).std(dim=1, unbiased=False) / density[:, 0].flatten(1).mean(dim=1).clamp_min(1e-8)
        widths = partition.boxes[..., 2] - partition.boxes[..., 0]
        heights = partition.boxes[..., 3] - partition.boxes[..., 1]
        areas = widths * heights
        area_cv = areas.std(dim=1, unbiased=False) / areas.mean(dim=1).clamp_min(1e-8)
        maximum_area_ratio = areas.max(dim=1).values / areas.mean(dim=1).clamp_min(1e-8)
        edge_fraction = density[:, 0, edge_mask].sum(dim=1) / density[:, 0].sum(dim=(1, 2)).clamp_min(1e-8)
        contrast = raw.flatten(1).std(dim=1, unbiased=False)
        feature_rows = torch.stack(
            [entropy, density_cv, area_cv, maximum_area_ratio, edge_fraction, contrast],
            dim=1,
        ).cpu()
        for path, features in zip(batch_paths, feature_rows):
            records.append(
                {
                    "path": str(path),
                    "synset": path.parent.name,
                    "features": features.tolist(),
                }
            )
    return records


def select_records(records):
    targets = [
        (2, True, "most adaptive token sizes"),
        (2, False, "most uniform token sizes"),
        (1, True, "highest density contrast"),
        (1, False, "lowest density contrast"),
        (4, True, "edge-distributed information"),
        (4, False, "center-concentrated information"),
        (0, True, "most spatially diffuse information"),
        (0, False, "most spatially concentrated information"),
    ]
    selected = []
    used_synsets = set()
    used_paths = set()
    for feature_index, descending, reason in targets:
        ordered = sorted(records, key=lambda record: record["features"][feature_index], reverse=descending)
        record = next(
            record
            for record in ordered
            if record["synset"] not in used_synsets and record["path"] not in used_paths
        )
        record = {**record, "selection_reason": reason}
        selected.append(record)
        used_synsets.add(record["synset"])
        used_paths.add(record["path"])
        if len(selected) == selected_image_count:
            break
    return selected


def token_outputs(model, selected, device):
    raw = torch.stack([load_image(Path(record["path"])) for record in selected]).to(device)
    normalized = (raw - mean.to(device)) / std.to(device)
    density = model.token_embed.density(raw)
    partition = model.token_embed.partitioner(density, target_tokens=token_count)
    content, geometry, _, _ = model.token_embed.region_encoder(
        normalized,
        boxes=partition.boxes,
        depth=partition.depth,
        mask=partition.mask,
    )
    latents = model.token_embed.norm(content + geometry)
    tiles = sample_region_tiles(raw, partition.boxes, model.config.tile_size)
    reconstruction = render_token_tiles(tiles, partition.boxes, partition.assignment, (image_size, image_size))
    centered = latents.reshape(-1, latents.shape[-1]).float()
    centered = centered - centered.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False)
    colors = centered @ basis[:, :3]
    low = torch.quantile(colors, 0.02, dim=0, keepdim=True)
    high = torch.quantile(colors, 0.98, dim=0, keepdim=True)
    colors = ((colors - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)
    colors = colors.reshape(len(selected), token_count, 3)
    assignment = F.interpolate(
        partition.assignment.unsqueeze(1).float(),
        size=(image_size, image_size),
        mode="nearest",
    )[:, 0].long()
    learned_maps = torch.stack(
        [colors[index][assignment[index]].permute(2, 0, 1) for index in range(len(selected))]
    )
    density_maps = F.interpolate(density, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return {
        "raw": raw.cpu(),
        "density": density.cpu(),
        "density_maps": density_maps.cpu(),
        "partition": partition.to("cpu"),
        "reconstruction": reconstruction.clamp(0.0, 1.0).cpu(),
        "learned_maps": learned_maps.cpu(),
    }


def image_array(tensor):
    return tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)


def draw_adaptive_boxes(axis, boxes, color="white", line_width=0.45):
    for box in boxes:
        x0, y0, x1, y1 = box.tolist()
        axis.add_patch(
            Rectangle(
                (x0 * image_size, y0 * image_size),
                (x1 - x0) * image_size,
                (y1 - y0) * image_size,
                fill=False,
                edgecolor=color,
                linewidth=line_width,
                alpha=0.9,
            )
        )


def draw_fixed_grid(axis):
    for x in range(0, image_size + 1, 32):
        axis.axvline(x - 0.5, color="white", linewidth=0.45, alpha=0.85)
    for y in range(0, image_size + 1, 16):
        axis.axhline(y - 0.5, color="white", linewidth=0.45, alpha=0.85)


def save_individual_plots(selected, outputs):
    individual_root = output_root / "individual"
    original_root = output_root / "selected_originals"
    individual_root.mkdir(parents=True, exist_ok=True)
    original_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(selected):
        raw = outputs["raw"][index]
        density = outputs["density_maps"][index, 0]
        boxes = outputs["partition"].boxes[index]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        masses = outputs["partition"].mass[index]
        number = index + 1
        stem = f"{number:02d}_{record['synset']}"
        Image.fromarray((image_array(raw) * 255).round().astype(np.uint8)).save(original_root / f"{stem}.png")
        figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        axes[0, 0].imshow(image_array(raw))
        axes[0, 0].set_title("Validation image")
        axes[0, 1].imshow(image_array(raw))
        axes[0, 1].imshow(density.numpy(), cmap="magma", alpha=0.62)
        axes[0, 1].set_title("Information density")
        axes[0, 2].imshow(image_array(raw))
        draw_adaptive_boxes(axes[0, 2], boxes)
        axes[0, 2].set_title("Adaptive 128-token mosaic")
        axes[1, 0].imshow(image_array(outputs["learned_maps"][index]))
        draw_adaptive_boxes(axes[1, 0], boxes, color="black", line_width=0.35)
        axes[1, 0].set_title("Learned token features, PCA color")
        axes[1, 1].imshow(image_array(outputs["reconstruction"][index]))
        draw_adaptive_boxes(axes[1, 1], boxes)
        axes[1, 1].set_title("Canonical 8×8 token-tile rendering")
        axes[1, 2].imshow(image_array(raw))
        draw_fixed_grid(axes[1, 2])
        axes[1, 2].set_title("Fixed 16×32 patches, 128 tokens")
        for axis in axes.flat:
            axis.set_xlim(-0.5, image_size - 0.5)
            axis.set_ylim(image_size - 0.5, -0.5)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"{record['synset']} · {record['selection_reason']}\n"
            f"token area CV {areas.std(unbiased=False) / areas.mean():.2f} · "
            f"area range {areas.min() * 100:.2f}–{areas.max() * 100:.2f}% · "
            f"information-mass CV {masses.std(unbiased=False) / masses.mean():.2f}",
            fontsize=15,
        )
        figure.savefig(individual_root / f"{stem}.png", dpi=160)
        plt.close(figure)
        manifest.append(
            {
                **record,
                "figure": str(individual_root / f"{stem}.png"),
                "original": str(original_root / f"{stem}.png"),
                "token_area_cv": float(areas.std(unbiased=False) / areas.mean()),
                "minimum_token_area_fraction": float(areas.min()),
                "maximum_token_area_fraction": float(areas.max()),
                "information_mass_cv": float(masses.std(unbiased=False) / masses.mean()),
            }
        )
    return manifest


def save_overview(selected, outputs):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        axis.imshow(image_array(outputs["raw"][index]))
        draw_adaptive_boxes(axis, outputs["partition"].boxes[index], line_width=0.55)
        axis.set_title(f"{selected[index]['synset']}\n{selected[index]['selection_reason']}", fontsize=10)
        axis.set_xlim(-0.5, image_size - 0.5)
        axis.set_ylim(image_size - 0.5, -0.5)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("FractalTokenizer: representative 128-token ImageNet100 validation mosaics", fontsize=16)
    figure.savefig(output_root / "representative_overview.png", dpi=180)
    plt.close(figure)


def save_area_plot(selected, outputs):
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, record in enumerate(selected):
        boxes = outputs["partition"].boxes[index]
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sort().values * 100
        axis.plot(np.arange(1, token_count + 1), areas.numpy(), label=f"{index + 1:02d} {record['synset']}", linewidth=1.5)
    axis.axhline(100 / token_count, color="black", linestyle="--", linewidth=1.2, label="uniform area")
    axis.set_yscale("log")
    axis.set_xlabel("token rank by spatial area")
    axis.set_ylabel("image area per token (%)")
    axis.set_title("Adaptive token-area distributions")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    figure.savefig(output_root / "token_area_distributions.png", dpi=180)
    plt.close(figure)


def main():
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_classifier(ClassifierConfig(variant="fractal_128")).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    paths = candidate_paths()
    with torch.inference_mode():
        records = score_candidates(model, paths, device)
        selected = select_records(records)
        outputs = token_outputs(model, selected, device)
    manifest = save_individual_plots(selected, outputs)
    save_overview(selected, outputs)
    save_area_plot(selected, outputs)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "checkpoint_best_val_top1": float(checkpoint["best_top1"]),
        "candidate_count": len(records),
        "selection_method": "extremes of density entropy, density contrast, token-area variation, and edge information across two validation images per class",
        "selected": manifest,
    }
    temporary = output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output_root / "manifest.json")
    readme = "\n".join(
        [
            "# Fractal token visualizations",
            "",
            "The overview shows eight automatically selected validation images spanning distinct adaptive-allocation regimes.",
            "Each individual panel compares the original image, information density, adaptive mosaic, trained token-feature PCA colors, canonical token-tile rendering, and the fixed 128-patch control.",
            "The white adaptive boundaries are computed by the exact analytic-density and quantile-partition path used by the trained Fractal classifier.",
            "The token-feature colors are a shared three-component PCA projection over the selected images and are only intended to show feature grouping, not RGB reconstruction.",
        ]
    )
    (output_root / "README.md").write_text(readme + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
