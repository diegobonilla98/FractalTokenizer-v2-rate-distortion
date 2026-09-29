from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from fractaltokenizer_v2.classification import ClassifierConfig, count_parameters, make_classifier
from fractaltokenizer_v2.cuda_ops import build_cuda_extension
from fractaltokenizer_v2.partition import QuantileGridPartitioner


data_root = Path("/dev/shm/ImageNet100")
fallback_data_root = Path("/mnt/hdd/ImageNet100")
output_root = Path("/mnt/hdd/fractaltokenizer_v21/density_audit")
sample_count_per_class = 5
loader_batch_size = 32
workers = 8
image_size = 256
token_count = 128
partition_stride = 8
visual_sample_count = 8
timing_batch_size = 32
timing_warmup = 3
timing_repetitions = 7
seed = 42
variants = (
    "fractal_analytic_128",
    "fractal_universal_128",
    "fractal_uniform_128",
)
timing_variants = (*variants, "vit_rect_128")


def boundary_map(assignment: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(assignment, dtype=torch.bool)
    boundary[:, 1:] |= assignment[:, 1:] != assignment[:, :-1]
    boundary[:, :-1] |= assignment[:, 1:] != assignment[:, :-1]
    boundary[:, :, 1:] |= assignment[:, :, 1:] != assignment[:, :, :-1]
    boundary[:, :, :-1] |= assignment[:, :, 1:] != assignment[:, :, :-1]
    return boundary


def token_values(values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(values.shape[0], token_count, device=values.device, dtype=values.dtype)
    result.scatter_add_(1, assignment.flatten(1), values[:, 0].flatten(1))
    return result


def coefficient_of_variation(values: torch.Tensor) -> torch.Tensor:
    return values.std(dim=1, unbiased=False) / values.mean(dim=1).clamp_min(1e-8)


def representative_indices(dataset: datasets.ImageFolder) -> list[int]:
    by_class = [[] for _ in dataset.classes]
    for index, (_, label) in enumerate(dataset.samples):
        by_class[label].append(index)
    selected_by_class = []
    for indices in by_class:
        positions = np.linspace(0, len(indices) - 1, sample_count_per_class, dtype=np.int64)
        selected_by_class.append([indices[int(position)] for position in positions])
    return [
        selected_by_class[class_index][position]
        for position in range(sample_count_per_class)
        for class_index in range(len(selected_by_class))
    ]


def create_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models = {}
    reference_state = None
    for variant in variants:
        torch.manual_seed(seed)
        model = make_classifier(
            ClassifierConfig(
                variant=variant,
                image_size=image_size,
                token_count=token_count,
                model_dim=384,
                depth=8,
                num_heads=6,
                partition_stride=partition_stride,
                density_temperature=1.0,
                density_coherence_floor=0.35,
                density_blur_sigma=0.0,
                kernel_backend="cuda",
                compile_density=False,
            )
        ).to(device).eval()
        if reference_state is None:
            reference_state = model.state_dict()
        else:
            model.load_state_dict(reference_state, strict=True)
        models[variant] = model
    return models


def add_partition_panel(axis, image: np.ndarray, density: np.ndarray, boxes: np.ndarray, title: str) -> None:
    low = float(np.quantile(density, 0.02))
    high = float(np.quantile(density, 0.98))
    normalized = np.clip((density - low) / max(high - low, 1e-8), 0.0, 1.0)
    normalized = np.asarray(
        F.interpolate(
            torch.from_numpy(normalized).view(1, 1, *normalized.shape),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    )
    axis.imshow(image)
    axis.imshow(normalized, cmap="magma", alpha=0.42, vmin=0.0, vmax=1.0)
    for x0, y0, x1, y1 in boxes:
        axis.add_patch(
            Rectangle(
                (x0 * image_size, y0 * image_size),
                (x1 - x0) * image_size,
                (y1 - y0) * image_size,
                fill=False,
                edgecolor="white",
                linewidth=0.32,
                alpha=0.82,
            )
        )
    axis.set_title(title)
    axis.axis("off")


def save_visuals(records: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(len(records), 4, figsize=(16, 3.8 * len(records)))
    for row, record in enumerate(records):
        axes[row, 0].imshow(record["image"])
        axes[row, 0].set_title(record["label"])
        axes[row, 0].axis("off")
        for column, variant in enumerate(variants, start=1):
            add_partition_panel(
                axes[row, column],
                record["image"],
                record[variant]["density"],
                record[variant]["boxes"],
                variant.replace("fractal_", "").replace("_128", ""),
            )
    figure.tight_layout()
    figure.savefig(output_root / "partition_contact_sheet.png", dpi=150)
    plt.close(figure)


def save_distribution_plot(per_image: dict[str, dict[str, list[float]]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [variant.replace("fractal_", "").replace("_128", "") for variant in variants]
    axes[0].boxplot([per_image[variant]["source_mass_cv"] for variant in variants], tick_labels=labels)
    axes[0].set_title("Own density mass CV")
    axes[0].set_ylabel("lower is more constant")
    axes[1].boxplot([per_image[variant]["universal_mass_cv"] for variant in variants], tick_labels=labels)
    axes[1].set_title("Universal codelength mass CV")
    axes[2].boxplot([per_image[variant]["token_area_cv"] for variant in variants], tick_labels=labels)
    axes[2].set_title("Token area CV")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "partition_statistics.png", dpi=160)
    plt.close(figure)


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def benchmark_models(device: torch.device) -> dict[str, dict[str, float]]:
    results = {}
    images = torch.randn(timing_batch_size, 3, image_size, image_size, device=device)
    for variant in timing_variants:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        model = make_classifier(
            ClassifierConfig(
                variant=variant,
                image_size=image_size,
                token_count=token_count,
                model_dim=384,
                depth=8,
                num_heads=6,
                partition_stride=partition_stride,
                density_temperature=1.0,
                density_coherence_floor=0.35,
                density_blur_sigma=0.0,
                kernel_backend="cuda",
                compile_density=True,
            )
        ).to(device).train()
        for _ in range(timing_warmup):
            torch.compiler.cudagraph_mark_step_begin()
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(images).square().mean().backward()
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(timing_repetitions):
            torch.compiler.cudagraph_mark_step_begin()
            model.zero_grad(set_to_none=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(images).square().mean().backward()
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
        trainable, total = count_parameters(model)
        results[variant] = {
            "median_forward_backward_ms": float(np.median(elapsed)),
            "images_per_second": float(timing_batch_size * 1000.0 / np.median(elapsed)),
            "peak_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30),
            "trainable_parameters": trainable,
            "total_parameters": total,
        }
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return results


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    build_cuda_extension()
    root = data_root if data_root.is_dir() else fallback_data_root
    output_root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = datasets.ImageFolder(root / "val", transform=transform)
    indices = representative_indices(dataset)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    models = create_models(device)
    parameter_counts = {variant: count_parameters(model) for variant, model in models.items()}
    state_shapes = {
        variant: {name: list(value.shape) for name, value in model.state_dict().items()}
        for variant, model in models.items()
    }
    per_image = {
        variant: {
            "source_mass_cv": [],
            "universal_mass_cv": [],
            "token_area_cv": [],
            "density_cv": [],
            "top_20_percent_density_mass_fraction": [],
            "low_information_token_count": [],
            "boundary_iou_with_analytic": [],
        }
        for variant in variants
    }
    visual_records = []
    all_masks_valid = True
    all_assignments_valid = True
    cuda_partition_parity = True
    uniform_boxes = []
    adaptive_box_variation = {variant: [] for variant in variants[:2]}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    seen = 0
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="v2.1 density audit", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            raw = (images.float() * std + mean).clamp(0.0, 1.0)
            densities = {variant: models[variant].token_embed.density(raw) for variant in variants}
            partitions = {
                variant: models[variant].token_embed.partitioner(density, target_tokens=token_count)
                for variant, density in densities.items()
            }
            universal_density = densities["fractal_universal_128"]
            analytic_boundary = boundary_map(partitions["fractal_analytic_128"].assignment)
            for variant in variants:
                density = densities[variant]
                partition = partitions[variant]
                torch_partition = QuantileGridPartitioner(
                    density_floor_ratio=0.02,
                    backend="torch",
                )(density, target_tokens=token_count)
                cuda_partition_parity &= bool(torch.equal(partition.assignment, torch_partition.assignment))
                cuda_partition_parity &= bool(torch.equal(partition.mask, torch_partition.mask))
                cuda_partition_parity &= bool(torch.allclose(partition.boxes, torch_partition.boxes, rtol=0.0, atol=0.0))
                unique_counts = torch.tensor(
                    [torch.unique(item).numel() for item in partition.assignment],
                    device=device,
                )
                all_masks_valid &= bool(partition.mask.all())
                all_assignments_valid &= bool(
                    (unique_counts == token_count).all()
                    and partition.assignment.min() == 0
                    and partition.assignment.max() == token_count - 1
                )
                areas = token_values(torch.ones_like(density), partition.assignment)
                source_mass = token_values(density, partition.assignment)
                universal_mass = token_values(universal_density, partition.assignment)
                source_cv = coefficient_of_variation(source_mass)
                universal_cv = coefficient_of_variation(universal_mass)
                area_cv = coefficient_of_variation(areas)
                flat_density = density.flatten(1)
                density_cv = coefficient_of_variation(flat_density)
                top_count = max(1, math.ceil(flat_density.shape[1] * 0.2))
                top_fraction = flat_density.topk(top_count, dim=1).values.sum(dim=1) / flat_density.sum(dim=1)
                threshold = torch.quantile(universal_density.flatten(1), 0.25, dim=1, keepdim=True)
                low_mask = universal_density[:, 0].flatten(1) <= threshold
                low_counts = []
                for batch_index in range(images.shape[0]):
                    low_counts.append(
                        torch.unique(partition.assignment[batch_index].flatten()[low_mask[batch_index]]).numel()
                    )
                current_boundary = boundary_map(partition.assignment)
                intersection = (current_boundary & analytic_boundary).flatten(1).sum(dim=1)
                union = (current_boundary | analytic_boundary).flatten(1).sum(dim=1).clamp_min(1)
                per_image[variant]["source_mass_cv"].extend(source_cv.cpu().tolist())
                per_image[variant]["universal_mass_cv"].extend(universal_cv.cpu().tolist())
                per_image[variant]["token_area_cv"].extend(area_cv.cpu().tolist())
                per_image[variant]["density_cv"].extend(density_cv.cpu().tolist())
                per_image[variant]["top_20_percent_density_mass_fraction"].extend(top_fraction.cpu().tolist())
                per_image[variant]["low_information_token_count"].extend(float(value) for value in low_counts)
                per_image[variant]["boundary_iou_with_analytic"].extend((intersection / union).cpu().tolist())
            uniform_boxes.append(partitions["fractal_uniform_128"].boxes.cpu())
            for variant in adaptive_box_variation:
                adaptive_box_variation[variant].append(partitions[variant].boxes.cpu())
            remaining = visual_sample_count - len(visual_records)
            for batch_index in range(min(remaining, images.shape[0])):
                record = {
                    "image": raw[batch_index].permute(1, 2, 0).cpu().numpy(),
                    "label": dataset.classes[int(labels[batch_index])],
                }
                for variant in variants:
                    record[variant] = {
                        "density": densities[variant][batch_index, 0].cpu().numpy(),
                        "boxes": partitions[variant].boxes[batch_index].cpu().numpy(),
                    }
                visual_records.append(record)
            seen += images.shape[0]
    uniform_boxes_tensor = torch.cat(uniform_boxes)
    uniform_identical = bool(torch.equal(uniform_boxes_tensor, uniform_boxes_tensor[:1].expand_as(uniform_boxes_tensor)))
    adaptive_varies = {
        variant: bool(
            torch.any(torch.cat(boxes) != torch.cat(boxes)[:1].expand_as(torch.cat(boxes)))
        )
        for variant, boxes in adaptive_box_variation.items()
    }
    summarized = {
        variant: {metric: percentile_summary(values) for metric, values in metrics.items()}
        for variant, metrics in per_image.items()
    }
    universal_cv = summarized["fractal_universal_128"]["universal_mass_cv"]["mean"]
    analytic_universal_cv = summarized["fractal_analytic_128"]["universal_mass_cv"]["mean"]
    uniform_universal_cv = summarized["fractal_uniform_128"]["universal_mass_cv"]["mean"]
    boundary_iou = summarized["fractal_universal_128"]["boundary_iou_with_analytic"]["mean"]
    architecture_equal = len(set(parameter_counts.values())) == 1 and len(
        {json.dumps(shapes, sort_keys=True) for shapes in state_shapes.values()}
    ) == 1
    gates = {
        "sample_count_is_500": seen == 500,
        "architecture_and_parameter_counts_equal": architecture_equal,
        "all_variants_have_exactly_128_valid_tokens": all_masks_valid and all_assignments_valid,
        "cuda_and_torch_partitioners_are_exact": cuda_partition_parity,
        "uniform_partition_is_image_independent": uniform_identical,
        "analytic_partition_is_image_adaptive": adaptive_varies["fractal_analytic_128"],
        "universal_partition_is_image_adaptive": adaptive_varies["fractal_universal_128"],
        "universal_signal_improves_universal_mass_balance_over_analytic": universal_cv < analytic_universal_cv,
        "universal_signal_improves_universal_mass_balance_over_uniform": universal_cv < uniform_universal_cv,
        "universal_partition_is_materially_different_from_analytic": boundary_iou < 0.95,
    }
    save_visuals(visual_records)
    save_distribution_plot(per_image)
    timing = benchmark_models(device)
    report = {
        "status": "pass" if all(gates.values()) else "fail",
        "dataset_root": str(root),
        "sample_selection": "five deterministic evenly spaced validation images per class",
        "sample_count": seen,
        "seed": seed,
        "image_size": image_size,
        "token_count": token_count,
        "parameter_counts": parameter_counts,
        "gates": gates,
        "statistics": summarized,
        "timing": timing,
        "artifacts": {
            "contact_sheet": str(output_root / "partition_contact_sheet.png"),
            "statistics_plot": str(output_root / "partition_statistics.png"),
        },
        "scientific_scope": "fixed-token density-only ablation; this does not claim variable-rate compression",
    }
    (output_root / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("v2.1 density audit failed one or more predefined gates")


main()
