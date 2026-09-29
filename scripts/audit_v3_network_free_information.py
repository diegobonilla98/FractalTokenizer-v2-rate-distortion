import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from fractaltokenizer_v2.cuda_ops import build_cuda_extension
from fractaltokenizer_v2.density import AnalyticInformationDensity
from fractaltokenizer_v2.partition import QuantileGridPartitioner
from fractaltokenizer_v3 import NetworkFreeInformationTokenizer, PredictiveInformationDensity, calibrate_capacity


DATASET_ROOT = Path("/mnt/hdd/ImageNet100")
OUTPUT_ROOT = Path("/mnt/hdd/fractaltokenizer_v3_clean/audit")
CALIBRATION_IMAGES = 2048
VALIDATION_IMAGES = 500
TARGET_MEAN_TOKENS = [64, 96]
MAXIMUM_TOKENS = 128
PARTITION_STRIDE = 8
RESIDUAL_SCALE = 1.0 / 255.0
SCAN_DIRECTIONS = 4
BATCH_SIZE = 64
WORKERS = 8
SEED = 42
VISUAL_IMAGES = 8


def balanced_indices(dataset: datasets.ImageFolder, count: int) -> list[int]:
    by_class = {}
    for index, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(index)
    selected = []
    per_class = math.ceil(count / len(by_class))
    positions_by_class = {
        label: np.linspace(0, len(values) - 1, per_class, dtype=np.int64)
        for label, values in by_class.items()
    }
    for position_index in range(per_class):
        for label in sorted(by_class):
            position = positions_by_class[label][position_index]
            selected.append(by_class[label][position])
            if len(selected) == count:
                return selected
    return selected[:count]


def collect_totals(
    dataset: datasets.ImageFolder,
    indices: list[int],
    density: PredictiveInformationDensity,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
    )
    totals = []
    for images, _ in tqdm(loader, desc="v3 calibration", leave=False):
        images = images.to(device, non_blocking=True)
        totals.append(density(images).sum(dim=(1, 2, 3)).cpu())
    return torch.cat(totals)


def mass_cv(mass: torch.Tensor, mask: torch.Tensor) -> list[float]:
    values = []
    for image_mass, image_mask in zip(mass, mask):
        active = image_mass[image_mask]
        mean = active.mean()
        values.append(float((active.std(unbiased=False) / mean.clamp_min(1e-8)).item()))
    return values


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def is_regular_grid(output, index: int) -> bool:
    boxes = output.partition.boxes_grid[index, output.partition.mask[index]]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return bool(widths.unique().numel() == 1 and heights.unique().numel() == 1)


def boundary_map(assignment: torch.Tensor, image_size: int) -> np.ndarray:
    grid = assignment.detach().cpu()
    expanded = F.interpolate(
        grid.float().view(1, 1, *grid.shape),
        size=(image_size, image_size),
        mode="nearest",
    )[0, 0].to(torch.int64)
    boundary = torch.zeros_like(expanded, dtype=torch.bool)
    boundary[:, 1:] |= expanded[:, 1:] != expanded[:, :-1]
    boundary[1:, :] |= expanded[1:, :] != expanded[:-1, :]
    return boundary.numpy()


def render_partition(axis, image: torch.Tensor, assignment: torch.Tensor, title: str) -> None:
    array = image.permute(1, 2, 0).cpu().numpy()
    boundary = boundary_map(assignment, array.shape[0])
    overlay = array.copy()
    overlay[boundary] = np.array([1.0, 0.1, 0.1])
    axis.imshow(overlay)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def synthetic_images() -> tuple[list[str], torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    uniform = torch.full((3, 256, 256), 0.35)
    ramp = torch.linspace(0.15, 0.85, 256).view(1, 1, 256).expand(3, 256, 256)
    eighth_noise = uniform.clone()
    eighth_noise[:, :, 224:] = (0.5 + 0.2 * torch.randn(3, 256, 32, generator=generator)).clamp(0.0, 1.0)
    half_noise = uniform.clone()
    half_noise[:, :, 128:] = (0.5 + 0.2 * torch.randn(3, 256, 128, generator=generator)).clamp(0.0, 1.0)
    gaussian = (0.5 + 0.2 * torch.randn(3, 256, 256, generator=generator)).clamp(0.0, 1.0)
    return ["uniform", "linear ramp", "1/8 noise", "1/2 noise", "Gaussian noise"], torch.stack(
        (uniform, ramp, eighth_noise, half_noise, gaussian)
    )


def benchmark(tokenizer: NetworkFreeInformationTokenizer, images: torch.Tensor) -> dict[str, float]:
    for _ in range(10):
        tokenizer(images)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(50):
        start.record()
        tokenizer(images)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    median = float(np.median(times))
    return {
        "batch_size": int(images.shape[0]),
        "median_milliseconds": median,
        "images_per_second": float(images.shape[0] * 1000.0 / median),
    }


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_cuda_extension()
    transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=transform)
    validation_dataset = datasets.ImageFolder(DATASET_ROOT / "val", transform=transform)
    density = PredictiveInformationDensity(
        partition_stride=PARTITION_STRIDE,
        residual_scale=RESIDUAL_SCALE,
        scan_directions=SCAN_DIRECTIONS,
    ).to(device)
    calibration_cache = OUTPUT_ROOT / "calibration_totals_v2_global_mean_boundary.pt"
    if calibration_cache.exists():
        calibration_totals = torch.load(calibration_cache, map_location="cpu", weights_only=True)
    else:
        calibration_indices = balanced_indices(train_dataset, CALIBRATION_IMAGES)
        calibration_totals = collect_totals(train_dataset, calibration_indices, density, device)
        torch.save(calibration_totals, calibration_cache)
    capacities = {
        target: calibrate_capacity(calibration_totals, target, MAXIMUM_TOKENS)
        for target in TARGET_MEAN_TOKENS
    }
    validation_indices = balanced_indices(validation_dataset, VALIDATION_IMAGES)
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
    )
    tokenizers = {
        target: NetworkFreeInformationTokenizer(
            capacity_bits=capacity,
            maximum_tokens=MAXIMUM_TOKENS,
            partition_stride=PARTITION_STRIDE,
            residual_scale=RESIDUAL_SCALE,
            scan_directions=SCAN_DIRECTIONS,
            backend="cuda",
        ).to(device)
        for target, capacity in capacities.items()
    }
    profile_values = {
        target: {"token_count": [], "mass_cv": [], "cap_hit": []}
        for target in TARGET_MEAN_TOKENS
    }
    validation_totals = []
    visual_batch = None
    for images, labels in tqdm(validation_loader, desc="v3 validation", leave=False):
        images = images.to(device, non_blocking=True)
        if visual_batch is None:
            visual_batch = (images[:VISUAL_IMAGES].detach().cpu(), labels[:VISUAL_IMAGES].clone())
        for target, tokenizer in tokenizers.items():
            output = tokenizer(images)
            profile_values[target]["token_count"].extend(output.token_count.cpu().tolist())
            profile_values[target]["mass_cv"].extend(mass_cv(output.partition.mass, output.partition.mask))
            profile_values[target]["cap_hit"].extend((output.token_count == MAXIMUM_TOKENS).cpu().tolist())
            if target == TARGET_MEAN_TOKENS[0]:
                validation_totals.extend(output.total_information.cpu().tolist())
    names, synthetic = synthetic_images()
    synthetic = synthetic.to(device)
    synthetic_outputs = {target: tokenizer(synthetic) for target, tokenizer in tokenizers.items()}
    analytic_density = AnalyticInformationDensity(
        partition_stride=PARTITION_STRIDE,
        temperature=0.4,
        coherence_floor=0.35,
        blur_sigma=0.0,
    ).to(device)
    analytic_partitioner = QuantileGridPartitioner(backend="cuda")
    visual_images, visual_labels = visual_batch
    visual_gpu = visual_images.to(device)
    visual_outputs = {target: tokenizer(visual_gpu) for target, tokenizer in tokenizers.items()}
    analytic_values = analytic_density(visual_gpu)
    analytic_output = analytic_partitioner(analytic_values, target_tokens=128)
    figure, axes = plt.subplots(VISUAL_IMAGES, 5, figsize=(17, 3.2 * VISUAL_IMAGES))
    for row in range(VISUAL_IMAGES):
        axes[row, 0].imshow(visual_images[row].permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"ImageNet100 class {int(visual_labels[row])}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(visual_outputs[96].density[row, 0].cpu().numpy(), cmap="magma")
        axes[row, 1].set_title("predictive information", fontsize=9)
        axes[row, 1].axis("off")
        render_partition(
            axes[row, 2],
            visual_images[row],
            visual_outputs[64].partition.assignment[row],
            f"v3-64 profile: {int(visual_outputs[64].token_count[row])} tokens",
        )
        render_partition(
            axes[row, 3],
            visual_images[row],
            visual_outputs[96].partition.assignment[row],
            f"v3-96 profile: {int(visual_outputs[96].token_count[row])} tokens",
        )
        render_partition(axes[row, 4], visual_images[row], analytic_output.assignment[row], "v2 analytic: 128 tokens")
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "imagenet100_comparison.png", dpi=170)
    plt.close(figure)
    synthetic_figure, synthetic_axes = plt.subplots(len(names), 4, figsize=(14, 3.2 * len(names)))
    for row, name in enumerate(names):
        synthetic_axes[row, 0].imshow(synthetic[row].permute(1, 2, 0).cpu().numpy())
        synthetic_axes[row, 0].set_title(name)
        synthetic_axes[row, 0].axis("off")
        synthetic_axes[row, 1].imshow(synthetic_outputs[96].density[row, 0].cpu().numpy(), cmap="magma")
        synthetic_axes[row, 1].set_title(f"information: {synthetic_outputs[96].total_information[row].item():.0f} bits")
        synthetic_axes[row, 1].axis("off")
        render_partition(
            synthetic_axes[row, 2],
            synthetic[row].cpu(),
            synthetic_outputs[64].partition.assignment[row],
            f"v3-64: {int(synthetic_outputs[64].token_count[row])}",
        )
        render_partition(
            synthetic_axes[row, 3],
            synthetic[row].cpu(),
            synthetic_outputs[96].partition.assignment[row],
            f"v3-96: {int(synthetic_outputs[96].token_count[row])}",
        )
    synthetic_figure.tight_layout()
    synthetic_figure.savefig(OUTPUT_ROOT / "synthetic_contract.png", dpi=170)
    plt.close(synthetic_figure)
    benchmark_images = next(iter(validation_loader))[0][:32].to(device)
    timing = {target: benchmark(tokenizer, benchmark_images) for target, tokenizer in tokenizers.items()}
    gaussian_assignments_equal = all(is_regular_grid(output, -1) for output in synthetic_outputs.values())
    report = {
        "status": "pass",
        "definition": "four-direction JPEG-LS Cauchy excess predictive codelength",
        "learned_parameters": sum(parameter.numel() for parameter in density.parameters()),
        "calibration": {
            "images": CALIBRATION_IMAGES,
            "labels_used": False,
            "total_information": summarize(calibration_totals.numpy()),
            "capacity_bits": {str(target): capacity for target, capacity in capacities.items()},
        },
        "validation": {
            "images": VALIDATION_IMAGES,
            "total_information": summarize(validation_totals),
            "profiles": {
                str(target): {
                    "token_count": summarize(values["token_count"]),
                    "mass_cv": summarize(values["mass_cv"]),
                    "cap_hit_fraction": float(np.mean(values["cap_hit"])),
                }
                for target, values in profile_values.items()
            },
        },
        "synthetic": {
            name: {
                "total_information": float(synthetic_outputs[96].total_information[index].item()),
                "token_counts": {
                    str(target): int(output.token_count[index].item())
                    for target, output in synthetic_outputs.items()
                },
            }
            for index, name in enumerate(names)
        },
        "gates": {
            "constant_is_one_token": all(output.token_count[0].item() == 1 for output in synthetic_outputs.values()),
            "linear_ramp_has_positive_subnoise_information": bool(
                0.0 < synthetic_outputs[96].total_information[1].item()
                < synthetic_outputs[96].total_information[2].item()
            ),
            "gaussian_is_maximum_tokens": all(
                output.token_count[-1].item() == MAXIMUM_TOKENS for output in synthetic_outputs.values()
            ),
            "gaussian_partition_is_regular_grid": gaussian_assignments_equal,
            "rates_are_monotonic_on_synthetic_sequence": all(
                bool(torch.all(output.token_count[1:] >= output.token_count[:-1]))
                for output in synthetic_outputs.values()
            ),
            "zero_learned_parameters": sum(parameter.numel() for parameter in density.parameters()) == 0,
        },
        "timing": {str(target): values for target, values in timing.items()},
        "artifacts": {
            "imagenet100_comparison": str(OUTPUT_ROOT / "imagenet100_comparison.png"),
            "synthetic_contract": str(OUTPUT_ROOT / "synthetic_contract.png"),
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "fail"
    with (OUTPUT_ROOT / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    calibrated = {
        "maximum_tokens": MAXIMUM_TOKENS,
        "partition_stride": PARTITION_STRIDE,
        "residual_scale": RESIDUAL_SCALE,
        "scan_directions": SCAN_DIRECTIONS,
        "profiles": {
            f"mean_{target}": {"capacity_bits": float(capacity)}
            for target, capacity in capacities.items()
        },
    }
    with (OUTPUT_ROOT / "calibrated_profiles.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calibrated, handle, sort_keys=False)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise RuntimeError("v3 information-tokenizer audit failed")


main()
