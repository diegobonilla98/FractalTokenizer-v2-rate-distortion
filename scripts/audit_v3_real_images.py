import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from fractaltokenizer_v2.cuda_ops import build_cuda_extension
from fractaltokenizer_v2.density import AnalyticInformationDensity
from fractaltokenizer_v2.regions import LocalDCTBasis, render_token_tiles, sample_region_tiles
from fractaltokenizer_v3 import NetworkFreeInformationTokenizer
from fractaltokenizer_v3.partition import VariableRateQuantilePartitioner


IMAGENET_ROOT = Path("/mnt/hdd/ImageNet100/val")
COCO_ROOT = Path("/mnt/hdd/coco2017/val2017")
AUDIT_ROOT = Path("/mnt/hdd/fractaltokenizer_v3_clean/real_image_audit")
CAPACITIES = {64: 11154.006347656252, 96: 6857.214111328126}
MAXIMUM_TOKENS = 128
IMAGES_PER_DATASET = 500
BATCH_SIZE = 32
WORKERS = 8
IMAGE_SIZE = 256
PARTITION_STRIDE = 8
TILE_SIZE = 16
BASIS_SIZE = 4
VISUAL_IMAGES = 6
SEED = 42


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), index


def balanced_indices(dataset: datasets.ImageFolder, count: int) -> list[int]:
    by_class = {}
    for index, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(index)
    selected = []
    positions = {
        label: np.linspace(0, len(indices) - 1, math.ceil(count / len(by_class)), dtype=np.int64)
        for label, indices in by_class.items()
    }
    for position_index in range(max(len(values) for values in positions.values())):
        for label in sorted(by_class):
            if position_index < len(positions[label]):
                selected.append(by_class[label][positions[label][position_index]])
                if len(selected) == count:
                    return selected
    return selected


def deterministic_indices(length: int, count: int) -> list[int]:
    return np.linspace(0, length - 1, min(length, count), dtype=np.int64).tolist()


def reconstruct(
    images: torch.Tensor,
    partition,
    basis: LocalDCTBasis,
) -> torch.Tensor:
    tiles = sample_region_tiles(
        images,
        partition.boxes,
        TILE_SIZE,
        mask=partition.mask,
        backend="cuda",
        sampling_ratio=2,
    )
    coefficients = basis.encode(tiles)
    decoded = basis.decode(coefficients)
    return render_token_tiles(
        decoded,
        partition.boxes,
        partition.assignment,
        (IMAGE_SIZE, IMAGE_SIZE),
        backend="cuda",
        clamp_output=True,
    )


def image_metrics(images: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, list[float]]:
    mse = (images - reconstruction).square().flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3) / 8.0
    gray_image = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    gray_reconstruction = (
        0.299 * reconstruction[:, 0:1]
        + 0.587 * reconstruction[:, 1:2]
        + 0.114 * reconstruction[:, 2:3]
    )
    image_edges = torch.cat(
        [F.conv2d(gray_image, sobel_x, padding=1), F.conv2d(gray_image, sobel_x.transpose(-1, -2), padding=1)],
        dim=1,
    )
    reconstruction_edges = torch.cat(
        [
            F.conv2d(gray_reconstruction, sobel_x, padding=1),
            F.conv2d(gray_reconstruction, sobel_x.transpose(-1, -2), padding=1),
        ],
        dim=1,
    )
    edge_mse = (image_edges - reconstruction_edges).square().flatten(1).mean(dim=1)
    image_arrays = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    reconstruction_arrays = reconstruction.permute(0, 2, 3, 1).detach().cpu().numpy()
    ssim = [
        structural_similarity(original, restored, channel_axis=2, data_range=1.0)
        for original, restored in zip(image_arrays, reconstruction_arrays)
    ]
    return {
        "mse": mse.cpu().tolist(),
        "psnr": psnr.cpu().tolist(),
        "ssim": [float(value) for value in ssim],
        "edge_mse": edge_mse.cpu().tolist(),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def information_mass_cv(density: torch.Tensor, partition) -> list[float]:
    values = []
    for index in range(density.shape[0]):
        count = int(partition.mask[index].sum().item())
        mass = torch.zeros(count, device=density.device, dtype=torch.float32)
        mass.scatter_add_(0, partition.assignment[index].flatten(), density[index].flatten().float())
        values.append(float((mass.std(unbiased=False) / mass.mean().clamp_min(1e-8)).item()))
    return values


def boundary_map(assignment: torch.Tensor) -> np.ndarray:
    expanded = F.interpolate(
        assignment.float().view(1, 1, *assignment.shape),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )[0, 0].to(torch.int64)
    boundary = torch.zeros_like(expanded, dtype=torch.bool)
    boundary[:, 1:] |= expanded[:, 1:] != expanded[:, :-1]
    boundary[1:, :] |= expanded[1:, :] != expanded[:-1, :]
    return boundary.cpu().numpy()


def partition_overlay(image: torch.Tensor, assignment: torch.Tensor) -> np.ndarray:
    array = image.permute(1, 2, 0).cpu().numpy().copy()
    array[boundary_map(assignment)] = np.array([1.0, 0.05, 0.05])
    return array


def evaluate_dataset(
    name: str,
    dataset: Dataset,
    device: torch.device,
    tokenizers: dict[int, NetworkFreeInformationTokenizer],
    analytic_density: AnalyticInformationDensity,
    partitioner: VariableRateQuantilePartitioner,
    basis: LocalDCTBasis,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
    )
    collected = {
        profile: {
            method: {metric: [] for metric in ("mse", "psnr", "ssim", "edge_mse", "information_mass_cv")}
            for method in ("v3_predictive", "v2_analytic", "equal_area")
        }
        for profile in CAPACITIES
    }
    token_counts = {profile: [] for profile in CAPACITIES}
    start = time.perf_counter()
    for images, _ in tqdm(loader, desc=f"real audit {name}"):
        images = images.to(device, non_blocking=True)
        analytic = analytic_density(images)
        uniform = torch.ones_like(analytic)
        for profile, tokenizer in tokenizers.items():
            v3_output = tokenizer(images)
            counts = v3_output.token_count
            partitions = {
                "v3_predictive": v3_output.partition,
                "v2_analytic": partitioner(analytic, counts),
                "equal_area": partitioner(uniform, counts),
            }
            token_counts[profile].extend(counts.cpu().tolist())
            for method, partition in partitions.items():
                reconstruction = reconstruct(images, partition, basis)
                metrics = image_metrics(images, reconstruction)
                for metric, values in metrics.items():
                    collected[profile][method][metric].extend(values)
                collected[profile][method]["information_mass_cv"].extend(
                    information_mass_cv(v3_output.density, partition)
                )
    elapsed = time.perf_counter() - start
    summaries = {}
    for profile in CAPACITIES:
        summaries[str(profile)] = {
            "token_count": summarize(token_counts[profile]),
            "methods": {
                method: {metric: summarize(values) for metric, values in metrics.items()}
                for method, metrics in collected[profile].items()
            },
        }
        paired_psnr = {
            method: np.asarray(collected[profile][method]["psnr"])
            for method in collected[profile]
        }
        summaries[str(profile)]["v3_psnr_win_fraction"] = {
            baseline: float((paired_psnr["v3_predictive"] > values).mean())
            for baseline, values in paired_psnr.items()
            if baseline != "v3_predictive"
        }
        summaries[str(profile)]["v3_psnr_delta"] = {
            baseline: summarize((paired_psnr["v3_predictive"] - values).tolist())
            for baseline, values in paired_psnr.items()
            if baseline != "v3_predictive"
        }
    make_visual(name, dataset, device, tokenizers, analytic_density, partitioner, basis)
    return {
        "images": len(dataset),
        "elapsed_seconds": elapsed,
        "images_per_second_all_six_reconstructions": len(dataset) / elapsed,
        "profiles": summaries,
    }


def make_visual(
    name: str,
    dataset: Dataset,
    device: torch.device,
    tokenizers: dict[int, NetworkFreeInformationTokenizer],
    analytic_density: AnalyticInformationDensity,
    partitioner: VariableRateQuantilePartitioner,
    basis: LocalDCTBasis,
) -> None:
    indices = deterministic_indices(len(dataset), VISUAL_IMAGES)
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    profile = 64
    output = tokenizers[profile](images)
    partitions = {
        "v3 predictive": output.partition,
        "v2 analytic": partitioner(analytic_density(images), output.token_count),
        "equal area": partitioner(torch.ones_like(output.density), output.token_count),
    }
    reconstructions = {method: reconstruct(images, partition, basis) for method, partition in partitions.items()}
    figure, axes = plt.subplots(VISUAL_IMAGES, 7, figsize=(22, 3.1 * VISUAL_IMAGES))
    for row in range(VISUAL_IMAGES):
        axes[row, 0].imshow(images[row].permute(1, 2, 0).cpu().numpy())
        axes[row, 0].set_title(f"original, N={int(output.token_count[row])}")
        axes[row, 0].axis("off")
        column = 1
        for method, partition in partitions.items():
            axes[row, column].imshow(partition_overlay(images[row].cpu(), partition.assignment[row]))
            axes[row, column].set_title(f"{method} partition")
            axes[row, column].axis("off")
            axes[row, column + 1].imshow(reconstructions[method][row].permute(1, 2, 0).cpu().numpy())
            score = image_metrics(images[row : row + 1], reconstructions[method][row : row + 1])
            axes[row, column + 1].set_title(f"PSNR {score['psnr'][0]:.2f}, SSIM {score['ssim'][0]:.3f}")
            axes[row, column + 1].axis("off")
            column += 2
    figure.tight_layout()
    figure.savefig(AUDIT_ROOT / f"{name}_matched_dct_reconstruction.png", dpi=150)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    build_cuda_extension()
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, antialias=True),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    imagenet_full = datasets.ImageFolder(IMAGENET_ROOT, transform=transform)
    imagenet = Subset(imagenet_full, balanced_indices(imagenet_full, IMAGES_PER_DATASET))
    coco_paths = sorted(COCO_ROOT.glob("*.jpg"))
    coco_full = ImagePathDataset(coco_paths, transform)
    coco = Subset(coco_full, deterministic_indices(len(coco_full), IMAGES_PER_DATASET))
    tokenizers = {
        profile: NetworkFreeInformationTokenizer(
            capacity_bits=capacity,
            maximum_tokens=MAXIMUM_TOKENS,
            partition_stride=PARTITION_STRIDE,
            residual_scale=1.0 / 255.0,
            scan_directions=4,
            backend="cuda",
        ).to(device)
        for profile, capacity in CAPACITIES.items()
    }
    analytic_density = AnalyticInformationDensity(
        partition_stride=PARTITION_STRIDE,
        temperature=1.0,
        coherence_floor=0.35,
        blur_sigma=0.0,
    ).to(device)
    partitioner = VariableRateQuantilePartitioner(
        maximum_tokens=MAXIMUM_TOKENS,
        backend="cuda",
        uniform_when_saturated=False,
    ).to(device)
    basis = LocalDCTBasis(tile_size=TILE_SIZE, basis_size=BASIS_SIZE).to(device)
    report = {
        "status": "complete",
        "contract": {
            "real_images": True,
            "learned_parameters": 0,
            "matched_per_image_token_counts": True,
            "content_numbers_per_token": 3 * BASIS_SIZE * BASIS_SIZE,
            "content_representation": "fixed orthonormal low-frequency DCT",
            "image_size": IMAGE_SIZE,
        },
        "datasets": {
            "imagenet100": evaluate_dataset(
                "imagenet100", imagenet, device, tokenizers, analytic_density, partitioner, basis
            ),
            "coco2017": evaluate_dataset(
                "coco2017", coco, device, tokenizers, analytic_density, partitioner, basis
            ),
        },
    }
    with (AUDIT_ROOT / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
