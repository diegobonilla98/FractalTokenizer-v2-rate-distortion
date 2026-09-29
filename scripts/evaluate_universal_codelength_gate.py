import datetime
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.ndimage import label
from skimage.metrics import structural_similarity
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import (
    AnalyticInformationDensity,
    HilbertEqualMassPartitioner,
    UniversalCodelengthDensity,
)


matplotlib.use("Agg", force=True)
configuration_path = Path(
    os.environ.get("FRACTAL_UNIVERSAL_GATE_CONFIGURATION", "configs/universal_codelength_gate.yaml")
)


class ImageRecordDataset(Dataset):
    def __init__(self, records: list[tuple[str, Path]], image_size: int):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        source, path = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, source, str(path)


class TelegramReporter:
    def __init__(self, enabled: bool, poll_updates: bool, state: dict[str, Any], output_root: Path):
        self.state = state
        self.bot = None
        if enabled and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.bot = TelegramBot(
                working_directory=str(output_root),
                poll_updates=poll_updates,
                enable_terminal_commands=False,
            )
            self.bot.set_state_provider(lambda: dict(self.state))
        elif enabled:
            print("Telegram disabled: credentials are not available")

    def update(self, **values: Any) -> None:
        self.state.update(values)
        if self.bot is not None:
            self.bot.update_state(**values)

    def send(self, message: str, wait: bool = False) -> None:
        if self.bot is not None:
            self.bot.send_message(text=message, wait=wait)

    def photo(self, path: Path, message: str) -> None:
        if self.bot is not None and path.is_file():
            self.bot.send_message(photo=str(path), text=message, wait=False)

    def close(self) -> None:
        if self.bot is not None:
            self.bot.stop(wait=True)


def atomic_json(path: Path, values: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_configuration() -> dict[str, Any]:
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    if configuration.get("schema_version") != 1:
        raise ValueError("unsupported configuration schema")
    return configuration


def image_paths(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    if not paths:
        raise FileNotFoundError(f"no images found under {root}")
    return paths


def stable_records(source: str, root: Path, count: int, seed: int) -> list[tuple[str, Path]]:
    paths = image_paths(root)
    paths.sort(key=lambda path: hashlib.sha256(f"{seed}:{source}:{path}".encode("utf-8")).hexdigest())
    return [(source, path) for path in paths[:count]]


def make_loader(
    records: list[tuple[str, Path]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        ImageRecordDataset(records, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def summed_cells(values: torch.Tensor, stride: int) -> torch.Tensor:
    return F.avg_pool2d(
        values,
        kernel_size=stride,
        stride=stride,
        ceil_mode=True,
        divisor_override=1,
    )


def sobel_density(image: torch.Tensor, stride: int) -> torch.Tensor:
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-2, -1)
    horizontal = F.conv2d(gray.float(), kernel_x, padding=1)
    vertical = F.conv2d(gray.float(), kernel_y, padding=1)
    return summed_cells(torch.sqrt(horizontal.square() + vertical.square() + 1e-12), stride) + 1e-6


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_densities(
    image: torch.Tensor,
    universal: UniversalCodelengthDensity,
    analytic: AnalyticInformationDensity,
    stride: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    timings = {}
    densities = {}
    for name, operation in (
        ("universal_codelength", lambda: universal(image)),
        ("analytic_residual", lambda: analytic(image)),
        ("sobel", lambda: sobel_density(image, stride)),
    ):
        synchronize(device)
        started = time.perf_counter()
        densities[name] = operation().float()
        synchronize(device)
        timings[name] = 1000.0 * (time.perf_counter() - started) / image.shape[0]
    height, width = densities["universal_codelength"].shape[-2:]
    densities["uniform"] = torch.ones(image.shape[0], 1, height, width, device=device)
    timings["uniform"] = 0.0
    return densities, timings


def full_assignment(assignment: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        assignment.unsqueeze(1).float(),
        size=image_shape,
        mode="nearest",
    )[:, 0].long()


def region_mean_reconstruction(image: torch.Tensor, assignment: torch.Tensor, token_count: int) -> torch.Tensor:
    batch_size, channels, height, width = image.shape
    labels = full_assignment(assignment, (height, width)).flatten(1)
    pixels = image.flatten(2).transpose(1, 2).float()
    sums = torch.zeros(batch_size, token_count, channels, device=image.device, dtype=torch.float32)
    sums.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, channels), pixels)
    counts = torch.zeros(batch_size, token_count, device=image.device, dtype=torch.float32)
    counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    reconstruction = torch.gather(means, 1, labels.unsqueeze(-1).expand(-1, -1, channels))
    return reconstruction.transpose(1, 2).reshape(batch_size, channels, height, width)


def assignment_mass(assignment: torch.Tensor, density: torch.Tensor, token_count: int) -> torch.Tensor:
    output = torch.zeros(assignment.shape[0], token_count, device=assignment.device, dtype=torch.float32)
    output.scatter_add_(1, assignment.flatten(1), density[:, 0].flatten(1))
    return output


def low_information_fragments(reference_density: np.ndarray, assignment: np.ndarray) -> float:
    threshold = float(np.quantile(reference_density, 0.25))
    components, count = label(reference_density <= threshold)
    if count == 0:
        return float("nan")
    masses = np.bincount(components.reshape(-1))[1:]
    component = int(np.argmax(masses)) + 1
    return float(np.unique(assignment[components == component]).size)


def evaluate_partition(
    image: torch.Tensor,
    reference_density: torch.Tensor,
    partition,
    compute_ssim: bool,
) -> tuple[list[dict[str, float]], torch.Tensor]:
    maximum_count = partition.mask.shape[1]
    reconstruction = region_mean_reconstruction(image, partition.assignment, maximum_count).clamp(0.0, 1.0)
    mse = (image.float() - reconstruction).square().flatten(1).mean(dim=1)
    reference_mass = assignment_mass(partition.assignment, reference_density, maximum_count)
    records = []
    for batch_index, count_value in enumerate(partition.token_count.tolist()):
        count = int(count_value)
        own = partition.mass[batch_index, :count]
        reference = reference_mass[batch_index, :count]
        own_mean = own.mean().clamp_min(1e-8)
        reference_mean = reference.mean().clamp_min(1e-8)
        raw_bits, enumerative_bits = HilbertEqualMassPartitioner.geometry_bits(
            count,
            partition.assignment.shape[-2] * partition.assignment.shape[-1],
        )
        if compute_ssim:
            source_np = image[batch_index].permute(1, 2, 0).float().cpu().numpy()
            reconstruction_np = reconstruction[batch_index].permute(1, 2, 0).cpu().numpy()
            ssim = float(structural_similarity(source_np, reconstruction_np, channel_axis=2, data_range=1.0))
        else:
            ssim = float("nan")
        records.append(
            {
                "tokens": float(count),
                "own_mass_cv": float(own.std(unbiased=False) / own_mean),
                "information_mass_cv": float(reference.std(unbiased=False) / reference_mean),
                "information_max_to_mean": float(reference.max() / reference_mean),
                "information_within_20_percent": float(
                    ((reference - reference_mean).abs() <= 0.2 * reference_mean).float().mean()
                ),
                "psnr": float(-10.0 * torch.log10(mse[batch_index].clamp_min(1e-12))),
                "ssim": ssim,
                "geometry_raw_bits": raw_bits,
                "geometry_enumerative_bits": enumerative_bits,
                "low_information_fragments": low_information_fragments(
                    reference_density[batch_index, 0].cpu().numpy(),
                    partition.assignment[batch_index].cpu().numpy(),
                ),
            }
        )
    return records, reconstruction


def aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    output = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = float(finite.mean()) if finite.size else float("nan")
        output[f"{key}_std"] = float(finite.std()) if finite.size else float("nan")
        if key == "tokens":
            output["tokens_min"] = float(finite.min())
            output["tokens_max"] = float(finite.max())
    return output


def boundary_segments(assignment: np.ndarray) -> list[list[tuple[float, float]]]:
    vertical = assignment[:, 1:] != assignment[:, :-1]
    horizontal = assignment[1:] != assignment[:-1]
    segments = []
    for y, x in np.argwhere(vertical):
        segments.append([(float(x + 1), float(y)), (float(x + 1), float(y + 1))])
    for y, x in np.argwhere(horizontal):
        segments.append([(float(x), float(y + 1)), (float(x + 1), float(y + 1))])
    return segments


def draw_partition(axis, image: np.ndarray, assignment: np.ndarray, title: str) -> None:
    side = assignment.shape[0]
    axis.imshow(image, extent=(0, side, side, 0))
    axis.add_collection(LineCollection(boundary_segments(assignment), colors="#ff1744", linewidths=0.45))
    axis.set_xlim(0, side)
    axis.set_ylim(side, 0)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def save_visuals(visuals: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(len(visuals), 6, figsize=(18, 3.0 * len(visuals)))
    if len(visuals) == 1:
        axes = axes[None, :]
    for row, visual in enumerate(visuals):
        image = visual["image"].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{visual['source']} original", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(visual["reference_density"], cmap="magma")
        axes[row, 1].set_title("Universal code bits", fontsize=8)
        axes[row, 1].axis("off")
        for column, method in enumerate(("uniform", "sobel", "analytic_residual", "universal_codelength"), start=2):
            assignment = visual["assignments"][method]
            count = int(assignment.max()) + 1
            draw_partition(axes[row, column], image, assignment, f"{method}: {count} tokens")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configuration = load_configuration()
    output_root = Path(configuration["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "starting",
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "objective": "network-free universal codelength and Hilbert equal-bit gate",
    }
    atomic_json(output_root / "run_state.json", state)
    telegram = configuration["telegram"]
    reporter = TelegramReporter(bool(telegram["enabled"]), bool(telegram["poll_updates"]), state, output_root)
    try:
        seed = int(configuration["seed"])
        dataset = configuration["dataset"]
        calibration_count = int(dataset["calibration_images_per_source"])
        evaluation_count = int(dataset["evaluation_images_per_source"])
        calibration_records = stable_records(
            "ImageNet100",
            Path(dataset["imagenet100_calibration"]),
            calibration_count,
            seed,
        ) + stable_records("COCO", Path(dataset["coco_calibration"]), calibration_count, seed)
        evaluation_records = stable_records(
            "ImageNet100",
            Path(dataset["imagenet100_evaluation"]),
            evaluation_count,
            seed + 1,
        ) + stable_records("COCO", Path(dataset["coco_evaluation"]), evaluation_count, seed + 1)
        image_settings = configuration["image"]
        evaluation = configuration["evaluation"]
        image_size = int(image_settings["size"])
        stride = int(image_settings["partition_stride"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader_values = (
            image_size,
            int(evaluation["batch_size"]),
            int(evaluation["num_workers"]),
            device,
        )
        calibration_loader = make_loader(calibration_records, *loader_values)
        evaluation_loader = make_loader(evaluation_records, *loader_values)
        coder = configuration["universal_coder"]
        universal = UniversalCodelengthDensity(
            partition_stride=stride,
            alpha=float(coder["alpha"]),
            scan_directions=int(coder["scan_directions"]),
        ).to(device).eval()
        analytic = AnalyticInformationDensity(partition_stride=stride).to(device).eval()
        partitioner = HilbertEqualMassPartitioner(maximum_tokens=int(evaluation["maximum_tokens"])).to(device)
        methods = ("uniform", "sobel", "analytic_residual", "universal_codelength")
        reporter.update(status="calibrating", device=str(device), calibration_images=len(calibration_records))
        reporter.send(
            f"Starting universal-codelength gate on {device}\n"
            f"calibration: {len(calibration_records)} images\nevaluation: {len(evaluation_records)} images"
        )
        calibration_totals = {method: [] for method in methods}
        density_timings = {method: [] for method in methods}
        with torch.inference_mode():
            for image, _, _ in tqdm(calibration_loader, desc="codelength calibration", dynamic_ncols=True):
                image = image.to(device, non_blocking=True)
                densities, timings = compute_densities(image, universal, analytic, stride, device)
                for method in methods:
                    calibration_totals[method].extend(densities[method].sum(dim=(1, 2, 3)).cpu().tolist())
                    density_timings[method].append(timings[method])
        target_average = float(evaluation["variable_target_average_tokens"])
        thresholds = {
            method: float(np.mean(calibration_totals[method]) / target_average)
            for method in methods
        }
        modes = [(f"fixed_{int(value)}", int(value), None) for value in evaluation["fixed_token_budgets"]]
        modes.append((f"variable_{int(target_average)}", None, target_average))
        records_by_experiment: dict[str, list[dict[str, float]]] = {
            f"{method}/{mode[0]}": [] for method in methods for mode in modes
        }
        per_image_records = []
        visuals = []
        processed = 0
        reporter.update(status="evaluating", evaluation_images=len(evaluation_records))
        with torch.inference_mode():
            iterator = tqdm(evaluation_loader, desc="universal codelength gate", dynamic_ncols=True)
            for image, sources, paths in iterator:
                image = image.to(device, non_blocking=True)
                densities, timings = compute_densities(image, universal, analytic, stride, device)
                for method in methods:
                    density_timings[method].append(timings[method])
                batch_visuals = [
                    {
                        "image": image[index].cpu(),
                        "source": sources[index],
                        "path": paths[index],
                        "reference_density": densities["universal_codelength"][index, 0].cpu().numpy(),
                        "assignments": {},
                    }
                    for index in range(min(image.shape[0], max(0, int(evaluation["visual_images"]) - len(visuals))))
                ]
                for method in methods:
                    for mode_name, fixed_tokens, _ in modes:
                        synchronize(device)
                        partition_started = time.perf_counter()
                        if fixed_tokens is None:
                            partition = partitioner(densities[method], mass_per_token=thresholds[method])
                        else:
                            partition = partitioner(densities[method], target_tokens=fixed_tokens)
                        synchronize(device)
                        partition_ms = 1000.0 * (time.perf_counter() - partition_started) / image.shape[0]
                        batch_records, _ = evaluate_partition(
                            image,
                            densities["universal_codelength"],
                            partition,
                            bool(evaluation["compute_ssim"]),
                        )
                        for index, record in enumerate(batch_records):
                            record["partition_ms"] = partition_ms
                            record["density_ms"] = timings[method]
                            record["universal_total_bits"] = float(
                                densities["universal_codelength"][index].sum()
                            )
                            records_by_experiment[f"{method}/{mode_name}"].append(record)
                            per_image_records.append(
                                {
                                    "source": sources[index],
                                    "path": paths[index],
                                    "method": method,
                                    "mode": mode_name,
                                    **record,
                                }
                            )
                        if fixed_tokens is None:
                            for index, visual in enumerate(batch_visuals):
                                visual["assignments"][method] = partition.assignment[index].cpu().numpy()
                visuals.extend(batch_visuals)
                processed += image.shape[0]
                iterator.set_postfix(processed=processed)
                if processed % 32 == 0 or processed == len(evaluation_records):
                    reporter.update(status="evaluating", processed=processed, total=len(evaluation_records))
        summary_methods = {
            key: aggregate(values) for key, values in records_by_experiment.items()
        }
        variable_name = f"variable_{int(target_average)}"
        universal_variable = summary_methods[f"universal_codelength/{variable_name}"]
        uniform_variable = summary_methods[f"uniform/{variable_name}"]
        black = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.inference_mode():
            black_density = universal(black)
            black_partition = partitioner(
                black_density,
                mass_per_token=thresholds["universal_codelength"],
            )
        black_tokens = int(black_partition.token_count.item())
        gate = configuration["gate"]
        checks = {
            "information_balance": universal_variable["information_mass_cv"]
            <= float(gate["maximum_information_mass_cv"]),
            "tokens_within_20_percent": universal_variable["information_within_20_percent"]
            >= float(gate["minimum_tokens_within_20_percent"]),
            "rate_distortion": universal_variable["psnr"] - uniform_variable["psnr"]
            >= float(gate["minimum_psnr_gain_over_uniform_db"]),
            "aggressive_uniform_compression": black_tokens <= int(gate["maximum_black_image_tokens"]),
        }
        summary = {
            "status": "complete",
            "scientific_gate_passed": all(checks.values()),
            "checks": checks,
            "definition": "rho(u) is the four-direction Bayesian mixture codelength of causal RGB prediction residuals under an online KT estimator",
            "network_training": False,
            "corpus_model_learning": False,
            "calibration_images": len(calibration_records),
            "evaluation_images": len(evaluation_records),
            "image_size": image_size,
            "partition_stride": stride,
            "variable_target_average_tokens": target_average,
            "mass_per_token": thresholds,
            "black_image_tokens": black_tokens,
            "density_ms_per_image": {
                method: float(np.mean(values)) for method, values in density_timings.items()
            },
            "methods": summary_methods,
        }
        atomic_json(output_root / "summary.json", summary)
        with (output_root / "per_image.jsonl").open("w", encoding="utf-8") as file:
            for record in per_image_records:
                file.write(json.dumps(record) + "\n")
        visual_path = output_root / "tokenization_comparison.png"
        save_visuals(visuals, visual_path)
        state.update(
            status="complete",
            completed_at=datetime.datetime.now().isoformat(timespec="seconds"),
            scientific_gate_passed=summary["scientific_gate_passed"],
            black_image_tokens=black_tokens,
            universal_tokens=f"{universal_variable['tokens']:.1f}",
            universal_psnr=f"{universal_variable['psnr']:.2f}",
        )
        atomic_json(output_root / "run_state.json", state)
        reporter.update(**state)
        reporter.send(
            "Universal-codelength gate complete\n"
            f"gate passed: {summary['scientific_gate_passed']}\n"
            f"black image: {black_tokens} tokens\n"
            f"universal variable: {universal_variable['tokens']:.1f} tokens, "
            f"{universal_variable['psnr']:.2f} dB, mass CV {universal_variable['information_mass_cv']:.3f}\n"
            f"artifacts: {output_root}",
            wait=True,
        )
        reporter.photo(visual_path, "Network-free constant-information gate: equal-bit Hilbert tokenizations")
        print(json.dumps(summary, indent=2))
    except Exception:
        failure = traceback.format_exc()
        state.update(
            status="failed",
            failed_at=datetime.datetime.now().isoformat(timespec="seconds"),
            error=failure[-3000:],
        )
        atomic_json(output_root / "run_state.json", state)
        reporter.update(**state)
        reporter.send(f"Universal-codelength gate FAILED\n{failure[-3000:]}", wait=True)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
