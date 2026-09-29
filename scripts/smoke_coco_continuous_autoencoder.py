import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from fractaltokenizer_v2 import (
    ContinuousAutoencoderConfig,
    ContinuousFractalAutoencoder,
    ContinuousReconstructionLoss,
)


configuration_path = Path("configs/coco_continuous_autoencoder.yaml")
output_root = Path("runs/coco_continuous_autoencoder/preflight")
sample_count = 8
overfit_steps = 40
benchmark_steps = 5


def load_images(root: Path, image_size: int, count: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.143), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    paths = sorted(root.glob("*.jpg"))[:count]
    if len(paths) < count:
        raise RuntimeError(f"needed {count} images under {root}, found {len(paths)}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images)


def tiny_overfit(device: torch.device, image_root: Path) -> dict:
    image = load_images(image_root, 64, 4).to(device)
    config = ContinuousAutoencoderConfig(
        latent_dim=96,
        input_tile_size=12,
        output_tile_size=20,
        encoder_hidden_dim=48,
        decoder_hidden_dim=64,
        transformer_depth=2,
        transformer_heads=4,
        coordinate_fourier_bands=5,
        geometry_fourier_bands=5,
        decoder_token_chunk_size=4,
        refiner_hidden_dim=48,
        refiner_blocks=2,
        target_tokens=12,
        max_tokens=16,
        partition_stride=4,
        bsp_workers=0,
    )
    model = ContinuousFractalAutoencoder(config).to(device)
    model.density_model.requires_grad_(False)
    criterion = ContinuousReconstructionLoss()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-3)
    losses = []
    psnr = []
    for _ in range(overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        metrics = criterion(output, image)
        metrics["loss"].backward()
        optimizer.step()
        losses.append(float(metrics["loss"].detach()))
        psnr.append(float(metrics["psnr"].detach()))
    if losses[-1] >= losses[0] * 0.75:
        raise AssertionError(f"tiny overfit did not improve enough: {losses[0]:.4f} to {losses[-1]:.4f}")
    with torch.no_grad():
        output = model(image)
    rows = []
    for index in range(image.shape[0]):
        rows.extend(
            [
                image[index].cpu(),
                output.base_reconstruction[index].float().cpu(),
                output.reconstruction[index].float().cpu(),
                (output.reconstruction[index].float() - image[index]).abs().cpu(),
            ]
        )
    save_image(make_grid(rows, nrow=4, padding=2), output_root / "tiny_overfit.png")
    return {
        "steps": overfit_steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "initial_psnr": psnr[0],
        "final_psnr": psnr[-1],
        "loss_ratio": losses[-1] / losses[0],
    }


def realistic_preflight(device: torch.device, image_root: Path, configuration: dict) -> dict:
    model_config = ContinuousAutoencoderConfig(**configuration["model"])
    model = ContinuousFractalAutoencoder(model_config).to(device)
    model.density_model.requires_grad_(False)
    criterion = ContinuousReconstructionLoss(**configuration["loss"])
    image_size = int(configuration["dataset"]["image_size"])
    image = load_images(image_root, image_size, sample_count).to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    output = None
    metrics = None
    for step in range(benchmark_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image, target_tokens=int(configuration["training"]["validation_tokens"]))
            metrics = criterion(output, image)
        metrics["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    rows = []
    for index in range(image.shape[0]):
        error = (output.reconstruction[index].float() - image[index]).abs()
        rows.extend(
            [
                image[index].cpu(),
                output.base_reconstruction[index].float().cpu(),
                output.reconstruction[index].float().cpu(),
                error.cpu(),
            ]
        )
    save_image(make_grid(rows, nrow=4, padding=2), output_root / "realistic_preflight.png")
    measured = timings[1:] if len(timings) > 1 else timings
    mean_time = sum(measured) / len(measured)
    return {
        "model_config": asdict(model_config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": image.shape[0],
        "tokens": float(metrics["tokens"].detach()),
        "loss": float(metrics["loss"].detach()),
        "psnr": float(metrics["psnr"].detach()),
        "seconds_per_step": mean_time,
        "images_per_second": image.shape[0] / mean_time,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
        "finite": all(torch.isfinite(value).item() for value in metrics.values()),
    }


configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("smoke test requires CUDA")
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
output_root.mkdir(parents=True, exist_ok=True)
dataset_root = Path(configuration["dataset"]["source_root"]) / "val2017"
results = {
    "status": "complete",
    "device": torch.cuda.get_device_name(),
    "tiny_overfit": tiny_overfit(device, dataset_root),
    "realistic_preflight": realistic_preflight(device, dataset_root, configuration),
}
(output_root / "smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
