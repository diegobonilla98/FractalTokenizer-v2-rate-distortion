import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import AnalyticInformationDensity, BalancedBSPPartitioner, FractalTokenizerV2, FractalTokenizerV2Config, QuantileGridPartitioner
from fractaltokenizer_v2.regions import render_token_tiles


image_root = Path("/mnt/hdd/supervisely_person_clean_2667_img/images")
output_path = Path("runs/supervisely_generation/token_audit.json")
image_size = 256
partition_stride = 8
sample_count = 128
calibration_count = 64
batch_size = 8
num_workers = 8
bsp_workers = 8
sample_seed = 42
target_average_tokens = 64.0
max_tokens = 128
binary_search_steps = 16
token_budgets = (32, 64, 128)


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, transform, limit: int):
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
        random_generator = random.Random(sample_seed)
        random_generator.shuffle(paths)
        self.paths = paths[:limit]
        if not self.paths:
            raise RuntimeError(f"no images found under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def deterministic_reconstruction(model, image, partition):
    _, _, coefficients, _ = model.region_encoder(
        image,
        boxes=partition.boxes,
        depth=partition.depth,
        mask=partition.mask,
        compact_valid_regions=partition.trees is not None,
    )
    tiles = model.region_encoder.basis.decode(coefficients)
    return render_token_tiles(
        tiles,
        partition.boxes,
        partition.assignment,
        tuple(image.shape[-2:]),
    ).clamp(0, 1)


torch.manual_seed(sample_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)
loader = DataLoader(
    ImagePathDataset(image_root, transform, sample_count),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=num_workers > 0,
)
density_model = AnalyticInformationDensity(
    partition_stride=partition_stride,
    coherence_floor=0.35,
    temperature=1.0,
    blur_sigma=0.0,
).to(device).eval()
model = FractalTokenizerV2(
    FractalTokenizerV2Config(
        latent_dim=64,
        partition_stride=partition_stride,
        use_vector_quantization=False,
        decoder_feature_dim=32,
        decoder_hidden_dim=64,
        decoder_residual_blocks=1,
    )
).to(device).eval()
bsp_partitioner = BalancedBSPPartitioner(max_tokens=max_tokens, workers=bsp_workers)
grid_partitioner = QuantileGridPartitioner()
images = []
densities = []
density_started = time.perf_counter()
with torch.inference_mode():
    for image in tqdm(loader, desc="Supervisely density", dynamic_ncols=True):
        image = image.to(device, non_blocking=True)
        images.append(image)
        densities.append(density_model(image))
if device.type == "cuda":
    torch.cuda.synchronize()
density_seconds = time.perf_counter() - density_started
calibration_density = torch.cat([density.cpu() for density in densities], dim=0)[:calibration_count]
total_mass = calibration_density.sum(dim=(-2, -1)).flatten()
low = max(float(total_mass.min()) / max_tokens, 1e-8)
high = max(float(total_mass.max()), low * 2.0)
for _ in range(binary_search_steps):
    threshold = math.sqrt(low * high)
    counts = bsp_partitioner(calibration_density, mass_per_token=threshold).token_count.float()
    if float(counts.mean()) > target_average_tokens:
        low = threshold
    else:
        high = threshold
mass_per_token = math.sqrt(low * high)
methods = {
    "fractal_bsp_variable_64": {"kind": "bsp_variable", "mass_per_token": mass_per_token},
    "fractal_bsp_fixed_64": {"kind": "bsp_fixed", "tokens": 64},
    "fractal_quantile_64": {"kind": "quantile", "tokens": 64},
    "standard_patch32_64": {"kind": "grid", "tokens": 64, "grid": (8, 8)},
    "standard_patch16_256": {"kind": "grid", "tokens": 256, "grid": (16, 16)},
}
for budget in token_budgets:
    methods[f"fractal_bsp_fixed_{budget}"] = {"kind": "bsp_fixed", "tokens": budget}
results = {}
with torch.inference_mode():
    for method_name, method in methods.items():
        squared_error = 0.0
        leaves = []
        elapsed = 0.0
        for image, density in tqdm(zip(images, densities), total=len(images), desc=method_name, dynamic_ncols=True):
            started = time.perf_counter()
            if method["kind"] == "bsp_variable":
                partition = bsp_partitioner(density, mass_per_token=method["mass_per_token"])
            elif method["kind"] == "bsp_fixed":
                partition = bsp_partitioner(density, target_tokens=method["tokens"])
            elif method["kind"] == "quantile":
                partition = grid_partitioner(density, target_tokens=method["tokens"])
            else:
                partition = grid_partitioner(
                    torch.ones_like(density),
                    target_tokens=method["tokens"],
                    token_grid=method["grid"],
                )
            reconstruction = deterministic_reconstruction(model, image, partition)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            squared_error += float((reconstruction - image).square().sum())
            leaves.extend(int(value) for value in partition.token_count.cpu())
        pixels = len(leaves) * 3 * image_size * image_size
        mse = squared_error / pixels
        leaf_tensor = torch.tensor(leaves, dtype=torch.float32)
        action_tensor = 2.0 * leaf_tensor - 1.0 if method["kind"].startswith("bsp") else leaf_tensor
        results[method_name] = {
            "mse": mse,
            "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
            "mean_content_tokens": float(leaf_tensor.mean()),
            "min_content_tokens": int(leaf_tensor.min()),
            "max_content_tokens": int(leaf_tensor.max()),
            "mean_autoregressive_events": float(action_tensor.mean()),
            "seconds": elapsed,
            "images_per_second": len(leaves) / elapsed,
        }
standard_events = results["standard_patch16_256"]["mean_autoregressive_events"]
variable = results["fractal_bsp_variable_64"]
summary = {
    "status": "complete",
    "dataset": str(image_root),
    "samples": sample_count,
    "image_size": image_size,
    "density_profile": {
        "name": "legacy_20260816",
        "temperature": 1.0,
        "coherence_floor": 0.35,
        "blur_sigma": 0.0,
    },
    "mass_per_token": mass_per_token,
    "density_seconds": density_seconds,
    "density_images_per_second": sample_count / density_seconds,
    "variable_content_token_reduction_vs_patch16": standard_events / variable["mean_content_tokens"],
    "variable_event_reduction_vs_patch16": standard_events / variable["mean_autoregressive_events"],
    "methods": results,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = output_path.with_suffix(".tmp")
temporary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
temporary_path.replace(output_path)
print(json.dumps(summary, indent=2))
