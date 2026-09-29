import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import FractalTokenizerV2, FractalTokenizerV2Config, QuantileGridPartitioner
from fractaltokenizer_v2.regions import render_token_tiles


image_root = Path("/mnt/hdd/ImageNet100/val")
output_path = Path("partition_rd_v2_results.json")
image_size = 256
batch_size = 8
num_workers = 8
max_images = 1000
token_budgets = (32, 64, 96, 128, 192, 256)
sample_seed = 42


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, transform, limit: int | None):
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
        if limit is not None and limit < len(paths):
            random_generator = random.Random(sample_seed)
            random_generator.shuffle(paths)
            paths = paths[:limit]
        self.paths = paths
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
    )
    tiles = model.region_encoder.basis.decode(coefficients)
    return render_token_tiles(
        tiles,
        partition.boxes,
        partition.assignment,
        tuple(image.shape[-2:]),
    ).clamp(0, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
loader = DataLoader(
    ImagePathDataset(image_root, transform, max_images),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=num_workers > 0,
)
config = FractalTokenizerV2Config(
    latent_dim=64,
    partition_stride=8,
    token_grid=None,
    use_vector_quantization=False,
    decoder_feature_dim=32,
    decoder_hidden_dim=64,
    decoder_residual_blocks=1,
)
model = FractalTokenizerV2(config).to(device).eval()
uniform_partitioner = QuantileGridPartitioner()
results = {}

with torch.inference_mode():
    for token_budget in token_budgets:
        sums = {
            "adaptive_mse": 0.0,
            "uniform_mse": 0.0,
            "adaptive_mass_cv": 0.0,
            "uniform_mass_cv": 0.0,
            "adaptive_wins": 0,
            "images": 0,
        }
        for image in tqdm(loader, desc=f"partition RD {token_budget} tokens", dynamic_ncols=True):
            image = image.to(device, non_blocking=True)
            density = model.density_model(image)
            adaptive = model.partitioner(density, target_tokens=token_budget, token_grid=None)
            uniform = uniform_partitioner(torch.ones_like(density), target_tokens=token_budget, token_grid=None)
            adaptive_reconstruction = deterministic_reconstruction(model, image, adaptive)
            uniform_reconstruction = deterministic_reconstruction(model, image, uniform)
            adaptive_mse = (adaptive_reconstruction - image).square().flatten(1).mean(dim=1)
            uniform_mse = (uniform_reconstruction - image).square().flatten(1).mean(dim=1)
            adaptive_mean = adaptive.mass.mean(dim=1, keepdim=True)
            uniform_mass = torch.zeros_like(uniform.mass)
            uniform_mass.scatter_add_(1, uniform.assignment.flatten(1), density[:, 0].flatten(1))
            uniform_mean = uniform_mass.mean(dim=1, keepdim=True)
            adaptive_cv = adaptive.mass.std(dim=1, unbiased=False) / adaptive_mean[:, 0].clamp_min(1e-8)
            uniform_cv = uniform_mass.std(dim=1, unbiased=False) / uniform_mean[:, 0].clamp_min(1e-8)
            count = image.shape[0]
            sums["adaptive_mse"] += float(adaptive_mse.sum())
            sums["uniform_mse"] += float(uniform_mse.sum())
            sums["adaptive_mass_cv"] += float(adaptive_cv.sum())
            sums["uniform_mass_cv"] += float(uniform_cv.sum())
            sums["adaptive_wins"] += int((adaptive_mse < uniform_mse).sum())
            sums["images"] += count
        count = sums.pop("images")
        adaptive_mse = sums["adaptive_mse"] / count
        uniform_mse = sums["uniform_mse"] / count
        results[str(token_budget)] = {
            "adaptive_mse": adaptive_mse,
            "uniform_mse": uniform_mse,
            "adaptive_psnr": -10.0 * math.log10(max(adaptive_mse, 1e-12)),
            "uniform_psnr": -10.0 * math.log10(max(uniform_mse, 1e-12)),
            "psnr_delta_db": 10.0 * math.log10(max(uniform_mse, 1e-12) / max(adaptive_mse, 1e-12)),
            "adaptive_win_fraction": sums["adaptive_wins"] / count,
            "adaptive_mass_cv": sums["adaptive_mass_cv"] / count,
            "uniform_mass_cv": sums["uniform_mass_cv"] / count,
            "images": count,
        }
        print(token_budget, results[str(token_budget)])

output_path.write_text(json.dumps(results, indent=2))
print(output_path)
