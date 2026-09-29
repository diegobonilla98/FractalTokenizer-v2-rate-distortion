import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import AnalyticInformationDensity, BalancedBSPPartitioner


image_root = Path("/mnt/hdd/ImageNet100/val")
output_path = Path("mass_threshold_calibration_v2.json")
image_size = 256
partition_stride = 8
sample_count = 512
target_average_tokens = 128.0
max_tokens = 256
batch_size = 16
num_workers = 8
binary_search_steps = 24
sample_seed = 42


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
loader = DataLoader(
    ImagePathDataset(image_root, transform, sample_count),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=num_workers > 0,
)
density_model = AnalyticInformationDensity(partition_stride=partition_stride).to(device).eval()
densities = []
with torch.inference_mode():
    for image in tqdm(loader, desc="density calibration", dynamic_ncols=True):
        densities.append(density_model(image.to(device, non_blocking=True)).cpu())
density = torch.cat(densities, dim=0)
partitioner = BalancedBSPPartitioner(max_tokens=max_tokens)
total_mass = density.sum(dim=(-2, -1)).flatten()
low = max(float(total_mass.min()) / max_tokens, 1e-8)
high = max(float(total_mass.max()), low * 2.0)

for _ in range(binary_search_steps):
    threshold = math.sqrt(low * high)
    average_tokens = float(partitioner(density, mass_per_token=threshold).token_count.float().mean())
    if average_tokens > target_average_tokens:
        low = threshold
    else:
        high = threshold

threshold = math.sqrt(low * high)
counts = partitioner(density, mass_per_token=threshold).token_count.float()
result = {
    "mass_per_token": threshold,
    "target_average_tokens": target_average_tokens,
    "measured_average_tokens": float(counts.mean()),
    "minimum_tokens": int(counts.min()),
    "maximum_tokens": int(counts.max()),
    "samples": int(counts.numel()),
    "image_size": image_size,
    "partition_stride": partition_stride,
    "max_tokens": max_tokens,
}
output_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
