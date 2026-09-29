from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import FractalSequenceBatch, FractalTokenizerV2, FractalTokenizerV2Config


image_root = Path("/mnt/hdd/ImageNet100/train")
checkpoint_path = Path("checkpoints_v2/fractaltokenizer_v2_vq_epoch_100.pt")
output_root = Path("sequence_cache_v2")
image_size = 256
batch_size = 32
num_workers = 8
target_tokens = None
mass_per_token = 1.0
resume = True


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, transform):
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
        if not self.paths:
            raise RuntimeError(f"no images found under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = FractalTokenizerV2Config(**checkpoint["config"])
config.partition_backend = "bsp"
config.max_tokens = max(config.max_tokens, config.target_tokens)
config.use_vector_quantization = True
model = FractalTokenizerV2(config)
model.load_state_dict(checkpoint["model"])
model = model.to(device).eval()
transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
loader = DataLoader(
    ImagePathDataset(image_root, transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=num_workers > 0,
)
output_root.mkdir(parents=True, exist_ok=True)

for step, image in enumerate(tqdm(loader, desc="cache sequences", dynamic_ncols=True), start=1):
    shard_path = output_root / f"sequence_shard_{step:07d}.ftseq"
    if resume and shard_path.exists():
        FractalSequenceBatch.load(shard_path)
        continue
    image = image.to(device, non_blocking=True)
    with torch.inference_mode():
        sequence = model.encode_sequence(
            image,
            target_tokens=target_tokens,
            mass_per_token=mass_per_token,
        )
    temporary_path = shard_path.with_suffix(".tmp")
    sequence.save(temporary_path)
    temporary_path.replace(shard_path)
    if step % 100 == 0:
        print(f"batches={step} sequences={min(step * batch_size, len(loader.dataset))}")
