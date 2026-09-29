from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import MaskedSurprisalDensity


image_root = Path("/mnt/hdd/coco2017/train2017")
checkpoint_path = Path("surprisal_density_v2.pt")
image_size = 256
partition_stride = 8
hidden_dim = 96
depth = 4
batch_size = 32
num_workers = 8
epochs = 20
learning_rate = 3e-4
weight_decay = 1e-4
seed = 42


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


torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
loader = DataLoader(
    ImagePathDataset(image_root, transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=num_workers > 0,
    drop_last=True,
)
model = MaskedSurprisalDensity(
    channels=3,
    hidden_dim=hidden_dim,
    depth=depth,
    partition_stride=partition_stride,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

for epoch in range(epochs):
    model.train()
    running = 0.0
    progress = tqdm(loader, desc=f"surprisal epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
    for step, image in enumerate(progress, start=1):
        image = image.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            loss = model.training_loss(image)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
        if step % 100 == 0:
            progress.set_postfix(nll=f"{running / step:.5f}")
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "partition_stride": partition_stride,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "epoch": epoch + 1,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
