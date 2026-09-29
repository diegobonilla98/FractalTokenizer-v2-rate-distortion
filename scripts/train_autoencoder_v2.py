import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from fractaltokenizer_v2 import FractalRateDistortionLoss, FractalTokenizerV2, FractalTokenizerV2Config
from fractaltokenizer_v2.configuration import load_density_profile


image_root = Path("/mnt/hdd/ImageNet100/train")
configuration_path = Path("configs/imagenet100.yaml")
checkpoint_dir = Path("checkpoints_v2")
metrics_path = checkpoint_dir / "autoencoder_metrics.jsonl"
run_state_path = checkpoint_dir / "run_state.json"
training_stage = "continuous"
initialization_checkpoint_path = None
resume_checkpoint_path = None
density_checkpoint_path = None
image_size = 256
batch_size = 64
num_workers = 8
epochs = 100
learning_rate = 2e-4
weight_decay = 0.05
save_every = 1
seed = 42
freeze_density_model = True
load_optimizer_state = True
token_budget_choices = (64, 96, 128, 192)
mass_per_token_choices = None
density_profile_name, density_parameters = load_density_profile(configuration_path)

if training_stage not in {"continuous", "vq"}:
    raise ValueError("training_stage must be continuous or vq")

config = FractalTokenizerV2Config(
    latent_dim=192,
    tile_size=8,
    basis_size=4,
    partition_stride=8,
    target_tokens=128,
    token_grid=None,
    partition_backend="quantile",
    kernel_backend="cuda",
    density_mode="analytic",
    **density_parameters,
    use_vector_quantization=training_stage == "vq",
    codebook_size=8192,
    num_quantizers=1,
    codebook_ema_decay=0.99,
    dead_code_threshold=0.1,
)


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
random.seed(seed)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
model = FractalTokenizerV2(config)
if density_checkpoint_path is not None:
    density_checkpoint = torch.load(density_checkpoint_path, map_location="cpu", weights_only=False)
    model.density_model.load_state_dict(density_checkpoint["model"])
if initialization_checkpoint_path is not None:
    initialization = torch.load(initialization_checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(initialization["model"], strict=False)
    print(f"initialization_missing={missing}")
    print(f"initialization_unexpected={unexpected}")
if freeze_density_model:
    model.density_model.requires_grad_(False)
model = model.to(device)
criterion = FractalRateDistortionLoss(
    codebook_size=config.codebook_size,
    num_quantizers=config.num_quantizers,
)
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=learning_rate,
    weight_decay=weight_decay,
)
start_epoch = 0
if resume_checkpoint_path is not None:
    checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    start_epoch = int(checkpoint.get("epoch", 0))
    if load_optimizer_state and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "random_state" in checkpoint:
        random.setstate(checkpoint["random_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
run_state = {
    "status": "starting",
    "epoch": start_epoch,
    "epochs": epochs,
    "training_stage": training_stage,
    "partition_backend": config.partition_backend,
    "latest_checkpoint": None,
}

for epoch in range(start_epoch, epochs):
    model.train()
    if freeze_density_model:
        model.density_model.eval()
    running = {}
    progress = tqdm(loader, desc=f"{training_stage} epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
    for step, image in enumerate(progress, start=1):
        image = image.to(device, non_blocking=True)
        target_tokens = None
        mass_per_token = None
        if token_budget_choices is not None:
            if training_stage == "vq" and epoch == start_epoch and step == 1:
                target_tokens = max(token_budget_choices)
            else:
                target_tokens = random.choice(token_budget_choices)
        if config.partition_backend == "bsp" and mass_per_token_choices is not None:
            mass_per_token = random.choice(mass_per_token_choices)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            output = model(image, target_tokens=target_tokens, mass_per_token=mass_per_token)
            losses = criterion(output, image)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + float(value.detach())
        if step % 50 == 0:
            progress.set_postfix({name: f"{value / step:.4f}" for name, value in running.items()})
    epoch_metrics = {
        "epoch": epoch + 1,
        "training_stage": training_stage,
        "partition_backend": config.partition_backend,
        **{name: value / max(len(loader), 1) for name, value in running.items()},
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(epoch_metrics) + "\n")
    if (epoch + 1) % save_every == 0:
        checkpoint_path = checkpoint_dir / f"fractaltokenizer_v2_{training_stage}_epoch_{epoch + 1:03d}.pt"
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "epoch": epoch + 1,
                "training_stage": training_stage,
                "random_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)
    run_state = {
        "status": "running",
        "epoch": epoch + 1,
        "epochs": epochs,
        "training_stage": training_stage,
        "partition_backend": config.partition_backend,
        "latest_checkpoint": str(checkpoint_path) if (epoch + 1) % save_every == 0 else None,
    }
    temporary_run_state_path = run_state_path.with_suffix(".tmp")
    temporary_run_state_path.write_text(json.dumps(run_state, indent=2), encoding="utf-8")
    temporary_run_state_path.replace(run_state_path)

run_state["status"] = "complete"
temporary_run_state_path = run_state_path.with_suffix(".tmp")
temporary_run_state_path.write_text(json.dumps(run_state, indent=2), encoding="utf-8")
temporary_run_state_path.replace(run_state_path)
