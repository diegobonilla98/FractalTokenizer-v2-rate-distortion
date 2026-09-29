from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from .classification import ClassifierConfig, count_parameters, make_classifier
from .cuda_ops import build_cuda_extension
from .universal_codelength import UniversalCodelengthDensity


matplotlib.use("Agg")


@dataclass
class TrainingConfig:
    variants: tuple[str, ...] = ("fractal_128", "vit_rect_128", "vit_patch16_256")
    seeds: tuple[int, ...] = (42,)
    source_dataset_root: str = "/mnt/hdd/ImageNet100"
    staged_dataset_root: str = "/dev/shm/ImageNet100"
    stage_to_shared_memory: bool = True
    output_root: str = "runs/imagenet100_comparison"
    image_size: int = 256
    num_classes: int = 100
    epochs: int = 100
    micro_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    workers: int = 16
    prefetch_factor: int = 4
    model_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    token_count: int = 128
    partition_stride: int = 8
    tile_size: int = 8
    basis_size: int = 4
    learning_rate: float = 5e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    gradient_clip_norm: float = 1.0
    random_erase_probability: float = 0.25
    use_bfloat16: bool = True
    compile_model: bool = False
    resume: bool = True
    telegram_enabled: bool = True
    telegram_poll_updates: bool = True
    telegram_plot_every: int = 5
    underfit_epoch: int = 20
    underfit_train_top1: float = 30.0
    overfit_gap: float = 15.0
    collapse_epoch: int = 5
    collapse_train_top1: float = 2.0
    stagnation_patience: int = 15
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    density_temperature: float = 1.0
    density_coherence_floor: float = 0.35
    density_blur_sigma: float = 0.0
    kernel_backend: str = "auto"
    partition_density_floor_ratio: float = 0.02
    region_sampling_ratio: int = 2
    compile_density: bool = True
    universal_mass_per_token: float = 9926.331293741861
    maximum_tokens: int = 160
    minimum_epochs: int = 30
    early_stopping_patience: int = 8
    mass_calibration_images: int = 2048
    mass_calibration_target_tokens: float = 96.0
    mass_validation_images: int = 5000
    configuration_file: str = ""
    run_profile: str = ""
    density_profile: str = "legacy_default"
    v3_capacity_bits_64: float = 11154.006347656252
    v3_capacity_bits_96: float = 6857.214111328126


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _image_inventory(root: Path) -> dict[str, Any]:
    extensions = {".jpeg", ".jpg", ".png", ".webp"}
    result: dict[str, Any] = {"root": str(root), "splits": {}}
    for split in ("train", "val"):
        split_root = root / split
        classes = sorted(path.name for path in split_root.iterdir() if path.is_dir()) if split_root.is_dir() else []
        image_paths = sorted(
            path.relative_to(split_root).as_posix()
            for path in split_root.rglob("*")
            if path.suffix.lower() in extensions
        )
        manifest = hashlib.sha256("\n".join(image_paths).encode("utf-8")).hexdigest()
        class_manifest = hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()
        result["splits"][split] = {
            "classes": len(classes),
            "images": len(image_paths),
            "class_manifest_sha256": class_manifest,
            "file_manifest_sha256": manifest,
        }
    return result


def validate_imagenet100(root: Path) -> dict[str, Any]:
    inventory = _image_inventory(root)
    train = inventory["splits"]["train"]
    validation = inventory["splits"]["val"]
    if train["classes"] != 100 or validation["classes"] != 100:
        raise RuntimeError(f"expected 100 classes in both splits, found {inventory['splits']}")
    if train["images"] != 130_000 or validation["images"] != 5_000:
        raise RuntimeError(f"ImageNet100 inventory is incomplete: {inventory['splits']}")
    return inventory


def prepare_imagenet100(config: TrainingConfig) -> tuple[Path, dict[str, Any]]:
    source = Path(config.source_dataset_root)
    source_inventory = validate_imagenet100(source)
    if not config.stage_to_shared_memory:
        return source, source_inventory
    target = Path(config.staged_dataset_root)
    marker = target / ".fractaltokenizer_stage.json"
    if marker.is_file():
        target_inventory = validate_imagenet100(target)
        if target_inventory["splits"] == source_inventory["splits"]:
            _atomic_json(
                marker,
                {
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "source": str(source),
                    "inventory": target_inventory,
                },
            )
            return target, target_inventory
    required_bytes = int(subprocess.check_output(["du", "-sb", str(source)], text=True).split()[0])
    free_bytes = shutil.disk_usage(target.parent).free
    if free_bytes < required_bytes * 1.1:
        raise RuntimeError(
            f"not enough shared memory to stage ImageNet100: need {required_bytes}, have {free_bytes}"
        )
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--info=progress2", f"{source}/", f"{target}/"],
        check=True,
    )
    target_inventory = validate_imagenet100(target)
    if target_inventory["splits"] != source_inventory["splits"]:
        raise RuntimeError("staged ImageNet100 inventory does not match its source")
    _atomic_json(
        marker,
        {
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": str(source),
            "inventory": target_inventory,
        },
    )
    return target, target_inventory


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _data_transforms(config: TrainingConfig) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=config.random_erase_probability, value="random"),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(config.image_size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, validation_transform


def make_loaders(
    config: TrainingConfig,
    data_root: Path,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_transform, validation_transform = _data_transforms(config)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(data_root / "val", transform=validation_transform)
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise RuntimeError("train and validation class mappings differ")
    if len(train_dataset.classes) != config.num_classes:
        raise RuntimeError(f"expected {config.num_classes} classes, found {len(train_dataset.classes)}")
    workers = min(config.workers, os.cpu_count() or 1)
    common = {
        "batch_size": config.micro_batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, train_dataset.class_to_idx


@torch.inference_mode()
def calibrate_universal_mass(
    config: TrainingConfig,
    data_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    if config.mass_calibration_images < 1:
        return {
            "enabled": False,
            "mass_per_token": config.universal_mass_per_token,
        }
    _, calibration_transform = _data_transforms(config)
    dataset = datasets.ImageFolder(data_root / "train", transform=calibration_transform)
    image_count = min(int(config.mass_calibration_images), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, image_count, dtype=np.int64).tolist()
    workers = min(config.workers, os.cpu_count() or 1)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=config.prefetch_factor if workers > 0 else None,
    )
    density_model = UniversalCodelengthDensity(
        partition_stride=config.partition_stride,
        backend=config.kernel_backend,
    ).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    totals = []
    started = time.perf_counter()
    for images, _ in tqdm(loader, desc="calibrate universal mass", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        raw_images = (images.float() * std + mean).clamp(0.0, 1.0)
        totals.append(density_model(raw_images).sum(dim=(1, 2, 3)).cpu())
    total_bits = torch.cat(totals).double()
    target = float(config.mass_calibration_target_tokens)
    lower = float(total_bits.min() / max(config.maximum_tokens, 1))
    upper = float(total_bits.max())
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        counts = torch.ceil(total_bits / midpoint).clamp(1, config.maximum_tokens)
        if float(counts.mean()) > target:
            lower = midpoint
        else:
            upper = midpoint
    threshold = upper
    counts = torch.ceil(total_bits / threshold).clamp(1, config.maximum_tokens)
    validation_dataset = datasets.ImageFolder(data_root / "val", transform=calibration_transform)
    validation_count = min(int(config.mass_validation_images), len(validation_dataset))
    validation_indices = np.linspace(
        0,
        len(validation_dataset) - 1,
        validation_count,
        dtype=np.int64,
    ).tolist()
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=config.prefetch_factor if workers > 0 else None,
    )
    validation_totals = []
    for images, _ in tqdm(validation_loader, desc="validate universal mass", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        raw_images = (images.float() * std + mean).clamp(0.0, 1.0)
        validation_totals.append(density_model(raw_images).sum(dim=(1, 2, 3)).cpu())
    validation_bits = torch.cat(validation_totals).double()
    validation_counts = torch.ceil(validation_bits / threshold).clamp(1, config.maximum_tokens)
    config.universal_mass_per_token = threshold
    elapsed = time.perf_counter() - started
    del density_model
    torch.cuda.empty_cache()
    return {
        "enabled": True,
        "source_split": "train",
        "labels_used": False,
        "images": image_count,
        "target_average_tokens": target,
        "mass_per_token": threshold,
        "measured_average_tokens": float(counts.mean()),
        "minimum_tokens": int(counts.min()),
        "maximum_tokens": int(counts.max()),
        "token_cap_fraction": float((counts >= config.maximum_tokens).double().mean()),
        "heldout_split": "val",
        "heldout_images": validation_count,
        "heldout_average_tokens": float(validation_counts.mean()),
        "heldout_minimum_tokens": int(validation_counts.min()),
        "heldout_maximum_tokens": int(validation_counts.max()),
        "heldout_token_cap_fraction": float(
            (validation_counts >= config.maximum_tokens).double().mean()
        ),
        "elapsed_seconds": elapsed,
    }


class TelegramReporter:
    def __init__(self, config: TrainingConfig, state: dict[str, Any], output_root: Path):
        self.state = state
        self.bot = None
        if not config.telegram_enabled:
            return
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("Telegram disabled: export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before launch")
            return
        self.bot = TelegramBot(
            working_directory=str(output_root),
            poll_updates=config.telegram_poll_updates,
            enable_terminal_commands=False,
        )
        self.bot.set_state_provider(lambda: dict(self.state))

    def update(self, **values: Any) -> None:
        self.state.update(values)
        if self.bot is not None:
            self.bot.update_state(**values)

    def send(self, text: str, wait: bool = False) -> None:
        if self.bot is None:
            return
        try:
            self.bot.send_message(text=text, wait=wait)
        except Exception as error:
            print(f"Telegram send failed: {error}")

    def photo(self, path: Path, text: str) -> None:
        if self.bot is None or not path.is_file():
            return
        try:
            self.bot.send_message(photo=str(path), text=text, wait=False)
        except Exception as error:
            print(f"Telegram photo failed: {error}")

    def close(self) -> None:
        if self.bot is not None:
            self.bot.stop(wait=True)


def _classifier_config(config: TrainingConfig, variant: str) -> ClassifierConfig:
    return ClassifierConfig(
        variant=variant,
        image_size=config.image_size,
        num_classes=config.num_classes,
        token_count=config.token_count,
        model_dim=config.model_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        partition_stride=config.partition_stride,
        tile_size=config.tile_size,
        basis_size=config.basis_size,
        density_temperature=config.density_temperature,
        density_coherence_floor=config.density_coherence_floor,
        density_blur_sigma=config.density_blur_sigma,
        kernel_backend=config.kernel_backend,
        partition_density_floor_ratio=config.partition_density_floor_ratio,
        region_sampling_ratio=config.region_sampling_ratio,
        compile_density=config.compile_density,
        universal_mass_per_token=config.universal_mass_per_token,
        maximum_tokens=config.maximum_tokens,
        v3_capacity_bits_64=config.v3_capacity_bits_64,
        v3_capacity_bits_96=config.v3_capacity_bits_96,
    )


def _optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "position_embedding" in name or "cls_token" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    effective_batch = config.micro_batch_size * config.gradient_accumulation_steps
    learning_rate = config.learning_rate * effective_batch / 512
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
    )


def _learning_rate(config: TrainingConfig, step: int, total_steps: int, warmup_steps: int) -> float:
    effective_batch = config.micro_batch_size * config.gradient_accumulation_steps
    peak = config.learning_rate * effective_batch / 512
    minimum = config.minimum_learning_rate * effective_batch / 512
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _mixed_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mixed_target: torch.Tensor,
    mixup_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    first = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    second = F.cross_entropy(logits, mixed_target, label_smoothing=label_smoothing)
    return mixup_weight * first + (1.0 - mixup_weight) * second


def _topk(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = logits.topk(5, dim=1).indices
    matches = prediction.eq(target[:, None])
    top1 = matches[:, :1].any(dim=1).float().sum().item()
    top5 = matches.any(dim=1).float().sum().item()
    return top1, top5


def _token_stats(model: nn.Module, batch_size: int) -> tuple[torch.Tensor, int, int]:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    counts = getattr(raw_model, "last_token_counts", None)
    padded = getattr(raw_model, "last_padded_token_count", None)
    if counts is None or padded is None:
        count = int(getattr(raw_model, "token_count"))
        counts = torch.full((batch_size,), count, device=next(raw_model.parameters()).device)
        padded = count
    capacity = int(getattr(getattr(raw_model, "token_embed", None), "maximum_tokens", 0))
    return counts, int(padded), capacity


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    sample_count = 0
    gradient_norm = 0.0
    active_token_sum = torch.zeros((), device=device)
    maximum_active_tokens = torch.zeros((), device=device)
    padded_token_sum = 0
    clipped_token_count = torch.zeros((), device=device)
    start = time.perf_counter()
    limit = len(loader) if config.max_train_batches is None else min(len(loader), config.max_train_batches)
    progress = tqdm(loader, total=limit, desc=f"train {epoch + 1:03d}", dynamic_ncols=True)
    for batch_index, (images, target) in enumerate(progress):
        if batch_index >= limit:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if config.mixup_alpha > 0:
            permutation = torch.randperm(images.shape[0], device=device)
            mixup_weight = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
            mixed_images = mixup_weight * images + (1.0 - mixup_weight) * images[permutation]
            mixed_target = target[permutation]
        else:
            mixup_weight = 1.0
            mixed_images = images
            mixed_target = target
        if config.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bfloat16):
            logits = model(mixed_images)
            loss = _mixed_loss(
                logits,
                target,
                mixed_target,
                mixup_weight,
                config.label_smoothing,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}, batch {batch_index}")
        (loss / config.gradient_accumulation_steps).backward()
        should_step = (batch_index + 1) % config.gradient_accumulation_steps == 0 or batch_index + 1 == limit
        if should_step:
            learning_rate = _learning_rate(config, global_step, total_steps, warmup_steps)
            _set_learning_rate(optimizer, learning_rate)
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            gradient_norm = float(norm.detach())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        batch_size = images.shape[0]
        token_counts, padded_tokens, token_capacity = _token_stats(model, batch_size)
        active_token_sum += token_counts.float().sum()
        maximum_active_tokens = torch.maximum(maximum_active_tokens, token_counts.max().float())
        padded_token_sum += padded_tokens * batch_size
        if token_capacity > 0:
            clipped_token_count += (token_counts >= token_capacity).sum()
        first_top1, first_top5 = _topk(logits.detach(), target)
        second_top1, second_top5 = _topk(logits.detach(), mixed_target)
        loss_sum += float(loss.detach()) * batch_size
        top1_sum += mixup_weight * first_top1 + (1.0 - mixup_weight) * second_top1
        top5_sum += mixup_weight * first_top5 + (1.0 - mixup_weight) * second_top5
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}", top1=f"{100 * top1_sum / sample_count:.1f}")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    mean_active_tokens = float(active_token_sum) / sample_count
    mean_padded_tokens = padded_token_sum / sample_count
    return (
        {
            "train_loss": loss_sum / sample_count,
            "train_top1": 100.0 * top1_sum / sample_count,
            "train_top5": 100.0 * top5_sum / sample_count,
            "train_images_per_second": sample_count / elapsed,
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_mean_active_tokens": mean_active_tokens,
            "train_mean_padded_tokens": mean_padded_tokens,
            "train_token_padding_fraction": 1.0 - mean_active_tokens / mean_padded_tokens,
            "train_max_active_tokens": float(maximum_active_tokens),
            "train_token_cap_fraction": float(clipped_token_count) / sample_count,
        },
        global_step,
    )


@torch.inference_mode()
def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    sample_count = 0
    active_token_sum = torch.zeros((), device=device)
    maximum_active_tokens = torch.zeros((), device=device)
    padded_token_sum = 0
    clipped_token_count = torch.zeros((), device=device)
    start = time.perf_counter()
    limit = len(loader) if config.max_val_batches is None else min(len(loader), config.max_val_batches)
    progress = tqdm(loader, total=limit, desc=f"valid {epoch + 1:03d}", dynamic_ncols=True)
    for batch_index, (images, target) in enumerate(progress):
        if batch_index >= limit:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if config.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch + 1}, batch {batch_index}")
        top1, top5 = _topk(logits, target)
        batch_size = images.shape[0]
        token_counts, padded_tokens, token_capacity = _token_stats(model, batch_size)
        active_token_sum += token_counts.float().sum()
        maximum_active_tokens = torch.maximum(maximum_active_tokens, token_counts.max().float())
        padded_token_sum += padded_tokens * batch_size
        if token_capacity > 0:
            clipped_token_count += (token_counts >= token_capacity).sum()
        loss_sum += float(loss) * batch_size
        top1_sum += top1
        top5_sum += top5
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}", top1=f"{100 * top1_sum / sample_count:.1f}")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    mean_active_tokens = float(active_token_sum) / sample_count
    mean_padded_tokens = padded_token_sum / sample_count
    val_top1 = 100.0 * top1_sum / sample_count
    return {
        "val_loss": loss_sum / sample_count,
        "val_top1": val_top1,
        "val_top5": 100.0 * top5_sum / sample_count,
        "val_images_per_second": sample_count / elapsed,
        "val_mean_active_tokens": mean_active_tokens,
        "val_mean_padded_tokens": mean_padded_tokens,
        "val_token_padding_fraction": 1.0 - mean_active_tokens / mean_padded_tokens,
        "val_max_active_tokens": float(maximum_active_tokens),
        "val_token_cap_fraction": float(clipped_token_count) / sample_count,
        "val_top1_per_100_tokens": val_top1 * 100.0 / mean_active_tokens,
    }


def _alerts(history: list[dict[str, Any]], config: TrainingConfig) -> list[str]:
    current = history[-1]
    epoch = int(current["epoch"])
    alerts = []
    if epoch >= config.collapse_epoch and current["train_top1"] < config.collapse_train_top1:
        alerts.append("possible collapse: training top-1 remains near chance")
    if epoch >= config.underfit_epoch and current["train_top1"] < config.underfit_train_top1:
        alerts.append("possible underfitting: training accuracy is below the configured floor")
    if current["train_top1"] - current["val_top1"] > config.overfit_gap:
        alerts.append("possible overfitting: train-validation accuracy gap is large")
    best_epoch = max(range(len(history)), key=lambda index: history[index]["val_top1"]) + 1
    if epoch - best_epoch >= config.stagnation_patience:
        alerts.append("validation has stagnated")
    return alerts


def _plot_history(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["train_top1"] for item in history], label="train top-1")
    axes[1].plot(epochs, [item["val_top1"] for item in history], label="validation top-1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].legend()
    axes[2].plot(epochs, [item["val_mean_active_tokens"] for item in history], label="active")
    axes[2].plot(epochs, [item["val_mean_padded_tokens"] for item in history], label="padded")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("tokens per image")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _raw_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_top1: float,
    history: list[dict[str, Any]],
    generator: torch.Generator,
    config: TrainingConfig,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": _raw_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_top1": best_top1,
            "history": history,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "loader_generator_rng": generator.get_state(),
            "config": asdict(config),
        },
        temporary,
    )
    os.replace(temporary, path)


def _restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> tuple[int, int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _raw_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    generator.set_state(checkpoint["loader_generator_rng"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint["best_top1"]),
        list(checkpoint["history"]),
    )


def train_experiment(
    config: TrainingConfig,
    variant: str,
    seed: int,
    data_root: Path,
    reporter: TelegramReporter,
) -> dict[str, Any]:
    experiment_name = f"{variant}_seed{seed}"
    output = Path(config.output_root).resolve() / experiment_name
    output.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader, validation_loader, class_to_idx = make_loaders(config, data_root, generator)
    _atomic_json(output / "class_to_idx.json", class_to_idx)
    device = torch.device("cuda")
    model = make_classifier(_classifier_config(config, variant)).to(device)
    optimizer = _optimizer(model, config)
    trainable_parameters, total_parameters = count_parameters(model)
    token_count = model.token_count
    if variant.startswith("fractal_") and config.kernel_backend != "torch":
        build_cuda_extension()
    if config.compile_model:
        model = torch.compile(model, mode="default")
    updates_per_epoch = math.ceil(
        (len(train_loader) if config.max_train_batches is None else min(len(train_loader), config.max_train_batches))
        / config.gradient_accumulation_steps
    )
    total_steps = config.epochs * updates_per_epoch
    warmup_steps = config.warmup_epochs * updates_per_epoch
    start_epoch = 0
    global_step = 0
    best_top1 = -math.inf
    history: list[dict[str, Any]] = []
    last_checkpoint = output / "last.pt"
    if config.resume and last_checkpoint.is_file():
        start_epoch, global_step, best_top1, history = _restore_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            generator,
        )
    metadata = {
        "experiment": experiment_name,
        "variant": variant,
        "seed": seed,
        "tokens": token_count,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "effective_batch_size": config.micro_batch_size * config.gradient_accumulation_steps,
        "data_root": str(data_root),
        "config": asdict(config),
    }
    _atomic_json(output / "config.json", metadata)
    reporter.update(
        status="running",
        experiment=experiment_name,
        epoch=f"{start_epoch}/{config.epochs}",
        tokens=token_count,
        parameters=f"{trainable_parameters / 1e6:.2f}M",
        output=str(output),
    )
    reporter.send(
        f"Started {experiment_name}\n"
        f"tokens: {token_count}\nparameters: {trainable_parameters / 1e6:.2f}M\n"
        f"batch: {config.micro_batch_size} x {config.gradient_accumulation_steps}\ndata: {data_root}",
        wait=True,
    )
    writer = SummaryWriter(output / "tensorboard")
    metrics_path = output / "metrics.jsonl"
    experiment_start = time.perf_counter()
    announced_alerts: set[str] = set()
    stopped_early = False
    for epoch in range(start_epoch, config.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics, global_step = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            epoch,
            global_step,
            total_steps,
            warmup_steps,
        )
        validation_metrics = _validate_epoch(model, validation_loader, device, config, epoch)
        elapsed = time.perf_counter() - experiment_start
        epochs_completed = epoch - start_epoch + 1
        eta_seconds = elapsed / epochs_completed * (config.epochs - epoch - 1)
        metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            **train_metrics,
            **validation_metrics,
            "accuracy_gap": train_metrics["train_top1"] - validation_metrics["val_top1"],
            "peak_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
        }
        history.append(metrics)
        is_best = metrics["val_top1"] > best_top1
        if is_best:
            best_top1 = metrics["val_top1"]
        current_alerts = _alerts(history, config)
        metrics["alerts"] = current_alerts
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, sort_keys=True) + "\n")
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and name not in {"epoch", "global_step"}:
                writer.add_scalar(name, value, epoch + 1)
        writer.flush()
        _save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch,
            global_step,
            best_top1,
            history,
            generator,
            config,
        )
        if is_best:
            shutil.copy2(last_checkpoint, output / "best.pt")
        state = {
            **metadata,
            "status": "running",
            "epoch": epoch + 1,
            "best_val_top1": best_top1,
            "latest": metrics,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _atomic_json(output / "run_state.json", state)
        reporter.update(
            epoch=f"{epoch + 1}/{config.epochs}",
            train_loss=f"{metrics['train_loss']:.4f}",
            train_top1=f"{metrics['train_top1']:.2f}%",
            val_loss=f"{metrics['val_loss']:.4f}",
            val_top1=f"{metrics['val_top1']:.2f}%",
            best_val_top1=f"{best_top1:.2f}%",
            eta=str(datetime.timedelta(seconds=round(eta_seconds))),
            alerts=", ".join(current_alerts) if current_alerts else "none",
        )
        message = (
            f"{experiment_name} epoch {epoch + 1}/{config.epochs}\n"
            f"train loss/top1: {metrics['train_loss']:.4f} / {metrics['train_top1']:.2f}%\n"
            f"val loss/top1/top5: {metrics['val_loss']:.4f} / {metrics['val_top1']:.2f}% / {metrics['val_top5']:.2f}%\n"
            f"best: {best_top1:.2f}% | gap: {metrics['accuracy_gap']:.2f} pp\n"
            f"tokens active/padded: {metrics['val_mean_active_tokens']:.1f} / {metrics['val_mean_padded_tokens']:.1f}\n"
            f"speed: {metrics['train_images_per_second']:.1f} img/s | peak CUDA: {metrics['peak_cuda_memory_gb']:.1f} GiB\n"
            f"ETA: {datetime.timedelta(seconds=round(eta_seconds))}"
        )
        if current_alerts:
            message += "\nalerts: " + "; ".join(current_alerts)
        reporter.send(message)
        new_alerts = set(current_alerts) - announced_alerts
        if new_alerts:
            reporter.send(f"ALERT {experiment_name}: " + "; ".join(sorted(new_alerts)), wait=True)
            announced_alerts.update(new_alerts)
        if (epoch + 1) % config.telegram_plot_every == 0 or epoch + 1 == config.epochs:
            plot_path = output / "training_curves.png"
            _plot_history(history, plot_path)
            reporter.photo(plot_path, f"{experiment_name} through epoch {epoch + 1}")
        best_epoch = max(range(len(history)), key=lambda index: history[index]["val_top1"]) + 1
        if (
            epoch + 1 >= config.minimum_epochs
            and epoch + 1 - best_epoch >= config.early_stopping_patience
        ):
            stopped_early = True
            reporter.send(
                f"Early stopping {experiment_name} after epoch {epoch + 1}; best epoch was {best_epoch}",
                wait=True,
            )
            break
    writer.close()
    final_state = {
        **metadata,
        "status": "complete",
        "best_val_top1": best_top1,
        "best_epoch": max(range(len(history)), key=lambda index: history[index]["val_top1"]) + 1,
        "latest": history[-1],
        "stopped_early": stopped_early,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _atomic_json(output / "run_state.json", final_state)
    reporter.update(status="complete", best_val_top1=f"{best_top1:.2f}%")
    reporter.send(
        f"Finished {experiment_name}\nbest val top-1: {best_top1:.2f}% at epoch {final_state['best_epoch']}\n"
        f"artifacts: {output}",
        wait=True,
    )
    return final_state


def run_comparison(config: TrainingConfig) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet100 training requires CUDA")
    if config.micro_batch_size < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("batch sizes must be positive")
    if config.model_dim % config.num_heads:
        raise ValueError("model_dim must be divisible by num_heads")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    output_root = Path(config.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "failure.json").unlink(missing_ok=True)
    state: dict[str, Any] = {"status": "preparing", "output": str(output_root)}
    reporter = TelegramReporter(config, state, output_root)
    results = []
    try:
        data_root, inventory = prepare_imagenet100(config)
        _atomic_json(output_root / "dataset_inventory.json", inventory)
        reporter.update(data_root=str(data_root), dataset=inventory["splits"])
        reporter.send(f"ImageNet100 ready at {data_root}\n{inventory['splits']}", wait=True)
        if "hilbert_universal_96" in config.variants:
            calibration = calibrate_universal_mass(config, data_root, torch.device("cuda"))
            _atomic_json(output_root / "mass_calibration.json", calibration)
            reporter.update(
                universal_mass_per_token=f"{calibration['mass_per_token']:.3f}",
                calibration_tokens=f"{calibration.get('measured_average_tokens', 0.0):.2f}",
            )
            reporter.send(
                f"Universal mass calibrated without labels\n"
                f"images: {calibration.get('images', 0)} | target/measured tokens: "
                f"{calibration.get('target_average_tokens', 0.0):.1f} / "
                f"{calibration.get('measured_average_tokens', 0.0):.2f}\n"
                f"held-out validation tokens: {calibration.get('heldout_average_tokens', 0.0):.2f}\n"
                f"mass per token: {calibration['mass_per_token']:.3f}",
                wait=True,
            )
        for seed in config.seeds:
            for variant in config.variants:
                results.append(train_experiment(config, variant, seed, data_root, reporter))
        summary = {
            "status": "complete",
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "results": results,
            "ranking": sorted(
                (
                    {
                        "variant": item["variant"],
                        "seed": item["seed"],
                        "best_val_top1": item["best_val_top1"],
                        "best_epoch": item["best_epoch"],
                        "mean_active_tokens_at_end": item["latest"]["val_mean_active_tokens"],
                        "val_top1_per_100_tokens_at_end": item["latest"]["val_top1_per_100_tokens"],
                    }
                    for item in results
                ),
                key=lambda item: item["best_val_top1"],
                reverse=True,
            ),
        }
        _atomic_json(output_root / "comparison_summary.json", summary)
        reporter.update(status="all complete")
        reporter.send(f"All ImageNet100 comparison runs finished\nartifacts: {output_root}", wait=True)
        return results
    except Exception:
        failure = traceback.format_exc()
        state.update(status="failed", error=failure[-3000:])
        _atomic_json(
            output_root / "failure.json",
            {
                "failed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "traceback": failure,
                "state": state,
            },
        )
        reporter.send(f"TRAINING FAILED\n{failure[-3500:]}\nartifacts: {output_root}", wait=True)
        raise
    finally:
        reporter.close()
