import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from fractaltokenizer_v2 import FractalAutoregressivePrior, FractalPriorConfig, FractalRateDistortionLoss, FractalSequenceBatch, FractalTokenizerV2, FractalTokenizerV2Config, RasterAutoregressivePrior, RasterPriorConfig, RasterSequenceBatch, collate_fractal_sequences, collate_raster_sequences
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot


configuration_path = Path("configs/supervisely_generation.yaml")
smoke_mode = os.environ.get("FRACTAL_SUPERVISELY_SMOKE", "0") == "1"
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageManifestDataset(Dataset):
    def __init__(self, root: Path, relative_paths: list[str], transform):
        self.root = root
        self.relative_paths = relative_paths
        self.transform = transform

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, index):
        path = self.root / self.relative_paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


class SequenceShardDataset(Dataset):
    def __init__(self, root: Path, suffix: str):
        self.paths = sorted(root.glob(f"*{suffix}"))
        if not self.paths:
            raise RuntimeError(f"no {suffix} shards found under {root}")
        self.suffix = suffix

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.suffix == ".ftseq":
            return FractalSequenceBatch.load(self.paths[index])
        return RasterSequenceBatch.load(self.paths[index])


class TelegramReporter:
    def __init__(self, enabled: bool, poll_updates: bool, output_root: Path):
        self.bot = None
        self.state = {}
        if enabled and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.bot = TelegramBot(
                working_directory=str(output_root),
                poll_updates=poll_updates,
                enable_terminal_commands=False,
            )
            self.bot.set_state_provider(lambda: dict(self.state))
        elif enabled:
            print("Telegram disabled: export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before launch")

    def update(self, **values):
        self.state.update(values)
        if self.bot is not None:
            self.bot.update_state(**values)

    def send(self, text: str, wait: bool = False):
        if self.bot is None:
            return
        try:
            self.bot.send_message(text=text, wait=wait)
        except Exception as error:
            print(f"Telegram send failed: {error}")

    def photo(self, path: Path, text: str):
        if self.bot is None or not path.is_file():
            return
        try:
            self.bot.send_message(photo=str(path), text=text, wait=False)
        except Exception as error:
            print(f"Telegram photo failed: {error}")

    def close(self):
        if self.bot is not None:
            self.bot.stop(wait=True)


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def append_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


def load_configuration():
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    if int(configuration.get("schema_version", 0)) != 1:
        raise ValueError("unsupported Supervisely configuration schema")
    if smoke_mode:
        configuration["run"]["output_root"] = "runs/supervisely_generation/smoke"
        configuration["tokenizer"]["continuous_epochs"] = 1
        configuration["tokenizer"]["vq_epochs"] = 1
        configuration["tokenizer"]["batch_size"] = 2
        configuration["tokenizer"]["validation_batch_size"] = 2
        configuration["tokenizer"]["workers"] = 0
        configuration["prior"]["epochs"] = 1
        configuration["prior"]["cache_batch_size"] = 2
        configuration["prior"]["workers"] = 0
        configuration["prior"]["model_dim"] = 64
        configuration["prior"]["depth"] = 2
        configuration["prior"]["heads"] = 4
        configuration["run"]["telegram_enabled"] = False
    return configuration


def resolve_image_root(configuration):
    staged = Path(configuration["dataset"]["staged_image_root"])
    source = Path(configuration["dataset"]["source_image_root"])
    root = staged if staged.is_dir() and any(staged.iterdir()) else source
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def create_manifest(configuration, image_root: Path, output_root: Path):
    manifest_path = output_root / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split in ("train", "validation", "test"):
            for relative_path in manifest[split]:
                if not (image_root / relative_path).is_file():
                    raise FileNotFoundError(image_root / relative_path)
        return manifest
    paths = sorted(path.relative_to(image_root).as_posix() for path in image_root.rglob("*") if path.suffix.lower() in extensions)
    if smoke_mode:
        paths = paths[:12]
    random_generator = random.Random(int(configuration["dataset"]["split_seed"]))
    random_generator.shuffle(paths)
    count = len(paths)
    train_end = int(round(count * float(configuration["dataset"]["train_fraction"])))
    validation_end = train_end + int(round(count * float(configuration["dataset"]["validation_fraction"])))
    train_end = min(max(train_end, 1), count - 2)
    validation_end = min(max(validation_end, train_end + 1), count - 1)
    manifest = {
        "version": 1,
        "seed": int(configuration["dataset"]["split_seed"]),
        "image_root": str(image_root),
        "train": paths[:train_end],
        "validation": paths[train_end:validation_end],
        "test": paths[validation_end:],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def build_loaders(configuration, image_root: Path, manifest):
    size = int(configuration["dataset"]["image_size"])
    tokenizer_values = configuration["tokenizer"]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(size, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    workers = int(tokenizer_values["workers"])
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        ImageManifestDataset(image_root, manifest["train"], train_transform),
        batch_size=int(tokenizer_values["batch_size"]),
        shuffle=True,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        ImageManifestDataset(image_root, manifest["validation"], evaluation_transform),
        batch_size=int(tokenizer_values["validation_batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, evaluation_transform


def tokenizer_config(configuration, variant_name: str, use_vector_quantization: bool):
    tokenizer_values = configuration["tokenizer"]
    density = configuration["density_profile"]
    variant = configuration["variants"][variant_name]
    standard = variant["kind"] == "standard"
    token_grid = tuple(variant["token_grid"]) if standard else None
    target_tokens = int(variant["target_tokens"]) if standard else int(variant["target_average_tokens"])
    max_tokens = target_tokens if standard else int(variant["max_tokens"])
    return FractalTokenizerV2Config(
        latent_dim=int(tokenizer_values["latent_dim"]),
        tile_size=int(tokenizer_values["tile_size"]),
        basis_size=int(tokenizer_values["basis_size"]),
        partition_stride=int(tokenizer_values["partition_stride"]),
        target_tokens=target_tokens,
        token_grid=token_grid,
        partition_backend=str(variant["partition_backend"]),
        kernel_backend="cuda",
        mass_per_token=None if standard else float(variant["mass_per_token"]),
        max_tokens=max_tokens,
        bsp_workers=min(8, int(tokenizer_values["workers"])),
        density_mode="uniform" if standard else "analytic",
        density_temperature=float(density["temperature"]),
        density_coherence_floor=float(density["coherence_floor"]),
        density_blur_sigma=float(density["blur_sigma"]),
        use_vector_quantization=use_vector_quantization,
        codebook_size=int(tokenizer_values["codebook_size"]),
        num_quantizers=int(tokenizer_values["num_quantizers"]),
        codebook_ema_decay=0.99,
        dead_code_threshold=0.1,
    )


def build_tokenizer(configuration, variant_name: str, use_vector_quantization: bool):
    return FractalTokenizerV2(tokenizer_config(configuration, variant_name, use_vector_quantization))


def forward_tokenizer(model, image, variant):
    if variant["kind"] == "fractal":
        return model(image, target_tokens=None, mass_per_token=float(variant["mass_per_token"]))
    return model(image, target_tokens=int(variant["target_tokens"]))


def save_reconstruction_sheet(path: Path, target: torch.Tensor, reconstruction: torch.Tensor):
    count = min(8, target.shape[0])
    paired = torch.stack([target[:count], reconstruction[:count]], dim=1).flatten(0, 1).cpu().clamp(0, 1)
    grid = make_grid(paired, nrow=4, padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(grid).save(path)


def evaluate_tokenizer(model, criterion, loader, variant, device, amp_enabled):
    model.eval()
    totals = {}
    images = 0
    sample = None
    with torch.inference_mode():
        for image in loader:
            image = image.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = forward_tokenizer(model, image, variant)
                losses = criterion(output, image)
            count = image.shape[0]
            images += count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value) * count
            mse = (output.reconstruction.float() - image.float()).square().flatten(1).mean(dim=1)
            totals["mse"] = totals.get("mse", 0.0) + float(mse.sum())
            if sample is None:
                sample = (image.detach(), output.reconstruction.detach())
    metrics = {name: value / images for name, value in totals.items()}
    metrics["psnr_db"] = -10.0 * math.log10(max(metrics["mse"], 1e-12))
    return metrics, sample


def save_tokenizer_checkpoint(path, model, optimizer, scheduler, scaler, epoch, stage, variant_name):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(model.config),
        "epoch": epoch,
        "training_stage": stage,
        "variant": variant_name,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def train_tokenizer_stage(configuration, variant_name, stage, train_loader, validation_loader, reporter, device):
    output_root = Path(configuration["run"]["output_root"]) / variant_name / stage
    output_root.mkdir(parents=True, exist_ok=True)
    run_state_path = output_root / "run_state.json"
    latest_path = output_root / "latest.pt"
    best_path = output_root / "best.pt"
    if run_state_path.is_file() and configuration["run"]["resume"]:
        state = json.loads(run_state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete" and best_path.is_file():
            print(f"skip complete tokenizer stage {variant_name} {stage}")
            return best_path
    use_vq = stage == "vq"
    model = build_tokenizer(configuration, variant_name, use_vq)
    start_epoch = 0
    best_psnr = -math.inf
    tokenizer_values = configuration["tokenizer"]
    epochs = int(tokenizer_values["vq_epochs"] if use_vq else tokenizer_values["continuous_epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tokenizer_values["learning_rate"]), weight_decay=float(tokenizer_values["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(tokenizer_values["minimum_learning_rate"]))
    amp_enabled = device.type == "cuda" and bool(configuration["run"]["use_bfloat16"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if latest_path.is_file() and configuration["run"]["resume"]:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_psnr = float(checkpoint.get("best_psnr", -math.inf))
    elif use_vq:
        initialization_path = Path(configuration["run"]["output_root"]) / variant_name / "continuous" / "best.pt"
        if not initialization_path.is_file():
            raise FileNotFoundError(initialization_path)
        initialization = torch.load(initialization_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(initialization["model"], strict=False)
        if unexpected or any(not name.startswith("quantizer.") for name in missing):
            raise RuntimeError(f"invalid VQ initialization missing={missing} unexpected={unexpected}")
    model = model.to(device)
    criterion = FractalRateDistortionLoss(codebook_size=model.config.codebook_size, num_quantizers=model.config.num_quantizers)
    variant = configuration["variants"][variant_name]
    reporter.send(f"Supervisely {variant_name} {stage} started: {epochs} epochs")
    for epoch in range(start_epoch, epochs):
        model.train()
        running = {}
        images = 0
        started = time.perf_counter()
        progress = tqdm(train_loader, desc=f"{variant_name} {stage} {epoch + 1}/{epochs}", dynamic_ncols=True)
        for image in progress:
            image = image.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = forward_tokenizer(model, image, variant)
                losses = criterion(output, image)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError(f"non-finite tokenizer loss for {variant_name} {stage}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tokenizer_values["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            count = image.shape[0]
            images += count
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach()) * count
        scheduler.step()
        train_metrics = {name: value / images for name, value in running.items()}
        validation_metrics, sample = evaluate_tokenizer(model, criterion, validation_loader, variant, device, amp_enabled)
        elapsed = time.perf_counter() - started
        metrics = {
            "epoch": epoch + 1,
            "variant": variant_name,
            "stage": stage,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": elapsed,
            "images_per_second": images / elapsed,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        append_json(output_root / "metrics.jsonl", metrics)
        improved = validation_metrics["psnr_db"] > best_psnr
        best_psnr = max(best_psnr, validation_metrics["psnr_db"])
        save_tokenizer_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch + 1, stage, variant_name)
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        checkpoint["best_psnr"] = best_psnr
        temporary_path = latest_path.with_suffix(".best.tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(latest_path)
        if improved:
            temporary_path = best_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(best_path)
        sheet_path = output_root / "samples" / f"epoch_{epoch + 1:03d}.png"
        if (epoch + 1) % int(tokenizer_values["sample_every"]) == 0 or epoch + 1 == epochs:
            save_reconstruction_sheet(sheet_path, *sample)
            reporter.photo(sheet_path, f"{variant_name} {stage} epoch {epoch + 1}: val PSNR {validation_metrics['psnr_db']:.2f} dB")
        state = {
            "status": "running",
            "variant": variant_name,
            "stage": stage,
            "epoch": epoch + 1,
            "epochs": epochs,
            "best_validation_psnr_db": best_psnr,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
        }
        write_json_atomic(run_state_path, state)
        reporter.update(**state)
        reporter.send(f"{variant_name} {stage} {epoch + 1}/{epochs}: train {train_metrics['loss']:.4f}, val {validation_metrics['loss']:.4f}, PSNR {validation_metrics['psnr_db']:.2f} dB, tokens {validation_metrics['tokens']:.1f}")
    state["status"] = "complete"
    write_json_atomic(run_state_path, state)
    reporter.send(f"Supervisely {variant_name} {stage} complete: best PSNR {best_psnr:.2f} dB")
    return best_path


def load_trained_tokenizer(configuration, variant_name, device):
    checkpoint_path = Path(configuration["run"]["output_root"]) / variant_name / "vq" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FractalTokenizerV2(FractalTokenizerV2Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint_path


def cache_sequences(configuration, variant_name, image_root, manifest, evaluation_transform, reporter, device):
    cache_root = Path(configuration["run"]["output_root"]) / variant_name / "sequence_cache"
    complete_path = cache_root / "complete.json"
    if complete_path.is_file() and configuration["run"]["resume"]:
        print(f"skip complete sequence cache {variant_name}")
        return cache_root
    model, checkpoint_path = load_trained_tokenizer(configuration, variant_name, device)
    variant = configuration["variants"][variant_name]
    batch_size = int(configuration["prior"]["cache_batch_size"])
    workers = int(configuration["prior"]["workers"])
    suffix = ".ftseq" if variant["kind"] == "fractal" else ".rtseq"
    summary = {"variant": variant_name, "tokenizer_checkpoint": str(checkpoint_path), "splits": {}}
    for split in ("train", "validation", "test"):
        split_root = cache_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(
            ImageManifestDataset(image_root, manifest[split], evaluation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        content_tokens = []
        events = []
        for step, image in enumerate(tqdm(loader, desc=f"cache {variant_name} {split}", dynamic_ncols=True)):
            shard_path = split_root / f"shard_{step:06d}{suffix}"
            image = image.to(device, non_blocking=True)
            with torch.inference_mode():
                if variant["kind"] == "fractal":
                    sequence = model.encode_sequence(image, target_tokens=None, mass_per_token=float(variant["mass_per_token"]))
                    if not torch.equal(sequence.sequence_length, 2 * sequence.token_count - 1):
                        raise RuntimeError("BSP sequence length does not equal 2N-1")
                    content_tokens.extend(int(value) for value in sequence.token_count.cpu())
                    events.extend(int(value) for value in sequence.sequence_length.cpu())
                else:
                    tokens, _, _ = model.encode(image, target_tokens=int(variant["target_tokens"]))
                    sequence = RasterSequenceBatch(
                        code_indices=tokens.code_indices,
                        token_mask=tokens.mask,
                        token_grid=tuple(variant["token_grid"]),
                        image_shape=tuple(image.shape[-2:]),
                    )
                    content_tokens.extend(int(value) for value in sequence.sequence_length.cpu())
                    events.extend(int(value) for value in sequence.sequence_length.cpu())
            temporary_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
            sequence.save(temporary_path)
            temporary_path.replace(shard_path)
        summary["splits"][split] = {
            "images": len(content_tokens),
            "shards": len(loader),
            "mean_content_tokens": sum(content_tokens) / len(content_tokens),
            "minimum_content_tokens": min(content_tokens),
            "maximum_content_tokens": max(content_tokens),
            "mean_autoregressive_events": sum(events) / len(events),
        }
    write_json_atomic(complete_path, summary)
    reporter.send(f"Cached {variant_name}: train mean {summary['splits']['train']['mean_content_tokens']:.1f} content tokens, {summary['splits']['train']['mean_autoregressive_events']:.1f} AR events")
    return cache_root


def prior_configuration(configuration, variant_name):
    values = configuration["prior"]
    tokenizer_values = configuration["tokenizer"]
    common = {
        "codebook_size": int(tokenizer_values["codebook_size"]),
        "num_quantizers": int(tokenizer_values["num_quantizers"]),
        "model_dim": int(values["model_dim"]),
        "depth": int(values["depth"]),
        "heads": int(values["heads"]),
        "mlp_ratio": float(values["mlp_ratio"]),
        "dropout": float(values["dropout"]),
    }
    variant = configuration["variants"][variant_name]
    if variant["kind"] == "fractal":
        return FractalPriorConfig(
            max_grid_size=int(configuration["dataset"]["image_size"]) // int(tokenizer_values["partition_stride"]),
            max_sequence_length=2 * int(variant["max_tokens"]) - 1,
            **common,
        )
    return RasterPriorConfig(max_sequence_length=int(variant["target_tokens"]), **common)


def build_prior(configuration, variant_name):
    config = prior_configuration(configuration, variant_name)
    if configuration["variants"][variant_name]["kind"] == "fractal":
        return FractalAutoregressivePrior(config)
    return RasterAutoregressivePrior(config)


def evaluate_prior(model, loader, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_bits = 0.0
    images = 0
    with torch.inference_mode():
        for sequence in loader:
            sequence = sequence.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                losses = model.loss(sequence)
                rate = model.estimated_bits(sequence)
            count = sequence.code_indices.shape[0]
            images += count
            total_loss += float(losses["loss"]) * count
            total_bits += float(rate["total_bits"].sum())
    return {"loss": total_loss / images, "bits_per_image": total_bits / images, "bits_per_pixel": total_bits / images / (sequence.image_shape[0] * sequence.image_shape[1])}


def train_prior(configuration, variant_name, cache_root, reporter, device):
    output_root = Path(configuration["run"]["output_root"]) / variant_name / "prior"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "run_state.json"
    latest_path = output_root / "latest.pt"
    best_path = output_root / "best.pt"
    if state_path.is_file() and configuration["run"]["resume"]:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete" and best_path.is_file():
            print(f"skip complete prior {variant_name}")
            return best_path
    variant = configuration["variants"][variant_name]
    suffix = ".ftseq" if variant["kind"] == "fractal" else ".rtseq"
    collate = collate_fractal_sequences if variant["kind"] == "fractal" else collate_raster_sequences
    values = configuration["prior"]
    workers = int(values["workers"])
    common = {
        "batch_size": int(values["shards_per_batch"]),
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "collate_fn": collate,
    }
    train_loader = DataLoader(SequenceShardDataset(cache_root / "train", suffix), shuffle=True, drop_last=True, **common)
    validation_loader = DataLoader(SequenceShardDataset(cache_root / "validation", suffix), shuffle=False, drop_last=False, **common)
    model = build_prior(configuration, variant_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(values["learning_rate"]), weight_decay=float(values["weight_decay"]))
    epochs = int(values["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(values["minimum_learning_rate"]))
    amp_enabled = device.type == "cuda" and bool(configuration["run"]["use_bfloat16"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_bits = math.inf
    if latest_path.is_file() and configuration["run"]["resume"]:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_bits = float(checkpoint.get("best_validation_bits_per_image", math.inf))
    reporter.send(f"Supervisely {variant_name} prior started: {epochs} epochs")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        images = 0
        started = time.perf_counter()
        progress = tqdm(train_loader, desc=f"{variant_name} prior {epoch + 1}/{epochs}", dynamic_ncols=True)
        for sequence in progress:
            sequence = sequence.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                losses = model.loss(sequence)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError(f"non-finite prior loss for {variant_name}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(values["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            count = sequence.code_indices.shape[0]
            images += count
            total_loss += float(losses["loss"].detach()) * count
        scheduler.step()
        validation = evaluate_prior(model, validation_loader, device, amp_enabled)
        elapsed = time.perf_counter() - started
        metrics = {
            "epoch": epoch + 1,
            "variant": variant_name,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": elapsed,
            "images_per_second": images / elapsed,
            "train_loss": total_loss / images,
            "validation": validation,
        }
        append_json(output_root / "metrics.jsonl", metrics)
        improved = validation["bits_per_image"] < best_bits
        best_bits = min(best_bits, validation["bits_per_image"])
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(model.config),
            "epoch": epoch + 1,
            "variant": variant_name,
            "prior_kind": variant["kind"],
            "best_validation_bits_per_image": best_bits,
        }
        temporary_path = latest_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(latest_path)
        if improved:
            temporary_path = best_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(best_path)
        state = {
            "status": "running",
            "variant": variant_name,
            "epoch": epoch + 1,
            "epochs": epochs,
            "best_validation_bits_per_image": best_bits,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
        }
        write_json_atomic(state_path, state)
        reporter.update(**state)
        reporter.send(f"{variant_name} prior {epoch + 1}/{epochs}: train {metrics['train_loss']:.4f}, val {validation['loss']:.4f}, {validation['bits_per_pixel']:.4f} bpp")
    state["status"] = "complete"
    write_json_atomic(state_path, state)
    reporter.send(f"Supervisely {variant_name} prior complete: best {best_bits:.1f} bits/image")
    return best_path


def generate_samples(configuration, variant_name, reporter, device):
    output_root = Path(configuration["run"]["output_root"]) / variant_name
    output_path = output_root / "generated_samples.png"
    model, _ = load_trained_tokenizer(configuration, variant_name, device)
    checkpoint_path = output_root / "prior" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = configuration["variants"][variant_name]
    if variant["kind"] == "fractal":
        prior = FractalAutoregressivePrior(FractalPriorConfig(**checkpoint["config"]))
    else:
        prior = RasterAutoregressivePrior(RasterPriorConfig(**checkpoint["config"]))
    prior.load_state_dict(checkpoint["model"])
    prior = prior.to(device).eval()
    batch_size = 2 if smoke_mode else 8
    image_size = int(configuration["dataset"]["image_size"])
    with torch.inference_mode():
        if variant["kind"] == "fractal":
            grid_size = image_size // int(configuration["tokenizer"]["partition_stride"])
            sequence = prior.generate(
                batch_size=batch_size,
                grid_shape=(grid_size, grid_size),
                image_shape=(image_size, image_size),
                min_tokens=16,
                max_tokens=int(variant["max_tokens"]),
                sample=True,
                temperature=1.0,
                top_k=64,
            )
            reconstruction = model.decode_sequence(sequence)
            details = {"content_tokens": sequence.token_count.tolist(), "autoregressive_events": sequence.sequence_length.tolist()}
        else:
            sequence = prior.generate(
                batch_size=batch_size,
                token_grid=tuple(variant["token_grid"]),
                image_shape=(image_size, image_size),
                sample=True,
                temperature=1.0,
                top_k=64,
            )
            blank = torch.zeros(batch_size, 3, image_size, image_size, device=device)
            density = model.density_model(blank)
            partition = model.partition(density, target_tokens=int(variant["target_tokens"]))
            reconstruction, _ = model.decode_codes(sequence.code_indices, partition, (image_size, image_size))
            details = {"content_tokens": sequence.sequence_length.tolist(), "autoregressive_events": sequence.sequence_length.tolist()}
    to_pil_image(make_grid(reconstruction.cpu().clamp(0, 1), nrow=4, padding=2)).save(output_path)
    write_json_atomic(output_root / "generated_samples.json", details)
    reporter.photo(output_path, f"Supervisely generated samples: {variant_name}")
    return output_path


def run():
    configuration = load_configuration()
    seed = int(configuration["run"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Supervisely comparison must run on the DGX GPU")
    output_root = Path(configuration["run"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    image_root = resolve_image_root(configuration)
    manifest = create_manifest(configuration, image_root, output_root)
    train_loader, validation_loader, evaluation_transform = build_loaders(configuration, image_root, manifest)
    reporter = TelegramReporter(
        bool(configuration["run"]["telegram_enabled"]),
        bool(configuration["run"]["telegram_poll_updates"]),
        output_root,
    )
    try:
        stages = configuration["run"]["stages"]
        variants = list(configuration["variants"])
        if "continuous" in stages:
            for variant_name in variants:
                train_tokenizer_stage(configuration, variant_name, "continuous", train_loader, validation_loader, reporter, device)
        if "vq" in stages:
            for variant_name in variants:
                train_tokenizer_stage(configuration, variant_name, "vq", train_loader, validation_loader, reporter, device)
        caches = {}
        if "cache" in stages:
            for variant_name in variants:
                caches[variant_name] = cache_sequences(configuration, variant_name, image_root, manifest, evaluation_transform, reporter, device)
        if "prior" in stages:
            for variant_name in variants:
                cache_root = caches.get(variant_name, output_root / variant_name / "sequence_cache")
                train_prior(configuration, variant_name, cache_root, reporter, device)
        if "sample" in stages:
            for variant_name in variants:
                generate_samples(configuration, variant_name, reporter, device)
        reporter.send("Supervisely fractal-versus-standard generation comparison complete", wait=True)
    finally:
        reporter.close()


run()
