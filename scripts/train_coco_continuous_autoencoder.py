import datetime
import json
import math
import random
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from fractaltokenizer_v2 import (
    ContinuousAutoencoderConfig,
    ContinuousFractalAutoencoder,
    ContinuousReconstructionLoss,
)


plt.switch_backend("Agg")


configuration_path = Path("configs/coco_continuous_autoencoder.yaml")


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, transform, limit: int | None = None):
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.paths = sorted(path for path in root.iterdir() if path.suffix.lower() in extensions)
        if limit is not None:
            self.paths = self.paths[: int(limit)]
        if not self.paths:
            raise RuntimeError(f"no images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


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


def atomic_json(path: Path, values: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_configuration() -> dict[str, Any]:
    values = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    if values.get("schema_version") != 1:
        raise ValueError("unsupported configuration schema")
    return values


def resolve_dataset_root(dataset_config: dict[str, Any]) -> Path:
    staged = Path(dataset_config["staged_root"])
    source = Path(dataset_config["source_root"])
    candidate = staged if dataset_config["use_staged_root"] else source
    if not (candidate / "train2017").is_dir() or not (candidate / "val2017").is_dir():
        candidate = source
    if not (candidate / "train2017").is_dir() or not (candidate / "val2017").is_dir():
        raise FileNotFoundError(f"COCO train2017 and val2017 were not found under {candidate}")
    return candidate


def make_loaders(configuration: dict[str, Any], device: torch.device) -> tuple[DataLoader, DataLoader, ImagePathDataset, ImagePathDataset, Path]:
    dataset_config = configuration["dataset"]
    training_config = configuration["training"]
    root = resolve_dataset_root(dataset_config)
    image_size = int(dataset_config["image_size"])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.143), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    train_dataset = ImagePathDataset(root / "train2017", train_transform, dataset_config["train_limit"])
    validation_dataset = ImagePathDataset(root / "val2017", validation_transform, dataset_config["validation_limit"])
    workers = int(training_config["workers"])
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = int(training_config["prefetch_factor"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["micro_batch_size"]),
        shuffle=True,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["validation_batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, train_dataset, validation_dataset, root


def learning_rate(step: int, total_steps: int, warmup_steps: int, maximum: float, minimum: float) -> float:
    if step < warmup_steps:
        return maximum * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def boundary_overlay(image: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    owner = torch.nn.functional.interpolate(
        assignment.unsqueeze(1).float(), size=image.shape[-2:], mode="nearest"
    )[:, 0].long()
    boundary = torch.zeros_like(owner, dtype=torch.bool)
    boundary[:, :, 1:] |= owner[:, :, 1:] != owner[:, :, :-1]
    boundary[:, :, :-1] |= owner[:, :, 1:] != owner[:, :, :-1]
    boundary[:, 1:, :] |= owner[:, 1:, :] != owner[:, :-1, :]
    boundary[:, :-1, :] |= owner[:, 1:, :] != owner[:, :-1, :]
    overlay = image.clone()
    overlay[:, 0] = torch.where(boundary, torch.ones_like(overlay[:, 0]), overlay[:, 0])
    overlay[:, 1] = torch.where(boundary, torch.zeros_like(overlay[:, 1]), overlay[:, 1])
    overlay[:, 2] = torch.where(boundary, torch.zeros_like(overlay[:, 2]), overlay[:, 2])
    return overlay


def save_reconstruction_sheet(
    path: Path,
    image: torch.Tensor,
    output,
    sample_count: int,
) -> None:
    count = min(sample_count, image.shape[0])
    original = image[:count].float().cpu()
    base = output.base_reconstruction[:count].float().cpu()
    reconstruction = output.reconstruction[:count].float().cpu()
    overlay = boundary_overlay(original, output.tokens.assignment[:count].cpu())
    error = (reconstruction - original).abs().mean(dim=1, keepdim=True)
    error = (error / error.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)).repeat(1, 3, 1, 1)
    rows = []
    for index in range(count):
        rows.extend([original[index], overlay[index], base[index], reconstruction[index], error[index]])
    grid = make_grid(rows, nrow=5, padding=2)
    save_image(grid, path)


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["val_loss"] for entry in history], label="validation")
    axes[0].set_title("Reconstruction loss")
    axes[0].legend()
    axes[1].plot(epochs, [entry["train_psnr"] for entry in history], label="train")
    axes[1].plot(epochs, [entry["val_psnr"] for entry in history], label="validation")
    axes[1].set_title("PSNR")
    axes[1].legend()
    axes[2].plot(epochs, [entry["val_seam"] for entry in history], label="seam")
    axes[2].plot(epochs, [entry["val_gradient"] for entry in history], label="gradient")
    axes[2].set_title("Validation structure errors")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def aggregate_metrics(sums: dict[str, float], metrics: dict[str, torch.Tensor], batch_size: int) -> None:
    for name, value in metrics.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size
    sums["images"] = sums.get("images", 0.0) + batch_size


def averaged_metrics(sums: dict[str, float]) -> dict[str, float]:
    count = max(sums.get("images", 0.0), 1.0)
    return {name: value / count for name, value in sums.items() if name != "images"}


@torch.no_grad()
def validate(
    model: ContinuousFractalAutoencoder,
    criterion: ContinuousReconstructionLoss,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
    target_tokens: int,
) -> tuple[dict[str, float], torch.Tensor, Any]:
    model.eval()
    sums: dict[str, float] = {}
    sample_image = None
    sample_output = None
    for image in tqdm(loader, desc="validation", dynamic_ncols=True):
        image = image.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            output = model(image, target_tokens=target_tokens)
            metrics = criterion(output, image)
        aggregate_metrics(sums, metrics, image.shape[0])
        if sample_image is None:
            sample_image = image.detach()
            sample_output = output
    return averaged_metrics(sums), sample_image, sample_output


def checkpoint_payload(
    model: ContinuousFractalAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_psnr: float,
    history: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "history": history,
        "configuration": configuration,
        "model_config": asdict(model.config),
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def diagnostic_alerts(history: list[dict[str, Any]], training_config: dict[str, Any]) -> list[str]:
    current = history[-1]
    alerts = []
    if current["epoch"] >= int(training_config["underfit_epoch"]) and current["val_psnr"] < float(training_config["underfit_psnr"]):
        alerts.append(f"underfitting: val PSNR {current['val_psnr']:.2f} dB")
    gap = current["train_psnr"] - current["val_psnr"]
    if gap > float(training_config["overfit_gap_db"]):
        alerts.append(f"overfitting gap: {gap:.2f} dB")
    patience = int(training_config["stagnation_patience"])
    if len(history) > patience:
        recent = [entry["val_psnr"] for entry in history[-patience:]]
        previous_best = max(entry["val_psnr"] for entry in history[:-patience])
        if max(recent) <= previous_best + 0.05:
            alerts.append(f"validation PSNR stagnant for {patience} epochs")
    return alerts


def train() -> None:
    configuration = load_configuration()
    training_config = configuration["training"]
    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("COCO training requires CUDA")
    device = torch.device("cuda")
    output_root = Path(training_config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root = output_root / "samples"
    samples_root.mkdir(exist_ok=True)
    state: dict[str, Any] = {"status": "preparing", "output": str(output_root)}
    reporter = TelegramReporter(
        bool(training_config["telegram_enabled"]),
        bool(training_config["telegram_poll_updates"]),
        state,
        output_root,
    )
    writer = SummaryWriter(output_root / "tensorboard")
    try:
        train_loader, validation_loader, train_dataset, validation_dataset, dataset_root = make_loaders(configuration, device)
        model_config = ContinuousAutoencoderConfig(**configuration["model"])
        model = ContinuousFractalAutoencoder(model_config).to(device)
        model.density_model.requires_grad_(False)
        criterion = ContinuousReconstructionLoss(**configuration["loss"])
        decay = []
        no_decay = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith("bias") or "mass_scale" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(training_config["weight_decay"])},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(training_config["learning_rate"]),
            betas=(0.9, 0.95),
        )
        epochs = int(training_config["epochs"])
        accumulation = int(training_config["gradient_accumulation_steps"])
        updates_per_epoch = math.ceil(len(train_loader) / accumulation)
        total_steps = epochs * updates_per_epoch
        warmup_steps = int(training_config["warmup_epochs"]) * updates_per_epoch
        use_bfloat16 = bool(training_config["use_bfloat16"])
        last_checkpoint = output_root / "last.pt"
        best_checkpoint = output_root / "best.pt"
        start_epoch = 0
        global_step = 0
        best_psnr = -math.inf
        history: list[dict[str, Any]] = []
        if bool(training_config["resume"]) and last_checkpoint.is_file():
            checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"])
            global_step = int(checkpoint["global_step"])
            best_psnr = float(checkpoint["best_psnr"])
            history = list(checkpoint["history"])
            random.setstate(checkpoint["random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        metadata = {
            "dataset_root": str(dataset_root),
            "train_images": len(train_dataset),
            "validation_images": len(validation_dataset),
            "parameters": parameter_count,
            "effective_batch_size": int(training_config["micro_batch_size"]) * accumulation,
            "model_config": asdict(model_config),
        }
        atomic_json(output_root / "run_config.json", {"configuration": configuration, "metadata": metadata})
        reporter.update(status="training", **metadata)
        reporter.send(
            f"COCO continuous fractal autoencoder ready\n"
            f"train/val: {len(train_dataset)}/{len(validation_dataset)}\n"
            f"parameters: {parameter_count / 1e6:.2f}M | effective batch: {metadata['effective_batch_size']}\n"
            f"tokens: {training_config['token_budget_choices']} | output: {output_root}",
            wait=True,
        )
        announced_alerts: set[str] = set()
        for epoch in range(start_epoch, epochs):
            model.train()
            model.density_model.eval()
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
            for batch_index, image in enumerate(progress):
                image = image.to(device, non_blocking=True)
                target_tokens = random.choice(training_config["token_budget_choices"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                    output = model(image, target_tokens=int(target_tokens))
                    metrics = criterion(output, image)
                    scaled_loss = metrics["loss"] / accumulation
                if not torch.isfinite(metrics["loss"]):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}")
                scaled_loss.backward()
                aggregate_metrics(sums, metrics, image.shape[0])
                update_now = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
                if update_now:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"]))
                    current_lr = learning_rate(
                        global_step,
                        total_steps,
                        warmup_steps,
                        float(training_config["learning_rate"]),
                        float(training_config["minimum_learning_rate"]),
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                if (batch_index + 1) % 50 == 0:
                    running = averaged_metrics(sums)
                    progress.set_postfix(loss=f"{running['loss']:.4f}", psnr=f"{running['psnr']:.2f}")
            train_metrics = averaged_metrics(sums)
            validation_metrics, sample_image, sample_output = validate(
                model,
                criterion,
                validation_loader,
                device,
                use_bfloat16,
                int(training_config["validation_tokens"]),
            )
            duration = time.perf_counter() - epoch_start
            images_per_second = len(train_dataset) / max(duration, 1e-6)
            peak_memory_gb = torch.cuda.max_memory_allocated() / 2**30
            metrics_record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"val_{name}": value for name, value in validation_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": duration,
                "train_images_per_second": images_per_second,
                "peak_cuda_memory_gb": peak_memory_gb,
            }
            history.append(metrics_record)
            is_best = validation_metrics["psnr"] > best_psnr
            best_psnr = max(best_psnr, validation_metrics["psnr"])
            payload = checkpoint_payload(model, optimizer, epoch + 1, global_step, best_psnr, history, configuration)
            save_checkpoint(last_checkpoint, payload)
            if is_best:
                shutil.copy2(last_checkpoint, best_checkpoint)
            with (output_root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics_record) + "\n")
            alerts = diagnostic_alerts(history, training_config)
            run_state = {
                **metadata,
                "status": "running",
                "epoch": epoch + 1,
                "epochs": epochs,
                "best_val_psnr": best_psnr,
                "latest": metrics_record,
                "alerts": alerts,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            atomic_json(output_root / "run_state.json", run_state)
            eta_seconds = duration * (epochs - epoch - 1)
            reporter.update(
                epoch=f"{epoch + 1}/{epochs}",
                train_psnr=f"{train_metrics['psnr']:.2f} dB",
                val_psnr=f"{validation_metrics['psnr']:.2f} dB",
                best_val_psnr=f"{best_psnr:.2f} dB",
                speed=f"{images_per_second:.1f} img/s",
                peak_cuda=f"{peak_memory_gb:.1f} GiB",
                eta=str(datetime.timedelta(seconds=round(eta_seconds))),
                alerts=", ".join(alerts) if alerts else "none",
            )
            message = (
                f"COCO fractal AE epoch {epoch + 1}/{epochs}\n"
                f"train loss/PSNR: {train_metrics['loss']:.4f} / {train_metrics['psnr']:.2f} dB\n"
                f"val loss/PSNR: {validation_metrics['loss']:.4f} / {validation_metrics['psnr']:.2f} dB\n"
                f"base/seam/gradient: {validation_metrics['base']:.4f} / {validation_metrics['seam']:.4f} / {validation_metrics['gradient']:.4f}\n"
                f"best: {best_psnr:.2f} dB | speed: {images_per_second:.1f} img/s | CUDA: {peak_memory_gb:.1f} GiB\n"
                f"ETA: {datetime.timedelta(seconds=round(eta_seconds))}"
            )
            if alerts:
                message += "\nalerts: " + "; ".join(alerts)
            reporter.send(message)
            new_alerts = set(alerts) - announced_alerts
            if new_alerts:
                reporter.send("ALERT COCO fractal AE: " + "; ".join(sorted(new_alerts)), wait=True)
                announced_alerts.update(new_alerts)
            for name, value in metrics_record.items():
                if isinstance(value, (int, float)) and name not in {"epoch", "global_step"}:
                    writer.add_scalar(name, value, epoch + 1)
            writer.flush()
            if (epoch + 1) % int(training_config["sample_every"]) == 0 or epoch == start_epoch:
                sample_path = samples_root / f"epoch_{epoch + 1:03d}.png"
                save_reconstruction_sheet(sample_path, sample_image, sample_output, int(training_config["sample_count"]))
                reporter.photo(sample_path, "Columns: original, fractal partition, local decoder, refined reconstruction, normalized error")
            if (epoch + 1) % int(training_config["telegram_plot_every"]) == 0 or epoch + 1 == epochs:
                plot_path = output_root / "training_curves.png"
                plot_history(history, plot_path)
                reporter.photo(plot_path, f"COCO fractal AE through epoch {epoch + 1}")
        final_state = {
            **metadata,
            "status": "complete",
            "epochs": epochs,
            "best_val_psnr": best_psnr,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        atomic_json(output_root / "run_state.json", final_state)
        reporter.update(status="complete", best_val_psnr=f"{best_psnr:.2f} dB")
        reporter.send(f"COCO continuous fractal autoencoder complete\nbest val PSNR: {best_psnr:.2f} dB\nartifacts: {output_root}", wait=True)
    except Exception:
        failure = traceback.format_exc()
        atomic_json(
            output_root / "failure.json",
            {
                "status": "failed",
                "traceback": failure,
                "failed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
        reporter.update(status="failed", error=failure[-2000:])
        reporter.send(f"COCO continuous fractal autoencoder FAILED\n{failure[-3000:]}", wait=True)
        raise
    finally:
        writer.close()
        reporter.close()


if __name__ == "__main__":
    train()
