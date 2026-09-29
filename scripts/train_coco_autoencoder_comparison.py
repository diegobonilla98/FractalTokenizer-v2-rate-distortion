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

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from fractaltokenizer_v2 import (
    ContinuousAutoencoderConfig,
    ContinuousFractalAutoencoder,
    ContinuousReconstructionLoss,
    GridAutoencoderConfig,
    GridViTAutoencoder,
)
from train_coco_continuous_autoencoder import (
    TelegramReporter,
    aggregate_metrics,
    atomic_json,
    averaged_metrics,
    diagnostic_alerts,
    learning_rate,
    make_loaders,
    plot_history,
    save_reconstruction_sheet,
)


configuration_path = Path("configs/coco_autoencoder_comparison.yaml")


def load_configuration() -> dict[str, Any]:
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    if configuration.get("schema_version") != 1:
        raise ValueError("unsupported configuration schema")
    variants = configuration["training"]["variants"]
    names = [variant["name"] for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("variant names must be unique")
    expected = {(family, 128) for family in ("grid", "fractal")}
    actual = {(variant["family"], int(variant["tokens"])) for variant in variants}
    if actual != expected:
        raise ValueError("variants must contain grid and fractal runs at 128 tokens")
    if int(configuration["training"]["epochs"]) != 10:
        raise ValueError("the comparison protocol requires exactly 10 epochs")
    return configuration


def model_for_variant(configuration: dict[str, Any], variant: dict[str, Any]):
    architecture = dict(configuration["architecture"])
    token_count = int(variant["tokens"])
    if variant["family"] == "grid":
        grid_config = GridAutoencoderConfig(
            image_height=int(configuration["dataset"]["image_size"]),
            image_width=int(configuration["dataset"]["image_size"]),
            token_count=token_count,
            **architecture,
        )
        return GridViTAutoencoder(grid_config), grid_config
    fractal_values = dict(configuration["fractal"])
    fractal_values["target_tokens"] = token_count
    fractal_values["max_tokens"] = max(256, token_count)
    fractal_config = ContinuousAutoencoderConfig(**architecture, **fractal_values)
    return ContinuousFractalAutoencoder(fractal_config), fractal_config


def optimizer_for_model(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "mass_scale" in name or "position_embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(training_config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(training_config["learning_rate"]),
        betas=(0.9, 0.95),
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_psnr: float,
    history: list[dict[str, Any]],
    configuration: dict[str, Any],
    variant: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "history": history,
        "configuration": configuration,
        "variant": variant,
        "model_config": asdict(model.config),
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(model, criterion, loader, device: torch.device, use_bfloat16: bool, token_count: int):
    model.eval()
    sums: dict[str, float] = {}
    sample_image = None
    sample_output = None
    for image in tqdm(loader, desc="validation", dynamic_ncols=True):
        image = image.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            output = model(image, target_tokens=token_count)
            metrics = criterion(output, image)
        aggregate_metrics(sums, metrics, image.shape[0])
        if sample_image is None:
            sample_image = image.detach()
            sample_output = output
    return averaged_metrics(sums), sample_image, sample_output


def completed_variant(output_root: Path, epochs: int) -> bool:
    state_path = output_root / "run_state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("status") == "complete" and int(state.get("epochs", 0)) == epochs


def run_variant(
    configuration: dict[str, Any],
    variant: dict[str, Any],
    device: torch.device,
    reporter: TelegramReporter,
) -> dict[str, Any]:
    training_config = configuration["training"]
    epochs = int(training_config["epochs"])
    output_root = Path(training_config["output_root"]).resolve() / variant["name"]
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root = output_root / "samples"
    samples_root.mkdir(exist_ok=True)
    if completed_variant(output_root, epochs):
        state = json.loads((output_root / "run_state.json").read_text(encoding="utf-8"))
        reporter.send(f"Skipping completed comparison run: {variant['name']}")
        return state
    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    train_loader, validation_loader, train_dataset, validation_dataset, dataset_root = make_loaders(configuration, device)
    model, model_config = model_for_variant(configuration, variant)
    model = model.to(device)
    if hasattr(model, "density_model"):
        model.density_model.requires_grad_(False)
    criterion = ContinuousReconstructionLoss(**configuration["loss"])
    optimizer = optimizer_for_model(model, training_config)
    accumulation = int(training_config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = epochs * updates_per_epoch
    warmup_steps = int(training_config["warmup_epochs"]) * updates_per_epoch
    use_bfloat16 = bool(training_config["use_bfloat16"])
    token_count = int(variant["tokens"])
    last_checkpoint = output_root / "last.pt"
    best_checkpoint = output_root / "best.pt"
    start_epoch = 0
    global_step = 0
    best_psnr = -math.inf
    history: list[dict[str, Any]] = []
    if bool(training_config["resume"]) and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["variant"] != variant:
            raise ValueError(f"checkpoint variant mismatch under {output_root}")
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
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    metadata = {
        "variant": variant["name"],
        "family": variant["family"],
        "tokens": token_count,
        "dataset_root": str(dataset_root),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "effective_batch_size": int(training_config["micro_batch_size"]) * accumulation,
        "model_config": asdict(model_config),
    }
    atomic_json(output_root / "run_config.json", {"configuration": configuration, "metadata": metadata})
    writer = SummaryWriter(output_root / "tensorboard")
    reporter.update(status="training", active_variant=variant["name"], **metadata)
    reporter.send(
        f"Starting COCO AE comparison: {variant['name']}\n"
        f"family/tokens: {variant['family']} / {token_count}\n"
        f"parameters: {parameter_count / 1e6:.2f}M | train/val: {len(train_dataset)}/{len(validation_dataset)}\n"
        f"resume epoch: {start_epoch} | output: {output_root}",
        wait=True,
    )
    announced_alerts: set[str] = set()
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            if hasattr(model, "density_model"):
                model.density_model.eval()
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            progress = tqdm(train_loader, desc=f"{variant['name']} epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
            for batch_index, image in enumerate(progress):
                image = image.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                    output = model(image, target_tokens=token_count)
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
                token_count,
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
            save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                epoch + 1,
                global_step,
                best_psnr,
                history,
                configuration,
                variant,
            )
            if is_best:
                shutil.copy2(last_checkpoint, best_checkpoint)
            with (output_root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics_record) + "\n")
            alerts = diagnostic_alerts(history, training_config)
            state = {
                **metadata,
                "status": "running",
                "epoch": epoch + 1,
                "epochs": epochs,
                "best_val_psnr": best_psnr,
                "latest": metrics_record,
                "alerts": alerts,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            atomic_json(output_root / "run_state.json", state)
            reporter.update(
                epoch=f"{epoch + 1}/{epochs}",
                train_psnr=f"{train_metrics['psnr']:.2f} dB",
                val_psnr=f"{validation_metrics['psnr']:.2f} dB",
                best_val_psnr=f"{best_psnr:.2f} dB",
                speed=f"{images_per_second:.1f} img/s",
                peak_cuda=f"{peak_memory_gb:.1f} GiB",
                alerts=", ".join(alerts) if alerts else "none",
            )
            reporter.send(
                f"{variant['name']} epoch {epoch + 1}/{epochs}\n"
                f"train loss/PSNR: {train_metrics['loss']:.4f} / {train_metrics['psnr']:.2f} dB\n"
                f"val loss/PSNR: {validation_metrics['loss']:.4f} / {validation_metrics['psnr']:.2f} dB\n"
                f"base/seam/gradient: {validation_metrics['base']:.4f} / {validation_metrics['seam']:.4f} / {validation_metrics['gradient']:.4f}\n"
                f"best: {best_psnr:.2f} dB | speed: {images_per_second:.1f} img/s | CUDA: {peak_memory_gb:.1f} GiB"
            )
            new_alerts = set(alerts) - announced_alerts
            if new_alerts:
                reporter.send(f"ALERT {variant['name']}: " + "; ".join(sorted(new_alerts)), wait=True)
                announced_alerts.update(new_alerts)
            for name, value in metrics_record.items():
                if isinstance(value, (int, float)) and name not in {"epoch", "global_step"}:
                    writer.add_scalar(name, value, epoch + 1)
            writer.flush()
            if (epoch + 1) % int(training_config["sample_every"]) == 0:
                sample_path = samples_root / f"epoch_{epoch + 1:03d}.png"
                save_reconstruction_sheet(sample_path, sample_image, sample_output, int(training_config["sample_count"]))
                reporter.photo(
                    sample_path,
                    f"{variant['name']}: original, token grid, base decoder, refined reconstruction, normalized error",
                )
            if (epoch + 1) % int(training_config["telegram_plot_every"]) == 0 or epoch + 1 == epochs:
                plot_path = output_root / "training_curves.png"
                plot_history(history, plot_path)
                reporter.photo(plot_path, f"{variant['name']} through epoch {epoch + 1}")
        final_state = {
            **metadata,
            "status": "complete",
            "epochs": epochs,
            "best_val_psnr": best_psnr,
            "final": history[-1],
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        atomic_json(output_root / "run_state.json", final_state)
        reporter.send(f"Completed {variant['name']} | best val PSNR {best_psnr:.2f} dB", wait=True)
        return final_state
    finally:
        writer.close()
        del model, optimizer, train_loader, validation_loader
        torch.cuda.empty_cache()


def write_comparison_summary(output_root: Path, results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda result: (int(result["tokens"]), result["family"]))
    summary = {
        "status": "complete" if all(result["status"] == "complete" for result in ordered) else "partial",
        "results": ordered,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json(output_root / "comparison_summary.json", summary)
    lines = [
        "# COCO autoencoder comparison",
        "",
        "| Tokens | Family | Parameters | Best val PSNR | Final val PSNR | Train img/s | Peak CUDA GiB |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in ordered:
        final = result.get("final", result.get("latest", {}))
        lines.append(
            f"| {result['tokens']} | {result['family']} | {result['parameters'] / 1e6:.2f}M | "
            f"{result['best_val_psnr']:.3f} | {final.get('val_psnr', float('nan')):.3f} | "
            f"{final.get('train_images_per_second', float('nan')):.1f} | "
            f"{final.get('peak_cuda_memory_gb', float('nan')):.1f} |"
        )
    (output_root / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train() -> None:
    configuration = load_configuration()
    training_config = configuration["training"]
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("COCO comparison training requires CUDA")
    device = torch.device("cuda")
    output_root = Path(training_config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"status": "preparing", "output": str(output_root)}
    reporter = TelegramReporter(
        bool(training_config["telegram_enabled"]),
        bool(training_config["telegram_poll_updates"]),
        state,
        output_root,
    )
    results = []
    try:
        reporter.send(
            "Starting matched COCO autoencoder sweep\n"
            "grid and A/B fractal at 128 tokens\n"
            "10 epochs each; completed variants resume or skip",
            wait=True,
        )
        for variant in training_config["variants"]:
            result = run_variant(configuration, dict(variant), device, reporter)
            results.append(result)
            write_comparison_summary(output_root, results)
        reporter.update(status="complete", active_variant="none")
        reporter.send(f"COCO autoencoder comparison complete\nartifacts: {output_root}", wait=True)
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
        reporter.send(f"COCO autoencoder comparison FAILED\n{failure[-3000:]}", wait=True)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    train()
