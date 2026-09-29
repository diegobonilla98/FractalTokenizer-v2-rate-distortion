import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from fractaltokenizer_v2 import ContinuousAutoencoderConfig, ContinuousFractalAutoencoder


project_root = Path("/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion")
run_root = project_root / "runs/coco_continuous_autoencoder"
checkpoint_path = run_root / "best.pt"
validation_root = Path("/mnt/hdd/coco2017/val2017")
output_path = run_root / "analysis_results.json"
sheet_path = run_root / "analysis_bottleneck_controls.png"
image_size = 256
batch_size = 48
workers = 12
token_budgets = (64, 96, 128)
control_budget = 96
sample_count = 6


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.glob("*.jpg"))
        self.transform = transforms.Compose(
            [
                transforms.Resize(round(image_size * 1.143), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    difference = prediction.float() - target.float()
    mse = difference.square().mean(dim=(1, 2, 3))
    l1 = difference.abs().mean(dim=(1, 2, 3))
    prediction_x = prediction[..., :, 1:].float() - prediction[..., :, :-1].float()
    prediction_y = prediction[..., 1:, :].float() - prediction[..., :-1, :].float()
    target_x = target[..., :, 1:].float() - target[..., :, :-1].float()
    target_y = target[..., 1:, :].float() - target[..., :-1, :].float()
    gradient_l1 = 0.5 * (
        (prediction_x - target_x).abs().mean(dim=(1, 2, 3))
        + (prediction_y - target_y).abs().mean(dim=(1, 2, 3))
    )
    return {
        "mse": mse,
        "psnr": -10.0 * torch.log10(mse.clamp_min(1e-10)),
        "l1": l1,
        "gradient_l1": gradient_l1,
    }


def update(accumulator: dict[str, list[float]], name: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
    metrics = image_metrics(prediction, target)
    for metric_name, values in metrics.items():
        accumulator.setdefault(f"{name}_{metric_name}", []).extend(values.detach().cpu().tolist())


def summarize(accumulator: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for key, values in accumulator.items():
        tensor = torch.tensor(values)
        summary[key] = {
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
            "p10": float(torch.quantile(tensor, 0.1)),
            "median": float(torch.quantile(tensor, 0.5)),
            "p90": float(torch.quantile(tensor, 0.9)),
        }
    return summary


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


checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
configuration = ContinuousAutoencoderConfig(**checkpoint["model_config"])
device = torch.device("cuda")
model = ContinuousFractalAutoencoder(configuration).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
loader = DataLoader(
    ImageDataset(validation_root),
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
accumulator: dict[str, list[float]] = {}
sample_rows = None
started = time.perf_counter()
with torch.inference_mode():
    for budget in token_budgets:
        for batch_index, image in enumerate(tqdm(loader, desc=f"budget {budget}")):
            image = image.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image, target_tokens=budget)
            update(accumulator, f"refined_{budget}", output.reconstruction, image)
            update(accumulator, f"base_{budget}", output.base_reconstruction, image)
            if budget == control_budget:
                shuffled_tokens = replace(output.tokens, latents=output.tokens.latents.roll(1, dims=0))
                zero_tokens = replace(output.tokens, latents=torch.zeros_like(output.tokens.latents))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    shuffled, _ = model.decode(shuffled_tokens, refine=True)
                    zero, _ = model.decode(zero_tokens, refine=True)
                update(accumulator, "shuffled_refined_96", shuffled, image)
                update(accumulator, "zero_refined_96", zero, image)
                token_change = (output.reconstruction.float() - shuffled.float()).abs().mean(dim=(1, 2, 3))
                accumulator.setdefault("shuffle_output_change_l1", []).extend(token_change.cpu().tolist())
                if batch_index == 0:
                    count = min(sample_count, image.shape[0])
                    original = image[:count].float().cpu()
                    overlay = boundary_overlay(original, output.tokens.assignment[:count].cpu())
                    base = output.base_reconstruction[:count].float().cpu()
                    refined = output.reconstruction[:count].float().cpu()
                    shuffled_image = shuffled[:count].float().cpu()
                    zero_image = zero[:count].float().cpu()
                    rows = []
                    for index in range(count):
                        rows.extend(
                            [
                                original[index],
                                overlay[index],
                                base[index],
                                refined[index],
                                shuffled_image[index],
                                zero_image[index],
                            ]
                        )
                    sample_rows = rows
if sample_rows is not None:
    save_image(make_grid(sample_rows, nrow=6, padding=2), sheet_path)
history = [json.loads(line) for line in (run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
results = {
    "status": "complete",
    "checkpoint": str(checkpoint_path),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "validation_images": len(loader.dataset),
    "token_budgets": list(token_budgets),
    "patch16_token_reference": 256,
    "content_token_reduction_at_96": 256 / 96,
    "elapsed_seconds": time.perf_counter() - started,
    "metrics": summarize(accumulator),
    "training_history": history,
}
output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
