import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from fractaltokenizer_v2 import FractalTokenizerV2, FractalTokenizerV2Config
from fractaltokenizer_v2.regions import render_token_tiles, sample_region_tiles


project = Path("/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion")
root = project / "runs/supervisely_generation/comparison"
mask_root = Path("/mnt/hdd/supervisely_person_clean_2667_img/masks")
validation_samples = 64
batch_size = 8
image_size = 256


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FractalTokenizerV2(FractalTokenizerV2Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def segmentation_metrics(prediction, target):
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).flatten(1).sum(dim=1).float()
    union = (prediction | target).flatten(1).sum(dim=1).float()
    predicted = prediction.flatten(1).sum(dim=1).float()
    actual = target.flatten(1).sum(dim=1).float()
    return {
        "iou": intersection / union.clamp_min(1),
        "dice": 2.0 * intersection / (predicted + actual).clamp_min(1),
        "pixel_accuracy": (prediction == target).flatten(1).float().mean(dim=1),
    }


manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
image_root = Path(manifest["image_root"])
if not image_root.is_dir():
    image_root = Path("/mnt/hdd/supervisely_person_clean_2667_img/images")
relative_paths = manifest["validation"][:validation_samples]
image_transform = transforms.Compose(
    [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)
mask_transform = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ]
)
images = torch.stack([image_transform(Image.open(image_root / path).convert("RGB")) for path in relative_paths])
masks = torch.stack([mask_transform(Image.open(mask_root / path).convert("L")).float() / 255.0 for path in relative_paths])
device = torch.device("cuda")
results = {}

for variant_name in ("fractal_bsp_variable_64", "standard_patch16_256"):
    model = load_model(root / variant_name / "continuous" / "best.pt")
    totals = {
        "region_constant": {"iou": 0.0, "dice": 0.0, "pixel_accuracy": 0.0},
        "region_tile_8x8": {"iou": 0.0, "dice": 0.0, "pixel_accuracy": 0.0},
    }
    token_total = 0
    count = 0
    sheet_rows = []
    for start in range(0, images.shape[0], batch_size):
        image = images[start : start + batch_size].to(device)
        target = masks[start : start + batch_size].to(device)
        with torch.inference_mode():
            tokens, _, _ = model.encode(image)
            owner = F.interpolate(
                tokens.assignment.unsqueeze(1).float(),
                size=(image_size, image_size),
                mode="nearest",
            )[:, 0].long()
            maximum_tokens = tokens.mask.shape[1]
            owner_flat = owner.flatten(1)
            target_flat = target[:, 0].flatten(1)
            foreground = torch.zeros(image.shape[0], maximum_tokens, device=device)
            pixels = torch.zeros_like(foreground)
            foreground.scatter_add_(1, owner_flat, target_flat)
            pixels.scatter_add_(1, owner_flat, torch.ones_like(target_flat))
            probability = foreground / pixels.clamp_min(1)
            constant = torch.gather(probability, 1, owner_flat).reshape(image.shape[0], 1, image_size, image_size)
            mask_tiles = sample_region_tiles(
                target,
                tokens.boxes,
                output_size=8,
                mask=tokens.mask,
            )
            tiled = render_token_tiles(
                mask_tiles,
                tokens.boxes,
                tokens.assignment,
                tokens.image_shape,
            ).clamp(0, 1)
        predictions = {
            "region_constant": constant >= 0.5,
            "region_tile_8x8": tiled >= 0.5,
        }
        for name, prediction in predictions.items():
            metrics = segmentation_metrics(prediction, target >= 0.5)
            for metric_name, values in metrics.items():
                totals[name][metric_name] += float(values.sum())
        token_total += int(tokens.mask.sum())
        count += image.shape[0]
        if not sheet_rows:
            for index in range(min(4, image.shape[0])):
                sheet_rows.extend(
                    [
                        image[index].cpu(),
                        target[index].repeat(3, 1, 1).cpu(),
                        constant[index].repeat(3, 1, 1).cpu(),
                        tiled[index].repeat(3, 1, 1).cpu(),
                    ]
                )
    results[variant_name] = {
        "mean_tokens": token_total / count,
        **{
            name: {metric_name: value / count for metric_name, value in metrics.items()}
            for name, metrics in totals.items()
        },
    }
    sheet_path = root / variant_name / "segmentation_oracle.png"
    to_pil_image(make_grid(torch.stack(sheet_rows).clamp(0, 1), nrow=4, padding=2)).save(sheet_path)

report = {
    "status": "complete",
    "samples": int(images.shape[0]),
    "columns": ["image", "ground truth mask", "one scalar per region", "8x8 mask tile per region"],
    "results": results,
}
output_path = root / "segmentation_oracle.json"
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
