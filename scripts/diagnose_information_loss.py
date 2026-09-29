import json
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from fractaltokenizer_v2 import FractalTokenizerV2, FractalTokenizerV2Config
from fractaltokenizer_v2.regions import render_token_tiles


project = Path("/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion")
root = project / "runs/supervisely_generation/comparison"
validation_samples = 64
batch_size = 8
image_size = 256


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FractalTokenizerV2(FractalTokenizerV2Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def add_error(sums, name, prediction, target):
    error = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    sums[name] = sums.get(name, 0.0) + float(error.sum())


manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
image_root = Path(manifest["image_root"])
if not image_root.is_dir():
    image_root = Path("/mnt/hdd/supervisely_person_clean_2667_img/images")
paths = [image_root / value for value in manifest["validation"][:validation_samples]]
transform = transforms.Compose(
    [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)
images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
device = torch.device("cuda")
results = {}
sheets = {}

for variant_name in ("fractal_bsp_variable_64", "standard_patch16_256"):
    continuous = load_model(root / variant_name / "continuous" / "best.pt")
    vq = load_model(root / variant_name / "vq" / "best.pt")
    sums = {}
    count = 0
    token_total = 0
    first = None
    for start in range(0, images.shape[0], batch_size):
        image = images[start : start + batch_size].to(device)
        with torch.inference_mode():
            continuous_tokens, _, _ = continuous.encode(image)
            _, _, _, sampled_tiles = continuous.region_encoder(
                image,
                boxes=continuous_tokens.boxes,
                depth=continuous_tokens.depth,
                mask=continuous_tokens.mask,
                compact_valid_regions=continuous.config.partition_backend == "bsp",
            )
            oracle_tile = render_token_tiles(
                sampled_tiles,
                continuous_tokens.boxes,
                continuous_tokens.assignment,
                continuous_tokens.image_shape,
            ).clamp(0, 1)
            oracle_dct = continuous.decode_basis(continuous_tokens)
            continuous_base, _ = continuous.decode(continuous_tokens, refine=False)
            continuous_refined, _ = continuous.decode(continuous_tokens, refine=True)
            vq_tokens, _, _ = vq.encode(image)
            vq_base, _ = vq.decode(vq_tokens, refine=False)
            vq_refined, _ = vq.decode(vq_tokens, refine=True)
        outputs = {
            "oracle_roi_8x8": oracle_tile,
            "oracle_dct_4x4": oracle_dct,
            "continuous_base": continuous_base,
            "continuous_refined": continuous_refined,
            "vq_base": vq_base,
            "vq_refined": vq_refined,
        }
        for name, prediction in outputs.items():
            add_error(sums, name, prediction, image)
        token_total += int(vq_tokens.mask.sum())
        count += image.shape[0]
        if first is None:
            first = (image.detach().cpu(), {name: value.detach().cpu() for name, value in outputs.items()})
    method_results = {}
    for name, total in sums.items():
        mse = total / count
        method_results[name] = {
            "mse": mse,
            "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        }
    mean_tokens = token_total / count
    method_results["mean_tokens"] = mean_tokens
    method_results["raw_vq_content_bits_per_image"] = (
        mean_tokens * math.log2(vq.config.codebook_size) * vq.config.num_quantizers
    )
    results[variant_name] = method_results
    target, outputs = first
    rows = []
    labels = ["oracle_roi_8x8", "oracle_dct_4x4", "continuous_refined", "vq_refined"]
    for index in range(min(4, target.shape[0])):
        rows.append(target[index])
        for label in labels:
            rows.append(outputs[label][index])
    sheet = make_grid(torch.stack(rows).clamp(0, 1), nrow=5, padding=2)
    sheet_path = root / variant_name / "information_loss_diagnostic.png"
    to_pil_image(sheet).save(sheet_path)
    sheets[variant_name] = str(sheet_path)

report = {
    "status": "complete",
    "samples": int(images.shape[0]),
    "columns": ["original", "oracle ROI 8x8", "oracle DCT 4x4", "continuous neural", "VQ neural"],
    "results": results,
    "sheets": sheets,
}
output_path = root / "information_loss_diagnostic.json"
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
