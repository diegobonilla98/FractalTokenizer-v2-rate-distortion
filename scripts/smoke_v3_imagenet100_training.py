import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fractaltokenizer_v2.classification import ClassifierConfig, make_classifier
from fractaltokenizer_v2.cuda_ops import build_cuda_extension


DATA_ROOT = Path("/mnt/hdd/ImageNet100/val")
OUTPUT_PATH = Path("/mnt/hdd/fractaltokenizer_v3_clean/imagenet100_classification/smoke.json")
VARIANTS = (
    ("fractal_v3_predictive_64", 64),
    ("fractal_analytic_64", 64),
    ("fractal_v3_predictive_96", 96),
    ("fractal_analytic_96", 96),
)
BATCH_SIZE = 8
REAL_IMAGES = 32
IMAGE_SIZE = 256
MODEL_DIM = 192
DEPTH = 2
HEADS = 6


device = torch.device("cuda")
build_cuda_extension()
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE + 32, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
indices = torch.linspace(0, len(dataset) - 1, REAL_IMAGES).long().tolist()
loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
results = {}
for variant, fixed_tokens in VARIANTS:
    config = ClassifierConfig(
        variant=variant,
        image_size=IMAGE_SIZE,
        num_classes=100,
        token_count=fixed_tokens,
        maximum_tokens=128,
        model_dim=MODEL_DIM,
        depth=DEPTH,
        num_heads=HEADS,
        partition_stride=8,
        tile_size=8,
        basis_size=4,
        kernel_backend="cuda",
        compile_density=True,
    )
    model = make_classifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    token_counts = []
    for batch_index, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        token_counts.extend(model.last_token_counts.cpu().tolist())
        if batch_index == 0 and not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite gradients for {variant}")
    results[variant] = {
        "real_images": len(token_counts),
        "losses": losses,
        "minimum_tokens": min(token_counts),
        "mean_tokens": sum(token_counts) / len(token_counts),
        "maximum_tokens": max(token_counts),
        "finite": all(torch.isfinite(torch.tensor(losses))),
    }
    if not results[variant]["finite"]:
        raise RuntimeError(f"non-finite loss for {variant}")
    if variant.startswith("fractal_analytic") and len(set(token_counts)) != 1:
        raise RuntimeError(f"fixed analytic token count changed for {variant}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary = OUTPUT_PATH.with_suffix(".tmp")
temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
temporary.replace(OUTPUT_PATH)
print(json.dumps(results, indent=2))
