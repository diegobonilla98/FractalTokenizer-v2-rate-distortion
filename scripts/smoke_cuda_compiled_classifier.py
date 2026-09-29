import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fractaltokenizer_v2 import build_cuda_extension
from fractaltokenizer_v2.classification import ClassifierConfig, make_classifier


OUTPUT_PATH = Path("runs/cuda_kernels/compiled_classifier_smoke.json")
BATCH_SIZE = 4
IMAGE_SIZE = 64
TOKEN_COUNT = 16
MODEL_DIM = 96
DEPTH = 2
HEADS = 4
STEPS = 2


torch.manual_seed(77)
device = torch.device("cuda")
build_cuda_extension()
config = ClassifierConfig(
    variant="fractal_128",
    image_size=IMAGE_SIZE,
    token_count=TOKEN_COUNT,
    model_dim=MODEL_DIM,
    depth=DEPTH,
    num_heads=HEADS,
    partition_stride=8,
    tile_size=8,
    basis_size=4,
    density_temperature=0.4,
    kernel_backend="cuda",
    compile_density=True,
)
raw_model = make_classifier(config).to(device)
model = torch.compile(raw_model, mode="default")
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-3)
images = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
target = torch.randint(0, config.num_classes, (BATCH_SIZE,), device=device)
losses = []
for _ in range(STEPS):
    torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images)
        loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
gradients = [parameter.grad for parameter in raw_model.parameters() if parameter.grad is not None]
if not losses or not all(torch.isfinite(torch.tensor(losses))):
    raise RuntimeError("compiled classifier produced a non-finite loss")
if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
    raise RuntimeError("compiled classifier produced invalid gradients")
result = {
    "steps": STEPS,
    "losses": losses,
    "parameters_with_gradients": len(gradients),
    "token_count": raw_model.token_count,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_path = OUTPUT_PATH.with_suffix(".tmp")
temporary_path.write_text(json.dumps(result, indent=2))
temporary_path.replace(OUTPUT_PATH)
print(json.dumps(result, indent=2))
