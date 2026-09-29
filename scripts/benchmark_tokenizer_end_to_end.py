import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from fractaltokenizer_v2.classification import AdaptiveFractalEmbed, ClassifierConfig, FixedPatchEmbed, make_classifier
from fractaltokenizer_v2.cuda_ops import build_cuda_extension


OUTPUT_PATH = Path("runs/cuda_kernels/tokenizer_end_to_end.json")
BATCH_SIZES = (1, 32)
IMAGE_SIZE = 256
TOKEN_COUNT = 128
MODEL_DIM = 384
DEPTH = 8
HEADS = 6
WARMUP_FORWARD = 12
ITERATIONS_FORWARD = 40
WARMUP_TRAINING = 5
ITERATIONS_TRAINING = 12


device = torch.device("cuda")
torch.manual_seed(321)
torch.backends.cudnn.benchmark = True
build_cuda_extension()
base_config = ClassifierConfig(
    variant="fractal_128",
    image_size=IMAGE_SIZE,
    token_count=TOKEN_COUNT,
    model_dim=MODEL_DIM,
    depth=DEPTH,
    num_heads=HEADS,
    density_temperature=0.4,
    density_coherence_floor=0.35,
    density_blur_sigma=0.0,
)
torch_config = replace(base_config, kernel_backend="torch", compile_density=False)
cuda_config = replace(base_config, kernel_backend="cuda", compile_density=True)
fullgraph_config = replace(base_config, kernel_backend="cuda", compile_density=False)
rect_config = replace(base_config, variant="vit_rect_128", patch_size=(16, 32))


def measure_forward(function) -> dict[str, float]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(WARMUP_FORWARD):
            torch.compiler.cudagraph_mark_step_begin()
            function()
        torch.cuda.synchronize()
        samples = []
        for _ in range(ITERATIONS_FORWARD):
            torch.compiler.cudagraph_mark_step_begin()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def measure_training(model: torch.nn.Module, image: torch.Tensor) -> dict[str, float]:
    model.train()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(image)
            tokens = output[0] if isinstance(output, tuple) else output
            loss = tokens.float().square().mean()
        loss.backward()

    for _ in range(WARMUP_TRAINING):
        torch.compiler.cudagraph_mark_step_begin()
        step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERATIONS_TRAINING):
        torch.compiler.cudagraph_mark_step_begin()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


results = {
    "hardware": {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    },
    "settings": {
        "image_size": IMAGE_SIZE,
        "token_count": TOKEN_COUNT,
        "model_dim": MODEL_DIM,
        "depth": DEPTH,
        "heads": HEADS,
        "dtype": "bfloat16 autocast",
    },
    "batches": {},
}

for batch_size in BATCH_SIZES:
    image = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    torch_embed = AdaptiveFractalEmbed(torch_config).to(device)
    cuda_embed = AdaptiveFractalEmbed(cuda_config).to(device)
    fullgraph_embed_base = AdaptiveFractalEmbed(fullgraph_config).to(device)
    rect_embed = FixedPatchEmbed(rect_config).to(device)
    cuda_embed.load_state_dict(torch_embed.state_dict())
    fullgraph_embed_base.load_state_dict(torch_embed.state_dict())
    fullgraph_embed = torch.compile(fullgraph_embed_base, fullgraph=True, mode="reduce-overhead")
    batch_result = {
        "embed_forward": {
            "torch_fractal": measure_forward(lambda: torch_embed(image)),
            "cuda_fractal": measure_forward(lambda: cuda_embed(image)),
            "fullgraph_cuda_fractal": measure_forward(lambda: fullgraph_embed(image)),
            "rectangular_128": measure_forward(lambda: rect_embed(image)),
        },
        "embed_forward_backward": {
            "torch_fractal": measure_training(torch_embed, image),
            "cuda_fractal": measure_training(cuda_embed, image),
            "fullgraph_cuda_fractal": measure_training(fullgraph_embed, image),
            "rectangular_128": measure_training(rect_embed, image),
        },
    }
    results["batches"][str(batch_size)] = batch_result
    print(batch_size, json.dumps(batch_result, indent=2))
    del torch_embed, cuda_embed, fullgraph_embed, fullgraph_embed_base, rect_embed
    torch.cuda.empty_cache()

classifier_batch = 32
classifier_image = torch.rand(classifier_batch, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
fractal_classifier = make_classifier(cuda_config).to(device)
rect_classifier = make_classifier(rect_config).to(device)
results["classifier_batch_32"] = {
    "forward": {
        "cuda_fractal": measure_forward(lambda: fractal_classifier(classifier_image)),
        "rectangular_128": measure_forward(lambda: rect_classifier(classifier_image)),
    },
    "forward_backward": {
        "cuda_fractal": measure_training(fractal_classifier, classifier_image),
        "rectangular_128": measure_training(rect_classifier, classifier_image),
    },
}
print("classifier", json.dumps(results["classifier_batch_32"], indent=2))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_path = OUTPUT_PATH.with_suffix(".tmp")
temporary_path.write_text(json.dumps(results, indent=2))
temporary_path.replace(OUTPUT_PATH)
print(OUTPUT_PATH)
