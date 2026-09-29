import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fractaltokenizer_v2 import (
    ContinuousAutoencoderConfig,
    ContinuousFractalAutoencoder,
    FractalDetokenizerSettings,
    build_cuda_extension,
)


CHECKPOINT_PATH = Path("runs/coco_continuous_autoencoder/best.pt")
DATASET_ROOT = Path("/dev/shm/coco2017/val2017")
OUTPUT_PATH = Path("runs/cuda_kernels/checkpoint_validation.json")
IMAGE_SIZE = 256
IMAGE_COUNT = 16
BATCH_SIZE = 2
TIMING_WARMUP = 3
TIMING_ITERATIONS = 10


device = torch.device("cuda")
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
build_cuda_extension()
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
base_config = ContinuousAutoencoderConfig(**checkpoint["model_config"])
reference_config = replace(
    base_config,
    kernel_backend="torch",
    compile_density=False,
    compile_decoder=False,
    compile_refiner=False,
)
cuda_config = replace(
    base_config,
    kernel_backend="cuda",
    compile_density=False,
    compile_decoder=False,
    compile_refiner=False,
)
optimized_config = replace(
    base_config,
    kernel_backend="cuda",
    compile_density=True,
    compile_decoder=True,
    compile_refiner=True,
)


def make_model(config: ContinuousAutoencoderConfig) -> ContinuousFractalAutoencoder:
    model = ContinuousFractalAutoencoder(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


reference_model = make_model(reference_config)
cuda_model = make_model(cuda_config)
optimized_model = make_model(optimized_config)
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)
image_paths = sorted(DATASET_ROOT.glob("*"))[:IMAGE_COUNT]
if len(image_paths) < IMAGE_COUNT:
    raise RuntimeError(f"expected {IMAGE_COUNT} validation images in {DATASET_ROOT}")
images = torch.stack([transform(Image.open(path).convert("RGB")) for path in image_paths])


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def timed(function) -> dict[str, float]:
    for _ in range(TIMING_WARMUP):
        torch.compiler.cudagraph_mark_step_begin()
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(TIMING_ITERATIONS):
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


statistics_accumulator = {
    "reference_psnr": [],
    "cuda_psnr": [],
    "optimized_psnr": [],
    "cuda_difference": [],
    "optimized_difference": [],
}
last_batch = None
with torch.inference_mode():
    for start in range(0, IMAGE_COUNT, BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE].to(device)
        last_batch = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = reference_model(batch).reconstruction
            cuda_output = cuda_model(batch).reconstruction
            torch.compiler.cudagraph_mark_step_begin()
            optimized = optimized_model(batch).reconstruction
        statistics_accumulator["reference_psnr"].extend(batch_psnr(reference, batch).cpu().tolist())
        statistics_accumulator["cuda_psnr"].extend(batch_psnr(cuda_output, batch).cpu().tolist())
        statistics_accumulator["optimized_psnr"].extend(batch_psnr(optimized, batch).cpu().tolist())
        statistics_accumulator["cuda_difference"].append((cuda_output - reference).abs().mean().item())
        statistics_accumulator["optimized_difference"].append((optimized - reference).abs().mean().item())

if last_batch is None:
    raise RuntimeError("validation batch is empty")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    tokens, _ = optimized_model.encode(last_batch)
    fast_settings = FractalDetokenizerSettings.from_profile("fast")
    quality_settings = FractalDetokenizerSettings.from_profile("quality")
    reference_settings = FractalDetokenizerSettings.from_profile("reference")
    fast_timing = timed(lambda: optimized_model.decode(tokens, settings=fast_settings)[0])
    quality_timing = timed(lambda: optimized_model.decode(tokens, settings=quality_settings)[0])
    reference_timing = timed(lambda: optimized_model.decode(tokens, settings=reference_settings)[0])

result = {
    "checkpoint": str(CHECKPOINT_PATH),
    "epoch": checkpoint["epoch"],
    "images": IMAGE_COUNT,
    "batch_size": BATCH_SIZE,
    "reference_psnr_db": statistics.mean(statistics_accumulator["reference_psnr"]),
    "cuda_psnr_db": statistics.mean(statistics_accumulator["cuda_psnr"]),
    "optimized_psnr_db": statistics.mean(statistics_accumulator["optimized_psnr"]),
    "cuda_vs_reference_mean_abs": statistics.mean(statistics_accumulator["cuda_difference"]),
    "optimized_vs_reference_mean_abs": statistics.mean(statistics_accumulator["optimized_difference"]),
    "decode_profiles": {
        "fast": fast_timing,
        "quality": quality_timing,
        "reference": reference_timing,
    },
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_path = OUTPUT_PATH.with_suffix(".tmp")
temporary_path.write_text(json.dumps(result, indent=2))
temporary_path.replace(OUTPUT_PATH)
print(json.dumps(result, indent=2))
