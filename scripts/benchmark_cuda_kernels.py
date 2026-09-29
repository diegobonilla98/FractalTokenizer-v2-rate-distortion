import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from fractaltokenizer_v2.cuda_ops import build_cuda_extension
from fractaltokenizer_v2.density import AnalyticInformationDensity
from fractaltokenizer_v2.partition import QuantileGridPartitioner
from fractaltokenizer_v2.regions import render_token_tiles, sample_region_tiles


OUTPUT_PATH = Path("runs/cuda_kernels/benchmark.json")
CUDA_ARCHITECTURE = "12.1"
BATCH_SIZES = (1, 8, 32)
IMAGE_SIZE = 256
PARTITION_SIZE = 32
TOKEN_COUNT = 96
INPUT_TILE_SIZE = 16
OUTPUT_TILE_SIZE = 32
LATENT_DIM = 256
WARMUP_ITERATIONS = 20
MEASUREMENT_GROUPS = 7
ITERATIONS_PER_GROUP = 50


os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCHITECTURE
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
build_cuda_extension()


def measure(function) -> dict[str, float]:
    for _ in range(WARMUP_ITERATIONS):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(MEASUREMENT_GROUPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERATIONS_PER_GROUP):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / ITERATIONS_PER_GROUP)
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
        "partition_size": PARTITION_SIZE,
        "token_count": TOKEN_COUNT,
        "input_tile_size": INPUT_TILE_SIZE,
        "output_tile_size": OUTPUT_TILE_SIZE,
        "latent_dim": LATENT_DIM,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measurement_groups": MEASUREMENT_GROUPS,
        "iterations_per_group": ITERATIONS_PER_GROUP,
    },
    "batches": {},
}

for batch_size in BATCH_SIZES:
    image = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    density_input = torch.rand(batch_size, 1, PARTITION_SIZE, PARTITION_SIZE, device=device)
    torch_partitioner = QuantileGridPartitioner(backend="torch").to(device)
    cuda_partitioner = QuantileGridPartitioner(backend="cuda").to(device)
    partition = cuda_partitioner(density_input, TOKEN_COUNT)
    tiles = torch.rand(batch_size, TOKEN_COUNT, 3, OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE, device=device)
    density_model = AnalyticInformationDensity(
        partition_stride=8,
        temperature=0.4,
        coherence_floor=0.35,
        blur_sigma=0.0,
    ).to(device)
    patch_projection = nn.Conv2d(3, LATENT_DIM, 16, stride=16).to(device).eval()

    def patch_embed():
        return patch_projection(image)

    def density_preprocess():
        return density_model(image)

    def partition_torch():
        return torch_partitioner(density_input, TOKEN_COUNT)

    def partition_cuda():
        return cuda_partitioner(density_input, TOKEN_COUNT)

    def sample_torch():
        return sample_region_tiles(
            image,
            partition.boxes,
            INPUT_TILE_SIZE,
            partition.mask,
            backend="torch",
        )

    def sample_cuda():
        return sample_region_tiles(
            image,
            partition.boxes,
            INPUT_TILE_SIZE,
            partition.mask,
            backend="cuda",
        )

    def render_torch():
        return render_token_tiles(
            tiles,
            partition.boxes,
            partition.assignment,
            (IMAGE_SIZE, IMAGE_SIZE),
            backend="torch",
        )

    def render_cuda():
        return render_token_tiles(
            tiles,
            partition.boxes,
            partition.assignment,
            (IMAGE_SIZE, IMAGE_SIZE),
            backend="cuda",
        )

    with torch.inference_mode():
        batch_results = {
            "vit_patch_projection": measure(patch_embed),
            "density_preprocess": measure(density_preprocess),
            "partition_torch": measure(partition_torch),
            "partition_cuda": measure(partition_cuda),
            "region_sample_torch": measure(sample_torch),
            "region_sample_cuda": measure(sample_cuda),
            "render_torch": measure(render_torch),
            "render_cuda": measure(render_cuda),
        }
    batch_results["partition_speedup"] = (
        batch_results["partition_torch"]["median_ms"] / batch_results["partition_cuda"]["median_ms"]
    )
    batch_results["region_sample_speedup"] = (
        batch_results["region_sample_torch"]["median_ms"] / batch_results["region_sample_cuda"]["median_ms"]
    )
    batch_results["render_speedup"] = (
        batch_results["render_torch"]["median_ms"] / batch_results["render_cuda"]["median_ms"]
    )
    results["batches"][str(batch_size)] = batch_results
    print(batch_size, json.dumps(batch_results, indent=2))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_path = OUTPUT_PATH.with_suffix(".tmp")
temporary_path.write_text(json.dumps(results, indent=2))
temporary_path.replace(OUTPUT_PATH)
print(OUTPUT_PATH)
