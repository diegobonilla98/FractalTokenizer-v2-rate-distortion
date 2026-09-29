import json
import platform
import statistics
import time
from pathlib import Path

import torch

from fractaltokenizer_v2 import FractalTokenizerV2, FractalTokenizerV2Config


output_path = Path("benchmark_v2_results.json")
image_size = 256
target_tokens = 128
cpu_threads = 5
warmup_steps = 3
benchmark_profiles = (
    {"device": "cuda", "batch_size": 1, "use_vector_quantization": False, "repeats": 5, "iterations": 20},
    {"device": "cuda", "batch_size": 1, "use_vector_quantization": True, "repeats": 5, "iterations": 20},
    {"device": "cuda", "batch_size": 8, "use_vector_quantization": False, "repeats": 5, "iterations": 10},
    {"device": "cuda", "batch_size": 8, "use_vector_quantization": True, "repeats": 5, "iterations": 10},
    {"device": "cpu", "batch_size": 1, "use_vector_quantization": False, "repeats": 5, "iterations": 5},
    {"device": "cpu", "batch_size": 1, "use_vector_quantization": True, "repeats": 3, "iterations": 2},
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, device: torch.device, batch_size: int, repeats: int, iterations: int) -> dict[str, object]:
    with torch.inference_mode():
        for _ in range(warmup_steps):
            function()
        synchronize(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                function()
            synchronize(device)
            samples.append(1000.0 * (time.perf_counter() - start) / iterations)
    median_ms = statistics.median(samples)
    return {
        "median_ms_batch": median_ms,
        "p90_ms_batch": percentile(samples, 0.9),
        "images_per_second": 1000.0 * batch_size / median_ms,
        "samples_ms": samples,
    }


def benchmark_profile(profile: dict[str, object]) -> dict[str, object]:
    device = torch.device(profile["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        return {**profile, "skipped": "CUDA is unavailable"}
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    batch_size = int(profile["batch_size"])
    repeats = int(profile["repeats"])
    iterations = int(profile["iterations"])
    use_vector_quantization = bool(profile["use_vector_quantization"])
    config = FractalTokenizerV2Config(
        latent_dim=192,
        target_tokens=target_tokens,
        token_grid=None,
        partition_stride=8,
        partition_backend="quantile",
        use_vector_quantization=use_vector_quantization,
        codebook_size=8192,
    )
    model = FractalTokenizerV2(config).to(device).eval()
    image = torch.rand(batch_size, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        density = model.density_model(image)
        tokens, _, _ = model.encode(image)
    settings = (device, batch_size, repeats, iterations)
    result = {
        **profile,
        "density": measure(lambda: model.density_model(image), *settings),
        "partition": measure(lambda: model.partition(density), *settings),
        "encode": measure(lambda: model.encode(image), *settings),
        "decode": measure(lambda: model.decode(tokens), *settings),
        "full_autoencoder": measure(lambda: model(image), *settings),
    }
    if batch_size == 1:
        bsp_config = FractalTokenizerV2Config(
            partition_backend="bsp",
            target_tokens=target_tokens,
            use_vector_quantization=False,
        )
        bsp_model = FractalTokenizerV2(bsp_config).to(device).eval()
        result["bsp_partition"] = measure(
            lambda: bsp_model.partition(density, target_tokens=target_tokens),
            *settings,
        )
    return result


torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
results = []
for benchmark_profile_config in benchmark_profiles:
    result = benchmark_profile(benchmark_profile_config)
    results.append(result)
    print(json.dumps(result), flush=True)

report = {
    "weights": "random_untrained",
    "torch_version": torch.__version__,
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_threads": cpu_threads,
    "image_size": image_size,
    "target_tokens": target_tokens,
    "profiles": results,
}
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(output_path)
