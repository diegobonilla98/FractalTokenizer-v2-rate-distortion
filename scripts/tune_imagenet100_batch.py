import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from fractaltokenizer_v2.classification import ClassifierConfig, count_parameters, make_classifier


variants = ("fractal_128", "vit_rect_128", "vit_patch16_256")
candidate_batch_sizes = (64, 96, 128, 192, 256)
image_size = 256
num_classes = 100
model_dim = 384
depth = 8
num_heads = 6
mlp_ratio = 4.0
token_count = 128
warmup_steps = 1
timed_steps = 2
target_effective_batch_size = 512
output_path = Path("runs/imagenet100_comparison/batch_plan.json")


def benchmark_variant(variant, batch_size):
    config = ClassifierConfig(
        variant=variant,
        image_size=image_size,
        num_classes=num_classes,
        token_count=token_count,
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    model = make_classifier(config).cuda().train()
    trainable_parameters, total_parameters = count_parameters(model)
    images = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
    target = torch.randint(0, num_classes, (batch_size,), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for step in range(warmup_steps + timed_steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, target)
        loss.backward()
        torch.cuda.synchronize()
        if step >= warmup_steps:
            durations.append(time.perf_counter() - start)
    result = {
        "variant": variant,
        "batch_size": batch_size,
        "tokens": model.token_count,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "images_per_second": batch_size * len(durations) / sum(durations),
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    del loss, logits, target, images, model
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("batch tuning requires CUDA")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    measurements = []
    viable_batches = []
    for batch_size in tqdm(candidate_batch_sizes, desc="batch candidates", dynamic_ncols=True):
        candidate_results = []
        candidate_viable = True
        for variant in variants:
            try:
                result = benchmark_variant(variant, batch_size)
                candidate_results.append(result)
                print(json.dumps(result, sort_keys=True))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                candidate_viable = False
                candidate_results.append({"variant": variant, "batch_size": batch_size, "oom": True})
                break
        measurements.extend(candidate_results)
        if candidate_viable:
            estimated_seconds = sum(130_000 / result["images_per_second"] for result in candidate_results)
            viable_batches.append((estimated_seconds, batch_size))
    if not viable_batches:
        raise RuntimeError("none of the candidate batch sizes fit all comparison models")
    _, recommended_batch_size = min(viable_batches)
    if target_effective_batch_size % recommended_batch_size:
        raise RuntimeError("target effective batch must be divisible by the selected micro-batch")
    gradient_accumulation_steps = target_effective_batch_size // recommended_batch_size
    result = {
        "recommended_micro_batch_size": recommended_batch_size,
        "recommended_gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": target_effective_batch_size,
        "selection_rule": "minimum summed one-epoch time across all three variants",
        "hardware": torch.cuda.get_device_name(0),
        "measurements": measurements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
