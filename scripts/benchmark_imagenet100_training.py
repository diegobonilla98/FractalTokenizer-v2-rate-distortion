import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fractaltokenizer_v2.classification import ClassifierConfig, count_parameters, make_classifier
from fractaltokenizer_v2.imagenet_training import TrainingConfig, make_loaders, prepare_imagenet100


variants = ("fractal_128", "vit_rect_128", "vit_patch16_256")
micro_batch_size = 64
workers = 16
warmup_batches = 5
timed_batches = 20
output_path = Path("runs/imagenet100_comparison/training_path_benchmark.json")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("training-path benchmark requires CUDA")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    training_config = TrainingConfig(
        micro_batch_size=micro_batch_size,
        workers=workers,
        stage_to_shared_memory=True,
        telegram_enabled=False,
    )
    data_root, inventory = prepare_imagenet100(training_config)
    measurements = []
    for variant in variants:
        generator = torch.Generator().manual_seed(42)
        train_loader, _, _ = make_loaders(training_config, data_root, generator)
        model = make_classifier(ClassifierConfig(variant=variant)).cuda().train()
        trainable_parameters, total_parameters = count_parameters(model)
        torch.cuda.reset_peak_memory_stats()
        timed_images = 0
        start = None
        for batch_index, (images, target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            model.zero_grad(set_to_none=True)
            if batch_index == warmup_batches:
                torch.cuda.synchronize()
                start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(images), target)
            loss.backward()
            if batch_index >= warmup_batches:
                timed_images += images.shape[0]
            if batch_index + 1 == warmup_batches + timed_batches:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        measurement = {
            "variant": variant,
            "tokens": model.token_count,
            "micro_batch_size": micro_batch_size,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "timed_images": timed_images,
            "elapsed_seconds": elapsed,
            "images_per_second": timed_images / elapsed,
            "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        print(json.dumps(measurement, sort_keys=True))
        measurements.append(measurement)
        del loss, target, images, model, train_loader
        torch.cuda.empty_cache()
    result = {
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": torch.cuda.get_device_name(0),
        "data_root": str(data_root),
        "dataset": inventory["splits"],
        "warmup_batches": warmup_batches,
        "timed_batches": timed_batches,
        "measurements": measurements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output_path)


if __name__ == "__main__":
    main()
