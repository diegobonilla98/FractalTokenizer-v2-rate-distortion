import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fractaltokenizer_v2.classification import ClassifierConfig, count_parameters, make_classifier
from fractaltokenizer_v2.imagenet_training import TrainingConfig, calibrate_universal_mass, make_loaders


data_root = Path("/dev/shm/ImageNet100")
output_path = Path("/mnt/hdd/fractaltokenizer_constant_information/imagenet100_overnight/preflight_benchmark.json")
variants = ("hilbert_uniform_96", "hilbert_universal_96", "vit_rect_128")
batch_size = 64
warmup_steps = 2
measured_steps = 5


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
loader_config = TrainingConfig(
    micro_batch_size=batch_size,
    workers=8,
    prefetch_factor=2,
    random_erase_probability=0.0,
    mixup_alpha=0.0,
)
calibration = calibrate_universal_mass(loader_config, data_root, torch.device("cuda"))
generator = torch.Generator().manual_seed(42)
train_loader, _, _ = make_loaders(loader_config, data_root, generator)
images, target = next(iter(train_loader))
images = images.cuda(non_blocking=True)
target = target.cuda(non_blocking=True)
results = []
for variant in variants:
    model_config = ClassifierConfig(
        variant=variant,
        image_size=256,
        num_classes=100,
        token_count=96,
        model_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        partition_stride=8,
        kernel_backend="cuda",
        universal_mass_per_token=loader_config.universal_mass_per_token,
        maximum_tokens=160,
    )
    model = make_classifier(model_config).cuda().train()
    trainable, total = count_parameters(model)
    for _ in range(warmup_steps):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(images), target)
        loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(measured_steps):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(images), target)
        loss.backward()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / measured_steps
    counts = model.last_token_counts.float()
    result = {
        "variant": variant,
        "batch_size": batch_size,
        "seconds_per_forward_backward": seconds,
        "images_per_second": batch_size / seconds,
        "mean_active_tokens": float(counts.mean()),
        "padded_tokens": model.last_padded_token_count,
        "maximum_active_tokens": float(counts.max()),
        "token_cap_fraction": float((counts >= model_config.maximum_tokens).float().mean()),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    print(json.dumps(result, sort_keys=True))
    results.append(result)
    del model, loss
    torch.cuda.empty_cache()
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps({"mass_calibration": calibration, "results": results}, indent=2, sort_keys=True),
    encoding="utf-8",
)
