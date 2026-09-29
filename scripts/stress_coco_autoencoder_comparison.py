import json
import time
from pathlib import Path

import torch
import yaml

from fractaltokenizer_v2 import ContinuousReconstructionLoss
from train_coco_autoencoder_comparison import model_for_variant


configuration_path = Path("configs/coco_autoencoder_comparison.yaml")
output_path = Path("runs/coco_autoencoder_comparison/batch_stress_test.json")
selected_variants = {"grid_128", "fractal_ab_128"}
measured_steps = 3


def main() -> None:
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    batch_size = int(configuration["training"]["micro_batch_size"])
    image_size = int(configuration["dataset"]["image_size"])
    criterion = ContinuousReconstructionLoss(**configuration["loss"])
    results = []
    for variant in configuration["training"]["variants"]:
        if variant["name"] not in selected_variants:
            continue
        torch.manual_seed(int(configuration["training"]["seed"]))
        model, _ = model_for_variant(configuration, variant)
        model = model.cuda().train()
        if hasattr(model, "density_model"):
            model.density_model.requires_grad_(False)
        image = torch.rand(batch_size, 3, image_size, image_size, device="cuda")
        durations = []
        loss_value = None
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad(set_to_none=True)
        for _ in range(measured_steps):
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image, target_tokens=int(variant["tokens"]))
                metrics = criterion(output, image)
            (metrics["loss"] / measured_steps).backward()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            loss_value = float(metrics["loss"].detach())
        results.append(
            {
                "variant": variant["name"],
                "batch_size": batch_size,
                "loss": loss_value,
                "seconds_forward_backward": durations,
                "images_per_second_steady": batch_size / min(durations[1:]),
                "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        del image, output, metrics, model
        torch.cuda.empty_cache()
    output_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
