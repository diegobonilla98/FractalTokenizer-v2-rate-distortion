import json
import time
from pathlib import Path

import torch
import yaml

from fractaltokenizer_v2 import ContinuousReconstructionLoss
from train_coco_autoencoder_comparison import model_for_variant


configuration_path = Path("configs/coco_autoencoder_comparison.yaml")
output_path = Path("runs/coco_autoencoder_comparison/smoke_test.json")
batch_size = 1


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("smoke test requires CUDA")
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    image_size = int(configuration["dataset"]["image_size"])
    criterion = ContinuousReconstructionLoss(**configuration["loss"])
    results = []
    for variant in configuration["training"]["variants"]:
        torch.manual_seed(int(configuration["training"]["seed"]))
        model, model_config = model_for_variant(configuration, variant)
        model = model.cuda().train()
        if hasattr(model, "density_model"):
            model.density_model.requires_grad_(False)
        image = torch.rand(batch_size, 3, image_size, image_size, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image, target_tokens=int(variant["tokens"]))
            metrics = criterion(output, image)
        metrics["loss"].backward()
        torch.cuda.synchronize()
        results.append(
            {
                "variant": variant["name"],
                "family": variant["family"],
                "tokens": int(variant["tokens"]),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "loss": float(metrics["loss"].detach()),
                "psnr": float(metrics["psnr"].detach()),
                "seconds_cold_forward_backward": time.perf_counter() - start,
                "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
                "output_shape": list(output.reconstruction.shape),
                "finite_gradients": all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ),
                "model_config": model_config.__dict__,
            }
        )
        del image, output, metrics, model
        torch.cuda.empty_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
