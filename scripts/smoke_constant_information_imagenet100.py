from dataclasses import replace
from pathlib import Path

from fractaltokenizer_v2.configuration import load_imagenet_training_values
from fractaltokenizer_v2.imagenet_training import TrainingConfig, run_comparison


configuration_path = Path("configs/constant_information_imagenet100_overnight.yaml")
run_profile = "overnight"


values = load_imagenet_training_values(configuration_path, run_profile)
config = TrainingConfig(**values)
config = replace(
    config,
    output_root="/tmp/fractaltokenizer_constant_information_smoke",
    epochs=1,
    minimum_epochs=1,
    micro_batch_size=4,
    gradient_accumulation_steps=1,
    workers=2,
    max_train_batches=2,
    max_val_batches=2,
    model_dim=96,
    depth=2,
    num_heads=4,
    maximum_tokens=160,
    mass_calibration_images=32,
    mass_validation_images=32,
    telegram_enabled=False,
    resume=False,
)
run_comparison(config)
