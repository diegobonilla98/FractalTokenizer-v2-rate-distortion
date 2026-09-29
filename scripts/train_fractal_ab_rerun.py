from pathlib import Path

from fractaltokenizer_v2.configuration import load_imagenet_training_values
from fractaltokenizer_v2.imagenet_training import TrainingConfig, run_comparison


configuration_path = Path("configs/imagenet100.yaml")
run_profile = "fractal_ab_preferred_rerun"


config = TrainingConfig(**load_imagenet_training_values(configuration_path, run_profile))
run_comparison(config)
