from pathlib import Path

from fractaltokenizer_v2.configuration import load_imagenet_training_values
from fractaltokenizer_v2.imagenet_training import TrainingConfig, run_comparison


configuration_path = Path("configs/imagenet100_v21_density_ablation.yaml")
run_profile = "missing_density_cells"


config = TrainingConfig(**load_imagenet_training_values(configuration_path, run_profile))
run_comparison(config)
