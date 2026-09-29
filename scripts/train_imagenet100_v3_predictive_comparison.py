from pathlib import Path

from fractaltokenizer_v2.configuration import load_imagenet_training_values
from fractaltokenizer_v2.imagenet_training import TrainingConfig, run_comparison


CONFIGURATION_PATH = Path("configs/imagenet100_v3_predictive_comparison.yaml")
RUN_PROFILES = ("mean64", "mean96")


for run_profile in RUN_PROFILES:
    config = TrainingConfig(**load_imagenet_training_values(CONFIGURATION_PATH, run_profile))
    run_comparison(config)
