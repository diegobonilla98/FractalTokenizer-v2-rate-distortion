from pathlib import Path

import torch

from fractaltokenizer_v2.classification import ClassifierConfig, make_classifier
from fractaltokenizer_v2.configuration import load_density_profile, load_imagenet_training_values
from fractaltokenizer_v2.density import AnalyticInformationDensity
from fractaltokenizer_v2.imagenet_training import TrainingConfig


configuration_path = Path("configs/imagenet100.yaml")
v21_configuration_path = Path("configs/imagenet100_v21_density_ablation.yaml")


def test_active_density_profile_matches_ab_result():
    profile_name, parameters = load_density_profile(configuration_path)
    assert profile_name == "ab_preferred_20260817"
    assert parameters == {
        "density_temperature": 0.4,
        "density_coherence_floor": 0.35,
        "density_blur_sigma": 0.0,
    }


def test_current_and_rerun_profiles_are_isolated():
    current = TrainingConfig(**load_imagenet_training_values(configuration_path, "current_vit_patch16_256"))
    rerun = TrainingConfig(**load_imagenet_training_values(configuration_path, "fractal_ab_preferred_rerun"))
    assert current.variants == ("vit_patch16_256",)
    assert current.output_root == "runs/imagenet100_comparison"
    assert rerun.variants == ("fractal_128",)
    assert rerun.output_root == "runs/imagenet100_fractal_ab_preferred"
    assert rerun.density_temperature == 0.4


def test_classifier_receives_configured_density():
    config = ClassifierConfig(
        variant="fractal_128",
        model_dim=96,
        depth=2,
        num_heads=4,
        density_temperature=0.4,
        density_coherence_floor=0.35,
        density_blur_sigma=0.0,
    )
    model = make_classifier(config)
    assert model.token_embed.density.temperature == 0.4
    assert model.token_embed.density.coherence_floor == 0.35


def test_temperature_changes_density_concentration():
    image = torch.rand(2, 3, 64, 64)
    legacy = AnalyticInformationDensity(partition_stride=8, temperature=1.0)(image)
    preferred = AnalyticInformationDensity(partition_stride=8, temperature=0.4)(image)
    legacy_cv = legacy.flatten(1).std(dim=1, unbiased=False) / legacy.flatten(1).mean(dim=1)
    preferred_cv = preferred.flatten(1).std(dim=1, unbiased=False) / preferred.flatten(1).mean(dim=1)
    assert torch.all(preferred_cv > legacy_cv)


def test_v21_density_ablation_reproduces_the_original_training_recipe():
    config = TrainingConfig(
        **load_imagenet_training_values(v21_configuration_path, "v21_density_ablation")
    )
    assert config.variants == (
        "fractal_universal_128",
        "fractal_analytic_128",
        "fractal_uniform_128",
    )
    assert config.seeds == (42,)
    assert config.epochs == 100
    assert config.micro_batch_size * config.gradient_accumulation_steps == 512
    assert config.mixup_alpha == 0.8
    assert config.random_erase_probability == 0.25
    assert config.dropout == 0.0
    assert config.token_count == 128
    assert config.kernel_backend == "cuda"
    assert config.minimum_epochs == 100


def test_v21_run_preserves_the_original_recipe_and_only_trains_missing_cells():
    config = TrainingConfig(
        **load_imagenet_training_values(v21_configuration_path, "missing_density_cells")
    )
    assert config.variants == ("fractal_universal_128", "fractal_uniform_128")
    assert config.epochs == 100
    assert config.micro_batch_size * config.gradient_accumulation_steps == 512
    assert config.mixup_alpha == 0.8
    assert config.random_erase_probability == 0.25
    assert config.dropout == 0.0
    assert config.token_count == 128
    assert config.density_temperature == 1.0
    assert config.kernel_backend == "cuda"
    assert config.minimum_epochs == 100
