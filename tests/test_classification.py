import torch

from fractaltokenizer_v2.classification import ClassifierConfig, count_parameters, make_classifier


def test_classifier_variants_forward_and_backward():
    expected_tokens = {
        "fractal_128": 16,
        "fractal_analytic_128": 16,
        "fractal_analytic_64": 16,
        "fractal_analytic_96": 16,
        "fractal_v3_predictive_64": 32,
        "fractal_v3_predictive_96": 32,
        "fractal_universal_128": 16,
        "fractal_uniform_128": 16,
        "vit_rect_128": 8,
        "vit_patch16_256": 16,
        "hilbert_uniform_96": 16,
        "hilbert_universal_96": 16,
    }
    image = torch.randn(2, 3, 64, 64)
    for variant, token_count in expected_tokens.items():
        config = ClassifierConfig(
            variant=variant,
            image_size=64,
            num_classes=7,
            token_count=16,
            model_dim=48,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            partition_stride=8,
            universal_mass_per_token=4000.0,
            maximum_tokens=32,
        )
        model = make_classifier(config)
        logits = model(image)
        logits.square().mean().backward()
        trainable, total = count_parameters(model)
        assert logits.shape == (2, 7)
        assert model.token_count == token_count
        assert trainable == total
        assert trainable > 0


def test_full_comparison_has_matching_transformer_dimensions():
    models = {
        variant: make_classifier(ClassifierConfig(variant=variant, model_dim=96, depth=2, num_heads=4))
        for variant in ("fractal_128", "vit_rect_128", "vit_patch16_256")
    }
    assert models["fractal_128"].token_count == 128
    assert models["vit_rect_128"].token_count == 128
    assert models["vit_patch16_256"].token_count == 256
    transformer_parameters = []
    for model in models.values():
        transformer_parameters.append(sum(parameter.numel() for parameter in model.transformer.parameters()))
    assert len(set(transformer_parameters)) == 1


def test_hilbert_partition_is_the_only_learned_architecture_difference():
    common = {
        "image_size": 64,
        "num_classes": 7,
        "token_count": 16,
        "model_dim": 48,
        "depth": 1,
        "num_heads": 4,
        "partition_stride": 8,
        "universal_mass_per_token": 4000.0,
        "maximum_tokens": 32,
    }
    uniform = make_classifier(ClassifierConfig(variant="hilbert_uniform_96", **common))
    universal = make_classifier(ClassifierConfig(variant="hilbert_universal_96", **common))
    assert uniform.state_dict().keys() == universal.state_dict().keys()
    assert count_parameters(uniform) == count_parameters(universal)


def test_v21_density_ablation_preserves_the_v2_architecture():
    common = {
        "image_size": 64,
        "num_classes": 7,
        "token_count": 16,
        "model_dim": 48,
        "depth": 1,
        "num_heads": 4,
        "partition_stride": 8,
        "compile_density": False,
        "kernel_backend": "torch",
    }
    variants = (
        "fractal_analytic_128",
        "fractal_universal_128",
        "fractal_uniform_128",
    )
    models = [make_classifier(ClassifierConfig(variant=variant, **common)) for variant in variants]
    assert [model.token_embed.density_mode for model in models] == ["analytic", "universal", "uniform"]
    assert len({count_parameters(model) for model in models}) == 1
    assert len({tuple(model.state_dict()) for model in models}) == 1


def test_v21_analytic_alias_is_exactly_legacy_v2():
    common = {
        "image_size": 64,
        "num_classes": 7,
        "token_count": 16,
        "model_dim": 48,
        "depth": 1,
        "num_heads": 4,
        "partition_stride": 8,
        "compile_density": False,
        "kernel_backend": "torch",
    }
    torch.manual_seed(123)
    legacy = make_classifier(ClassifierConfig(variant="fractal_128", **common)).eval()
    torch.manual_seed(123)
    analytic = make_classifier(ClassifierConfig(variant="fractal_analytic_128", **common)).eval()
    image = torch.randn(2, 3, 64, 64)
    torch.testing.assert_close(legacy(image), analytic(image), rtol=0.0, atol=0.0)


def test_v3_predictive_rate_changes_without_learned_density_parameters():
    config = ClassifierConfig(
        variant="fractal_v3_predictive_64",
        image_size=64,
        num_classes=7,
        model_dim=48,
        depth=1,
        num_heads=4,
        partition_stride=8,
        maximum_tokens=32,
        v3_capacity_bits_64=1000.0,
        compile_density=False,
        kernel_backend="torch",
    )
    model = make_classifier(config).eval()
    normalized_uniform = torch.zeros(1, 3, 64, 64)
    normalized_noise = torch.randn(1, 3, 64, 64)
    model(normalized_uniform)
    uniform_tokens = int(model.last_token_counts.item())
    model(normalized_noise)
    noise_tokens = int(model.last_token_counts.item())
    assert sum(parameter.numel() for parameter in model.token_embed.density.parameters()) == 0
    assert uniform_tokens < noise_tokens
    assert noise_tokens == config.maximum_tokens
