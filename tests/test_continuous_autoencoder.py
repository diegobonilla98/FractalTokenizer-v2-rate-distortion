import torch

from fractaltokenizer_v2 import (
    ContinuousAutoencoderConfig,
    ContinuousFractalAutoencoder,
    ContinuousReconstructionLoss,
    FractalDetokenizerSettings,
)


def small_config(**updates) -> ContinuousAutoencoderConfig:
    values = {
        "latent_dim": 48,
        "input_tile_size": 8,
        "output_tile_size": 10,
        "encoder_hidden_dim": 24,
        "decoder_hidden_dim": 32,
        "transformer_depth": 2,
        "transformer_heads": 4,
        "coordinate_fourier_bands": 3,
        "geometry_fourier_bands": 3,
        "decoder_token_chunk_size": 3,
        "refiner_hidden_dim": 24,
        "refiner_blocks": 1,
        "target_tokens": 6,
        "max_tokens": 12,
        "partition_stride": 4,
        "bsp_workers": 0,
    }
    values.update(updates)
    return ContinuousAutoencoderConfig(**values)


def test_forward_shapes_partition_coverage_and_finiteness():
    model = ContinuousFractalAutoencoder(small_config())
    image = torch.rand(2, 3, 32, 32)
    output = model(image)
    assert output.reconstruction.shape == image.shape
    assert output.base_reconstruction.shape == image.shape
    assert output.local_tiles.shape == (2, 6, 3, 10, 10)
    assert output.tokens.token_count.tolist() == [6, 6]
    assert output.tokens.assignment.amin().item() == 0
    assert output.tokens.assignment.amax().item() == 5
    assert torch.isfinite(output.reconstruction).all()
    assert output.reconstruction.min() >= 0
    assert output.reconstruction.max() <= 1


def test_variable_token_budgets_use_the_same_model():
    model = ContinuousFractalAutoencoder(small_config())
    image = torch.rand(1, 3, 32, 32)
    output_four = model(image, target_tokens=4)
    output_nine = model(image, target_tokens=9)
    assert output_four.tokens.latents.shape == (1, 4, 48)
    assert output_nine.tokens.latents.shape == (1, 9, 48)
    assert output_four.reconstruction.shape == output_nine.reconstruction.shape == image.shape


def test_bsp_backend_roundtrip_has_tree_and_exact_coverage():
    model = ContinuousFractalAutoencoder(small_config(partition_backend="bsp", target_tokens=7))
    image = torch.rand(2, 3, 32, 32)
    output = model(image)
    assert output.tokens.trees is not None
    assert len(output.tokens.trees) == 2
    assert output.tokens.token_count.tolist() == [7, 7]
    for batch_index in range(2):
        identifiers = torch.unique(output.tokens.assignment[batch_index])
        assert identifiers.tolist() == list(range(7))


def test_backward_reaches_encoder_transformer_decoder_and_refiner():
    model = ContinuousFractalAutoencoder(small_config())
    criterion = ContinuousReconstructionLoss()
    image = torch.rand(2, 3, 32, 32)
    metrics = criterion(model(image), image)
    metrics["loss"].backward()
    groups = {
        "content_encoder": model.content_encoder,
        "transformer": model.transformer,
        "local_decoder": model.local_decoder,
        "refiner": model.refiner,
    }
    for module in groups.values():
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_initial_refiner_is_identity_and_local_decoder_uses_coordinates():
    model = ContinuousFractalAutoencoder(small_config())
    image = torch.rand(1, 3, 32, 32)
    output = model(image)
    assert torch.allclose(output.reconstruction, output.base_reconstruction, atol=2e-6, rtol=0)
    spatial_variation = output.local_tiles.var(dim=(-2, -1)).mean()
    assert spatial_variation > 0


def test_loss_is_finite_and_psnr_matches_mse():
    model = ContinuousFractalAutoencoder(small_config())
    criterion = ContinuousReconstructionLoss()
    image = torch.rand(1, 3, 32, 32)
    metrics = criterion(model(image), image)
    expected_psnr = -10.0 * torch.log10(metrics["mse"])
    assert torch.allclose(metrics["psnr"], expected_psnr)
    assert all(torch.isfinite(value) for value in metrics.values())


def test_configuration_rejects_invalid_geometry():
    try:
        ContinuousFractalAutoencoder(small_config(latent_dim=50, transformer_heads=4))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid head geometry was accepted")


def test_detokenizer_profiles_expose_fast_quality_and_reference_modes():
    fast = FractalDetokenizerSettings.from_profile("fast")
    quality = FractalDetokenizerSettings.from_profile("quality")
    reference = FractalDetokenizerSettings.from_profile("reference")
    assert fast.refine is False
    assert fast.compile_decoder is True
    assert quality.refine is True
    assert quality.backend == "auto"
    assert reference.backend == "torch"
    assert reference.compile_decoder is False
    assert reference.compile_refiner is False


def test_fast_detokenizer_profile_returns_unrefined_base():
    model = ContinuousFractalAutoencoder(small_config())
    image = torch.rand(1, 3, 32, 32)
    tokens, _ = model.encode(image)
    fast_reconstruction, _ = model.decode(tokens, settings=FractalDetokenizerSettings.from_profile("fast"))
    reference_output = model(image)
    assert torch.allclose(fast_reconstruction, reference_output.base_reconstruction, atol=2e-6, rtol=0)
