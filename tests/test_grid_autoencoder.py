import pytest
import torch

from fractaltokenizer_v2 import (
    ContinuousReconstructionLoss,
    GridAutoencoderConfig,
    GridViTAutoencoder,
    grid_shape_for_tokens,
)


@pytest.mark.parametrize(
    ("tokens", "expected_grid"),
    [(64, (8, 8)), (128, (16, 8)), (256, (16, 16))],
)
def test_comparison_grid_shapes(tokens: int, expected_grid: tuple[int, int]) -> None:
    assert grid_shape_for_tokens(256, 256, tokens) == expected_grid


@pytest.mark.parametrize("tokens", [4, 8, 16])
def test_grid_autoencoder_forward_and_loss(tokens: int) -> None:
    config = GridAutoencoderConfig(
        image_height=32,
        image_width=32,
        token_count=tokens,
        latent_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        transformer_mlp_ratio=2.0,
        refiner_hidden_dim=16,
        refiner_blocks=1,
        compile_refiner=False,
    )
    model = GridViTAutoencoder(config)
    image = torch.rand(2, 3, 32, 32)
    output = model(image, target_tokens=tokens)
    assert output.reconstruction.shape == image.shape
    assert output.base_reconstruction.shape == image.shape
    assert output.tokens.latents.shape == (2, tokens, 32)
    assert output.tokens.token_count.tolist() == [tokens, tokens]
    assert sorted(output.tokens.assignment[0].unique().tolist()) == list(range(tokens))
    assert output.local_tiles.shape[1] == tokens
    metrics = ContinuousReconstructionLoss()(output, image)
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics["loss"].backward()
    assert model.patch_projection.weight.grad is not None
    assert model.patch_decoder.weight.grad is not None
    assert model.refiner.input.weight.grad is not None


def test_grid_autoencoder_rejects_runtime_token_change() -> None:
    model = GridViTAutoencoder(
        GridAutoencoderConfig(
            image_height=32,
            image_width=32,
            token_count=4,
            latent_dim=32,
            transformer_depth=1,
            transformer_heads=4,
            refiner_hidden_dim=16,
            refiner_blocks=1,
            compile_refiner=False,
        )
    )
    with pytest.raises(ValueError, match="cannot change"):
        model(torch.rand(1, 3, 32, 32), target_tokens=8)


def test_grid_shape_requires_exact_divisibility() -> None:
    with pytest.raises(ValueError, match="factor"):
        grid_shape_for_tokens(31, 29, 8)
