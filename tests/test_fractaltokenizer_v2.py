import torch

from fractaltokenizer_v2 import (
    BalancedBSPPartitioner,
    FractalAutoregressivePrior,
    FractalPriorConfig,
    FractalRateDistortionLoss,
    FractalTokenPacket,
    FractalTokenizerV2,
    FractalTokenizerV2Config,
    QuantileGridPartitioner,
    RasterAutoregressivePrior,
    RasterPriorConfig,
    RasterSequenceBatch,
    UniformInformationDensity,
    assignment_from_tree,
    deserialize_tree,
    serialize_tree,
)
from fractaltokenizer_v2.quantization import VectorQuantizer


def small_config(partition_backend="quantile", use_vector_quantization=False):
    return FractalTokenizerV2Config(
        latent_dim=32,
        tile_size=4,
        basis_size=2,
        partition_stride=4,
        target_tokens=8,
        partition_backend=partition_backend,
        max_tokens=16,
        use_vector_quantization=use_vector_quantization,
        codebook_size=32,
        decoder_feature_dim=16,
        decoder_hidden_dim=24,
        decoder_residual_blocks=1,
        gaussian_chunk_size=8,
    )


def test_quantile_partition_has_exact_count_and_coverage():
    density = torch.rand(3, 1, 8, 8)
    partition = QuantileGridPartitioner()(density, target_tokens=13)
    assert partition.mask.sum(dim=1).tolist() == [13, 13, 13]
    assert partition.assignment.min().item() == 0
    assert partition.assignment.max().item() == 12
    for batch_index in range(3):
        counts = torch.bincount(partition.assignment[batch_index].flatten(), minlength=13)
        assert bool((counts > 0).all())


def test_quantile_partition_reduces_information_mass_variation():
    x = torch.linspace(0.05, 2.0, 16)
    density = x.view(1, 1, 1, 16).expand(1, 1, 16, 16)
    adaptive = QuantileGridPartitioner()(density, target_tokens=16)
    uniform = QuantileGridPartitioner()(torch.ones_like(density), target_tokens=16)
    uniform_mass = torch.zeros_like(uniform.mass)
    uniform_mass.scatter_add_(1, uniform.assignment.flatten(1), density[:, 0].flatten(1))
    adaptive_cv = adaptive.mass.std(dim=1, unbiased=False) / adaptive.mass.mean(dim=1)
    uniform_cv = uniform_mass.std(dim=1, unbiased=False) / uniform_mass.mean(dim=1)
    assert float(adaptive_cv) < float(uniform_cv)


def test_uniform_density_produces_a_fixed_patch_grid():
    image = torch.rand(2, 3, 32, 32)
    density = UniformInformationDensity(partition_stride=4)(image)
    partition = QuantileGridPartitioner()(density, target_tokens=64, token_grid=(8, 8))
    expected = torch.arange(64).reshape(8, 8)
    assert density.shape == (2, 1, 8, 8)
    assert torch.equal(partition.assignment[0], expected)
    assert torch.equal(partition.assignment[1], expected)


def test_bsp_variable_rate_responds_to_total_information_mass():
    partitioner = BalancedBSPPartitioner(max_tokens=16)
    low = partitioner(torch.full((1, 1, 8, 8), 0.05), mass_per_token=1.0)
    high = partitioner(torch.full((1, 1, 8, 8), 1.0), mass_per_token=1.0)
    assert int(low.token_count[0]) < int(high.token_count[0])
    assert int(high.token_count[0]) == 16


def test_bsp_tree_roundtrip_preserves_assignment():
    density = torch.rand(1, 1, 8, 8)
    partition = BalancedBSPPartitioner(max_tokens=16)(density, target_tokens=11)
    tree = partition.trees[0]
    restored = deserialize_tree(serialize_tree(tree), tree.grid_shape)
    assert torch.equal(assignment_from_tree(tree), assignment_from_tree(restored))


def test_continuous_forward_backward_is_finite():
    model = FractalTokenizerV2(small_config())
    criterion = FractalRateDistortionLoss(codebook_size=32)
    image = torch.rand(2, 3, 32, 32)
    output = model(image)
    losses = criterion(output, image)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert float(losses["content_rate_bpp"].detach()) == 0.0
    assert any(parameter.grad is not None for parameter in model.detokenizer.parameters())


def test_vq_indices_decode_to_the_quantized_content():
    model = FractalTokenizerV2(small_config(use_vector_quantization=True)).eval()
    image = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        tokens, density, quantizer_output = model.encode(image)
        partition = model.partition(density)
        decoded = model.quantizer.decode_indices(tokens.code_indices, tokens.mask)
    assert torch.equal(decoded, quantizer_output.quantized)
    assert partition.mask.shape == tokens.mask.shape


def test_ema_codebook_initializes_from_first_latent_batch():
    quantizer = VectorQuantizer(codebook_size=32, embedding_dim=8, ema_decay=0.99).train()
    latents = torch.randn(2, 32, 8)
    mask = torch.ones(2, 32, dtype=torch.bool)
    output = quantizer(latents, mask)
    assert int(output.indices.unique().numel()) > 8
    assert quantizer.ema_updates == 1
    restored = VectorQuantizer(codebook_size=32, embedding_dim=8, ema_decay=0.99)
    restored.load_state_dict(quantizer.state_dict())
    assert restored.ema_updates == 1
    assert torch.equal(restored.codebook.weight, quantizer.codebook.weight)


def test_fixed_box_packet_roundtrip_matches_direct_decode(tmp_path):
    model = FractalTokenizerV2(small_config(use_vector_quantization=True)).eval()
    image = torch.rand(1, 3, 32, 32)
    packet_path = tmp_path / "sample.ftk2"
    with torch.inference_mode():
        packet = model.encode_packet(image)
        packet.save(packet_path)
        restored = FractalTokenPacket.load(packet_path)
        direct = model.decode_packet(packet)
        loaded = model.decode_packet(restored)
    assert torch.equal(direct, loaded)
    assert restored.boxes_grid is not None


def test_bsp_sequence_roundtrip_matches_direct_decode():
    model = FractalTokenizerV2(small_config("bsp", True)).eval()
    image = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        sequence = model.encode_sequence(image, target_tokens=8)
        tokens = model.tokens_from_sequence(sequence)
        direct, _ = model.decode(tokens)
        restored = model.decode_sequence(sequence)
    assert torch.equal(direct, restored)
    assert sequence.sequence_length.tolist() == [15, 15]


def test_prior_loss_and_generation_are_structurally_valid():
    tokenizer = FractalTokenizerV2(small_config("bsp", True)).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        sequence = tokenizer.encode_sequence(image, target_tokens=5)
    prior = FractalAutoregressivePrior(
        FractalPriorConfig(
            codebook_size=32,
            num_quantizers=1,
            max_grid_size=8,
            model_dim=32,
            depth=2,
            heads=4,
            max_sequence_length=31,
        )
    )
    losses = prior.loss(sequence)
    assert all(torch.isfinite(value) for value in losses.values())
    generated = prior.generate(
        batch_size=2,
        grid_shape=(8, 8),
        image_shape=(32, 32),
        min_tokens=3,
        max_tokens=6,
        sample=False,
    )
    assert bool((generated.token_count >= 3).all())
    assert bool((generated.token_count <= 6).all())
    with torch.inference_mode():
        reconstruction = tokenizer.decode_sequence(generated)
    assert reconstruction.shape == (2, 3, 32, 32)
    assert torch.isfinite(reconstruction).all()


def test_raster_prior_loss_and_generation_are_structurally_valid():
    sequence = RasterSequenceBatch(
        code_indices=torch.randint(0, 32, (2, 16, 1)),
        token_mask=torch.ones(2, 16, dtype=torch.bool),
        token_grid=(4, 4),
        image_shape=(32, 32),
    )
    prior = RasterAutoregressivePrior(
        RasterPriorConfig(
            codebook_size=32,
            num_quantizers=1,
            model_dim=32,
            depth=2,
            heads=4,
            max_sequence_length=16,
        )
    )
    losses = prior.loss(sequence)
    assert all(torch.isfinite(value) for value in losses.values())
    generated = prior.generate(
        batch_size=2,
        token_grid=(4, 4),
        image_shape=(32, 32),
        sample=False,
    )
    assert generated.code_indices.shape == (2, 16, 1)
    assert generated.sequence_length.tolist() == [16, 16]
