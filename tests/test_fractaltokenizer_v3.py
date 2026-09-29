import torch

from fractaltokenizer_v3 import (
    NetworkFreeInformationTokenizer,
    PredictiveInformationDensity,
    calibrate_capacity,
    token_counts_from_information,
)


def _noise(seed: int, standard_deviation: float, batch: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (0.5 + standard_deviation * torch.randn(batch, 3, 256, 256, generator=generator)).clamp(0.0, 1.0)


def test_density_has_no_learned_state() -> None:
    density = PredictiveInformationDensity()
    assert list(density.parameters()) == []
    assert density.state_dict() == {}


def test_every_uniform_color_has_zero_information_and_one_token() -> None:
    colors = torch.tensor([0.0, 0.17, 0.5, 0.83, 1.0]).view(-1, 1, 1, 1)
    images = colors.expand(-1, 3, 256, 256).contiguous()
    tokenizer = NetworkFreeInformationTokenizer(capacity_bits=100.0, backend="torch")
    output = tokenizer(images)
    assert torch.equal(output.density, torch.zeros_like(output.density))
    assert torch.equal(output.token_count, torch.ones(5, dtype=torch.int64))
    assert output.partition.mask.shape == (5, 1)
    assert torch.equal(output.partition.boxes_grid[:, 0], torch.tensor([0, 0, 32, 32]).expand(5, -1))


def test_density_is_invariant_to_color_offset_and_channel_permutation() -> None:
    generator = torch.Generator().manual_seed(7)
    image = 0.25 + 0.5 * torch.rand(2, 3, 128, 128, generator=generator)
    density = PredictiveInformationDensity(partition_stride=8)
    reference = density(image)
    shifted = density(image + 0.1)
    permuted = density(image[:, [2, 0, 1]])
    torch.testing.assert_close(reference, shifted, atol=2e-3, rtol=2e-5)
    torch.testing.assert_close(reference, permuted, atol=2e-3, rtol=2e-5)


def test_gaussian_noise_saturates_to_exact_regular_grid() -> None:
    maximum_tokens = 128
    tokenizer = NetworkFreeInformationTokenizer(
        capacity_bits=1.0,
        maximum_tokens=maximum_tokens,
        backend="torch",
    )
    output = tokenizer(_noise(11, 0.2, batch=2))
    assert torch.equal(output.token_count, torch.full((2,), maximum_tokens, dtype=torch.int64))
    boxes = output.partition.boxes_grid
    widths = boxes[..., 2] - boxes[..., 0]
    heights = boxes[..., 3] - boxes[..., 1]
    assert torch.equal(widths, torch.full_like(widths, 2))
    assert torch.equal(heights, torch.full_like(heights, 4))
    assert torch.equal(output.partition.assignment[0], output.partition.assignment[1])


def test_nonstationary_saturated_image_remains_adaptive() -> None:
    image = torch.full((1, 3, 256, 256), 0.5)
    image[..., :, 128:] = _noise(13, 0.2)[..., :, 128:]
    tokenizer = NetworkFreeInformationTokenizer(capacity_bits=1.0, backend="torch")
    half_noise = tokenizer(image)
    gaussian = tokenizer(_noise(13, 0.2))
    assert half_noise.token_count.item() == 128
    assert gaussian.token_count.item() == 128
    assert not torch.equal(half_noise.partition.assignment, gaussian.partition.assignment)


def test_information_and_token_count_increase_with_unpredictability() -> None:
    uniform = torch.full((1, 3, 256, 256), 0.5)
    ramp = torch.linspace(0.2, 0.8, 256).view(1, 1, 1, 256).expand(1, 3, 256, 256)
    low_noise = _noise(19, 0.02)
    high_noise = _noise(19, 0.2)
    density = PredictiveInformationDensity()
    totals = torch.cat([density(image).sum().view(1) for image in (uniform, ramp, low_noise, high_noise)])
    assert torch.all(totals[1:] > totals[:-1])
    capacity = float(totals[-1].item()) / 128.0
    counts = token_counts_from_information(totals, capacity, maximum_tokens=128)
    assert counts[0].item() == 1
    assert counts[-1].item() == 128
    assert torch.all(counts[1:] >= counts[:-1])


def test_half_noise_has_intermediate_rate_and_mass_balancing() -> None:
    image = torch.full((1, 3, 256, 256), 0.5)
    image[..., :, 128:] = _noise(23, 0.2)[..., :, 128:]
    density_module = PredictiveInformationDensity()
    total = density_module(image).sum().item()
    tokenizer = NetworkFreeInformationTokenizer(
        capacity_bits=total / 48.0,
        maximum_tokens=128,
        backend="torch",
    )
    output = tokenizer(image)
    assert output.token_count.item() == 48
    masses = output.partition.mass[0, output.partition.mask[0]]
    coefficient_of_variation = masses.std() / masses.mean()
    assert coefficient_of_variation.item() < 0.45


def test_variable_rate_batch_padding_is_consistent() -> None:
    images = torch.cat((torch.full((1, 3, 256, 256), 0.5), _noise(29, 0.02), _noise(31, 0.2)))
    density = PredictiveInformationDensity()
    totals = density(images).sum(dim=(1, 2, 3))
    capacity = float(totals[-1].item()) / 128.0
    tokenizer = NetworkFreeInformationTokenizer(capacity_bits=capacity, backend="torch")
    output = tokenizer(images)
    assert output.token_count.tolist()[0] == 1
    assert output.token_count.tolist()[-1] == 128
    assert torch.equal(output.partition.mask.sum(dim=1), output.token_count)
    assert output.partition.assignment.shape == (3, 32, 32)


def test_capacity_calibration_hits_requested_mean() -> None:
    totals = torch.linspace(100.0, 10000.0, 1000)
    capacity = calibrate_capacity(totals, target_mean_tokens=64.0, maximum_tokens=128)
    mean_tokens = token_counts_from_information(totals, capacity, 128).float().mean().item()
    assert abs(mean_tokens - 64.0) <= 0.1


def test_cuda_and_torch_partition_paths_are_exact() -> None:
    if not torch.cuda.is_available():
        return
    images = torch.cat((torch.full((1, 3, 256, 256), 0.5), _noise(41, 0.02), _noise(43, 0.2))).cuda()
    density = PredictiveInformationDensity().cuda()(images)
    capacity = float(density[-1].sum().item()) / 128.0
    torch_tokenizer = NetworkFreeInformationTokenizer(capacity_bits=capacity, backend="torch").cuda()
    cuda_tokenizer = NetworkFreeInformationTokenizer(capacity_bits=capacity, backend="cuda").cuda()
    torch_output = torch_tokenizer(images)
    cuda_output = cuda_tokenizer(images)
    assert torch.equal(torch_output.token_count, cuda_output.token_count)
    assert torch.equal(torch_output.partition.assignment, cuda_output.partition.assignment)
    assert torch.equal(torch_output.partition.boxes_grid, cuda_output.partition.boxes_grid)
    torch.testing.assert_close(torch_output.partition.mass, cuda_output.partition.mass, atol=1e-3, rtol=1e-6)
