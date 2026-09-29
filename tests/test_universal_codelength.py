import numpy as np
import pytest
import torch
from scipy.ndimage import label

from fractaltokenizer_v2.universal_codelength import (
    HilbertEqualMassPartitioner,
    UniversalCodelengthDensity,
    hilbert_coordinates,
)


@pytest.mark.parametrize("side", [1, 2, 4, 8, 32])
def test_hilbert_coordinates_are_connected_permutation(side):
    coordinates = hilbert_coordinates(side)
    assert len(coordinates) == side * side
    assert len(set(coordinates)) == side * side
    assert all(0 <= y < side and 0 <= x < side for y, x in coordinates)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(coordinates, coordinates[1:])
    )


def test_universal_codelength_is_network_free_and_detects_randomness():
    generator = torch.Generator().manual_seed(41)
    uniform = torch.zeros(1, 3, 64, 64)
    random_image = torch.rand(1, 3, 64, 64, generator=generator)
    estimator = UniversalCodelengthDensity(partition_stride=8, scan_directions=4)
    values = estimator(torch.cat((uniform, random_image), dim=0))
    assert sum(parameter.numel() for parameter in estimator.parameters()) == 0
    assert values.shape == (2, 1, 8, 8)
    assert torch.isfinite(values).all()
    assert float(values[1].sum()) > 4.0 * float(values[0].sum())


def test_kt_probabilities_match_sequential_counts():
    estimator = UniversalCodelengthDensity(partition_stride=1, alpha=0.5, scan_directions=1)
    symbols = torch.tensor([[[[4, 4, 2, 4, 2]]]], dtype=torch.int64)
    measured = estimator._kt_probabilities(symbols).flatten()
    expected = torch.tensor(
        [
            0.5 / 255.5,
            1.5 / 256.5,
            0.5 / 257.5,
            2.5 / 258.5,
            1.5 / 259.5,
        ]
    )
    torch.testing.assert_close(measured, expected)


def test_four_direction_density_is_rotation_equivariant():
    generator = torch.Generator().manual_seed(83)
    image = torch.rand(1, 3, 32, 32, generator=generator)
    estimator = UniversalCodelengthDensity(partition_stride=4, scan_directions=4)
    first = estimator(image)
    second = torch.rot90(estimator(torch.rot90(image, 1, dims=(-2, -1))), -1, dims=(-2, -1))
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-4)


def test_fixed_hilbert_partition_balances_mass_and_is_connected():
    generator = torch.Generator().manual_seed(97)
    density = torch.rand(3, 1, 16, 16, generator=generator) + 0.1
    result = HilbertEqualMassPartitioner()(density, target_tokens=37)
    assert result.assignment.shape == (3, 16, 16)
    assert result.mask.shape == (3, 37)
    assert result.token_count.tolist() == [37, 37, 37]
    torch.testing.assert_close(result.mass.sum(dim=1), density.sum(dim=(1, 2, 3)))
    coefficient = result.mass.std(dim=1, unbiased=False) / result.mass.mean(dim=1)
    assert float(coefficient.max()) < 0.2
    for assignment in result.assignment.numpy():
        for token in range(37):
            _, components = label(assignment == token)
            assert components == 1


def test_variable_hilbert_partition_uses_shared_mass_capacity():
    density = torch.stack(
        (
            torch.ones(1, 8, 8),
            torch.full((1, 8, 8), 2.0),
            torch.full((1, 8, 8), 4.0),
        )
    )
    result = HilbertEqualMassPartitioner(maximum_tokens=20)(density, mass_per_token=16.0)
    assert result.token_count.tolist() == [4, 8, 16]
    for batch_index, count in enumerate(result.token_count.tolist()):
        active = result.mass[batch_index, :count]
        assert float(active.max()) <= 16.0
        assert torch.all(result.intervals[batch_index, 1:count, 0] == result.intervals[batch_index, : count - 1, 1])


def test_hilbert_geometry_rate_is_smaller_than_raw_boundaries():
    raw, enumerative = HilbertEqualMassPartitioner.geometry_bits(96, 1024)
    assert raw == 961.0
    assert 0.0 < enumerative < raw


def test_vectorized_strict_boundaries_match_sequential_definition():
    generator = torch.Generator().manual_seed(111)
    profile = torch.rand(1024, generator=generator)
    token_count = 173
    measured = HilbertEqualMassPartitioner._strict_boundaries(profile, token_count)
    cumulative = profile.cumsum(dim=0)
    fractions = torch.arange(1, token_count, dtype=profile.dtype) / token_count
    raw = torch.searchsorted(cumulative.contiguous(), (cumulative[-1] * fractions).contiguous()) + 1
    expected = []
    previous = 0
    for index in range(token_count - 1):
        remaining = token_count - 1 - index
        current = int(raw[index].clamp(previous + 1, profile.numel() - remaining))
        expected.append(current)
        previous = current
    assert measured.tolist() == expected
