from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuda_ops import cuda_universal_codelength_density


@dataclass
class HilbertPartitionBatch:
    assignment: torch.Tensor
    mask: torch.Tensor
    mass: torch.Tensor
    intervals: torch.Tensor
    boxes_grid: torch.Tensor
    boxes: torch.Tensor
    token_count: torch.Tensor

    def to(self, device: torch.device | str) -> HilbertPartitionBatch:
        return HilbertPartitionBatch(
            assignment=self.assignment.to(device),
            mask=self.mask.to(device),
            mass=self.mass.to(device),
            intervals=self.intervals.to(device),
            boxes_grid=self.boxes_grid.to(device),
            boxes=self.boxes.to(device),
            token_count=self.token_count.to(device),
        )


@lru_cache(maxsize=16)
def hilbert_coordinates(side: int) -> tuple[tuple[int, int], ...]:
    side = int(side)
    if side < 1 or side & (side - 1):
        raise ValueError("Hilbert side length must be a positive power of two")
    coordinates = []
    for distance in range(side * side):
        x = 0
        y = 0
        value = distance
        scale = 1
        while scale < side:
            rx = 1 & (value // 2)
            ry = 1 & (value ^ rx)
            if ry == 0:
                if rx == 1:
                    x = scale - 1 - x
                    y = scale - 1 - y
                x, y = y, x
            x += scale * rx
            y += scale * ry
            value //= 4
            scale *= 2
        coordinates.append((y, x))
    return tuple(coordinates)


class UniversalCodelengthDensity(nn.Module):
    def __init__(
        self,
        partition_stride: int = 8,
        alpha: float = 0.5,
        scan_directions: int = 4,
        backend: str = "auto",
    ):
        super().__init__()
        if partition_stride < 1:
            raise ValueError("partition_stride must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if scan_directions not in {1, 2, 4}:
            raise ValueError("scan_directions must be one, two, or four")
        self.partition_stride = int(partition_stride)
        self.alpha = float(alpha)
        self.scan_directions = int(scan_directions)
        if backend not in {"auto", "cuda", "torch"}:
            raise ValueError("backend must be auto, cuda, or torch")
        self.backend = backend
        self.alphabet_size = 511

    @staticmethod
    def _predictors(image: torch.Tensor) -> torch.Tensor:
        neutral = torch.full_like(image[..., :1, :1], 128)
        left = torch.cat((neutral.expand(*image.shape[:-1], 1), image[..., :-1]), dim=-1)
        top = torch.cat((neutral.expand(*image.shape[:-2], 1, image.shape[-1]), image[..., :-1, :]), dim=-2)
        top_left = torch.cat(
            (
                neutral.expand(*image.shape[:-2], 1, image.shape[-1]),
                torch.cat((neutral.expand(*image.shape[:-1], 1), image[..., :-1]), dim=-1)[..., :-1, :],
            ),
            dim=-2,
        )
        gradient = (left + top - top_left).clamp(0, 255)
        lower = torch.minimum(left, top)
        upper = torch.maximum(left, top)
        median = torch.where(top_left >= upper, lower, torch.where(top_left <= lower, upper, gradient))
        return torch.stack((left, top, gradient, median), dim=1)

    def _kt_probabilities(self, symbols: torch.Tensor) -> torch.Tensor:
        sequence_length = symbols.shape[-1]
        flat = symbols.reshape(-1, sequence_length)
        positions = torch.arange(sequence_length, device=symbols.device, dtype=torch.int64)
        keys = flat * sequence_length + positions.unsqueeze(0)
        order = torch.argsort(keys, dim=1)
        sorted_symbols = torch.gather(flat, 1, order)
        sequence_positions = positions.unsqueeze(0).expand_as(flat)
        starts = torch.ones_like(sorted_symbols, dtype=torch.bool)
        starts[:, 1:] = sorted_symbols[:, 1:] != sorted_symbols[:, :-1]
        group_start = torch.where(starts, sequence_positions, torch.zeros_like(sequence_positions))
        group_start = torch.cummax(group_start, dim=1).values
        ranks_sorted = sequence_positions - group_start
        prior_count = torch.empty_like(ranks_sorted)
        prior_count.scatter_(1, order, ranks_sorted)
        denominator = positions.float() + self.alphabet_size * self.alpha
        return (prior_count.float() + self.alpha) / denominator.unsqueeze(0)

    def _direction_bits(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = image.shape
        predictors = self._predictors(image)
        residual_symbols = image.unsqueeze(1) - predictors + 255
        probabilities = self._kt_probabilities(
            residual_symbols.reshape(batch_size, 4, channels, height * width)
        ).reshape(batch_size, 4, channels, height, width)
        log_mixture = torch.logsumexp(probabilities.clamp_min(1e-20).log(), dim=1) - math.log(4.0)
        return (-log_mixture / math.log(2.0)).sum(dim=1, keepdim=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] not in {1, 3}:
            raise ValueError("image must have shape [B,1,H,W] or [B,3,H,W]")
        cuda_density = cuda_universal_codelength_density(
            image,
            partition_stride=self.partition_stride,
            scan_directions=self.scan_directions,
            alpha=self.alpha,
            backend=self.backend,
        )
        if cuda_density is not None:
            return cuda_density
        quantized = image.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.int64)
        bits = torch.zeros(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=torch.float32,
        )
        for rotation in range(self.scan_directions):
            oriented = torch.rot90(quantized, rotation, dims=(-2, -1))
            oriented_bits = self._direction_bits(oriented)
            bits = bits + torch.rot90(oriented_bits, -rotation, dims=(-2, -1))
        bits = bits / self.scan_directions
        return F.avg_pool2d(
            bits,
            kernel_size=self.partition_stride,
            stride=self.partition_stride,
            ceil_mode=True,
            divisor_override=1,
        )


class HilbertEqualMassPartitioner(nn.Module):
    def __init__(self, maximum_tokens: int | None = None):
        super().__init__()
        if maximum_tokens is not None and maximum_tokens < 1:
            raise ValueError("maximum_tokens must be positive")
        self.maximum_tokens = maximum_tokens

    @staticmethod
    def _strict_boundaries(profile: torch.Tensor, token_count: int) -> torch.Tensor:
        length = profile.numel()
        if token_count == 1:
            return torch.empty(0, device=profile.device, dtype=torch.int64)
        cumulative = profile.cumsum(dim=0)
        total = cumulative[-1]
        fractions = torch.arange(1, token_count, device=profile.device, dtype=profile.dtype) / token_count
        raw = torch.searchsorted(cumulative.contiguous(), (total * fractions).contiguous(), right=False) + 1
        indices = torch.arange(token_count - 1, device=profile.device, dtype=torch.int64)
        adjusted = torch.maximum(raw - indices, torch.ones_like(indices))
        adjusted = torch.cummax(adjusted, dim=0).values.clamp_max(length - token_count + 1)
        return adjusted + indices

    @staticmethod
    def geometry_bits(token_count: int, cells: int) -> tuple[float, float]:
        token_count = int(token_count)
        cells = int(cells)
        count_bits = math.ceil(math.log2(cells + 1))
        raw_bits = count_bits + max(token_count - 1, 0) * math.ceil(math.log2(max(cells, 2)))
        if token_count <= 1:
            return float(raw_bits), float(count_bits)
        enumerative = (
            math.lgamma(cells)
            - math.lgamma(token_count)
            - math.lgamma(cells - token_count + 1)
        ) / math.log(2.0)
        return float(raw_bits), float(count_bits + enumerative)

    def forward(
        self,
        density: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> HilbertPartitionBatch:
        if density.ndim != 4 or density.shape[1] != 1:
            raise ValueError("density must have shape [B,1,H,W]")
        if (target_tokens is None) == (mass_per_token is None):
            raise ValueError("provide exactly one of target_tokens or mass_per_token")
        if mass_per_token is not None and mass_per_token <= 0.0:
            raise ValueError("mass_per_token must be positive")
        batch_size, _, height, width = density.shape
        if height != width:
            raise ValueError("Hilbert partitioning requires a square density grid")
        coordinates = hilbert_coordinates(height)
        order = torch.tensor(
            [y * width + x for y, x in coordinates],
            device=density.device,
            dtype=torch.int64,
        )
        values = density[:, 0].float().clamp_min(0.0)
        ordered = values.flatten(1)[:, order]
        cells = height * width
        if target_tokens is not None:
            if not 1 <= int(target_tokens) <= cells:
                raise ValueError("target_tokens must be within the Hilbert grid cell count")
            token_counts = torch.full(
                (batch_size,), int(target_tokens), device=density.device, dtype=torch.int64
            )
        else:
            token_counts = torch.ceil(ordered.sum(dim=1) / float(mass_per_token)).to(torch.int64).clamp_min(1)
            maximum = cells if self.maximum_tokens is None else min(cells, int(self.maximum_tokens))
            token_counts = token_counts.clamp_max(maximum)
        maximum_count = int(token_counts.max().item())
        assignments = torch.empty(batch_size, height, width, device=density.device, dtype=torch.int64)
        mask = torch.zeros(batch_size, maximum_count, device=density.device, dtype=torch.bool)
        mass = torch.zeros(batch_size, maximum_count, device=density.device, dtype=torch.float32)
        intervals = torch.zeros(batch_size, maximum_count, 2, device=density.device, dtype=torch.int64)
        boxes_grid = torch.zeros(batch_size, maximum_count, 4, device=density.device, dtype=torch.int64)
        positions = torch.arange(cells, device=density.device, dtype=torch.int64)
        coordinate_tensor = torch.tensor(coordinates, device=density.device, dtype=torch.int64)
        y_coordinates = coordinate_tensor[:, 0]
        x_coordinates = coordinate_tensor[:, 1]
        for batch_index in range(batch_size):
            count = int(token_counts[batch_index].item())
            boundaries = self._strict_boundaries(ordered[batch_index], count)
            segment = torch.searchsorted(boundaries, positions, right=True)
            assignment_flat = torch.empty(cells, device=density.device, dtype=torch.int64)
            assignment_flat[order] = segment
            assignments[batch_index] = assignment_flat.reshape(height, width)
            mask[batch_index, :count] = True
            mass[batch_index, :count].scatter_add_(0, segment, ordered[batch_index])
            starts = torch.cat((torch.zeros(1, device=density.device, dtype=torch.int64), boundaries))
            ends = torch.cat((boundaries, torch.full((1,), cells, device=density.device, dtype=torch.int64)))
            intervals[batch_index, :count, 0] = starts
            intervals[batch_index, :count, 1] = ends
            x0 = torch.full((count,), width, device=density.device, dtype=torch.int64)
            y0 = torch.full((count,), height, device=density.device, dtype=torch.int64)
            x1 = torch.full((count,), -1, device=density.device, dtype=torch.int64)
            y1 = torch.full((count,), -1, device=density.device, dtype=torch.int64)
            x0.scatter_reduce_(0, segment, x_coordinates, reduce="amin", include_self=True)
            y0.scatter_reduce_(0, segment, y_coordinates, reduce="amin", include_self=True)
            x1.scatter_reduce_(0, segment, x_coordinates, reduce="amax", include_self=True)
            y1.scatter_reduce_(0, segment, y_coordinates, reduce="amax", include_self=True)
            boxes_grid[batch_index, :count] = torch.stack((x0, y0, x1 + 1, y1 + 1), dim=1)
        scale = torch.tensor([width, height, width, height], device=density.device, dtype=torch.float32)
        boxes = boxes_grid.float() / scale
        return HilbertPartitionBatch(
            assignment=assignments,
            mask=mask,
            mass=mass,
            intervals=intervals,
            boxes_grid=boxes_grid,
            boxes=boxes,
            token_count=token_counts,
        )
