from __future__ import annotations

import math

import torch
import torch.nn as nn

from fractaltokenizer_v2.partition import QuantileGridPartitioner
from fractaltokenizer_v2.structures import PartitionBatch


def _factor_grid(token_count: int, height: int, width: int) -> tuple[int, int] | None:
    target_ratio = height / width
    best = None
    for rows in range(1, min(height, token_count) + 1):
        if token_count % rows:
            continue
        columns = token_count // rows
        if columns > width:
            continue
        score = abs(math.log((rows / columns) / target_ratio))
        if best is None or score < best[0]:
            best = (score, rows, columns)
    if best is None:
        return None
    return int(best[1]), int(best[2])


class VariableRateQuantilePartitioner(nn.Module):
    def __init__(
        self,
        maximum_tokens: int = 128,
        density_floor_ratio: float = 0.0,
        backend: str = "auto",
        uniform_when_saturated: bool = True,
        stationarity_cv_threshold: float = 0.1,
    ):
        super().__init__()
        if maximum_tokens < 1:
            raise ValueError("maximum_tokens must be positive")
        if stationarity_cv_threshold < 0.0:
            raise ValueError("stationarity_cv_threshold cannot be negative")
        self.maximum_tokens = int(maximum_tokens)
        self.uniform_when_saturated = bool(uniform_when_saturated)
        self.stationarity_cv_threshold = float(stationarity_cv_threshold)
        self.partitioner = QuantileGridPartitioner(
            density_floor_ratio=density_floor_ratio,
            backend=backend,
        )

    def forward(self, density: torch.Tensor, token_counts: torch.Tensor) -> PartitionBatch:
        if density.ndim != 4 or density.shape[1] != 1:
            raise ValueError("density must have shape [B,1,H,W]")
        counts = token_counts.to(device=density.device, dtype=torch.int64).flatten()
        if counts.numel() != density.shape[0]:
            raise ValueError("token_counts must contain one value per image")
        height, width = density.shape[-2:]
        if bool(((counts < 1) | (counts > min(self.maximum_tokens, height * width))).any()):
            raise ValueError("token counts exceed the supported range")
        batch_size = density.shape[0]
        padded_tokens = int(counts.max().item())
        boxes_grid = torch.zeros(batch_size, padded_tokens, 4, device=density.device, dtype=torch.int64)
        boxes = torch.zeros(batch_size, padded_tokens, 4, device=density.device, dtype=torch.float32)
        mask = torch.zeros(batch_size, padded_tokens, device=density.device, dtype=torch.bool)
        mass = torch.zeros(batch_size, padded_tokens, device=density.device, dtype=torch.float32)
        depth = torch.zeros(batch_size, padded_tokens, device=density.device, dtype=torch.int64)
        assignment = torch.zeros(batch_size, height, width, device=density.device, dtype=torch.int64)
        flat_density = density.flatten(1).float()
        density_cv = flat_density.std(dim=1, unbiased=False) / flat_density.mean(dim=1).clamp_min(1e-8)
        regular_grid = (
            (counts == self.maximum_tokens)
            & (density_cv <= self.stationarity_cv_threshold)
            & self.uniform_when_saturated
        )
        for count_tensor in counts.unique(sorted=True):
            count = int(count_tensor.item())
            for use_regular_grid in (False, True):
                selected = (counts == count) & (regular_grid == use_regular_grid)
                indices = torch.nonzero(selected, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                group_density = density.index_select(0, indices)
                token_grid = None
                if use_regular_grid:
                    group_density = torch.ones_like(group_density)
                    token_grid = _factor_grid(count, height, width)
                result = self.partitioner(group_density, target_tokens=count, token_grid=token_grid)
                boxes_grid[indices, :count] = result.boxes_grid
                boxes[indices, :count] = result.boxes.float()
                mask[indices, :count] = True
                depth[indices, :count] = result.depth
                assignment[indices] = result.assignment
                true_density = density.index_select(0, indices).flatten(1)
                true_mass = torch.zeros(indices.numel(), count, device=density.device, dtype=torch.float32)
                true_mass.scatter_add_(1, result.assignment.flatten(1), true_density)
                mass[indices, :count] = true_mass
        return PartitionBatch(
            boxes_grid=boxes_grid,
            boxes=boxes,
            mask=mask,
            mass=mass,
            depth=depth,
            assignment=assignment,
            trees=None,
        )
