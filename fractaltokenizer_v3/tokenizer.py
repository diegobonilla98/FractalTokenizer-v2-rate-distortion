from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fractaltokenizer_v2.structures import PartitionBatch

from .information import PredictiveInformationDensity, token_counts_from_information
from .partition import VariableRateQuantilePartitioner


@dataclass
class InformationTokenizerOutput:
    density: torch.Tensor
    total_information: torch.Tensor
    token_count: torch.Tensor
    partition: PartitionBatch


class NetworkFreeInformationTokenizer(nn.Module):
    def __init__(
        self,
        capacity_bits: float,
        maximum_tokens: int = 128,
        partition_stride: int = 8,
        residual_scale: float = 1.0 / 255.0,
        scan_directions: int = 4,
        backend: str = "auto",
        stationarity_cv_threshold: float = 0.1,
    ):
        super().__init__()
        if capacity_bits <= 0.0:
            raise ValueError("capacity_bits must be positive")
        self.capacity_bits = float(capacity_bits)
        self.maximum_tokens = int(maximum_tokens)
        self.density = PredictiveInformationDensity(
            partition_stride=partition_stride,
            residual_scale=residual_scale,
            scan_directions=scan_directions,
        )
        self.partitioner = VariableRateQuantilePartitioner(
            maximum_tokens=maximum_tokens,
            density_floor_ratio=0.0,
            backend=backend,
            uniform_when_saturated=True,
            stationarity_cv_threshold=stationarity_cv_threshold,
        )

    def forward(self, image: torch.Tensor) -> InformationTokenizerOutput:
        density = self.density(image)
        total_information = density.sum(dim=(1, 2, 3))
        token_count = token_counts_from_information(
            total_information,
            capacity_bits=self.capacity_bits,
            maximum_tokens=self.maximum_tokens,
        )
        partition = self.partitioner(density, token_count)
        return InformationTokenizerOutput(
            density=density,
            total_information=total_information,
            token_count=token_count,
            partition=partition,
        )
