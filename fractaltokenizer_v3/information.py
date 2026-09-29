from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def token_counts_from_information(
    total_information: torch.Tensor,
    capacity_bits: float,
    maximum_tokens: int,
) -> torch.Tensor:
    if capacity_bits <= 0.0:
        raise ValueError("capacity_bits must be positive")
    if maximum_tokens < 1:
        raise ValueError("maximum_tokens must be positive")
    return torch.ceil(total_information.float() / float(capacity_bits)).to(torch.int64).clamp(1, maximum_tokens)


def calibrate_capacity(
    total_information: torch.Tensor,
    target_mean_tokens: float,
    maximum_tokens: int,
    iterations: int = 80,
) -> float:
    values = total_information.detach().float().flatten()
    if values.numel() == 0:
        raise ValueError("total_information cannot be empty")
    if not 1.0 <= target_mean_tokens <= float(maximum_tokens):
        raise ValueError("target_mean_tokens must lie within the token range")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    positive = values[values > 0.0]
    if positive.numel() == 0:
        return 1.0
    lower = torch.finfo(torch.float32).tiny
    upper = float(positive.max().item()) * 2.0
    for _ in range(iterations):
        middle = math.sqrt(lower * upper)
        mean_tokens = token_counts_from_information(values, middle, maximum_tokens).float().mean().item()
        if mean_tokens > target_mean_tokens:
            lower = middle
        else:
            upper = middle
    return float(upper)


class PredictiveInformationDensity(nn.Module):
    def __init__(
        self,
        partition_stride: int = 8,
        residual_scale: float = 1.0 / 255.0,
        scan_directions: int = 4,
    ):
        super().__init__()
        if partition_stride < 1:
            raise ValueError("partition_stride must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        if scan_directions not in {1, 2, 4}:
            raise ValueError("scan_directions must be one, two, or four")
        self.partition_stride = int(partition_stride)
        self.residual_scale = float(residual_scale)
        self.scan_directions = int(scan_directions)

    @staticmethod
    def _jpeg_ls_predict(image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        neutral = image.mean(dim=(-2, -1), keepdim=True)
        left = torch.cat((neutral.expand(batch, channels, height, 1), image[..., :-1]), dim=-1)
        top = torch.cat((neutral.expand(batch, channels, 1, width), image[..., :-1, :]), dim=-2)
        top_left_interior = torch.cat(
            (neutral.expand(batch, channels, height - 1, 1), image[..., :-1, :-1]),
            dim=-1,
        )
        top_left = torch.cat((neutral.expand(batch, channels, 1, width), top_left_interior), dim=-2)
        lower = torch.minimum(left, top)
        upper = torch.maximum(left, top)
        gradient = left + top - top_left
        return torch.where(top_left >= upper, lower, torch.where(top_left <= lower, upper, gradient))

    def _direction_information(self, image: torch.Tensor) -> torch.Tensor:
        residual = image - self._jpeg_ls_predict(image)
        scaled_squared = residual.square() / (self.residual_scale * self.residual_scale)
        return torch.log1p(scaled_squared).sum(dim=1, keepdim=True) / math.log(2.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] not in {1, 3}:
            raise ValueError("image must have shape [B,1,H,W] or [B,3,H,W]")
        source = image.detach().float()
        information = torch.zeros(
            source.shape[0],
            1,
            source.shape[2],
            source.shape[3],
            device=source.device,
            dtype=torch.float32,
        )
        for rotation in range(self.scan_directions):
            oriented = torch.rot90(source, rotation, dims=(-2, -1))
            oriented_information = self._direction_information(oriented)
            information = information + torch.rot90(oriented_information, -rotation, dims=(-2, -1))
        information = information / self.scan_directions
        return F.avg_pool2d(
            information,
            kernel_size=self.partition_stride,
            stride=self.partition_stride,
            ceil_mode=True,
            divisor_override=1,
        )
