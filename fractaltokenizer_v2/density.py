from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class UniformInformationDensity(nn.Module):
    def __init__(self, partition_stride: int = 8):
        super().__init__()
        if partition_stride < 1:
            raise ValueError("partition_stride must be positive")
        self.partition_stride = int(partition_stride)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        height = math.ceil(image.shape[-2] / self.partition_stride)
        width = math.ceil(image.shape[-1] / self.partition_stride)
        return torch.ones(
            image.shape[0],
            1,
            height,
            width,
            device=image.device,
            dtype=image.dtype,
        )


class AnalyticInformationDensity(nn.Module):
    def __init__(
        self,
        partition_stride: int = 8,
        pyramid_levels: int = 4,
        residual_power: float = 0.7,
        coherence_floor: float = 0.35,
        coherence_power: float = 0.5,
        uniform_floor: float = 0.02,
        chroma_weight: float = 0.5,
        temperature: float = 1.0,
        blur_sigma: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= coherence_floor <= 1.0:
            raise ValueError("coherence_floor must be in [0,1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if blur_sigma < 0.0:
            raise ValueError("blur_sigma cannot be negative")
        self.partition_stride = int(partition_stride)
        self.pyramid_levels = int(pyramid_levels)
        self.residual_power = float(residual_power)
        self.coherence_floor = float(coherence_floor)
        self.coherence_power = float(coherence_power)
        self.uniform_floor = float(uniform_floor)
        self.chroma_weight = float(chroma_weight)
        self.temperature = float(temperature)
        self.blur_sigma = float(blur_sigma)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().view(1, 1, 3, 3), persistent=False)

    def _structure_coherence(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(gray.dtype), padding=1)
        jxx = F.avg_pool2d(gx.square(), 5, stride=1, padding=2)
        jyy = F.avg_pool2d(gy.square(), 5, stride=1, padding=2)
        jxy = F.avg_pool2d(gx * gy, 5, stride=1, padding=2)
        numerator = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1e-12)
        return (numerator / (jxx + jyy + 1e-6)).clamp(0.0, 1.0)

    def components(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        image = image.float()
        current = image
        density = torch.zeros(
            image.shape[0], 1, image.shape[2], image.shape[3], device=image.device, dtype=image.dtype
        )
        for level in range(self.pyramid_levels):
            if min(current.shape[-2:]) < 2:
                break
            low = F.avg_pool2d(current, 2, stride=2, ceil_mode=True)
            up = F.interpolate(low, size=current.shape[-2:], mode="bilinear", align_corners=False)
            residual = current - up
            if residual.shape[1] == 3:
                luminance = 0.299 * residual[:, 0:1] + 0.587 * residual[:, 1:2] + 0.114 * residual[:, 2:3]
                chroma = residual - luminance
                energy = luminance.square() + self.chroma_weight * chroma.square().mean(dim=1, keepdim=True)
            else:
                energy = residual.square().mean(dim=1, keepdim=True)
            energy = torch.pow(energy + 1e-8, 0.5 * self.residual_power)
            energy = F.interpolate(energy, size=image.shape[-2:], mode="bilinear", align_corners=False)
            density = density + energy * (2.0 ** (0.5 * level))
            current = low
        coherence = self._structure_coherence(image)
        variation = image.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        return density, coherence, variation

    def compose(
        self,
        residual_density: torch.Tensor,
        coherence: torch.Tensor,
        variation: torch.Tensor,
        coherence_floor: float | None = None,
        coherence_power: float | None = None,
    ) -> torch.Tensor:
        floor = self.coherence_floor if coherence_floor is None else float(coherence_floor)
        power = self.coherence_power if coherence_power is None else float(coherence_power)
        structure_weight = floor + (1.0 - floor) * torch.pow(coherence + 1e-6, power)
        density = F.avg_pool2d(
            residual_density * structure_weight,
            kernel_size=self.partition_stride,
            stride=self.partition_stride,
            ceil_mode=True,
        )
        return density + self.uniform_floor * (variation + 0.01)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual_density, coherence, variation = self.components(image)
        density = self.compose(residual_density, coherence, variation)
        if self.blur_sigma > 0.0:
            radius = max(1, math.ceil(3.0 * self.blur_sigma))
            maximum_radius = max(1, (min(density.shape[-2:]) - 1) // 2)
            radius = min(radius, maximum_radius)
            density = gaussian_blur(density, kernel_size=2 * radius + 1, sigma=self.blur_sigma)
        if self.temperature != 1.0:
            mean = density.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            density = (density / mean).clamp_min(1e-6).pow(1.0 / self.temperature)
            density = density / density.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return density


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be A or B")
        mask = torch.ones_like(self.weight)
        center_y = self.kernel_size[0] // 2
        center_x = self.kernel_size[1] // 2
        mask[:, :, center_y + 1 :, :] = 0
        start = center_x if mask_type == "A" else center_x + 1
        mask[:, :, center_y, start:] = 0
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MaskedSurprisalDensity(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        hidden_dim: int = 96,
        depth: int = 4,
        partition_stride: int = 8,
        min_log_scale: float = -5.0,
        max_log_scale: float = 1.5,
    ):
        super().__init__()
        self.partition_stride = int(partition_stride)
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.input = MaskedConv2d("A", channels, hidden_dim, 7, padding=3)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GELU(),
                    MaskedConv2d("B", hidden_dim, hidden_dim, 3, padding=1),
                    nn.GELU(),
                    MaskedConv2d("B", hidden_dim, hidden_dim, 1),
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(nn.GELU(), MaskedConv2d("B", hidden_dim, 2 * channels, 1))

    def low_resolution_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            image,
            kernel_size=self.partition_stride,
            stride=self.partition_stride,
            ceil_mode=True,
        )

    def distribution(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.low_resolution_image(image.float())
        hidden = self.input(target)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        mean, log_scale = self.output(hidden).chunk(2, dim=1)
        return target, mean, log_scale.clamp(self.min_log_scale, self.max_log_scale)

    def negative_log_likelihood(self, image: torch.Tensor) -> torch.Tensor:
        target, mean, log_scale = self.distribution(image)
        nll = 0.5 * ((target - mean) * torch.exp(-log_scale)).square()
        nll = nll + log_scale + 0.5 * math.log(2.0 * math.pi)
        return nll.mean(dim=1, keepdim=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.negative_log_likelihood(image)) + 1e-4

    def training_loss(self, image: torch.Tensor) -> torch.Tensor:
        return self.negative_log_likelihood(image).mean()


class DensityStudent(nn.Module):
    def __init__(self, channels: int = 3, hidden_dim: int = 48, partition_stride: int = 8):
        super().__init__()
        if partition_stride not in {2, 4, 8, 16}:
            raise ValueError("partition_stride must be one of 2, 4, 8, 16")
        layers: list[nn.Module] = []
        in_channels = channels
        stride = 1
        width = hidden_dim
        while stride < partition_stride:
            layers.extend(
                [
                    nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GELU(),
                ]
            )
            in_channels = width
            stride *= 2
            width = min(width * 2, 192)
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self.encoder(image.float()))) + 1e-4

    @staticmethod
    def distillation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(target, size=prediction.shape[-2:], mode="area")
        prediction_log = torch.log(prediction + 1e-6)
        target_log = torch.log(target + 1e-6)
        shape_loss = F.smooth_l1_loss(
            prediction_log - prediction_log.mean(dim=(-2, -1), keepdim=True),
            target_log - target_log.mean(dim=(-2, -1), keepdim=True),
        )
        total_loss = F.smooth_l1_loss(
            prediction.sum(dim=(-2, -1)).log1p(),
            target.sum(dim=(-2, -1)).log1p(),
        )
        return shape_loss + 0.25 * total_loss
