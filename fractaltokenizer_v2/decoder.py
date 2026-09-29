from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import LocalDCTBasis, render_token_tiles
from .structures import FractalTokens


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(16, channels)
        while channels % groups != 0:
            groups -= 1
        self.network = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(x)


class GaussianSplatRenderer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int = 128,
        sigma_scale: float = 0.35,
        token_chunk_size: int = 64,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(latent_dim, feature_dim)
        self.geometry_delta = nn.Linear(latent_dim, 5)
        nn.init.zeros_(self.geometry_delta.weight)
        nn.init.zeros_(self.geometry_delta.bias)
        self.feature_dim = int(feature_dim)
        self.sigma_scale = float(sigma_scale)
        self.token_chunk_size = int(token_chunk_size)

    def forward(
        self,
        latents: torch.Tensor,
        boxes: torch.Tensor,
        mask: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = latents.shape
        grid_height, grid_width = grid_shape
        features = self.feature_projection(latents)
        delta = self.geometry_delta(latents)
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        width = (x1 - x0).clamp_min(1e-4)
        height = (y1 - y0).clamp_min(1e-4)
        center_x = 0.5 * (x0 + x1) + 0.15 * width * torch.tanh(delta[..., 0])
        center_y = 0.5 * (y0 + y1) + 0.15 * height * torch.tanh(delta[..., 1])
        sigma_x = width * self.sigma_scale * torch.exp(0.35 * torch.tanh(delta[..., 2]))
        sigma_y = height * self.sigma_scale * torch.exp(0.35 * torch.tanh(delta[..., 3]))
        rho = 0.65 * torch.tanh(delta[..., 4])
        x = (torch.arange(grid_width, device=latents.device, dtype=latents.dtype) + 0.5) / grid_width
        y = (torch.arange(grid_height, device=latents.device, dtype=latents.dtype) + 0.5) / grid_height
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx.view(1, 1, grid_height, grid_width)
        yy = yy.view(1, 1, grid_height, grid_width)
        numerator = torch.zeros(
            batch_size,
            self.feature_dim,
            grid_height,
            grid_width,
            device=latents.device,
            dtype=latents.dtype,
        )
        denominator = torch.zeros(batch_size, 1, grid_height, grid_width, device=latents.device, dtype=latents.dtype)
        for start in range(0, token_count, self.token_chunk_size):
            end = min(start + self.token_chunk_size, token_count)
            dx = (xx - center_x[:, start:end, None, None]) / sigma_x[:, start:end, None, None].clamp_min(1e-4)
            dy = (yy - center_y[:, start:end, None, None]) / sigma_y[:, start:end, None, None].clamp_min(1e-4)
            local_rho = rho[:, start:end, None, None]
            inverse = 1.0 / (1.0 - local_rho.square()).clamp_min(0.2)
            exponent = -0.5 * inverse * (dx.square() - 2.0 * local_rho * dx * dy + dy.square())
            weight = torch.exp(exponent.clamp_min(-18.0)) * mask[:, start:end, None, None]
            numerator = numerator + torch.einsum("bnhw,bnf->bfhw", weight, features[:, start:end])
            denominator = denominator + weight.sum(dim=1, keepdim=True)
        dense = numerator / denominator.clamp_min(1e-6)
        parameters = torch.stack([center_x, center_y, sigma_x, sigma_y, rho], dim=-1)
        return dense, denominator, parameters


class NeuralDetokenizer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 192,
        channels: int = 3,
        tile_size: int = 8,
        basis_size: int = 4,
        feature_dim: int = 128,
        hidden_dim: int = 160,
        residual_blocks: int = 4,
        sigma_scale: float = 0.35,
        token_chunk_size: int = 64,
        kernel_backend: str = "auto",
    ):
        super().__init__()
        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.basis_size = int(basis_size)
        self.kernel_backend = kernel_backend
        self.basis = LocalDCTBasis(tile_size=tile_size, basis_size=basis_size)
        self.basis_head = nn.Linear(latent_dim, channels * basis_size * basis_size)
        nn.init.normal_(self.basis_head.weight, std=1e-3)
        nn.init.zeros_(self.basis_head.bias)
        with torch.no_grad():
            self.basis_head.bias.view(channels, basis_size, basis_size)[:, 0, 0] = 0.5 * tile_size
        self.splat = GaussianSplatRenderer(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            sigma_scale=sigma_scale,
            token_chunk_size=token_chunk_size,
        )
        input_channels = feature_dim + channels + 3
        self.input_projection = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(residual_blocks)])
        groups = min(16, hidden_dim)
        while hidden_dim % groups != 0:
            groups -= 1
        self.output = nn.Sequential(
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )

    def predict_local_basis(self, tokens: FractalTokens) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = tokens.latents.shape
        predicted_basis = self.basis_head(tokens.latents).reshape(
            batch_size,
            token_count,
            self.channels,
            self.basis_size,
            self.basis_size,
        )
        predicted_basis = predicted_basis * tokens.mask[:, :, None, None, None]
        basis_tiles = self.basis.decode(predicted_basis)
        base = render_token_tiles(
            basis_tiles,
            boxes=tokens.boxes,
            assignment=tokens.assignment,
            image_shape=tokens.image_shape,
            backend=self.kernel_backend,
        ).clamp(0.0, 1.0)
        return base, predicted_basis

    def forward(
        self,
        tokens: FractalTokens,
        refine: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base, predicted_basis = self.predict_local_basis(tokens)
        if not refine:
            return base, predicted_basis
        batch_size = tokens.latents.shape[0]
        grid_shape = tuple(tokens.assignment.shape[-2:])
        dense, coverage, _ = self.splat(
            tokens.latents,
            boxes=tokens.boxes,
            mask=tokens.mask,
            grid_shape=grid_shape,
        )
        base_low = F.interpolate(base, size=grid_shape, mode="bilinear", align_corners=False)
        y = torch.linspace(-1.0, 1.0, grid_shape[0], device=base.device, dtype=base.dtype)
        x = torch.linspace(-1.0, 1.0, grid_shape[1], device=base.device, dtype=base.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        feature_map = torch.cat([dense, base_low, torch.log1p(coverage), coordinates], dim=1)
        residual = self.output(self.blocks(self.input_projection(feature_map)))
        residual = F.interpolate(residual, size=tokens.image_shape, mode="bilinear", align_corners=False)
        reconstruction = torch.sigmoid(torch.logit(base.clamp(0.01, 0.99)) + residual)
        return reconstruction, predicted_basis
