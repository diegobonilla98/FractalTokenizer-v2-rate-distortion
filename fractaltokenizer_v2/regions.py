from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .cuda_ops import cuda_render_token_tiles, cuda_sample_region_tiles


def sample_region_tiles(
    image: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int,
    mask: torch.Tensor | None = None,
    backend: str = "auto",
    sampling_ratio: int = 2,
) -> torch.Tensor:
    batch_size, token_count, _ = boxes.shape
    height, width = image.shape[-2:]
    cuda_mask = torch.ones(boxes.shape[:2], device=boxes.device, dtype=torch.bool) if mask is None else mask
    cuda_tiles = cuda_sample_region_tiles(
        image,
        boxes,
        output_size,
        cuda_mask,
        sampling_ratio=sampling_ratio,
        backend=backend,
    )
    if cuda_tiles is not None:
        return cuda_tiles
    scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
    pixel_boxes = boxes * scale
    if mask is None:
        batch_indices = torch.arange(batch_size, device=boxes.device, dtype=boxes.dtype)
        batch_indices = batch_indices[:, None].expand(batch_size, token_count).reshape(-1, 1)
        rois = torch.cat([batch_indices, pixel_boxes.reshape(-1, 4)], dim=1)
        tiles = roi_align(
            image,
            rois,
            output_size=(output_size, output_size),
            spatial_scale=1.0,
            sampling_ratio=sampling_ratio,
            aligned=True,
        )
        return tiles.reshape(batch_size, token_count, image.shape[1], output_size, output_size)
    if mask.shape != boxes.shape[:2]:
        raise ValueError("mask must have shape [B,N]")
    valid = torch.nonzero(mask, as_tuple=False)
    tiles = torch.zeros(
        batch_size,
        token_count,
        image.shape[1],
        output_size,
        output_size,
        device=image.device,
        dtype=image.dtype,
    )
    if valid.numel() == 0:
        return tiles
    valid_boxes = pixel_boxes[valid[:, 0], valid[:, 1]]
    rois = torch.cat([valid[:, :1].to(boxes.dtype), valid_boxes], dim=1)
    valid_tiles = roi_align(
        image,
        rois,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=sampling_ratio,
        aligned=True,
    )
    tiles[valid[:, 0], valid[:, 1]] = valid_tiles
    return tiles


def _dct_matrix(size: int) -> torch.Tensor:
    position = torch.arange(size, dtype=torch.float64)
    frequency = torch.arange(size, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos(math.pi / size * (position + 0.5) * frequency)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix.float()


class LocalDCTBasis(nn.Module):
    def __init__(self, tile_size: int = 8, basis_size: int = 4):
        super().__init__()
        if basis_size > tile_size:
            raise ValueError("basis_size cannot exceed tile_size")
        self.tile_size = int(tile_size)
        self.basis_size = int(basis_size)
        self.register_buffer("matrix", _dct_matrix(tile_size), persistent=False)

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        shape = tiles.shape
        flat = tiles.reshape(-1, self.tile_size, self.tile_size)
        matrix = self.matrix.to(device=tiles.device, dtype=tiles.dtype)
        coefficients = matrix @ flat @ matrix.t()
        coefficients = coefficients[:, : self.basis_size, : self.basis_size]
        return coefficients.reshape(*shape[:-2], self.basis_size, self.basis_size)

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        shape = coefficients.shape
        flat = coefficients.reshape(-1, self.basis_size, self.basis_size)
        padded = torch.zeros(flat.shape[0], self.tile_size, self.tile_size, device=flat.device, dtype=flat.dtype)
        padded[:, : self.basis_size, : self.basis_size] = flat
        matrix = self.matrix.to(device=coefficients.device, dtype=coefficients.dtype)
        tiles = matrix.t() @ padded @ matrix
        return tiles.reshape(*shape[:-2], self.tile_size, self.tile_size)


class GeometryEncoder(nn.Module):
    def __init__(self, output_dim: int, fourier_bands: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.fourier_bands = int(fourier_bands)
        feature_dim = 7 + 4 * 2 * self.fourier_bands
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, boxes: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        width = (x1 - x0).clamp_min(1e-6)
        height = (y1 - y0).clamp_min(1e-6)
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        log_area = torch.log(width * height)
        normalized_depth = depth.to(boxes.dtype) / depth.max(dim=1, keepdim=True).values.clamp_min(1).to(boxes.dtype)
        raw = torch.stack(
            [
                center_x,
                center_y,
                torch.log(width),
                torch.log(height),
                torch.log(width / height),
                log_area,
                normalized_depth,
            ],
            dim=-1,
        )
        coordinates = torch.stack([center_x, center_y, width, height], dim=-1)
        frequencies = 2.0 ** torch.arange(self.fourier_bands, device=boxes.device, dtype=boxes.dtype)
        angles = 2.0 * math.pi * coordinates.unsqueeze(-1) * frequencies
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=-2)
        return self.network(torch.cat([raw, fourier], dim=-1)) * mask.unsqueeze(-1)


class RegionEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        tile_size: int = 8,
        basis_size: int = 4,
        latent_dim: int = 192,
        hidden_dim: int = 128,
        fourier_bands: int = 6,
        kernel_backend: str = "auto",
        sampling_ratio: int = 2,
    ):
        super().__init__()
        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.basis_size = int(basis_size)
        self.kernel_backend = kernel_backend
        self.sampling_ratio = int(sampling_ratio)
        self.basis = LocalDCTBasis(tile_size, basis_size)
        coefficient_dim = channels * basis_size * basis_size
        self.coefficient_projection = nn.Sequential(
            nn.Linear(coefficient_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(channels * tile_size * tile_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.geometry_encoder = GeometryEncoder(latent_dim, fourier_bands=fourier_bands, hidden_dim=hidden_dim)
        self.content_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        compact_valid_regions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sampling_mask = mask if compact_valid_regions else None
        tiles = sample_region_tiles(
            image,
            boxes,
            self.tile_size,
            mask=sampling_mask,
            backend=self.kernel_backend,
            sampling_ratio=self.sampling_ratio,
        )
        coefficients = self.basis.encode(tiles)
        batch_size, token_count = boxes.shape[:2]
        coefficient_features = self.coefficient_projection(coefficients.flatten(start_dim=2))
        low_frequency_tiles = self.basis.decode(coefficients)
        residual_tiles = tiles - low_frequency_tiles
        residual_features = self.residual_projection(residual_tiles.flatten(start_dim=2))
        content_latents = self.content_norm(coefficient_features + residual_features)
        geometry_latents = self.geometry_encoder(boxes, depth, mask)
        content_latents = content_latents * mask.unsqueeze(-1)
        geometry_latents = geometry_latents * mask.unsqueeze(-1)
        coefficients = coefficients * mask[:, :, None, None, None]
        tiles = tiles * mask[:, :, None, None, None]
        return content_latents, geometry_latents, coefficients, tiles


def render_token_tiles(
    tiles: torch.Tensor,
    boxes: torch.Tensor,
    assignment: torch.Tensor,
    image_shape: tuple[int, int],
    backend: str = "auto",
    clamp_output: bool = False,
) -> torch.Tensor:
    cuda_output = cuda_render_token_tiles(
        tiles,
        boxes,
        assignment,
        image_shape,
        clamp_output=clamp_output,
        backend=backend,
    )
    if cuda_output is not None:
        return cuda_output
    batch_size, token_count, channels, tile_height, tile_width = tiles.shape
    height, width = image_shape
    owner = F.interpolate(assignment.unsqueeze(1).float(), size=(height, width), mode="nearest")[:, 0].long()
    owner_flat = owner.reshape(batch_size, -1).clamp(0, token_count - 1)
    geometry_dtype = torch.float64 if tiles.dtype == torch.float64 or boxes.dtype == torch.float64 else torch.float32
    geometry_boxes = boxes.to(geometry_dtype)
    selected_boxes = torch.gather(
        geometry_boxes,
        1,
        owner_flat.unsqueeze(-1).expand(batch_size, owner_flat.shape[1], 4),
    )
    x = (torch.arange(width, device=tiles.device, dtype=geometry_dtype) + 0.5) / width
    y = (torch.arange(height, device=tiles.device, dtype=geometry_dtype) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.reshape(1, -1).expand(batch_size, -1)
    yy = yy.reshape(1, -1).expand(batch_size, -1)
    x0, y0, x1, y1 = selected_boxes.unbind(dim=-1)
    u = ((xx - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0.0, 1.0)
    v = ((yy - y0) / (y1 - y0).clamp_min(1e-6)).clamp(0.0, 1.0)
    tile_x = (u * tile_width - 0.5).clamp(0.0, tile_width - 1.0)
    tile_y = (v * tile_height - 0.5).clamp(0.0, tile_height - 1.0)
    x_low = tile_x.floor().long()
    y_low = tile_y.floor().long()
    x_high = (x_low + 1).clamp_max(tile_width - 1)
    y_high = (y_low + 1).clamp_max(tile_height - 1)
    weight_x = (tile_x - x_low.to(tile_x.dtype)).unsqueeze(-1)
    weight_y = (tile_y - y_low.to(tile_y.dtype)).unsqueeze(-1)
    bank = tiles.permute(0, 1, 3, 4, 2).reshape(batch_size, token_count * tile_height * tile_width, channels)

    def gather(y_index: torch.Tensor, x_index: torch.Tensor) -> torch.Tensor:
        index = owner_flat * (tile_height * tile_width) + y_index * tile_width + x_index
        return torch.gather(bank, 1, index.unsqueeze(-1).expand(-1, -1, channels))

    value_00 = gather(y_low, x_low)
    value_01 = gather(y_low, x_high)
    value_10 = gather(y_high, x_low)
    value_11 = gather(y_high, x_high)
    top = value_00 * (1.0 - weight_x) + value_01 * weight_x
    bottom = value_10 * (1.0 - weight_x) + value_11 * weight_x
    output = top * (1.0 - weight_y) + bottom * weight_y
    output = output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    return output.clamp(0.0, 1.0) if clamp_output else output
