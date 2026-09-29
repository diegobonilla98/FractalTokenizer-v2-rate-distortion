from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuda_ops import FractalCudaSettings
from .density import AnalyticInformationDensity
from .partition import BalancedBSPPartitioner, QuantileGridPartitioner
from .regions import render_token_tiles, sample_region_tiles
from .structures import FractalTokens, PartitionBatch


@dataclass(frozen=True)
class FractalDetokenizerSettings:
    backend: str = "auto"
    refine: bool = True
    compile_decoder: bool = True
    compile_refiner: bool = True
    clamp_output: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "cuda", "torch"}:
            raise ValueError("backend must be auto, cuda, or torch")

    @classmethod
    def from_profile(cls, profile: str) -> FractalDetokenizerSettings:
        if profile == "fast":
            return cls(refine=False)
        if profile == "quality":
            return cls()
        if profile == "reference":
            return cls(backend="torch", compile_decoder=False, compile_refiner=False)
        raise ValueError("profile must be fast, quality, or reference")


@dataclass(frozen=True)
class ContinuousAutoencoderConfig:
    channels: int = 3
    latent_dim: int = 256
    input_tile_size: int = 16
    output_tile_size: int = 32
    encoder_hidden_dim: int = 128
    decoder_hidden_dim: int = 192
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    attention_dropout: float = 0.0
    dropout: float = 0.0
    coordinate_fourier_bands: int = 8
    geometry_fourier_bands: int = 8
    decoder_token_chunk_size: int = 16
    refiner_hidden_dim: int = 128
    refiner_blocks: int = 4
    target_tokens: int = 96
    max_tokens: int = 192
    partition_stride: int = 8
    partition_backend: str = "quantile"
    bsp_min_size: int = 1
    bsp_workers: int = 12
    density_temperature: float = 0.4
    density_coherence_floor: float = 0.35
    density_blur_sigma: float = 0.0
    kernel_backend: str = "auto"
    partition_density_floor_ratio: float = 0.02
    region_sampling_ratio: int = 2
    detokenizer_postprocess_clamp: bool = False
    compile_density: bool = True
    compile_decoder: bool = True
    compile_refiner: bool = True

    @property
    def cuda_settings(self) -> FractalCudaSettings:
        return FractalCudaSettings(
            backend=self.kernel_backend,
            sampling_ratio=self.region_sampling_ratio,
            partition_density_floor_ratio=self.partition_density_floor_ratio,
            postprocess_clamp=self.detokenizer_postprocess_clamp,
        )

    @property
    def detokenizer_settings(self) -> FractalDetokenizerSettings:
        return FractalDetokenizerSettings(
            backend=self.kernel_backend,
            refine=True,
            compile_decoder=self.compile_decoder,
            compile_refiner=self.compile_refiner,
            clamp_output=self.detokenizer_postprocess_clamp,
        )


@dataclass
class ContinuousAutoencoderOutput:
    reconstruction: torch.Tensor
    base_reconstruction: torch.Tensor
    tokens: FractalTokens
    density: torch.Tensor
    local_tiles: torch.Tensor


def _fourier_features(values: torch.Tensor, bands: int) -> torch.Tensor:
    frequencies = 2.0 ** torch.arange(bands, device=values.device, dtype=values.dtype)
    angles = 2.0 * math.pi * values.unsqueeze(-1) * frequencies
    return torch.cat([values, torch.sin(angles).flatten(start_dim=-2), torch.cos(angles).flatten(start_dim=-2)], dim=-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class ContinuousRegionEncoder(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        hidden = config.encoder_hidden_dim
        self.tile_size = int(config.input_tile_size)
        self.kernel_backend = config.kernel_backend
        self.sampling_ratio = int(config.region_sampling_ratio)
        self.compact_valid_regions = config.partition_backend == "bsp"
        self.network = nn.Sequential(
            nn.Conv2d(config.channels, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            ResidualConvBlock(hidden),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            ResidualConvBlock(hidden),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 16, config.latent_dim * 2),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, image: torch.Tensor, partition: PartitionBatch) -> torch.Tensor:
        tiles = sample_region_tiles(
            image,
            partition.boxes,
            self.tile_size,
            mask=partition.mask if self.compact_valid_regions else None,
            backend=self.kernel_backend,
            sampling_ratio=self.sampling_ratio,
        )
        batch_size, token_count = tiles.shape[:2]
        features = self.network(tiles.flatten(0, 1))
        latents = self.norm(self.projection(features)).reshape(batch_size, token_count, -1)
        return latents * partition.mask.unsqueeze(-1)


class AdaptiveGeometryEncoder(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        self.fourier_bands = int(config.geometry_fourier_bands)
        feature_dim = 9 + 4 * 2 * self.fourier_bands
        self.network = nn.Sequential(
            nn.Linear(feature_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def forward(self, partition: PartitionBatch) -> torch.Tensor:
        boxes = partition.boxes
        mask = partition.mask
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        width = (x1 - x0).clamp_min(1e-6)
        height = (y1 - y0).clamp_min(1e-6)
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        area = width * height
        maximum_depth = partition.depth.max(dim=1, keepdim=True).values.clamp_min(1)
        normalized_depth = partition.depth.to(boxes.dtype) / maximum_depth.to(boxes.dtype)
        mass_sum = (partition.mass * mask).sum(dim=1, keepdim=True).clamp_min(1e-8)
        normalized_mass = partition.mass / mass_sum
        token_count = mask.sum(dim=1, keepdim=True).to(boxes.dtype).clamp_min(1)
        relative_mass = (normalized_mass * token_count).clamp_min(1e-8)
        raw = torch.stack(
            [
                center_x,
                center_y,
                torch.log(width),
                torch.log(height),
                torch.log(width / height),
                torch.log(area),
                normalized_depth,
                torch.log(relative_mass),
                area,
            ],
            dim=-1,
        )
        coordinates = torch.stack([center_x, center_y, width, height], dim=-1)
        frequencies = 2.0 ** torch.arange(self.fourier_bands, device=boxes.device, dtype=boxes.dtype)
        angles = 2.0 * math.pi * coordinates.unsqueeze(-1) * frequencies
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=-2)
        return self.network(torch.cat([raw, fourier], dim=-1)) * mask.unsqueeze(-1)


class GeometryAwareAttention(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        if config.latent_dim % config.transformer_heads:
            raise ValueError("latent_dim must be divisible by transformer_heads")
        self.heads = int(config.transformer_heads)
        self.head_dim = config.latent_dim // self.heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.latent_dim, config.latent_dim * 3)
        self.relative_bias = nn.Sequential(
            nn.Linear(8, config.latent_dim // 2),
            nn.GELU(),
            nn.Linear(config.latent_dim // 2, self.heads),
        )
        self.mass_scale = nn.Parameter(torch.full((self.heads,), 0.25))
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output = nn.Linear(config.latent_dim, config.latent_dim)
        self.output_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _relative_features(partition: PartitionBatch) -> torch.Tensor:
        boxes = partition.boxes
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        width = (x1 - x0).clamp_min(1e-6)
        height = (y1 - y0).clamp_min(1e-6)
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        delta_x = (center_x[:, None, :] - center_x[:, :, None]) / width[:, :, None]
        delta_y = (center_y[:, None, :] - center_y[:, :, None]) / height[:, :, None]
        log_width_ratio = torch.log(width[:, None, :] / width[:, :, None])
        log_height_ratio = torch.log(height[:, None, :] / height[:, :, None])
        log_area_ratio = log_width_ratio + log_height_ratio
        horizontal_overlap = torch.minimum(y1[:, :, None], y1[:, None, :]) - torch.maximum(y0[:, :, None], y0[:, None, :])
        vertical_overlap = torch.minimum(x1[:, :, None], x1[:, None, :]) - torch.maximum(x0[:, :, None], x0[:, None, :])
        horizontal_touch = torch.isclose(x1[:, :, None], x0[:, None, :], atol=1e-5, rtol=0.0) | torch.isclose(
            x0[:, :, None], x1[:, None, :], atol=1e-5, rtol=0.0
        )
        vertical_touch = torch.isclose(y1[:, :, None], y0[:, None, :], atol=1e-5, rtol=0.0) | torch.isclose(
            y0[:, :, None], y1[:, None, :], atol=1e-5, rtol=0.0
        )
        adjacent = ((horizontal_touch & (horizontal_overlap > 0)) | (vertical_touch & (vertical_overlap > 0))).to(boxes.dtype)
        maximum_depth = partition.depth.max(dim=1, keepdim=True).values.clamp_min(1).to(boxes.dtype)
        depth_delta = (partition.depth[:, None, :] - partition.depth[:, :, None]).abs().to(boxes.dtype)
        depth_delta = depth_delta / maximum_depth.unsqueeze(-1)
        distance = torch.sqrt(delta_x.square() + delta_y.square() + 1e-8)
        return torch.stack(
            [delta_x, delta_y, log_width_ratio, log_height_ratio, log_area_ratio, adjacent, depth_delta, distance],
            dim=-1,
        ).clamp(-8.0, 8.0)

    def forward(self, values: torch.Tensor, partition: PartitionBatch) -> torch.Tensor:
        batch_size, token_count, dimension = values.shape
        qkv = self.qkv(values).reshape(batch_size, token_count, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        relative = self.relative_bias(self._relative_features(partition)).permute(0, 3, 1, 2)
        mass_sum = (partition.mass * partition.mask).sum(dim=1, keepdim=True).clamp_min(1e-8)
        normalized_mass = partition.mass / mass_sum
        token_average = partition.mask.sum(dim=1, keepdim=True).to(values.dtype).clamp_min(1)
        log_mass = torch.log((normalized_mass * token_average).clamp_min(1e-8)).clamp(-4.0, 4.0)
        scores = scores + relative + self.mass_scale.view(1, -1, 1, 1) * log_mass[:, None, None, :]
        scores = scores.masked_fill(~partition.mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch_size, token_count, dimension)
        output = self.output_dropout(self.output(output))
        return output * partition.mask.unsqueeze(-1)


class GeometryTransformerBlock(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        hidden = round(config.latent_dim * config.transformer_mlp_ratio)
        self.attention_norm = nn.LayerNorm(config.latent_dim)
        self.attention = GeometryAwareAttention(config)
        self.mlp_norm = nn.LayerNorm(config.latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.latent_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: torch.Tensor, partition: PartitionBatch) -> torch.Tensor:
        mask = partition.mask.unsqueeze(-1)
        values = (values + self.attention(self.attention_norm(values), partition)) * mask
        values = (values + self.mlp(self.mlp_norm(values))) * mask
        return values


class GeometryAwareTransformer(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        self.blocks = nn.ModuleList([GeometryTransformerBlock(config) for _ in range(config.transformer_depth)])
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, values: torch.Tensor, partition: PartitionBatch) -> torch.Tensor:
        for block in self.blocks:
            values = block(values, partition)
        return self.norm(values) * partition.mask.unsqueeze(-1)


class LocalCoordinateDecoder(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        self.channels = int(config.channels)
        self.tile_size = int(config.output_tile_size)
        self.chunk_size = int(config.decoder_token_chunk_size)
        self.coordinate_bands = int(config.coordinate_fourier_bands)
        coordinate_dim = 2 + 2 * 2 * self.coordinate_bands
        self.token_projection = nn.Linear(config.latent_dim, config.decoder_hidden_dim)
        self.coordinate_projection = nn.Linear(coordinate_dim, config.decoder_hidden_dim)
        self.network = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.channels),
        )
        coordinates = torch.linspace(-0.5, 0.5, self.tile_size)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy], dim=-1).reshape(-1, 2), persistent=False)

    def forward(self, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, token_count = latents.shape[:2]
        coordinates = _fourier_features(self.coordinates.to(latents.dtype), self.coordinate_bands)
        coordinate_features = self.coordinate_projection(coordinates)
        token_features = self.token_projection(latents)
        chunks = []
        for start in range(0, token_count, self.chunk_size):
            end = min(start + self.chunk_size, token_count)
            hidden = token_features[:, start:end, None, :] + coordinate_features[None, None, :, :]
            pixels = torch.sigmoid(self.network(hidden))
            chunks.append(pixels)
        tiles = torch.cat(chunks, dim=1)
        tiles = tiles.reshape(batch_size, token_count, self.tile_size, self.tile_size, self.channels)
        tiles = tiles.permute(0, 1, 4, 2, 3)
        return tiles * mask[:, :, None, None, None]


class SpatialRefiner(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig):
        super().__init__()
        self.input = nn.Conv2d(config.channels + 3, config.refiner_hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualConvBlock(config.refiner_hidden_dim) for _ in range(config.refiner_blocks)])
        groups = min(16, config.refiner_hidden_dim)
        while config.refiner_hidden_dim % groups:
            groups -= 1
        self.output = nn.Sequential(
            nn.GroupNorm(groups, config.refiner_hidden_dim),
            nn.SiLU(),
            nn.Conv2d(config.refiner_hidden_dim, config.channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _boundary_map(assignment: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
        owner = F.interpolate(assignment.unsqueeze(1).float(), size=image_shape, mode="nearest")[:, 0].long()
        boundary = torch.zeros_like(owner, dtype=torch.float32)
        boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], (owner[:, :, 1:] != owner[:, :, :-1]).float())
        boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], (owner[:, :, 1:] != owner[:, :, :-1]).float())
        boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], (owner[:, 1:, :] != owner[:, :-1, :]).float())
        boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], (owner[:, 1:, :] != owner[:, :-1, :]).float())
        return boundary.unsqueeze(1)

    def forward(self, base: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = base.shape
        y = torch.linspace(-1.0, 1.0, height, device=base.device, dtype=base.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=base.device, dtype=base.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        boundary = self._boundary_map(assignment, (height, width)).to(base.dtype)
        residual = self.output(self.blocks(self.input(torch.cat([base, coordinates, boundary], dim=1))))
        return torch.sigmoid(torch.logit(base.clamp(0.001, 0.999)) + residual)


class ContinuousFractalAutoencoder(nn.Module):
    def __init__(self, config: ContinuousAutoencoderConfig | None = None):
        super().__init__()
        self.config = ContinuousAutoencoderConfig() if config is None else config
        self._validate_config(self.config)
        self.density_model = AnalyticInformationDensity(
            partition_stride=self.config.partition_stride,
            temperature=self.config.density_temperature,
            coherence_floor=self.config.density_coherence_floor,
            blur_sigma=self.config.density_blur_sigma,
        )
        object.__setattr__(self, "_compiled_density_forward", None)
        if self.config.partition_backend == "quantile":
            self.partitioner = QuantileGridPartitioner(
                density_floor_ratio=self.config.partition_density_floor_ratio,
                backend=self.config.kernel_backend,
            )
        else:
            self.partitioner = BalancedBSPPartitioner(
                min_size=self.config.bsp_min_size,
                max_tokens=self.config.max_tokens,
                workers=self.config.bsp_workers,
            )
        self.content_encoder = ContinuousRegionEncoder(self.config)
        self.geometry_encoder = AdaptiveGeometryEncoder(self.config)
        self.input_norm = nn.LayerNorm(self.config.latent_dim)
        self.transformer = GeometryAwareTransformer(self.config)
        self.local_decoder = LocalCoordinateDecoder(self.config)
        self.refiner = SpatialRefiner(self.config)
        object.__setattr__(self, "_compiled_decoder_forward", None)
        object.__setattr__(self, "_compiled_refiner_forward", None)

    @staticmethod
    def _validate_config(config: ContinuousAutoencoderConfig) -> None:
        integer_values = {
            "channels": config.channels,
            "latent_dim": config.latent_dim,
            "input_tile_size": config.input_tile_size,
            "output_tile_size": config.output_tile_size,
            "transformer_depth": config.transformer_depth,
            "transformer_heads": config.transformer_heads,
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "partition_stride": config.partition_stride,
            "decoder_token_chunk_size": config.decoder_token_chunk_size,
            "region_sampling_ratio": config.region_sampling_ratio,
        }
        invalid = [name for name, value in integer_values.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if config.latent_dim % config.transformer_heads:
            raise ValueError("latent_dim must be divisible by transformer_heads")
        if config.partition_backend not in {"quantile", "bsp"}:
            raise ValueError("partition_backend must be quantile or bsp")
        if config.kernel_backend not in {"auto", "cuda", "torch"}:
            raise ValueError("kernel_backend must be auto, cuda, or torch")
        if config.partition_density_floor_ratio < 0:
            raise ValueError("partition_density_floor_ratio cannot be negative")
        if config.target_tokens > config.max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")

    def partition(self, density: torch.Tensor, target_tokens: int | None = None) -> PartitionBatch:
        target = self.config.target_tokens if target_tokens is None else int(target_tokens)
        if target < 1 or target > self.config.max_tokens:
            raise ValueError("target_tokens must be within [1,max_tokens]")
        return self.partitioner(density.detach(), target_tokens=target)

    def _density_forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.density_model(image.float()).to(image.dtype)
        if not self.config.compile_density or not image.is_cuda:
            return self.density_model(image.float()).to(image.dtype)
        compiled = object.__getattribute__(self, "_compiled_density_forward")
        if compiled is None:
            compiled = torch.compile(self.density_model.forward, fullgraph=True, mode="reduce-overhead")
            object.__setattr__(self, "_compiled_density_forward", compiled)
        return compiled(image.float()).to(image.dtype)

    def encode(
        self,
        image: torch.Tensor,
        target_tokens: int | None = None,
    ) -> tuple[FractalTokens, torch.Tensor]:
        if (
            image.is_cuda
            and not torch.compiler.is_compiling()
            and (self.config.compile_density or self.config.compile_decoder or self.config.compile_refiner)
        ):
            torch.compiler.cudagraph_mark_step_begin()
        density = self._density_forward(image)
        partition = self.partition(density, target_tokens=target_tokens)
        content = self.content_encoder(image, partition)
        geometry = self.geometry_encoder(partition)
        latents = self.transformer(self.input_norm(content + geometry), partition)
        tokens = FractalTokens(
            latents=latents,
            boxes=partition.boxes,
            boxes_grid=partition.boxes_grid,
            mask=partition.mask,
            mass=partition.mass,
            depth=partition.depth,
            assignment=partition.assignment,
            image_shape=tuple(image.shape[-2:]),
            trees=partition.trees,
        )
        return tokens, density

    def _decode_components(
        self,
        tokens: FractalTokens,
        settings: FractalDetokenizerSettings,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_tiles = self._local_decoder_forward(
            tokens.latents,
            tokens.mask,
            compile_enabled=settings.compile_decoder,
        )
        base = render_token_tiles(
            local_tiles,
            tokens.boxes,
            tokens.assignment,
            tokens.image_shape,
            backend=settings.backend,
            clamp_output=settings.clamp_output,
        )
        reconstruction = (
            self._refiner_forward(base, tokens.assignment, compile_enabled=settings.compile_refiner)
            if settings.refine
            else base
        )
        return reconstruction, base, local_tiles

    def _local_decoder_forward(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        compile_enabled: bool,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.local_decoder(latents, mask)
        if not compile_enabled or not latents.is_cuda:
            return self.local_decoder(latents, mask)
        compiled = object.__getattribute__(self, "_compiled_decoder_forward")
        if compiled is None:
            compiled = torch.compile(
                self.local_decoder.forward,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
            object.__setattr__(self, "_compiled_decoder_forward", compiled)
        return compiled(latents, mask).clone()

    def _refiner_forward(
        self,
        base: torch.Tensor,
        assignment: torch.Tensor,
        compile_enabled: bool,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.refiner(base, assignment)
        if not compile_enabled or not base.is_cuda:
            return self.refiner(base, assignment)
        compiled = object.__getattribute__(self, "_compiled_refiner_forward")
        if compiled is None:
            compiled = torch.compile(
                self.refiner.forward,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
            object.__setattr__(self, "_compiled_refiner_forward", compiled)
        return compiled(base, assignment).clone()

    def decode(
        self,
        tokens: FractalTokens,
        refine: bool | None = None,
        settings: FractalDetokenizerSettings | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.latents.is_cuda and not torch.compiler.is_compiling():
            torch.compiler.cudagraph_mark_step_begin()
        if settings is not None and refine is not None:
            raise ValueError("pass either refine or settings, not both")
        if settings is None:
            settings = self.config.detokenizer_settings
            if refine is not None:
                settings = FractalDetokenizerSettings(
                    backend=settings.backend,
                    refine=refine,
                    compile_decoder=settings.compile_decoder,
                    compile_refiner=settings.compile_refiner,
                    clamp_output=settings.clamp_output,
                )
        reconstruction, _, local_tiles = self._decode_components(tokens, settings)
        return reconstruction, local_tiles

    def forward(self, image: torch.Tensor, target_tokens: int | None = None) -> ContinuousAutoencoderOutput:
        tokens, density = self.encode(image, target_tokens=target_tokens)
        reconstruction, base, local_tiles = self._decode_components(tokens, self.config.detokenizer_settings)
        return ContinuousAutoencoderOutput(
            reconstruction=reconstruction,
            base_reconstruction=base,
            tokens=tokens,
            density=density,
            local_tiles=local_tiles,
        )


class ContinuousReconstructionLoss(nn.Module):
    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        mse_weight: float = 0.5,
        multiscale_weight: float = 0.5,
        gradient_weight: float = 0.25,
        seam_weight: float = 0.1,
        base_weight: float = 0.25,
        charbonnier_epsilon: float = 1e-3,
    ):
        super().__init__()
        self.charbonnier_weight = float(charbonnier_weight)
        self.mse_weight = float(mse_weight)
        self.multiscale_weight = float(multiscale_weight)
        self.gradient_weight = float(gradient_weight)
        self.seam_weight = float(seam_weight)
        self.base_weight = float(base_weight)
        self.charbonnier_epsilon = float(charbonnier_epsilon)

    def _charbonnier(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.charbonnier_epsilon**2).mean()

    @staticmethod
    def _gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image[..., :, 1:] - image[..., :, :-1], image[..., 1:, :] - image[..., :-1, :]

    def _multiscale(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for _ in range(4):
            losses.append(F.l1_loss(prediction, target))
            if min(prediction.shape[-2:]) < 8:
                break
            prediction = F.avg_pool2d(prediction, 2, stride=2)
            target = F.avg_pool2d(target, 2, stride=2)
        return torch.stack(losses).mean()

    def _gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_x, prediction_y = self._gradients(prediction)
        target_x, target_y = self._gradients(target)
        return 0.5 * (F.l1_loss(prediction_x, target_x) + F.l1_loss(prediction_y, target_y))

    def _seam(self, prediction: torch.Tensor, target: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        owner = F.interpolate(assignment.unsqueeze(1).float(), size=prediction.shape[-2:], mode="nearest")[:, 0].long()
        boundary_x = (owner[:, :, 1:] != owner[:, :, :-1]).unsqueeze(1)
        boundary_y = (owner[:, 1:, :] != owner[:, :-1, :]).unsqueeze(1)
        error_x, error_y = self._gradients(prediction - target)
        numerator = (error_x.abs() * boundary_x).sum() + (error_y.abs() * boundary_y).sum()
        denominator = (boundary_x.sum() + boundary_y.sum()).clamp_min(1) * prediction.shape[1]
        return numerator / denominator

    def forward(self, output: ContinuousAutoencoderOutput, target: torch.Tensor) -> dict[str, torch.Tensor]:
        charbonnier = self._charbonnier(output.reconstruction, target)
        mse = F.mse_loss(output.reconstruction, target)
        multiscale = self._multiscale(output.reconstruction, target)
        gradient = self._gradient(output.reconstruction, target)
        seam = self._seam(output.reconstruction, target, output.tokens.assignment)
        base = self._charbonnier(output.base_reconstruction, target)
        total = (
            self.charbonnier_weight * charbonnier
            + self.mse_weight * mse
            + self.multiscale_weight * multiscale
            + self.gradient_weight * gradient
            + self.seam_weight * seam
            + self.base_weight * base
        )
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
        return {
            "loss": total,
            "charbonnier": charbonnier,
            "mse": mse,
            "psnr": psnr,
            "multiscale": multiscale,
            "gradient": gradient,
            "seam": seam,
            "base": base,
            "tokens": output.tokens.token_count.float().mean(),
        }
