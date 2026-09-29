from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .density import AnalyticInformationDensity, UniformInformationDensity
from .partition import QuantileGridPartitioner
from .regions import RegionEncoder
from .universal_codelength import HilbertEqualMassPartitioner, UniversalCodelengthDensity
from fractaltokenizer_v3.information import PredictiveInformationDensity, token_counts_from_information
from fractaltokenizer_v3.partition import VariableRateQuantilePartitioner


@dataclass(frozen=True)
class ClassifierConfig:
    variant: str = "fractal_128"
    image_size: int = 256
    num_classes: int = 100
    token_count: int = 128
    model_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    partition_stride: int = 8
    tile_size: int = 8
    basis_size: int = 4
    density_temperature: float = 1.0
    density_coherence_floor: float = 0.35
    density_blur_sigma: float = 0.0
    patch_size: tuple[int, int] = (16, 32)
    kernel_backend: str = "auto"
    partition_density_floor_ratio: float = 0.02
    region_sampling_ratio: int = 2
    compile_density: bool = True
    universal_mass_per_token: float = 9926.331293741861
    maximum_tokens: int = 160
    v3_capacity_bits_64: float = 11154.006347656252
    v3_capacity_bits_96: float = 6857.214111328126


class AdaptiveFractalEmbed(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.token_count = int(config.token_count)
        self.compile_density = bool(config.compile_density)
        object.__setattr__(self, "_compiled_density_forward", None)
        density_mode = {
            "fractal_128": "analytic",
            "fractal_analytic_128": "analytic",
            "fractal_analytic_64": "analytic",
            "fractal_analytic_96": "analytic",
            "fractal_universal_128": "universal",
            "fractal_uniform_128": "uniform",
        }.get(config.variant)
        if density_mode is None:
            raise ValueError(f"invalid adaptive fractal variant: {config.variant}")
        self.density_mode = density_mode
        if density_mode == "analytic":
            self.density = AnalyticInformationDensity(
                partition_stride=config.partition_stride,
                coherence_floor=config.density_coherence_floor,
                temperature=config.density_temperature,
                blur_sigma=config.density_blur_sigma,
            )
        elif density_mode == "universal":
            self.density = UniversalCodelengthDensity(
                partition_stride=config.partition_stride,
                backend=config.kernel_backend,
            )
        else:
            self.density = UniformInformationDensity(partition_stride=config.partition_stride)
        self.partitioner = QuantileGridPartitioner(
            density_floor_ratio=config.partition_density_floor_ratio,
            backend=config.kernel_backend,
        )
        self.region_encoder = RegionEncoder(
            channels=3,
            tile_size=config.tile_size,
            basis_size=config.basis_size,
            latent_dim=config.model_dim,
            hidden_dim=max(128, config.model_dim // 2),
            kernel_backend=config.kernel_backend,
            sampling_ratio=config.region_sampling_ratio,
        )
        self.norm = nn.LayerNorm(config.model_dim)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _density_forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.density(image)
        if not self.compile_density or not image.is_cuda:
            return self.density(image)
        compiled = object.__getattribute__(self, "_compiled_density_forward")
        if compiled is None:
            compiled = torch.compile(self.density.forward, fullgraph=True, mode="reduce-overhead")
            object.__setattr__(self, "_compiled_density_forward", compiled)
        return compiled(image)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        density_image = (image.float() * self.image_std + self.image_mean).clamp(0.0, 1.0)
        density = self._density_forward(density_image)
        partition = self.partitioner(density.detach(), target_tokens=self.token_count)
        content, geometry, _, _ = self.region_encoder(
            image,
            boxes=partition.boxes,
            depth=partition.depth,
            mask=partition.mask,
        )
        tokens = self.norm(content + geometry) * partition.mask.unsqueeze(-1)
        return tokens, partition.mask


class VariableInformationEmbed(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        capacities = {
            "fractal_v3_predictive_64": config.v3_capacity_bits_64,
            "fractal_v3_predictive_96": config.v3_capacity_bits_96,
        }
        if config.variant not in capacities:
            raise ValueError(f"invalid v3 information variant: {config.variant}")
        self.token_count = int(config.maximum_tokens)
        self.maximum_tokens = int(config.maximum_tokens)
        self.capacity_bits = float(capacities[config.variant])
        self.compile_density = bool(config.compile_density)
        object.__setattr__(self, "_compiled_density_forward", None)
        self.density = PredictiveInformationDensity(
            partition_stride=config.partition_stride,
            residual_scale=1.0 / 255.0,
            scan_directions=4,
        )
        self.partitioner = VariableRateQuantilePartitioner(
            maximum_tokens=config.maximum_tokens,
            density_floor_ratio=0.0,
            backend=config.kernel_backend,
            uniform_when_saturated=True,
        )
        self.region_encoder = RegionEncoder(
            channels=3,
            tile_size=config.tile_size,
            basis_size=config.basis_size,
            latent_dim=config.model_dim,
            hidden_dim=max(128, config.model_dim // 2),
            kernel_backend=config.kernel_backend,
            sampling_ratio=config.region_sampling_ratio,
        )
        self.norm = nn.LayerNorm(config.model_dim)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _density_forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.density(image)
        if not self.compile_density or not image.is_cuda:
            return self.density(image)
        compiled = object.__getattribute__(self, "_compiled_density_forward")
        if compiled is None:
            compiled = torch.compile(self.density.forward, fullgraph=True, mode="reduce-overhead")
            object.__setattr__(self, "_compiled_density_forward", compiled)
        return compiled(image)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        density_image = (image.float() * self.image_std + self.image_mean).clamp(0.0, 1.0)
        density = self._density_forward(density_image)
        total_information = density.sum(dim=(1, 2, 3))
        token_counts = token_counts_from_information(
            total_information,
            capacity_bits=self.capacity_bits,
            maximum_tokens=self.maximum_tokens,
        )
        partition = self.partitioner(density.detach(), token_counts)
        content, geometry, _, _ = self.region_encoder(
            image,
            boxes=partition.boxes,
            depth=partition.depth,
            mask=partition.mask,
            compact_valid_regions=True,
        )
        tokens = self.norm(content + geometry) * partition.mask.unsqueeze(-1)
        return tokens, partition.mask


class FixedPatchEmbed(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        patch_height, patch_width = config.patch_size
        if config.image_size % patch_height or config.image_size % patch_width:
            raise ValueError("image_size must be divisible by both patch dimensions")
        self.grid_size = (config.image_size // patch_height, config.image_size // patch_width)
        self.token_count = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(
            3,
            config.model_dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.projection(image).flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        return tokens, mask


class HilbertPatchEmbed(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        if config.image_size % config.partition_stride:
            raise ValueError("image_size must be divisible by partition_stride")
        if config.variant not in {"hilbert_uniform_96", "hilbert_universal_96"}:
            raise ValueError("invalid Hilbert tokenizer variant")
        self.variant = config.variant
        self.token_count = int(config.token_count)
        self.maximum_tokens = int(config.maximum_tokens)
        self.partition_stride = int(config.partition_stride)
        self.density = UniversalCodelengthDensity(
            partition_stride=config.partition_stride,
            backend=config.kernel_backend,
        )
        self.partitioner = HilbertEqualMassPartitioner(maximum_tokens=config.maximum_tokens)
        self.mass_per_token = float(config.universal_mass_per_token)
        self.projection = nn.Conv2d(
            3,
            config.model_dim,
            kernel_size=config.partition_stride,
            stride=config.partition_stride,
        )
        self.moment_projection = nn.Linear(config.model_dim * 3, config.model_dim)
        self.geometry_projection = nn.Sequential(
            nn.Linear(8, config.model_dim // 2),
            nn.GELU(),
            nn.Linear(config.model_dim // 2, config.model_dim),
        )
        self.norm = nn.LayerNorm(config.model_dim)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        density_image = (image.float() * self.image_std + self.image_mean).clamp(0.0, 1.0)
        if self.variant == "hilbert_uniform_96":
            density = torch.ones(
                image.shape[0],
                1,
                image.shape[2] // self.partition_stride,
                image.shape[3] // self.partition_stride,
                device=image.device,
            )
            partition = self.partitioner(density, target_tokens=self.token_count)
        else:
            density = self.density(density_image)
            partition = self.partitioner(density, mass_per_token=self.mass_per_token)
        atomic = self.projection(image).flatten(2).transpose(1, 2)
        batch_size, cells, channels = atomic.shape
        token_capacity = partition.mask.shape[1]
        assignment = partition.assignment.flatten(1)
        index = assignment.unsqueeze(-1).expand(-1, -1, channels)
        coordinates = torch.linspace(
            0.5 / partition.assignment.shape[-1],
            1.0 - 0.5 / partition.assignment.shape[-1],
            partition.assignment.shape[-1],
            device=image.device,
            dtype=atomic.dtype,
        )
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        x = x.flatten().view(1, cells, 1)
        y = y.flatten().view(1, cells, 1)
        pooled = torch.zeros(batch_size, token_capacity, channels * 3, device=image.device, dtype=atomic.dtype)
        pooled[:, :, :channels].scatter_add_(1, index, atomic)
        pooled[:, :, channels : 2 * channels].scatter_add_(1, index, atomic * x)
        pooled[:, :, 2 * channels :].scatter_add_(1, index, atomic * y)
        counts = torch.zeros(batch_size, token_capacity, 1, device=image.device, dtype=atomic.dtype)
        counts.scatter_add_(1, assignment.unsqueeze(-1), torch.ones(batch_size, cells, 1, device=image.device, dtype=atomic.dtype))
        pooled = pooled / counts.clamp_min(1.0)
        total_mass = partition.mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
        cells_float = float(cells)
        geometry = torch.cat(
            (
                partition.intervals.float() / cells_float,
                partition.boxes,
                (partition.intervals[..., 1:] - partition.intervals[..., :1]).float() / cells_float,
                (partition.mass / total_mass).unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = self.norm(self.moment_projection(pooled) + self.geometry_projection(geometry))
        return tokens * partition.mask.unsqueeze(-1), partition.mask


class TransformerClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig, token_embed: nn.Module):
        super().__init__()
        self.config = config
        self.token_embed = token_embed
        token_count = int(token_embed.token_count)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        self.position_embedding = None if isinstance(token_embed, HilbertPatchEmbed) else nn.Parameter(
            torch.zeros(1, token_count + 1, config.model_dim)
        )
        self.position_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=round(config.model_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        layer.self_attn.dropout = config.attention_dropout
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.depth,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(config.model_dim, config.num_classes)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            nn.init.trunc_normal_(module.weight, std=math.sqrt(1.0 / fan_in))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens, mask = self.token_embed(image)
        self.last_token_counts = mask.sum(dim=1).detach()
        self.last_padded_token_count = int(mask.shape[1])
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = self.position_dropout(tokens)
        cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)
        tokens = self.transformer(tokens, src_key_padding_mask=~mask)
        return self.head(tokens[:, 0])

    @property
    def token_count(self) -> int:
        return int(self.token_embed.token_count)


def make_classifier(config: ClassifierConfig) -> TransformerClassifier:
    if config.variant in {
        "fractal_128",
        "fractal_analytic_128",
        "fractal_analytic_64",
        "fractal_analytic_96",
        "fractal_universal_128",
        "fractal_uniform_128",
    }:
        token_embed = AdaptiveFractalEmbed(config)
    elif config.variant in {"fractal_v3_predictive_64", "fractal_v3_predictive_96"}:
        token_embed = VariableInformationEmbed(config)
    elif config.variant == "vit_rect_128":
        exact_config = ClassifierConfig(**{**config.__dict__, "patch_size": (16, 32)})
        token_embed = FixedPatchEmbed(exact_config)
    elif config.variant == "vit_patch16_256":
        standard_config = ClassifierConfig(**{**config.__dict__, "patch_size": (16, 16)})
        token_embed = FixedPatchEmbed(standard_config)
    elif config.variant in {"hilbert_uniform_96", "hilbert_universal_96"}:
        token_embed = HilbertPatchEmbed(config)
    else:
        raise ValueError(f"unknown classifier variant: {config.variant}")
    return TransformerClassifier(config, token_embed)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total
