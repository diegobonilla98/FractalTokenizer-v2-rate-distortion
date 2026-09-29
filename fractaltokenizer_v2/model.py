from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .codec import FractalTokenPacket
from .decoder import NeuralDetokenizer
from .density import AnalyticInformationDensity, DensityStudent, MaskedSurprisalDensity, UniformInformationDensity
from .generative import FractalSequenceBatch, pack_fractal_sequence, unpack_fractal_sequence
from .partition import (
    BalancedBSPPartitioner,
    QuantileGridPartitioner,
    deserialize_tree,
    partition_batch_from_trees,
)
from .quantization import ResidualVectorQuantizer
from .regions import RegionEncoder, render_token_tiles
from .structures import FractalOutput, FractalTokens, PartitionBatch, QuantizerOutput


@dataclass
class FractalTokenizerV2Config:
    channels: int = 3
    latent_dim: int = 192
    tile_size: int = 8
    basis_size: int = 4
    partition_stride: int = 8
    target_tokens: int = 128
    token_grid: Optional[tuple[int, int]] = None
    partition_backend: str = "quantile"
    mass_per_token: Optional[float] = None
    max_tokens: int = 256
    bsp_min_size: int = 1
    bsp_workers: int = 0
    density_mode: str = "analytic"
    density_pyramid_levels: int = 4
    density_hidden_dim: int = 64
    density_temperature: float = 1.0
    density_coherence_floor: float = 0.35
    density_blur_sigma: float = 0.0
    kernel_backend: str = "auto"
    partition_density_floor_ratio: float = 0.02
    region_sampling_ratio: int = 2
    detach_density_for_partition: bool = True
    use_vector_quantization: bool = True
    codebook_size: int = 8192
    num_quantizers: int = 1
    commitment_weight: float = 0.25
    codebook_ema_decay: Optional[float] = 0.99
    codebook_ema_epsilon: float = 1e-5
    dead_code_threshold: float = 0.1
    dead_code_check_interval: int = 100
    decoder_feature_dim: int = 128
    decoder_hidden_dim: int = 160
    decoder_residual_blocks: int = 4
    gaussian_sigma_scale: float = 0.35
    gaussian_chunk_size: int = 64


class FractalTokenizerV2(nn.Module):
    @staticmethod
    def _validate_config(config: FractalTokenizerV2Config) -> None:
        positive = {
            "channels": config.channels,
            "latent_dim": config.latent_dim,
            "tile_size": config.tile_size,
            "basis_size": config.basis_size,
            "partition_stride": config.partition_stride,
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "bsp_min_size": config.bsp_min_size,
            "density_pyramid_levels": config.density_pyramid_levels,
            "density_hidden_dim": config.density_hidden_dim,
            "decoder_feature_dim": config.decoder_feature_dim,
            "decoder_hidden_dim": config.decoder_hidden_dim,
            "decoder_residual_blocks": config.decoder_residual_blocks,
            "gaussian_chunk_size": config.gaussian_chunk_size,
            "dead_code_check_interval": config.dead_code_check_interval,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if config.basis_size > config.tile_size:
            raise ValueError("basis_size cannot exceed tile_size")
        if config.mass_per_token is not None and config.mass_per_token <= 0:
            raise ValueError("mass_per_token must be positive")
        if config.density_temperature <= 0:
            raise ValueError("density_temperature must be positive")
        if not 0.0 <= config.density_coherence_floor <= 1.0:
            raise ValueError("density_coherence_floor must be in [0,1]")
        if config.density_blur_sigma < 0:
            raise ValueError("density_blur_sigma cannot be negative")
        if config.kernel_backend not in {"auto", "cuda", "torch"}:
            raise ValueError("kernel_backend must be auto, cuda, or torch")
        if config.partition_density_floor_ratio < 0:
            raise ValueError("partition_density_floor_ratio cannot be negative")
        if config.region_sampling_ratio < 1:
            raise ValueError("region_sampling_ratio must be positive")
        if config.gaussian_sigma_scale <= 0:
            raise ValueError("gaussian_sigma_scale must be positive")
        if config.use_vector_quantization:
            if config.codebook_size < 2:
                raise ValueError("codebook_size must be at least 2")
            if config.num_quantizers < 1:
                raise ValueError("num_quantizers must be positive")
            if config.commitment_weight < 0:
                raise ValueError("commitment_weight cannot be negative")
            if config.codebook_ema_decay is not None and not 0.0 <= config.codebook_ema_decay < 1.0:
                raise ValueError("codebook_ema_decay must be in [0,1) or None")
            if config.codebook_ema_epsilon <= 0:
                raise ValueError("codebook_ema_epsilon must be positive")
            if config.dead_code_threshold < 0:
                raise ValueError("dead_code_threshold cannot be negative")
        if config.token_grid is not None:
            rows, columns = config.token_grid
            if rows < 1 or columns < 1:
                raise ValueError("token_grid values must be positive")

    def __init__(
        self,
        config: FractalTokenizerV2Config | None = None,
        density_model: nn.Module | None = None,
    ):
        super().__init__()
        self.config = FractalTokenizerV2Config() if config is None else config
        config = self.config
        self._validate_config(config)
        if density_model is not None:
            self.density_model = density_model
        elif config.density_mode == "analytic":
            self.density_model = AnalyticInformationDensity(
                partition_stride=config.partition_stride,
                pyramid_levels=config.density_pyramid_levels,
                coherence_floor=config.density_coherence_floor,
                temperature=config.density_temperature,
                blur_sigma=config.density_blur_sigma,
            )
        elif config.density_mode == "student":
            self.density_model = DensityStudent(
                channels=config.channels,
                hidden_dim=config.density_hidden_dim,
                partition_stride=config.partition_stride,
            )
        elif config.density_mode == "surprisal":
            self.density_model = MaskedSurprisalDensity(
                channels=config.channels,
                hidden_dim=config.density_hidden_dim,
                partition_stride=config.partition_stride,
            )
        elif config.density_mode == "uniform":
            self.density_model = UniformInformationDensity(
                partition_stride=config.partition_stride,
            )
        else:
            raise ValueError(f"unknown density_mode={config.density_mode}")
        if config.partition_backend == "quantile":
            self.partitioner = QuantileGridPartitioner(
                density_floor_ratio=config.partition_density_floor_ratio,
                backend=config.kernel_backend,
            )
        elif config.partition_backend == "bsp":
            self.partitioner = BalancedBSPPartitioner(
                min_size=config.bsp_min_size,
                max_tokens=config.max_tokens,
                workers=config.bsp_workers,
            )
        else:
            raise ValueError(f"unknown partition_backend={config.partition_backend}")
        self.region_encoder = RegionEncoder(
            channels=config.channels,
            tile_size=config.tile_size,
            basis_size=config.basis_size,
            latent_dim=config.latent_dim,
            hidden_dim=max(96, config.latent_dim // 2),
            kernel_backend=config.kernel_backend,
            sampling_ratio=config.region_sampling_ratio,
        )
        self.quantizer = None
        if config.use_vector_quantization:
            self.quantizer = ResidualVectorQuantizer(
                codebook_size=config.codebook_size,
                embedding_dim=config.latent_dim,
                num_quantizers=config.num_quantizers,
                commitment_weight=config.commitment_weight,
                ema_decay=config.codebook_ema_decay,
                ema_epsilon=config.codebook_ema_epsilon,
                dead_code_threshold=config.dead_code_threshold,
                dead_code_check_interval=config.dead_code_check_interval,
            )
        self.token_norm = nn.LayerNorm(config.latent_dim)
        self.detokenizer = NeuralDetokenizer(
            latent_dim=config.latent_dim,
            channels=config.channels,
            tile_size=config.tile_size,
            basis_size=config.basis_size,
            feature_dim=config.decoder_feature_dim,
            hidden_dim=config.decoder_hidden_dim,
            residual_blocks=config.decoder_residual_blocks,
            sigma_scale=config.gaussian_sigma_scale,
            token_chunk_size=config.gaussian_chunk_size,
            kernel_backend=config.kernel_backend,
        )

    def partition(
        self,
        density: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> PartitionBatch:
        target = self.config.target_tokens if target_tokens is None else int(target_tokens)
        partition_density = density.detach() if self.config.detach_density_for_partition else density
        if self.config.partition_backend == "quantile":
            return self.partitioner(partition_density, target_tokens=target, token_grid=self.config.token_grid)
        if mass_per_token is not None:
            return self.partitioner(
                partition_density,
                target_tokens=None,
                mass_per_token=float(mass_per_token),
            )
        if target_tokens is not None:
            return self.partitioner(partition_density, target_tokens=target, mass_per_token=None)
        if self.config.mass_per_token is not None:
            return self.partitioner(
                partition_density,
                target_tokens=None,
                mass_per_token=float(self.config.mass_per_token),
            )
        return self.partitioner(partition_density, target_tokens=target, mass_per_token=None)

    def _combine_content_geometry(
        self,
        content_latents: torch.Tensor,
        partition: PartitionBatch,
    ) -> torch.Tensor:
        geometry_latents = self.region_encoder.geometry_encoder(
            partition.boxes,
            partition.depth,
            partition.mask,
        )
        return self.token_norm(content_latents + geometry_latents) * partition.mask.unsqueeze(-1)

    def encode(
        self,
        image: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> tuple[FractalTokens, torch.Tensor, QuantizerOutput]:
        density = self.density_model(image)
        partition = self.partition(density, target_tokens=target_tokens, mass_per_token=mass_per_token)
        content_latents, geometry_latents, coefficients, _ = self.region_encoder(
            image,
            boxes=partition.boxes,
            depth=partition.depth,
            mask=partition.mask,
            compact_valid_regions=self.config.partition_backend == "bsp",
        )
        if self.quantizer is None:
            zero = content_latents.sum() * 0.0
            quantizer_output = QuantizerOutput(
                quantized=content_latents,
                indices=torch.zeros(*partition.mask.shape, 1, device=image.device, dtype=torch.long),
                loss=zero,
                perplexity=torch.ones((), device=image.device, dtype=image.dtype),
            )
            code_indices = None
        else:
            quantizer_output = self.quantizer(content_latents, partition.mask)
            code_indices = quantizer_output.indices
        token_latents = self.token_norm(quantizer_output.quantized + geometry_latents)
        token_latents = token_latents * partition.mask.unsqueeze(-1)
        tokens = FractalTokens(
            latents=token_latents,
            boxes=partition.boxes,
            boxes_grid=partition.boxes_grid,
            mask=partition.mask,
            mass=partition.mass,
            depth=partition.depth,
            assignment=partition.assignment,
            image_shape=tuple(image.shape[-2:]),
            basis_coefficients=coefficients,
            code_indices=code_indices,
            trees=partition.trees,
        )
        return tokens, density, quantizer_output

    def encode_sequence(
        self,
        image: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> FractalSequenceBatch:
        if self.config.partition_backend != "bsp":
            raise RuntimeError("generative tree sequences require partition_backend='bsp'")
        if self.quantizer is None:
            raise RuntimeError("generative tree sequences require use_vector_quantization=True")
        tokens, _, _ = self.encode(
            image,
            target_tokens=target_tokens,
            mass_per_token=mass_per_token,
        )
        return pack_fractal_sequence(tokens)

    def tokens_from_sequence(self, sequence: FractalSequenceBatch) -> FractalTokens:
        if self.quantizer is None:
            raise RuntimeError("sequence decoding requires use_vector_quantization=True")
        device = self.quantizer.quantizers[0].codebook.weight.device
        content_tokens = unpack_fractal_sequence(sequence.to(device), self.quantizer)
        partition = PartitionBatch(
            boxes_grid=content_tokens.boxes_grid,
            boxes=content_tokens.boxes,
            mask=content_tokens.mask,
            mass=content_tokens.mass,
            depth=content_tokens.depth,
            assignment=content_tokens.assignment,
            trees=content_tokens.trees,
        )
        content_tokens.latents = self._combine_content_geometry(content_tokens.latents, partition)
        return content_tokens

    def decode_sequence(
        self,
        sequence: FractalSequenceBatch,
        refine: bool = True,
    ) -> torch.Tensor:
        tokens = self.tokens_from_sequence(sequence)
        reconstruction, _ = self.decode(tokens, refine=refine)
        return reconstruction

    def tokens_from_codes(
        self,
        code_indices: torch.Tensor,
        partition: PartitionBatch,
        image_shape: tuple[int, int],
    ) -> FractalTokens:
        if self.quantizer is None:
            raise RuntimeError("decode from code indices requires use_vector_quantization=True")
        device = self.quantizer.quantizers[0].codebook.weight.device
        dtype = self.quantizer.quantizers[0].codebook.weight.dtype
        code_indices = code_indices.to(device=device)
        partition = partition.to(device=device, dtype=dtype)
        if code_indices.shape[:-1] != partition.mask.shape:
            raise ValueError("code index prefix must match partition mask shape")
        content_latents = self.quantizer.decode_indices(code_indices, partition.mask)
        token_latents = self._combine_content_geometry(content_latents, partition)
        return FractalTokens(
            latents=token_latents,
            boxes=partition.boxes,
            boxes_grid=partition.boxes_grid,
            mask=partition.mask,
            mass=partition.mass,
            depth=partition.depth,
            assignment=partition.assignment,
            image_shape=tuple(image_shape),
            basis_coefficients=None,
            code_indices=code_indices,
            trees=partition.trees,
        )

    def decode_basis(self, tokens: FractalTokens) -> torch.Tensor:
        if tokens.basis_coefficients is None:
            raise ValueError("tokens do not contain deterministic basis coefficients")
        basis_tiles = self.region_encoder.basis.decode(tokens.basis_coefficients)
        return render_token_tiles(
            basis_tiles,
            boxes=tokens.boxes,
            assignment=tokens.assignment,
            image_shape=tokens.image_shape,
            backend=self.config.kernel_backend,
        ).clamp(0.0, 1.0)

    def decode(
        self,
        tokens: FractalTokens,
        refine: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.detokenizer(tokens, refine=refine)

    def decode_codes(
        self,
        code_indices: torch.Tensor,
        partition: PartitionBatch,
        image_shape: tuple[int, int],
        refine: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokens_from_codes(code_indices, partition, image_shape)
        return self.decode(tokens, refine=refine)

    def decode_tree_codes(
        self,
        code_indices: torch.Tensor,
        tree_sequences: list[list[tuple[int, int]]],
        grid_shape: tuple[int, int],
        image_shape: tuple[int, int],
        refine: bool = True,
    ) -> torch.Tensor:
        if self.quantizer is None:
            raise RuntimeError("tree-code decoding requires use_vector_quantization=True")
        trees = [deserialize_tree(sequence, grid_shape) for sequence in tree_sequences]
        device = self.quantizer.quantizers[0].codebook.weight.device
        dtype = self.quantizer.quantizers[0].codebook.weight.dtype
        code_indices = code_indices.to(device=device)
        partition = partition_batch_from_trees(trees, device=device, dtype=dtype)
        maximum = partition.mask.shape[1]
        if code_indices.shape[0] != len(trees) or code_indices.shape[1] < maximum:
            raise ValueError("code_indices do not contain one code sequence per generated tree leaf")
        reconstruction, _ = self.decode_codes(
            code_indices[:, :maximum],
            partition,
            image_shape,
            refine=refine,
        )
        return reconstruction

    def encode_packet(
        self,
        image: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> FractalTokenPacket:
        if self.quantizer is None:
            raise RuntimeError("packet encoding requires use_vector_quantization=True")
        tokens, _, _ = self.encode(
            image,
            target_tokens=target_tokens,
            mass_per_token=mass_per_token,
        )
        return FractalTokenPacket.from_tokens(tokens, codebook_size=self.config.codebook_size)

    def decode_packet(
        self,
        packet: FractalTokenPacket,
        refine: bool = True,
    ) -> torch.Tensor:
        if self.quantizer is None:
            raise RuntimeError("packet decoding requires use_vector_quantization=True")
        if packet.num_quantizers != self.config.num_quantizers:
            raise ValueError("packet and tokenizer use different numbers of quantizers")
        if packet.codebook_size is not None and packet.codebook_size != self.config.codebook_size:
            raise ValueError("packet and tokenizer use different codebook sizes")
        device = self.quantizer.quantizers[0].codebook.weight.device
        dtype = self.quantizer.quantizers[0].codebook.weight.dtype
        partition = packet.partition(device=device, dtype=dtype)
        reconstruction, _ = self.decode_codes(
            packet.codes(device=device),
            partition,
            packet.image_shape,
            refine=refine,
        )
        return reconstruction

    def forward(
        self,
        image: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> FractalOutput:
        tokens, density, quantizer_output = self.encode(
            image,
            target_tokens=target_tokens,
            mass_per_token=mass_per_token,
        )
        reconstruction, predicted_basis = self.decode(tokens)
        if tokens.basis_coefficients is None:
            basis_reconstruction = torch.zeros_like(image)
        else:
            basis_reconstruction = self.decode_basis(tokens)
        return FractalOutput(
            tokens=tokens,
            density=density,
            reconstruction=reconstruction,
            basis_reconstruction=basis_reconstruction,
            quantizer_loss=quantizer_output.loss,
            quantizer_perplexity=quantizer_output.perplexity,
            predicted_basis=predicted_basis,
        )
