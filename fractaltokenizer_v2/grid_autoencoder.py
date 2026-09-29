from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .continuous_autoencoder import ContinuousAutoencoderOutput, ContinuousAutoencoderConfig, SpatialRefiner
from .structures import FractalTokens


@dataclass(frozen=True)
class GridAutoencoderConfig:
    image_height: int = 256
    image_width: int = 256
    channels: int = 3
    token_count: int = 128
    latent_dim: int = 256
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    attention_dropout: float = 0.0
    dropout: float = 0.0
    refiner_hidden_dim: int = 128
    refiner_blocks: int = 4
    compile_refiner: bool = True


def grid_shape_for_tokens(image_height: int, image_width: int, token_count: int) -> tuple[int, int]:
    candidates = []
    target_ratio = image_height / image_width
    for rows in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % rows:
            continue
        columns = token_count // rows
        for candidate_rows, candidate_columns in ((rows, columns), (columns, rows)):
            if image_height % candidate_rows or image_width % candidate_columns:
                continue
            ratio_error = abs(math.log((candidate_rows / candidate_columns) / target_ratio))
            tie_break = 0 if candidate_rows >= candidate_columns else 1
            candidates.append((ratio_error, tie_break, candidate_rows, candidate_columns))
    if not candidates:
        raise ValueError("token_count must factor into a grid that divides the image shape")
    _, _, rows, columns = min(candidates)
    return rows, columns


class GridViTAutoencoder(nn.Module):
    def __init__(self, config: GridAutoencoderConfig | None = None):
        super().__init__()
        self.config = GridAutoencoderConfig() if config is None else config
        self._validate_config(self.config)
        self.grid_rows, self.grid_columns = grid_shape_for_tokens(
            self.config.image_height,
            self.config.image_width,
            self.config.token_count,
        )
        self.patch_height = self.config.image_height // self.grid_rows
        self.patch_width = self.config.image_width // self.grid_columns
        self.patch_projection = nn.Conv2d(
            self.config.channels,
            self.config.latent_dim,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.config.token_count, self.config.latent_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.latent_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=round(self.config.latent_dim * self.config.transformer_mlp_ratio),
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_depth,
            norm=nn.LayerNorm(self.config.latent_dim),
        )
        self.patch_decoder = nn.Linear(
            self.config.latent_dim,
            self.config.channels * self.patch_height * self.patch_width,
        )
        refiner_config = ContinuousAutoencoderConfig(
            channels=self.config.channels,
            latent_dim=self.config.latent_dim,
            transformer_depth=self.config.transformer_depth,
            transformer_heads=self.config.transformer_heads,
            transformer_mlp_ratio=self.config.transformer_mlp_ratio,
            attention_dropout=self.config.attention_dropout,
            dropout=self.config.dropout,
            refiner_hidden_dim=self.config.refiner_hidden_dim,
            refiner_blocks=self.config.refiner_blocks,
        )
        self.refiner = SpatialRefiner(refiner_config)
        object.__setattr__(self, "_compiled_refiner_forward", None)
        boxes_grid, boxes, assignment = self._geometry()
        self.register_buffer("boxes_grid", boxes_grid, persistent=False)
        self.register_buffer("boxes", boxes, persistent=False)
        self.register_buffer("assignment", assignment, persistent=False)

    @staticmethod
    def _validate_config(config: GridAutoencoderConfig) -> None:
        values = {
            "image_height": config.image_height,
            "image_width": config.image_width,
            "channels": config.channels,
            "token_count": config.token_count,
            "latent_dim": config.latent_dim,
            "transformer_depth": config.transformer_depth,
            "transformer_heads": config.transformer_heads,
            "refiner_hidden_dim": config.refiner_hidden_dim,
            "refiner_blocks": config.refiner_blocks,
        }
        invalid = [name for name, value in values.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if config.latent_dim % config.transformer_heads:
            raise ValueError("latent_dim must be divisible by transformer_heads")
        if config.attention_dropout < 0.0 or config.dropout < 0.0:
            raise ValueError("dropout values cannot be negative")

    def _geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = []
        assignment = torch.empty(self.grid_rows, self.grid_columns, dtype=torch.long)
        for row in range(self.grid_rows):
            for column in range(self.grid_columns):
                token = row * self.grid_columns + column
                boxes.append((column, row, column + 1, row + 1))
                assignment[row, column] = token
        boxes_grid = torch.tensor(boxes, dtype=torch.long)
        scale = torch.tensor(
            [self.grid_columns, self.grid_rows, self.grid_columns, self.grid_rows],
            dtype=torch.float32,
        )
        return boxes_grid, boxes_grid.float() / scale, assignment

    def _tokens(self, latents: torch.Tensor) -> FractalTokens:
        batch_size = latents.shape[0]
        boxes_grid = self.boxes_grid.unsqueeze(0).expand(batch_size, -1, -1)
        boxes = self.boxes.unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.ones(batch_size, self.config.token_count, device=latents.device, dtype=torch.bool)
        mass = torch.ones(batch_size, self.config.token_count, device=latents.device, dtype=torch.float32)
        depth = torch.zeros(batch_size, self.config.token_count, device=latents.device, dtype=torch.long)
        assignment = self.assignment.unsqueeze(0).expand(batch_size, -1, -1)
        return FractalTokens(
            latents=latents,
            boxes=boxes,
            boxes_grid=boxes_grid,
            mask=mask,
            mass=mass,
            depth=depth,
            assignment=assignment,
            image_shape=(self.config.image_height, self.config.image_width),
        )

    def encode(self, image: torch.Tensor) -> FractalTokens:
        self._validate_image(image)
        latents = self.patch_projection(image).flatten(2).transpose(1, 2)
        latents = self.transformer(latents + self.position_embedding.to(latents.dtype))
        return self._tokens(latents)

    def _base_reconstruction(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latents.shape[0]
        patches = torch.sigmoid(self.patch_decoder(latents))
        patches = patches.reshape(
            batch_size,
            self.grid_rows,
            self.grid_columns,
            self.config.channels,
            self.patch_height,
            self.patch_width,
        )
        base = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            self.config.channels,
            self.config.image_height,
            self.config.image_width,
        )
        local_tiles = patches.permute(0, 1, 2, 3, 4, 5).reshape(
            batch_size,
            self.config.token_count,
            self.config.channels,
            self.patch_height,
            self.patch_width,
        )
        return base, local_tiles

    def _refiner_forward(self, base: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.refiner(base, self.assignment.unsqueeze(0).expand(base.shape[0], -1, -1))
        if not self.config.compile_refiner or not base.is_cuda:
            return self.refiner(base, self.assignment.unsqueeze(0).expand(base.shape[0], -1, -1))
        compiled = object.__getattribute__(self, "_compiled_refiner_forward")
        if compiled is None:
            compiled = torch.compile(
                self.refiner.forward,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
            object.__setattr__(self, "_compiled_refiner_forward", compiled)
        return compiled(base, self.assignment.unsqueeze(0).expand(base.shape[0], -1, -1)).clone()

    def decode(self, tokens: FractalTokens, refine: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        base, local_tiles = self._base_reconstruction(tokens.latents)
        reconstruction = self._refiner_forward(base) if refine else base
        return reconstruction, local_tiles

    def _validate_image(self, image: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != self.config.channels:
            raise ValueError("image must be [B,C,H,W] with the configured channel count")
        if tuple(image.shape[-2:]) != (self.config.image_height, self.config.image_width):
            raise ValueError("image shape does not match GridAutoencoderConfig")

    def forward(self, image: torch.Tensor, target_tokens: int | None = None) -> ContinuousAutoencoderOutput:
        if target_tokens is not None and int(target_tokens) != self.config.token_count:
            raise ValueError("fixed-grid token count cannot change at runtime")
        if image.is_cuda and not torch.compiler.is_compiling() and self.config.compile_refiner:
            torch.compiler.cudagraph_mark_step_begin()
        tokens = self.encode(image)
        base, local_tiles = self._base_reconstruction(tokens.latents)
        reconstruction = self._refiner_forward(base)
        density = torch.ones(
            image.shape[0],
            1,
            self.grid_rows,
            self.grid_columns,
            device=image.device,
            dtype=image.dtype,
        )
        return ContinuousAutoencoderOutput(
            reconstruction=reconstruction,
            base_reconstruction=base,
            tokens=tokens,
            density=density,
            local_tiles=local_tiles,
        )
