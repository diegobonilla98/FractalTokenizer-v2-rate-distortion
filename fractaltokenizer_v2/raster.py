from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RasterSequenceBatch:
    code_indices: torch.Tensor
    token_mask: torch.Tensor
    token_grid: tuple[int, int]
    image_shape: tuple[int, int]

    @property
    def sequence_length(self) -> torch.Tensor:
        return self.token_mask.sum(dim=1)

    def to(self, device: torch.device | str) -> RasterSequenceBatch:
        return RasterSequenceBatch(
            code_indices=self.code_indices.to(device),
            token_mask=self.token_mask.to(device),
            token_grid=self.token_grid,
            image_shape=self.image_shape,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "code_indices": self.code_indices.detach().cpu(),
            "token_mask": self.token_mask.detach().cpu(),
            "token_grid": self.token_grid,
            "image_shape": self.image_shape,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> RasterSequenceBatch:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported raster sequence version")
        return cls(
            code_indices=state["code_indices"],
            token_mask=state["token_mask"],
            token_grid=tuple(state["token_grid"]),
            image_shape=tuple(state["image_shape"]),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RasterSequenceBatch:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls.from_state_dict(state)


def collate_raster_sequences(sequences: list[RasterSequenceBatch]) -> RasterSequenceBatch:
    if not sequences:
        raise ValueError("cannot collate an empty raster sequence list")
    token_grid = sequences[0].token_grid
    image_shape = sequences[0].image_shape
    num_quantizers = sequences[0].code_indices.shape[-1]
    for sequence in sequences:
        if sequence.token_grid != token_grid or sequence.image_shape != image_shape:
            raise ValueError("all raster sequences must share token and image shapes")
        if sequence.code_indices.shape[-1] != num_quantizers:
            raise ValueError("all raster sequences must share the quantizer count")
    return RasterSequenceBatch(
        code_indices=torch.cat([sequence.code_indices for sequence in sequences], dim=0),
        token_mask=torch.cat([sequence.token_mask for sequence in sequences], dim=0),
        token_grid=token_grid,
        image_shape=image_shape,
    )


@dataclass
class RasterPriorConfig:
    codebook_size: int = 8192
    num_quantizers: int = 1
    model_dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_sequence_length: int = 256


class RasterAutoregressivePrior(nn.Module):
    def __init__(self, config: RasterPriorConfig | None = None):
        super().__init__()
        self.config = RasterPriorConfig() if config is None else config
        config = self.config
        values = (
            config.codebook_size,
            config.num_quantizers,
            config.model_dim,
            config.depth,
            config.heads,
            config.max_sequence_length,
        )
        if min(values) < 1:
            raise ValueError("raster prior configuration values must be positive")
        if config.codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if config.model_dim % config.heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        if config.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(config.codebook_size, config.model_dim) for _ in range(config.num_quantizers)]
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.model_dim)
        )
        self.bos = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=int(config.model_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.depth,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        self.code_heads = nn.ModuleList(
            [nn.Linear(config.model_dim, config.codebook_size) for _ in range(config.num_quantizers)]
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.bos, std=0.02)

    def event_embeddings(self, code_indices: torch.Tensor) -> torch.Tensor:
        embedding = torch.zeros(
            *code_indices.shape[:-1],
            self.config.model_dim,
            device=code_indices.device,
            dtype=self.position_embedding.dtype,
        )
        for level, code_embedding in enumerate(self.code_embeddings):
            embedding = embedding + code_embedding(code_indices[..., level])
        return embedding

    def hidden_states(self, sequence: RasterSequenceBatch) -> torch.Tensor:
        length = sequence.code_indices.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError("raster sequence exceeds max_sequence_length")
        events = self.event_embeddings(sequence.code_indices)
        inputs = torch.empty_like(events)
        inputs[:, :1] = self.bos
        if length > 1:
            inputs[:, 1:] = events[:, :-1]
        inputs = inputs + self.position_embedding[:, :length]
        causal_mask = torch.triu(
            torch.ones(length, length, device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.transformer(
            inputs,
            mask=causal_mask,
            src_key_padding_mask=~sequence.token_mask,
        )

    def logits(self, hidden: torch.Tensor, code_indices: torch.Tensor | None = None) -> list[torch.Tensor]:
        context = hidden
        output = []
        for level, head in enumerate(self.code_heads):
            output.append(head(context))
            if code_indices is not None and level + 1 < self.config.num_quantizers:
                context = context + self.code_embeddings[level](code_indices[..., level])
        return output

    def loss(self, sequence: RasterSequenceBatch) -> dict[str, torch.Tensor]:
        hidden = self.hidden_states(sequence)
        logits = self.logits(hidden, sequence.code_indices)
        losses = []
        for level, level_logits in enumerate(logits):
            error = F.cross_entropy(
                level_logits.flatten(0, 1),
                sequence.code_indices[..., level].flatten(),
                reduction="none",
            ).reshape_as(sequence.token_mask)
            losses.append((error * sequence.token_mask).sum() / sequence.token_mask.sum().clamp_min(1))
        content = torch.stack(losses).sum()
        return {"loss": content, "content": content}

    def estimated_bits(self, sequence: RasterSequenceBatch) -> dict[str, torch.Tensor]:
        hidden = self.hidden_states(sequence)
        logits = self.logits(hidden, sequence.code_indices)
        nats = torch.zeros(sequence.code_indices.shape[0], device=hidden.device)
        for level, level_logits in enumerate(logits):
            error = F.cross_entropy(
                level_logits.flatten(0, 1),
                sequence.code_indices[..., level].flatten(),
                reduction="none",
            ).reshape_as(sequence.token_mask)
            nats = nats + (error * sequence.token_mask).sum(dim=1)
        bits = nats / math.log(2.0)
        pixels = float(sequence.image_shape[0] * sequence.image_shape[1])
        return {"total_bits": bits, "content_bits": bits, "bits_per_pixel": bits / pixels}

    @staticmethod
    def _choose(logits: torch.Tensor, sample: bool, temperature: float, top_k: int | None) -> torch.Tensor:
        if not sample:
            return logits.argmax(dim=-1)
        logits = logits / max(float(temperature), 1e-5)
        if top_k is not None and 0 < top_k < logits.shape[-1]:
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        probability = torch.softmax(logits, dim=-1)
        return torch.multinomial(probability, 1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        token_grid: tuple[int, int],
        image_shape: tuple[int, int],
        sample: bool = True,
        temperature: float = 1.0,
        top_k: int | None = 64,
    ) -> RasterSequenceBatch:
        token_count = int(token_grid[0] * token_grid[1])
        if token_count > self.config.max_sequence_length:
            raise ValueError("token grid exceeds max_sequence_length")
        device = self.bos.device
        codes = torch.empty(
            batch_size,
            0,
            self.config.num_quantizers,
            device=device,
            dtype=torch.long,
        )
        for index in range(token_count):
            if index == 0:
                inputs = self.bos.expand(batch_size, 1, -1)
            else:
                events = self.event_embeddings(codes)
                inputs = torch.cat([self.bos.expand(batch_size, 1, -1), events], dim=1)
            inputs = inputs + self.position_embedding[:, : index + 1]
            causal_mask = torch.triu(
                torch.ones(index + 1, index + 1, device=device, dtype=torch.bool),
                diagonal=1,
            )
            hidden = self.transformer(inputs, mask=causal_mask)[:, -1]
            context = hidden
            current = []
            for level, head in enumerate(self.code_heads):
                code = self._choose(head(context), sample, temperature, top_k)
                current.append(code)
                if level + 1 < self.config.num_quantizers:
                    context = context + self.code_embeddings[level](code)
            current_codes = torch.stack(current, dim=-1).unsqueeze(1)
            codes = torch.cat([codes, current_codes], dim=1)
        return RasterSequenceBatch(
            code_indices=codes,
            token_mask=torch.ones(batch_size, token_count, device=device, dtype=torch.bool),
            token_grid=token_grid,
            image_shape=image_shape,
        )
