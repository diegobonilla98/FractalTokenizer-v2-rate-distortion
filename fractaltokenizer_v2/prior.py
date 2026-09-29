from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generative import (
    LEAF_ACTION,
    PAD_ACTION,
    SPLIT_X_ACTION,
    SPLIT_Y_ACTION,
    FractalSequenceBatch,
    collate_fractal_sequences,
)


@dataclass
class FractalPriorConfig:
    codebook_size: int = 8192
    num_quantizers: int = 1
    max_grid_size: int = 64
    model_dim: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_sequence_length: int = 511


class FractalAutoregressivePrior(nn.Module):
    def __init__(self, config: FractalPriorConfig | None = None):
        super().__init__()
        self.config = FractalPriorConfig() if config is None else config
        config = self.config
        positive = {
            "codebook_size": config.codebook_size,
            "num_quantizers": config.num_quantizers,
            "max_grid_size": config.max_grid_size,
            "model_dim": config.model_dim,
            "depth": config.depth,
            "heads": config.heads,
            "max_sequence_length": config.max_sequence_length,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"prior configuration values must be positive: {', '.join(invalid)}")
        if config.codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if config.model_dim % config.heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        if config.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.action_embedding = nn.Embedding(4, config.model_dim)
        self.action_condition = nn.Embedding(4, config.model_dim)
        self.split_embedding = nn.Embedding(config.max_grid_size + 1, config.model_dim)
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
        self.action_head = nn.Linear(config.model_dim, 4)
        self.split_head = nn.Linear(config.model_dim, config.max_grid_size + 1)
        self.code_heads = nn.ModuleList(
            [nn.Linear(config.model_dim, config.codebook_size) for _ in range(config.num_quantizers)]
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.bos, std=0.02)

    def event_code_indices(self, sequence: FractalSequenceBatch) -> torch.Tensor:
        leaf_events = sequence.action_mask & sequence.actions.eq(LEAF_ACTION)
        leaf_rank = leaf_events.long().cumsum(dim=1) - 1
        gather_index = leaf_rank.clamp_min(0).unsqueeze(-1).expand(
            -1,
            -1,
            self.config.num_quantizers,
        )
        event_codes = torch.gather(sequence.code_indices, 1, gather_index)
        return event_codes * leaf_events.unsqueeze(-1)

    def event_embeddings(
        self,
        actions: torch.Tensor,
        splits: torch.Tensor,
        event_codes: torch.Tensor,
    ) -> torch.Tensor:
        split_events = actions.eq(SPLIT_X_ACTION) | actions.eq(SPLIT_Y_ACTION)
        leaf_events = actions.eq(LEAF_ACTION)
        embedding = self.action_embedding(actions)
        split_ids = splits.clamp(0, self.config.max_grid_size)
        embedding = embedding + self.split_embedding(split_ids) * split_events.unsqueeze(-1)
        for level, code_embedding in enumerate(self.code_embeddings):
            embedding = embedding + code_embedding(event_codes[..., level]) * leaf_events.unsqueeze(-1)
        return embedding

    def hidden_states(self, sequence: FractalSequenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        length = sequence.actions.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        event_codes = self.event_code_indices(sequence)
        events = self.event_embeddings(sequence.actions, sequence.splits, event_codes)
        inputs = torch.empty_like(events)
        inputs[:, :1] = self.bos
        if length > 1:
            inputs[:, 1:] = events[:, :-1]
        inputs = inputs + self.position_embedding[:, :length]
        causal_mask = torch.triu(
            torch.ones(length, length, device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(
            inputs,
            mask=causal_mask,
            src_key_padding_mask=~sequence.action_mask,
        )
        return hidden, event_codes

    def conditional_logits(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        event_codes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        context = hidden + self.action_condition(actions)
        split_logits = self.split_head(context)
        code_logits = []
        for level, head in enumerate(self.code_heads):
            code_logits.append(head(context))
            if event_codes is not None and level + 1 < self.config.num_quantizers:
                context = context + self.code_embeddings[level](event_codes[..., level])
        return split_logits, code_logits

    def forward(self, sequence: FractalSequenceBatch) -> dict[str, object]:
        hidden, event_codes = self.hidden_states(sequence)
        action_logits = self.action_head(hidden)
        split_logits, code_logits = self.conditional_logits(hidden, sequence.actions, event_codes)
        return {
            "action_logits": action_logits,
            "split_logits": split_logits,
            "code_logits": code_logits,
            "event_codes": event_codes,
        }

    def negative_log_likelihood(self, sequence: FractalSequenceBatch) -> dict[str, torch.Tensor]:
        output = self(sequence)
        valid = sequence.action_mask
        split_mask = valid & (
            sequence.actions.eq(SPLIT_X_ACTION) | sequence.actions.eq(SPLIT_Y_ACTION)
        )
        leaf_mask = valid & sequence.actions.eq(LEAF_ACTION)
        action_error = F.cross_entropy(
            output["action_logits"].flatten(0, 1),
            sequence.actions.flatten(),
            reduction="none",
        ).reshape_as(sequence.actions)
        split_target = sequence.splits.clamp(0, self.config.max_grid_size)
        split_error = F.cross_entropy(
            output["split_logits"].flatten(0, 1),
            split_target.flatten(),
            reduction="none",
        ).reshape_as(sequence.splits)
        event_codes = output["event_codes"]
        content_error = torch.zeros_like(action_error)
        for level, logits in enumerate(output["code_logits"]):
            code_error = F.cross_entropy(
                logits.flatten(0, 1),
                event_codes[..., level].flatten(),
                reduction="none",
            ).reshape_as(sequence.actions)
            content_error = content_error + code_error
        action_nats = (action_error * valid).sum(dim=1)
        split_nats = (split_error * split_mask).sum(dim=1)
        content_nats = (content_error * leaf_mask).sum(dim=1)
        total_nats = action_nats + split_nats + content_nats
        return {
            "total_nats": total_nats,
            "action_nats": action_nats,
            "split_nats": split_nats,
            "content_nats": content_nats,
        }

    def estimated_bits(self, sequence: FractalSequenceBatch) -> dict[str, torch.Tensor]:
        nll = self.negative_log_likelihood(sequence)
        inverse_log_two = 1.0 / math.log(2.0)
        pixels = float(sequence.image_shape[0] * sequence.image_shape[1])
        result = {name.replace("_nats", "_bits"): value * inverse_log_two for name, value in nll.items()}
        result["bits_per_pixel"] = result["total_bits"] / pixels
        return result

    def loss(self, sequence: FractalSequenceBatch) -> dict[str, torch.Tensor]:
        output = self(sequence)
        valid = sequence.action_mask
        split_mask = valid & (
            sequence.actions.eq(SPLIT_X_ACTION) | sequence.actions.eq(SPLIT_Y_ACTION)
        )
        leaf_mask = valid & sequence.actions.eq(LEAF_ACTION)
        action_error = F.cross_entropy(
            output["action_logits"].flatten(0, 1),
            sequence.actions.flatten(),
            reduction="none",
        ).reshape_as(sequence.actions)
        action_loss = (action_error * valid).sum() / valid.sum().clamp_min(1)
        split_target = sequence.splits.clamp(0, self.config.max_grid_size)
        split_error = F.cross_entropy(
            output["split_logits"].flatten(0, 1),
            split_target.flatten(),
            reduction="none",
        ).reshape_as(sequence.splits)
        split_loss = (split_error * split_mask).sum() / split_mask.sum().clamp_min(1)
        code_losses = []
        event_codes = output["event_codes"]
        for level, logits in enumerate(output["code_logits"]):
            code_error = F.cross_entropy(
                logits.flatten(0, 1),
                event_codes[..., level].flatten(),
                reduction="none",
            ).reshape_as(sequence.actions)
            code_losses.append((code_error * leaf_mask).sum() / leaf_mask.sum().clamp_min(1))
        code_loss = torch.stack(code_losses).sum()
        total = action_loss + split_loss + code_loss
        return {
            "loss": total,
            "action": action_loss,
            "split": split_loss,
            "content": code_loss,
        }

    def _history_hidden(
        self,
        actions: list[int],
        splits: list[int],
        event_codes: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        history_length = len(actions)
        input_length = history_length + 1
        if input_length > self.config.max_sequence_length:
            raise RuntimeError("generation exceeded max_sequence_length")
        if history_length == 0:
            inputs = self.bos.expand(1, 1, -1)
        else:
            action_tensor = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(0)
            split_tensor = torch.tensor(splits, device=device, dtype=torch.long).unsqueeze(0)
            code_tensor = torch.tensor(event_codes, device=device, dtype=torch.long).unsqueeze(0)
            events = self.event_embeddings(action_tensor, split_tensor, code_tensor)
            inputs = torch.cat([self.bos, events], dim=1)
        inputs = inputs + self.position_embedding[:, :input_length]
        causal_mask = torch.triu(
            torch.ones(input_length, input_length, device=device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(inputs, mask=causal_mask)
        return hidden[:, -1]

    @staticmethod
    def _choose(
        logits: torch.Tensor,
        sample: bool,
        temperature: float,
        top_k: int | None,
    ) -> int:
        if not sample:
            return int(logits.argmax(dim=-1).item())
        logits = logits / max(float(temperature), 1e-5)
        if top_k is not None and 0 < top_k < logits.shape[-1]:
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        probability = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probability, 1).item())

    @torch.no_grad()
    def generate_one(
        self,
        grid_shape: tuple[int, int],
        image_shape: tuple[int, int],
        min_tokens: int = 1,
        max_tokens: int = 128,
        sample: bool = True,
        temperature: float = 1.0,
        top_k: int | None = 64,
    ) -> FractalSequenceBatch:
        grid_height, grid_width = grid_shape
        if min(grid_shape) < 1 or max(grid_shape) > self.config.max_grid_size:
            raise ValueError("grid shape is invalid or exceeds max_grid_size")
        capacity = grid_height * grid_width
        if min_tokens < 1 or max_tokens < min_tokens or max_tokens > capacity:
            raise ValueError("token limits are invalid for this grid")
        if 2 * min_tokens - 1 > self.config.max_sequence_length:
            raise ValueError("max_sequence_length cannot represent min_tokens")
        if min_tokens > grid_height * grid_width:
            raise ValueError("min_tokens exceeds the number of partition-grid cells")
        max_tokens = min(max_tokens, grid_height * grid_width)
        device = self.bos.device
        actions: list[int] = []
        splits: list[int] = []
        event_codes: list[list[int]] = []
        leaf_codes: list[list[int]] = []
        stack = [(0, 0, grid_width, grid_height)]
        leaves = 0
        while stack:
            current = stack.pop()
            x0, y0, x1, y1 = current
            hidden = self._history_hidden(actions, splits, event_codes, device)
            action_logits = self.action_head(hidden)[0]
            feasible = torch.zeros(4, device=device, dtype=torch.bool)
            feasible[LEAF_ACTION] = True
            minimum_after_split = leaves + len(stack) + 2
            sequence_after_split = len(actions) + 1 + len(stack) + 2
            can_add_leaf = minimum_after_split <= max_tokens
            can_add_action = sequence_after_split <= self.config.max_sequence_length
            if x1 - x0 > 1 and can_add_leaf and can_add_action:
                feasible[SPLIT_X_ACTION] = True
            if y1 - y0 > 1 and can_add_leaf and can_add_action:
                feasible[SPLIT_Y_ACTION] = True
            minimum_after_leaf = leaves + len(stack) + 1
            if minimum_after_leaf < min_tokens and feasible[2:].any():
                feasible[LEAF_ACTION] = False
            action_logits = action_logits.masked_fill(~feasible, -torch.inf)
            action = self._choose(action_logits, sample, temperature, top_k=None)
            action_tensor = torch.tensor([action], device=device, dtype=torch.long)
            context = hidden + self.action_condition(action_tensor)
            split_logits = self.split_head(context)[0]
            chosen_split = -1
            codes = [0] * self.config.num_quantizers
            if action == SPLIT_X_ACTION:
                valid = torch.zeros_like(split_logits, dtype=torch.bool)
                valid[x0 + 1 : x1] = True
                chosen_split = self._choose(
                    split_logits.masked_fill(~valid, -torch.inf),
                    sample,
                    temperature,
                    top_k,
                )
                stack.append((chosen_split, y0, x1, y1))
                stack.append((x0, y0, chosen_split, y1))
            elif action == SPLIT_Y_ACTION:
                valid = torch.zeros_like(split_logits, dtype=torch.bool)
                valid[y0 + 1 : y1] = True
                chosen_split = self._choose(
                    split_logits.masked_fill(~valid, -torch.inf),
                    sample,
                    temperature,
                    top_k,
                )
                stack.append((x0, chosen_split, x1, y1))
                stack.append((x0, y0, x1, chosen_split))
            else:
                code_context = context
                codes = []
                for level, head in enumerate(self.code_heads):
                    code_logits = head(code_context)[0]
                    code = self._choose(code_logits, sample, temperature, top_k)
                    codes.append(code)
                    if level + 1 < self.config.num_quantizers:
                        code_context = code_context + self.code_embeddings[level](
                            torch.tensor([code], device=device)
                        )
                leaf_codes.append(codes)
                leaves += 1
            actions.append(action)
            splits.append(chosen_split)
            event_codes.append(codes)
        action_tensor = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(0)
        split_tensor = torch.tensor(splits, device=device, dtype=torch.long).unsqueeze(0)
        code_tensor = torch.tensor(leaf_codes, device=device, dtype=torch.long).unsqueeze(0)
        return FractalSequenceBatch(
            actions=action_tensor,
            splits=split_tensor,
            action_mask=torch.ones_like(action_tensor, dtype=torch.bool),
            code_indices=code_tensor,
            leaf_mask=torch.ones(code_tensor.shape[:2], device=device, dtype=torch.bool),
            grid_shape=grid_shape,
            image_shape=image_shape,
        )

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        grid_shape: tuple[int, int],
        image_shape: tuple[int, int],
        min_tokens: int = 1,
        max_tokens: int = 128,
        sample: bool = True,
        temperature: float = 1.0,
        top_k: int | None = 64,
    ) -> FractalSequenceBatch:
        sequences = [
            self.generate_one(
                grid_shape=grid_shape,
                image_shape=image_shape,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
            )
            for _ in range(batch_size)
        ]
        return collate_fractal_sequences(sequences)
