from __future__ import annotations

import torch
import torch.nn as nn

from .model import FractalTokenizerV2, FractalTokenizerV2Config


class FractalTokenizerLayerV2(nn.Module):
    def __init__(self, config: FractalTokenizerV2Config | None = None):
        super().__init__()
        self.tokenizer = FractalTokenizerV2(config)

    def forward(
        self,
        image: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        tokens, density, quantizer_output = self.tokenizer.encode(
            image,
            target_tokens=target_tokens,
            mass_per_token=mass_per_token,
        )
        metadata: dict[str, object] = {
            "boxes": tokens.boxes,
            "boxes_grid": tokens.boxes_grid,
            "mass": tokens.mass,
            "depth": tokens.depth,
            "assignment": tokens.assignment,
            "density": density,
            "code_indices": tokens.code_indices,
            "trees": tokens.trees,
            "quantizer_loss": quantizer_output.loss,
            "quantizer_perplexity": quantizer_output.perplexity,
        }
        return tokens.latents, tokens.mask, metadata

    def decode_codes(
        self,
        code_indices: torch.Tensor,
        partition,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokenizer.decode_codes(code_indices, partition, image_shape)
