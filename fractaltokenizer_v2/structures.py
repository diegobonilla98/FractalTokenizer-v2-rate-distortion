from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BSPNode:
    box: tuple[int, int, int, int]
    depth: int
    mass: float
    axis: int = -1
    split: int = -1
    left: int = -1
    right: int = -1
    leaf_index: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.axis < 0


@dataclass
class PartitionTree:
    grid_shape: tuple[int, int]
    nodes: list[BSPNode] = field(default_factory=list)
    root: int = 0

    def leaf_nodes(self) -> list[BSPNode]:
        return sorted(
            (node for node in self.nodes if node.is_leaf and node.leaf_index >= 0),
            key=lambda node: node.leaf_index,
        )


@dataclass
class PartitionBatch:
    boxes_grid: torch.Tensor
    boxes: torch.Tensor
    mask: torch.Tensor
    mass: torch.Tensor
    depth: torch.Tensor
    assignment: torch.Tensor
    trees: Optional[list[PartitionTree]] = None

    @property
    def token_count(self) -> torch.Tensor:
        return self.mask.sum(dim=1)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> PartitionBatch:
        target_dtype = self.boxes.dtype if dtype is None else dtype
        return PartitionBatch(
            boxes_grid=self.boxes_grid.to(device=device),
            boxes=self.boxes.to(device=device, dtype=target_dtype),
            mask=self.mask.to(device=device),
            mass=self.mass.to(device=device, dtype=target_dtype),
            depth=self.depth.to(device=device),
            assignment=self.assignment.to(device=device),
            trees=self.trees,
        )


@dataclass
class FractalTokens:
    latents: torch.Tensor
    boxes: torch.Tensor
    boxes_grid: torch.Tensor
    mask: torch.Tensor
    mass: torch.Tensor
    depth: torch.Tensor
    assignment: torch.Tensor
    image_shape: tuple[int, int]
    basis_coefficients: Optional[torch.Tensor] = None
    code_indices: Optional[torch.Tensor] = None
    trees: Optional[list[PartitionTree]] = None

    @property
    def token_count(self) -> torch.Tensor:
        return self.mask.sum(dim=1)

    def to(self, device: torch.device | str) -> FractalTokens:
        return FractalTokens(
            latents=self.latents.to(device),
            boxes=self.boxes.to(device),
            boxes_grid=self.boxes_grid.to(device),
            mask=self.mask.to(device),
            mass=self.mass.to(device),
            depth=self.depth.to(device),
            assignment=self.assignment.to(device),
            image_shape=self.image_shape,
            basis_coefficients=None if self.basis_coefficients is None else self.basis_coefficients.to(device),
            code_indices=None if self.code_indices is None else self.code_indices.to(device),
            trees=self.trees,
        )


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor


@dataclass
class FractalOutput:
    tokens: FractalTokens
    density: torch.Tensor
    reconstruction: torch.Tensor
    basis_reconstruction: torch.Tensor
    quantizer_loss: torch.Tensor
    quantizer_perplexity: torch.Tensor
    predicted_basis: torch.Tensor
