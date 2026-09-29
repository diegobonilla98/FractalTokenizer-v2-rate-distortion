from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .partition import (
    deserialize_tree,
    estimate_tree_bits,
    partition_batch_from_boxes,
    partition_batch_from_trees,
    serialize_tree,
)
from .structures import FractalTokens, PartitionBatch


@dataclass
class FractalTokenPacket:
    image_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    code_indices: torch.Tensor
    mask: torch.Tensor
    boxes_grid: torch.Tensor | None = None
    mass: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    tree_sequences: list[list[tuple[int, int]]] | None = None
    codebook_size: int | None = None

    @property
    def num_quantizers(self) -> int:
        return int(self.code_indices.shape[-1])

    @classmethod
    def from_tokens(
        cls,
        tokens: FractalTokens,
        codebook_size: int | None = None,
    ) -> FractalTokenPacket:
        if tokens.code_indices is None:
            raise ValueError("tokens do not contain discrete code indices")
        tree_sequences = None
        boxes_grid = tokens.boxes_grid.detach().cpu()
        if tokens.trees is not None:
            tree_sequences = [serialize_tree(tree) for tree in tokens.trees]
            boxes_grid = None
        return cls(
            image_shape=tuple(tokens.image_shape),
            grid_shape=tuple(tokens.assignment.shape[-2:]),
            code_indices=tokens.code_indices.detach().cpu(),
            mask=tokens.mask.detach().cpu(),
            boxes_grid=boxes_grid,
            mass=tokens.mass.detach().cpu(),
            depth=tokens.depth.detach().cpu(),
            tree_sequences=tree_sequences,
            codebook_size=None if codebook_size is None else int(codebook_size),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "image_shape": self.image_shape,
            "grid_shape": self.grid_shape,
            "code_indices": self.code_indices,
            "mask": self.mask,
            "boxes_grid": self.boxes_grid,
            "mass": self.mass,
            "depth": self.depth,
            "tree_sequences": self.tree_sequences,
            "codebook_size": self.codebook_size,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> FractalTokenPacket:
        version = int(state.get("version", 0))
        if version not in {1, 2}:
            raise ValueError("unsupported token packet version")
        return cls(
            image_shape=tuple(state["image_shape"]),
            grid_shape=tuple(state["grid_shape"]),
            code_indices=state["code_indices"],
            mask=state["mask"],
            boxes_grid=state.get("boxes_grid"),
            mass=state.get("mass"),
            depth=state.get("depth"),
            tree_sequences=state.get("tree_sequences"),
            codebook_size=state.get("codebook_size"),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> FractalTokenPacket:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls.from_state_dict(state)

    def partition(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PartitionBatch:
        if self.tree_sequences is not None:
            trees = [deserialize_tree(sequence, self.grid_shape) for sequence in self.tree_sequences]
            partition = partition_batch_from_trees(trees, device=device, dtype=dtype)
            stored_mask = self.mask.to(device=device)
            if stored_mask.shape != partition.mask.shape or not torch.equal(stored_mask, partition.mask):
                raise ValueError("stored packet mask disagrees with tree geometry")
            if self.mass is not None:
                if self.mass.shape != partition.mass.shape:
                    raise ValueError("stored packet mass shape disagrees with tree geometry")
                partition.mass.copy_(self.mass.to(device=device, dtype=dtype))
            if self.depth is not None:
                if self.depth.shape != partition.depth.shape:
                    raise ValueError("stored packet depth shape disagrees with tree geometry")
                partition.depth.copy_(self.depth.to(device=device))
            return partition
        if self.boxes_grid is None:
            raise ValueError("packet has neither tree geometry nor explicit boxes")
        return partition_batch_from_boxes(
            boxes_grid=self.boxes_grid.to(device=device),
            mask=self.mask.to(device=device),
            grid_shape=self.grid_shape,
            mass=None if self.mass is None else self.mass.to(device=device, dtype=dtype),
            depth=None if self.depth is None else self.depth.to(device=device),
        )

    def codes(self, device: torch.device | str | None = None) -> torch.Tensor:
        if self.code_indices.ndim != 3:
            raise ValueError("packet code indices must have shape [B,N,Q]")
        if self.code_indices.shape[:2] != self.mask.shape:
            raise ValueError("packet code indices and mask shapes disagree")
        return self.code_indices.to(device=device)

    def estimated_bits(self, codebook_size: int | None = None) -> dict[str, float]:
        resolved_codebook_size = self.codebook_size if codebook_size is None else int(codebook_size)
        if resolved_codebook_size is None:
            raise ValueError("codebook_size must be stored in the packet or supplied explicitly")
        valid_tokens = int(self.mask.sum())
        content_bits = valid_tokens * self.num_quantizers * math.log2(resolved_codebook_size)
        if self.tree_sequences is not None:
            geometry_bits = 0
            for sequence in self.tree_sequences:
                tree = deserialize_tree(sequence, self.grid_shape)
                geometry_bits += estimate_tree_bits(tree)
        else:
            height, width = self.grid_shape
            x_bits = math.ceil(math.log2(width + 1))
            y_bits = math.ceil(math.log2(height + 1))
            geometry_bits = valid_tokens * 2 * (x_bits + y_bits)
        header_bits = 128.0
        total_bits = float(content_bits + geometry_bits + header_bits)
        pixels = self.mask.shape[0] * self.image_shape[0] * self.image_shape[1]
        return {
            "content_bits": float(content_bits),
            "geometry_bits": float(geometry_bits),
            "header_bits": header_bits,
            "total_bits": total_bits,
            "bits_per_pixel": total_bits / pixels,
        }
