from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .partition import deserialize_tree, partition_batch_from_trees
from .quantization import ResidualVectorQuantizer
from .structures import FractalTokens, PartitionBatch, PartitionTree


PAD_ACTION = 0
LEAF_ACTION = 1
SPLIT_X_ACTION = 2
SPLIT_Y_ACTION = 3


@dataclass
class FractalSequenceBatch:
    actions: torch.Tensor
    splits: torch.Tensor
    action_mask: torch.Tensor
    code_indices: torch.Tensor
    leaf_mask: torch.Tensor
    grid_shape: tuple[int, int]
    image_shape: tuple[int, int]

    @property
    def sequence_length(self) -> torch.Tensor:
        return self.action_mask.sum(dim=1)

    @property
    def token_count(self) -> torch.Tensor:
        return self.leaf_mask.sum(dim=1)

    def to(self, device: torch.device | str) -> FractalSequenceBatch:
        return FractalSequenceBatch(
            actions=self.actions.to(device),
            splits=self.splits.to(device),
            action_mask=self.action_mask.to(device),
            code_indices=self.code_indices.to(device),
            leaf_mask=self.leaf_mask.to(device),
            grid_shape=self.grid_shape,
            image_shape=self.image_shape,
        )

    def select(self, index: int) -> FractalSequenceBatch:
        action_length = int(self.sequence_length[index])
        token_count = int(self.token_count[index])
        return FractalSequenceBatch(
            actions=self.actions[index : index + 1, :action_length],
            splits=self.splits[index : index + 1, :action_length],
            action_mask=self.action_mask[index : index + 1, :action_length],
            code_indices=self.code_indices[index : index + 1, :token_count],
            leaf_mask=self.leaf_mask[index : index + 1, :token_count],
            grid_shape=self.grid_shape,
            image_shape=self.image_shape,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "actions": self.actions.detach().cpu(),
            "splits": self.splits.detach().cpu(),
            "action_mask": self.action_mask.detach().cpu(),
            "code_indices": self.code_indices.detach().cpu(),
            "leaf_mask": self.leaf_mask.detach().cpu(),
            "grid_shape": self.grid_shape,
            "image_shape": self.image_shape,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> FractalSequenceBatch:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported sequence version")
        return cls(
            actions=state["actions"],
            splits=state["splits"],
            action_mask=state["action_mask"],
            code_indices=state["code_indices"],
            leaf_mask=state["leaf_mask"],
            grid_shape=tuple(state["grid_shape"]),
            image_shape=tuple(state["image_shape"]),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> FractalSequenceBatch:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls.from_state_dict(state)


def trees_to_partition(
    trees: list[PartitionTree],
    device: torch.device | str,
    dtype: torch.dtype,
) -> PartitionBatch:
    return partition_batch_from_trees(trees, device=device, dtype=dtype)


def _serialize_generative_tree(tree: PartitionTree) -> tuple[list[tuple[int, int]], list[int]]:
    actions: list[tuple[int, int]] = []
    leaf_order: list[int] = []

    def visit(node_index: int) -> None:
        node = tree.nodes[node_index]
        if node.is_leaf:
            actions.append((LEAF_ACTION, -1))
            leaf_order.append(node.leaf_index)
            return
        action = SPLIT_X_ACTION if node.axis == 0 else SPLIT_Y_ACTION
        actions.append((action, node.split))
        visit(node.left)
        visit(node.right)

    visit(tree.root)
    return actions, leaf_order


def pack_fractal_sequence(tokens: FractalTokens) -> FractalSequenceBatch:
    if tokens.trees is None:
        raise ValueError("BSP trees are required to serialize adaptive geometry")
    if tokens.code_indices is None:
        raise ValueError("discrete code indices are required for generative serialization")
    serialized = [_serialize_generative_tree(tree) for tree in tokens.trees]
    maximum_actions = max(len(item[0]) for item in serialized)
    maximum_leaves = max(len(item[1]) for item in serialized)
    batch_size = len(serialized)
    num_quantizers = tokens.code_indices.shape[-1]
    actions = torch.full(
        (batch_size, maximum_actions),
        PAD_ACTION,
        device=tokens.code_indices.device,
        dtype=torch.long,
    )
    splits = torch.full_like(actions, -1)
    action_mask = torch.zeros_like(actions, dtype=torch.bool)
    code_indices = torch.zeros(
        batch_size,
        maximum_leaves,
        num_quantizers,
        device=tokens.code_indices.device,
        dtype=torch.long,
    )
    leaf_mask = torch.zeros(batch_size, maximum_leaves, device=tokens.code_indices.device, dtype=torch.bool)
    for batch_index, (tree_actions, leaf_order) in enumerate(serialized):
        for position, (action, split) in enumerate(tree_actions):
            actions[batch_index, position] = action
            splits[batch_index, position] = split
            action_mask[batch_index, position] = True
        order = torch.tensor(leaf_order, device=tokens.code_indices.device, dtype=torch.long)
        count = len(leaf_order)
        code_indices[batch_index, :count] = tokens.code_indices[batch_index, order]
        leaf_mask[batch_index, :count] = True
    return FractalSequenceBatch(
        actions=actions,
        splits=splits,
        action_mask=action_mask,
        code_indices=code_indices,
        leaf_mask=leaf_mask,
        grid_shape=tokens.trees[0].grid_shape,
        image_shape=tokens.image_shape,
    )


def unpack_fractal_sequence(
    sequence: FractalSequenceBatch,
    quantizer: ResidualVectorQuantizer,
) -> FractalTokens:
    trees = []
    for batch_index in range(sequence.actions.shape[0]):
        valid_actions = sequence.actions[batch_index, sequence.action_mask[batch_index]]
        valid_splits = sequence.splits[batch_index, sequence.action_mask[batch_index]]
        serialized = []
        leaf_index = 0
        for action_id, split in zip(valid_actions.tolist(), valid_splits.tolist()):
            if action_id == LEAF_ACTION:
                serialized.append((0, leaf_index))
                leaf_index += 1
            elif action_id == SPLIT_X_ACTION:
                serialized.append((1, int(split)))
            elif action_id == SPLIT_Y_ACTION:
                serialized.append((2, int(split)))
            else:
                raise ValueError(f"invalid action id {action_id}")
        trees.append(deserialize_tree(serialized, sequence.grid_shape))
    dtype = quantizer.quantizers[0].codebook.weight.dtype
    device = sequence.code_indices.device
    partition = trees_to_partition(trees, device=device, dtype=dtype)
    if sequence.code_indices.shape[0] != partition.mask.shape[0]:
        raise ValueError("tree batch and content-code batch disagree")
    maximum = partition.mask.shape[1]
    if sequence.code_indices.shape[1] < maximum or sequence.leaf_mask.shape[1] < maximum:
        raise ValueError("content-code tensor is shorter than the largest generated tree")
    code_indices = sequence.code_indices[:, :maximum]
    leaf_mask = sequence.leaf_mask[:, :maximum]
    if not torch.equal(partition.mask.cpu(), leaf_mask.cpu()):
        raise ValueError("tree leaves and content-code mask disagree")
    if bool(sequence.leaf_mask[:, maximum:].any()):
        raise ValueError("content-code mask contains leaves not represented by the tree")
    mask = partition.mask
    latents = quantizer.decode_indices(code_indices, mask)
    return FractalTokens(
        latents=latents,
        boxes=partition.boxes,
        boxes_grid=partition.boxes_grid,
        mask=mask,
        mass=partition.mass,
        depth=partition.depth,
        assignment=partition.assignment,
        image_shape=sequence.image_shape,
        basis_coefficients=None,
        code_indices=code_indices,
        trees=trees,
    )


def collate_fractal_sequences(sequences: list[FractalSequenceBatch]) -> FractalSequenceBatch:
    if not sequences:
        raise ValueError("at least one sequence is required")
    grid_shape = sequences[0].grid_shape
    image_shape = sequences[0].image_shape
    num_quantizers = sequences[0].code_indices.shape[-1]
    for sequence in sequences:
        if sequence.grid_shape != grid_shape or sequence.image_shape != image_shape:
            raise ValueError("all sequences in a batch must share image and grid shapes")
        if sequence.code_indices.shape[-1] != num_quantizers:
            raise ValueError("all sequences must use the same number of quantizers")
    maximum_actions = max(sequence.actions.shape[1] for sequence in sequences)
    maximum_leaves = max(sequence.code_indices.shape[1] for sequence in sequences)
    device = sequences[0].actions.device
    batch_size = sum(sequence.actions.shape[0] for sequence in sequences)
    actions = torch.full((batch_size, maximum_actions), PAD_ACTION, device=device, dtype=torch.long)
    splits = torch.full((batch_size, maximum_actions), -1, device=device, dtype=torch.long)
    action_mask = torch.zeros((batch_size, maximum_actions), device=device, dtype=torch.bool)
    code_indices = torch.zeros(
        batch_size,
        maximum_leaves,
        num_quantizers,
        device=device,
        dtype=torch.long,
    )
    leaf_mask = torch.zeros((batch_size, maximum_leaves), device=device, dtype=torch.bool)
    cursor = 0
    for sequence in sequences:
        count = sequence.actions.shape[0]
        action_length = sequence.actions.shape[1]
        leaf_length = sequence.code_indices.shape[1]
        actions[cursor : cursor + count, :action_length] = sequence.actions.to(device)
        splits[cursor : cursor + count, :action_length] = sequence.splits.to(device)
        action_mask[cursor : cursor + count, :action_length] = sequence.action_mask.to(device)
        code_indices[cursor : cursor + count, :leaf_length] = sequence.code_indices.to(device)
        leaf_mask[cursor : cursor + count, :leaf_length] = sequence.leaf_mask.to(device)
        cursor += count
    return FractalSequenceBatch(
        actions=actions,
        splits=splits,
        action_mask=action_mask,
        code_indices=code_indices,
        leaf_mask=leaf_mask,
        grid_shape=grid_shape,
        image_shape=image_shape,
    )
