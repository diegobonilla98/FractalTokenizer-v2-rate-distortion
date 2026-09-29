from __future__ import annotations

import heapq
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from .cuda_ops import cuda_quantile_partition
from .structures import BSPNode, PartitionBatch, PartitionTree


def _token_row_counts(token_count: int, height: int, width: int) -> list[int]:
    token_count = int(token_count)
    if token_count < 1 or token_count > height * width:
        raise ValueError(
            f"target_tokens must satisfy 1 <= target_tokens <= {height * width}, got {token_count}"
        )
    target_ratio = height / max(width, 1)
    best = None
    for rows in range(1, min(height, token_count) + 1):
        base, remainder = divmod(token_count, rows)
        maximum_columns = base + int(remainder > 0)
        if base < 1 or maximum_columns > width:
            continue
        average_columns = token_count / rows
        ratio = rows / average_columns
        score = abs(math.log((ratio + 1e-8) / (target_ratio + 1e-8)))
        score += 0.01 * remainder / rows
        if best is None or score < best[0]:
            best = (score, rows, base, remainder)
    if best is None:
        raise ValueError(f"cannot allocate {token_count} tokens over a {height} by {width} grid")
    _, rows, base, remainder = best
    counts = [base] * rows
    if remainder:
        for index in range(remainder):
            row = round((index + 0.5) * rows / remainder - 0.5)
            counts[min(max(row, 0), rows - 1)] += 1
    if sum(counts) != token_count or min(counts) < 1 or max(counts) > width:
        raise RuntimeError("invalid token-row allocation")
    return counts


def _strict_profile_boundaries(profile: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    batch, length = profile.shape
    bins = int(fractions.numel()) + 1
    if bins < 1 or bins > length:
        raise ValueError("the number of bins must not exceed the profile length")
    if bins == 1:
        return torch.empty(batch, 0, device=profile.device, dtype=torch.long)
    cumulative = profile.cumsum(dim=-1)
    total = cumulative[:, -1:].clamp_min(1e-8)
    targets = total * fractions.to(device=profile.device, dtype=profile.dtype).unsqueeze(0)
    raw = torch.searchsorted(cumulative.contiguous(), targets.contiguous(), right=False) + 1
    boundaries = []
    previous = torch.zeros(batch, device=profile.device, dtype=torch.long)
    for index in range(bins - 1):
        remaining = bins - 1 - index
        low = previous + 1
        high = torch.full_like(low, length - remaining)
        current = torch.minimum(torch.maximum(raw[:, index], low), high)
        boundaries.append(current)
        previous = current
    return torch.stack(boundaries, dim=-1)

def _strict_quantile_boundaries(profile: torch.Tensor, bins: int) -> torch.Tensor:
    fractions = torch.arange(1, bins, device=profile.device, dtype=profile.dtype) / bins
    return _strict_profile_boundaries(profile, fractions)


def _boxes_from_assignment(
    assignment: torch.Tensor,
    density: torch.Tensor,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, height, width = assignment.shape
    device = assignment.device
    flat_ids = assignment.reshape(batch_size, -1)
    y = torch.arange(height, device=device, dtype=torch.long).view(1, height, 1).expand(batch_size, height, width)
    x = torch.arange(width, device=device, dtype=torch.long).view(1, 1, width).expand(batch_size, height, width)
    flat_x = x.reshape(batch_size, -1)
    flat_y = y.reshape(batch_size, -1)
    x0 = torch.full((batch_size, token_count), width, device=device, dtype=torch.long)
    y0 = torch.full((batch_size, token_count), height, device=device, dtype=torch.long)
    x1 = torch.full((batch_size, token_count), -1, device=device, dtype=torch.long)
    y1 = torch.full((batch_size, token_count), -1, device=device, dtype=torch.long)
    x0.scatter_reduce_(1, flat_ids, flat_x, reduce="amin", include_self=True)
    y0.scatter_reduce_(1, flat_ids, flat_y, reduce="amin", include_self=True)
    x1.scatter_reduce_(1, flat_ids, flat_x, reduce="amax", include_self=True)
    y1.scatter_reduce_(1, flat_ids, flat_y, reduce="amax", include_self=True)
    count = torch.zeros((batch_size, token_count), device=device, dtype=density.dtype)
    count.scatter_add_(1, flat_ids, torch.ones_like(flat_ids, dtype=density.dtype))
    mass = torch.zeros((batch_size, token_count), device=device, dtype=density.dtype)
    mass.scatter_add_(1, flat_ids, density.reshape(batch_size, -1))
    mask = count > 0
    boxes_grid = torch.stack([x0, y0, x1 + 1, y1 + 1], dim=-1)
    boxes_grid = torch.where(mask.unsqueeze(-1), boxes_grid, torch.zeros_like(boxes_grid))
    area = ((boxes_grid[..., 2] - boxes_grid[..., 0]) * (boxes_grid[..., 3] - boxes_grid[..., 1])).clamp_min(1)
    depth = torch.round(0.5 * torch.log2(torch.tensor(height * width, device=device) / area.float())).long()
    depth = torch.where(mask, depth.clamp_min(0), torch.zeros_like(depth))
    return boxes_grid, mask, mass, depth


class QuantileGridPartitioner(nn.Module):
    def __init__(self, density_floor_ratio: float = 0.02, backend: str = "auto"):
        super().__init__()
        if density_floor_ratio < 0:
            raise ValueError("density_floor_ratio cannot be negative")
        if backend not in {"auto", "cuda", "torch"}:
            raise ValueError("backend must be auto, cuda, or torch")
        self.density_floor_ratio = float(density_floor_ratio)
        self.backend = backend

    def forward(
        self,
        density: torch.Tensor,
        target_tokens: int,
        token_grid: tuple[int, int] | None = None,
    ) -> PartitionBatch:
        if density.ndim != 4 or density.shape[1] != 1:
            raise ValueError("density must have shape [B,1,H,W]")
        values = density[:, 0].float().clamp_min(0)
        batch_size, height, width = values.shape
        if token_grid is None:
            columns_per_row = _token_row_counts(int(target_tokens), height, width)
        else:
            rows, columns = int(token_grid[0]), int(token_grid[1])
            if rows * columns != int(target_tokens):
                raise ValueError("token_grid product must equal target_tokens")
            if rows < 1 or columns < 1 or rows > height or columns > width:
                raise ValueError("token_grid cannot be empty or finer than the density grid")
            columns_per_row = [columns] * rows
        rows = len(columns_per_row)
        token_count = sum(columns_per_row)
        cuda_result = cuda_quantile_partition(
            values,
            columns_per_row,
            self.density_floor_ratio,
            backend=self.backend,
        )
        if cuda_result is not None:
            boxes_grid, boxes, mask, mass, depth, assignment = cuda_result
            return PartitionBatch(
                boxes_grid=boxes_grid,
                boxes=boxes,
                mask=mask,
                mass=mass,
                depth=depth,
                assignment=assignment,
                trees=None,
            )
        mean = values.mean(dim=(-2, -1), keepdim=True)
        values = values + self.density_floor_ratio * (mean + 1e-6)
        row_profile = values.sum(dim=-1)
        cumulative_tokens = torch.tensor(
            columns_per_row[:-1],
            device=values.device,
            dtype=values.dtype,
        ).cumsum(dim=0)
        row_fractions = cumulative_tokens / token_count
        row_boundaries = _strict_profile_boundaries(row_profile, row_fractions)
        y_coordinates = torch.arange(height, device=values.device).view(1, height, 1)
        row_ids = (y_coordinates >= row_boundaries.unsqueeze(1)).sum(dim=-1)
        if len(set(columns_per_row)) == 1:
            columns = columns_per_row[0]
            band_profile = torch.zeros(batch_size, rows, width, device=values.device, dtype=values.dtype)
            band_profile.scatter_add_(
                1,
                row_ids.unsqueeze(-1).expand(batch_size, height, width),
                values,
            )
            flat_band_profile = band_profile.reshape(batch_size * rows, width)
            column_boundaries = _strict_quantile_boundaries(flat_band_profile, columns).reshape(
                batch_size,
                rows,
                -1,
            )
            x_coordinates = torch.arange(width, device=values.device).view(1, 1, width, 1)
            column_ids_by_band = (x_coordinates >= column_boundaries.unsqueeze(2)).sum(dim=-1)
            column_ids = torch.gather(
                column_ids_by_band,
                1,
                row_ids.unsqueeze(-1).expand(batch_size, height, width),
            )
            assignment = row_ids.unsqueeze(-1) * columns + column_ids
        else:
            assignment = torch.zeros(batch_size, height, width, device=values.device, dtype=torch.long)
            x_coordinates = torch.arange(width, device=values.device).view(1, width, 1)
            offset = 0
            for row, columns in enumerate(columns_per_row):
                row_mask = row_ids.eq(row)
                profile = (values * row_mask.unsqueeze(-1)).sum(dim=1)
                boundaries = _strict_quantile_boundaries(profile, columns)
                column_ids = (x_coordinates >= boundaries.unsqueeze(1)).sum(dim=-1)
                assignment = torch.where(
                    row_mask.unsqueeze(-1),
                    offset + column_ids.unsqueeze(1),
                    assignment,
                )
                offset += columns
        boxes_grid, mask, mass, depth = _boxes_from_assignment(assignment, values, token_count)
        scale = torch.tensor([width, height, width, height], device=values.device, dtype=values.dtype)
        boxes = boxes_grid.to(values.dtype) / scale
        return PartitionBatch(
            boxes_grid=boxes_grid,
            boxes=boxes,
            mask=mask,
            mass=mass,
            depth=depth,
            assignment=assignment,
            trees=None,
        )


def _weighted_median(profile: np.ndarray, low: int, high: int, min_size: int) -> int | None:
    length = high - low
    if length < 2 * min_size:
        return None
    total = float(profile.sum())
    if total <= 1e-12:
        split = low + length // 2
    else:
        cumulative = np.cumsum(profile)
        split_local = int(np.searchsorted(cumulative, 0.5 * total, side="left")) + 1
        split = low + split_local
    return int(np.clip(split, low + min_size, high - min_size))


def _axis_candidate(
    values: np.ndarray,
    box: tuple[int, int, int, int],
    axis: int,
    min_size: int,
) -> tuple[float, int] | None:
    x0, y0, x1, y1 = box
    region = values[y0:y1, x0:x1]
    if axis == 0:
        profile = region.sum(axis=0)
        split = _weighted_median(profile, x0, x1, min_size)
        if split is None:
            return None
        local_split = split - x0
        mass_a = float(region[:, :local_split].sum())
        mass_b = float(region[:, local_split:].sum())
        child_shapes = ((y1 - y0, split - x0), (y1 - y0, x1 - split))
        coordinate = np.arange(x0, x1, dtype=np.float64) + 0.5
    else:
        profile = region.sum(axis=1)
        split = _weighted_median(profile, y0, y1, min_size)
        if split is None:
            return None
        local_split = split - y0
        mass_a = float(region[:local_split, :].sum())
        mass_b = float(region[local_split:, :].sum())
        child_shapes = ((split - y0, x1 - x0), (y1 - split, x1 - x0))
        coordinate = np.arange(y0, y1, dtype=np.float64) + 0.5
    total = mass_a + mass_b
    imbalance = abs(mass_a - mass_b) / max(total, 1e-12)
    profile_total = float(profile.sum())
    if profile_total > 1e-12:
        mean = float((profile * coordinate).sum() / profile_total)
        variance = float((profile * (coordinate - mean) ** 2).sum() / profile_total)
        normalized_variance = variance / max(float(len(profile) ** 2), 1.0)
    else:
        normalized_variance = 1.0 / 12.0
    aspect_cost = 0.0
    for child_height, child_width in child_shapes:
        ratio = child_width / max(child_height, 1)
        aspect_cost += abs(math.log(max(ratio, 1e-8)))
    return normalized_variance - imbalance - 0.025 * aspect_cost, split


def _best_split(
    values: np.ndarray,
    box: tuple[int, int, int, int],
    min_size: int,
) -> tuple[int, int] | None:
    candidates = []
    for axis in (0, 1):
        candidate = _axis_candidate(values, box, axis, min_size)
        if candidate is not None:
            quality, split = candidate
            candidates.append((quality, axis, split))
    if not candidates:
        return None
    _, axis, split = max(candidates, key=lambda item: item[0])
    return axis, split


def _partition_single(
    density: np.ndarray,
    target_tokens: int | None,
    mass_per_token: float | None,
    max_tokens: int,
    min_size: int,
    density_floor_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PartitionTree]:
    values = np.asarray(density, dtype=np.float64)
    values = np.maximum(values, 0.0) + density_floor_ratio * (float(values.mean()) + 1e-8)
    height, width = values.shape
    root_mass = float(values.sum())
    nodes = [BSPNode(box=(0, 0, width, height), depth=0, mass=root_mass)]
    leaves = {0}
    heap: list[tuple[float, int, int]] = []
    root_score = root_mass if _best_split(values, nodes[0].box, min_size) is not None else -math.inf
    heapq.heappush(heap, (-root_score, 0, 0))
    order = 1
    desired = max_tokens if target_tokens is None else min(int(target_tokens), max_tokens)
    while heap and len(leaves) < desired:
        negative_score, _, node_index = heapq.heappop(heap)
        if node_index not in leaves:
            continue
        node = nodes[node_index]
        if target_tokens is None and mass_per_token is not None and node.mass <= mass_per_token:
            break
        split_spec = _best_split(values, node.box, min_size)
        if split_spec is None or not math.isfinite(-negative_score):
            continue
        axis, split = split_spec
        x0, y0, x1, y1 = node.box
        boxes = (
            ((x0, y0, split, y1), (split, y0, x1, y1))
            if axis == 0
            else ((x0, y0, x1, split), (x0, split, x1, y1))
        )
        child_indices = []
        for child_box in boxes:
            cx0, cy0, cx1, cy1 = child_box
            mass = float(values[cy0:cy1, cx0:cx1].sum())
            child_indices.append(len(nodes))
            nodes.append(BSPNode(box=child_box, depth=node.depth + 1, mass=mass))
        node.axis = axis
        node.split = split
        node.left, node.right = child_indices
        leaves.remove(node_index)
        leaves.update(child_indices)
        for child_index in child_indices:
            child = nodes[child_index]
            splittable = _best_split(values, child.box, min_size) is not None
            heapq.heappush(heap, (-(child.mass if splittable else -math.inf), order, child_index))
            order += 1
        if target_tokens is None and mass_per_token is None:
            break
    ordered_leaves: list[int] = []

    def visit(node_index: int) -> None:
        node = nodes[node_index]
        if node.is_leaf:
            ordered_leaves.append(node_index)
            return
        visit(node.left)
        visit(node.right)

    visit(0)
    token_count = len(ordered_leaves)
    boxes_grid = np.zeros((token_count, 4), dtype=np.int64)
    mass = np.zeros(token_count, dtype=np.float32)
    depth = np.zeros(token_count, dtype=np.int64)
    assignment = np.full((height, width), -1, dtype=np.int64)
    for leaf_index, node_index in enumerate(ordered_leaves):
        node = nodes[node_index]
        node.leaf_index = leaf_index
        boxes_grid[leaf_index] = node.box
        mass[leaf_index] = node.mass
        depth[leaf_index] = node.depth
        x0, y0, x1, y1 = node.box
        assignment[y0:y1, x0:x1] = leaf_index
    if (assignment < 0).any():
        raise RuntimeError("partition did not cover the density grid")
    return (
        boxes_grid,
        np.ones(token_count, dtype=np.bool_),
        mass,
        depth,
        assignment,
        PartitionTree(grid_shape=(height, width), nodes=nodes),
    )


class BalancedBSPPartitioner:
    def __init__(
        self,
        min_size: int = 1,
        max_tokens: int = 256,
        density_floor_ratio: float = 0.01,
        workers: int = 0,
    ):
        if min_size < 1:
            raise ValueError("min_size must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if density_floor_ratio < 0:
            raise ValueError("density_floor_ratio cannot be negative")
        if workers < 0:
            raise ValueError("workers cannot be negative")
        self.min_size = int(min_size)
        self.max_tokens = int(max_tokens)
        self.density_floor_ratio = float(density_floor_ratio)
        self.workers = int(workers)

    def __call__(
        self,
        density: torch.Tensor,
        target_tokens: int | None = None,
        mass_per_token: float | None = None,
    ) -> PartitionBatch:
        if density.ndim != 4 or density.shape[1] != 1:
            raise ValueError("density must have shape [B,1,H,W]")
        if target_tokens is None and mass_per_token is None:
            raise ValueError("provide target_tokens or mass_per_token")
        if target_tokens is not None and target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if mass_per_token is not None and mass_per_token <= 0:
            raise ValueError("mass_per_token must be positive")
        device = density.device
        dtype = density.dtype
        arrays = list(density[:, 0].detach().float().cpu().numpy())

        def partition(array: np.ndarray):
            return _partition_single(
                array,
                target_tokens=target_tokens,
                mass_per_token=mass_per_token,
                max_tokens=self.max_tokens,
                min_size=self.min_size,
                density_floor_ratio=self.density_floor_ratio,
            )

        if self.workers > 1 and len(arrays) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = list(executor.map(partition, arrays))
        else:
            results = [partition(array) for array in arrays]
        maximum = max(result[0].shape[0] for result in results)
        batch_size = len(results)
        height, width = arrays[0].shape
        boxes_grid = torch.zeros(batch_size, maximum, 4, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, maximum, dtype=torch.bool, device=device)
        mass = torch.zeros(batch_size, maximum, dtype=dtype, device=device)
        depth = torch.zeros(batch_size, maximum, dtype=torch.long, device=device)
        assignment = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
        trees = []
        for batch_index, result in enumerate(results):
            boxes_np, mask_np, mass_np, depth_np, assignment_np, tree = result
            count = boxes_np.shape[0]
            boxes_grid[batch_index, :count] = torch.from_numpy(boxes_np).to(device)
            mask[batch_index, :count] = torch.from_numpy(mask_np).to(device)
            mass[batch_index, :count] = torch.from_numpy(mass_np).to(device=device, dtype=dtype)
            depth[batch_index, :count] = torch.from_numpy(depth_np).to(device)
            assignment[batch_index] = torch.from_numpy(assignment_np).to(device)
            trees.append(tree)
        scale = torch.tensor([width, height, width, height], device=device, dtype=dtype)
        return PartitionBatch(
            boxes_grid=boxes_grid,
            boxes=boxes_grid.to(dtype) / scale,
            mask=mask,
            mass=mass,
            depth=depth,
            assignment=assignment,
            trees=trees,
        )


def serialize_tree(tree: PartitionTree) -> list[tuple[int, int]]:
    sequence: list[tuple[int, int]] = []

    def visit(node_index: int) -> None:
        node = tree.nodes[node_index]
        if node.is_leaf:
            sequence.append((0, node.leaf_index))
            return
        sequence.append((node.axis + 1, node.split))
        visit(node.left)
        visit(node.right)

    visit(tree.root)
    return sequence


def deserialize_tree(sequence: list[tuple[int, int]], grid_shape: tuple[int, int]) -> PartitionTree:
    height, width = grid_shape
    nodes: list[BSPNode] = []
    cursor = 0
    leaf_index = 0

    def parse(box: tuple[int, int, int, int], depth: int) -> int:
        nonlocal cursor, leaf_index
        if cursor >= len(sequence):
            raise ValueError("incomplete tree sequence")
        action, split = sequence[cursor]
        cursor += 1
        node_index = len(nodes)
        nodes.append(BSPNode(box=box, depth=depth, mass=0.0))
        node = nodes[node_index]
        if action == 0:
            node.leaf_index = split
            leaf_index = max(leaf_index, split + 1)
            return node_index
        axis = action - 1
        x0, y0, x1, y1 = box
        if axis == 0:
            if not x0 < split < x1:
                raise ValueError("invalid x split")
            left_box = (x0, y0, split, y1)
            right_box = (split, y0, x1, y1)
        elif axis == 1:
            if not y0 < split < y1:
                raise ValueError("invalid y split")
            left_box = (x0, y0, x1, split)
            right_box = (x0, split, x1, y1)
        else:
            raise ValueError("invalid split action")
        node.axis = axis
        node.split = split
        node.left = parse(left_box, depth + 1)
        node.right = parse(right_box, depth + 1)
        return node_index

    root = parse((0, 0, width, height), 0)
    if cursor != len(sequence):
        raise ValueError("tree sequence has trailing entries")
    return PartitionTree(grid_shape=grid_shape, nodes=nodes, root=root)


def assignment_from_tree(tree: PartitionTree, device: torch.device | str | None = None) -> torch.Tensor:
    height, width = tree.grid_shape
    assignment = torch.full((height, width), -1, dtype=torch.long, device=device)
    for node in tree.leaf_nodes():
        x0, y0, x1, y1 = node.box
        assignment[y0:y1, x0:x1] = node.leaf_index
    if bool((assignment < 0).any()):
        raise ValueError("tree leaves do not cover the full grid")
    return assignment


def partition_batch_from_trees(
    trees: list[PartitionTree],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PartitionBatch:
    if not trees:
        raise ValueError("trees cannot be empty")
    grid_shape = trees[0].grid_shape
    if any(tree.grid_shape != grid_shape for tree in trees):
        raise ValueError("all trees in a batch must use the same grid shape")
    height, width = grid_shape
    token_counts = [len(tree.leaf_nodes()) for tree in trees]
    maximum = max(token_counts)
    batch_size = len(trees)
    boxes_grid = torch.zeros(batch_size, maximum, 4, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, maximum, dtype=torch.bool, device=device)
    mass = torch.zeros(batch_size, maximum, dtype=dtype, device=device)
    depth = torch.zeros(batch_size, maximum, dtype=torch.long, device=device)
    assignment = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
    for batch_index, tree in enumerate(trees):
        leaves = tree.leaf_nodes()
        seen = set()
        for node in leaves:
            if node.leaf_index in seen:
                raise ValueError("tree has duplicate leaf indices")
            seen.add(node.leaf_index)
            if node.leaf_index < 0 or node.leaf_index >= len(leaves):
                raise ValueError("tree leaf indices must be contiguous from zero")
            boxes_grid[batch_index, node.leaf_index] = torch.tensor(node.box, device=device)
            mask[batch_index, node.leaf_index] = True
            mass[batch_index, node.leaf_index] = float(node.mass)
            depth[batch_index, node.leaf_index] = int(node.depth)
        assignment[batch_index] = assignment_from_tree(tree, device=device)
    scale = torch.tensor([width, height, width, height], device=device, dtype=dtype)
    return PartitionBatch(
        boxes_grid=boxes_grid,
        boxes=boxes_grid.to(dtype) / scale,
        mask=mask,
        mass=mass,
        depth=depth,
        assignment=assignment,
        trees=trees,
    )


def estimate_tree_bits(tree: PartitionTree) -> int:
    total = 0
    for node in tree.nodes:
        total += 1
        if node.is_leaf:
            continue
        x0, y0, x1, y1 = node.box
        length = (x1 - x0) if node.axis == 0 else (y1 - y0)
        total += 1
        total += max(1, math.ceil(math.log2(max(length - 1, 1))))
    return total


def assignment_from_boxes_grid(
    boxes_grid: torch.Tensor,
    mask: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    if boxes_grid.ndim != 3 or boxes_grid.shape[-1] != 4:
        raise ValueError("boxes_grid must have shape [B,N,4]")
    if mask.shape != boxes_grid.shape[:2]:
        raise ValueError("mask must have shape [B,N]")
    height, width = grid_shape
    assignment = torch.full(
        (boxes_grid.shape[0], height, width),
        -1,
        dtype=torch.long,
        device=boxes_grid.device,
    )
    for batch_index in range(boxes_grid.shape[0]):
        for token_index in torch.nonzero(mask[batch_index], as_tuple=False).flatten().tolist():
            x0, y0, x1, y1 = [int(value) for value in boxes_grid[batch_index, token_index].tolist()]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("box is outside the partition grid")
            region = assignment[batch_index, y0:y1, x0:x1]
            if bool((region >= 0).any()):
                raise ValueError("boxes overlap")
            region.fill_(token_index)
    if bool((assignment < 0).any()):
        raise ValueError("boxes do not cover the full partition grid")
    return assignment


def partition_batch_from_boxes(
    boxes_grid: torch.Tensor,
    mask: torch.Tensor,
    grid_shape: tuple[int, int],
    mass: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
) -> PartitionBatch:
    height, width = grid_shape
    if mass is None:
        mass = torch.zeros(mask.shape, device=boxes_grid.device, dtype=torch.float32)
    if depth is None:
        area = (
            (boxes_grid[..., 2] - boxes_grid[..., 0]).clamp_min(1)
            * (boxes_grid[..., 3] - boxes_grid[..., 1]).clamp_min(1)
        )
        depth = torch.round(
            0.5 * torch.log2(torch.tensor(height * width, device=boxes_grid.device) / area.float())
        ).long()
        depth = torch.where(mask, depth.clamp_min(0), torch.zeros_like(depth))
    assignment = assignment_from_boxes_grid(boxes_grid, mask, grid_shape)
    dtype = mass.dtype
    scale = torch.tensor([width, height, width, height], device=boxes_grid.device, dtype=dtype)
    return PartitionBatch(
        boxes_grid=boxes_grid,
        boxes=boxes_grid.to(dtype) / scale,
        mask=mask,
        mass=mass,
        depth=depth,
        assignment=assignment,
        trees=None,
    )
