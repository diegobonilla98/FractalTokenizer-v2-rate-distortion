from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class FractalCudaSettings:
    backend: str = "auto"
    sampling_ratio: int = 2
    partition_density_floor_ratio: float = 0.02
    postprocess_clamp: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "cuda", "torch"}:
            raise ValueError("backend must be auto, cuda, or torch")
        if self.sampling_ratio < 1:
            raise ValueError("sampling_ratio must be positive")
        if self.partition_density_floor_ratio < 0:
            raise ValueError("partition_density_floor_ratio cannot be negative")


_extension = None
_extension_error: Exception | None = None
_extension_lock = threading.Lock()
_warning_emitted = False
_columns_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
_columns_lock = threading.Lock()
_fake_ops_registered = False


def _extension_sources() -> list[str]:
    source_root = Path(__file__).resolve().parent / "csrc"
    return [str(source_root / "fractal_ops.cpp"), str(source_root / "fractal_ops_cuda.cu")]


def build_cuda_extension():
    global _extension
    global _extension_error
    if _extension is not None:
        _register_fake_ops()
        return _extension
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    with _extension_lock:
        if _extension is not None:
            _register_fake_ops()
            return _extension
        if _extension_error is not None:
            raise RuntimeError("the FractalTokenizer CUDA extension previously failed to build") from _extension_error
        verbose = os.environ.get("FRACTAL_CUDA_VERBOSE", "0") == "1"
        try:
            _extension = load(
                name="fractaltokenizer_v2_cuda_v1",
                sources=_extension_sources(),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=verbose,
            )
        except Exception as error:
            _extension_error = error
            raise
    _register_fake_ops()
    return _extension


def _fake_quantile_partition(
    density: torch.Tensor,
    columns_per_row: torch.Tensor,
    token_count: int,
    density_floor_ratio: float,
):
    batch_size, height, width = density.shape
    long_options = {"device": density.device, "dtype": torch.int64}
    return [
        torch.empty(batch_size, token_count, 4, **long_options),
        torch.empty(batch_size, token_count, 4, device=density.device, dtype=torch.float32),
        torch.empty(batch_size, token_count, device=density.device, dtype=torch.bool),
        torch.empty(batch_size, token_count, device=density.device, dtype=torch.float32),
        torch.empty(batch_size, token_count, **long_options),
        torch.empty(batch_size, height, width, **long_options),
    ]


def _fake_region_sample_forward(
    image: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor,
    output_size: int,
    sampling_ratio: int,
) -> torch.Tensor:
    return torch.empty(
        image.shape[0],
        boxes.shape[1],
        image.shape[1],
        output_size,
        output_size,
        device=image.device,
        dtype=image.dtype,
    )


def _fake_region_sample_backward(
    gradient: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor,
    image_height: int,
    image_width: int,
    sampling_ratio: int,
) -> torch.Tensor:
    return torch.empty(
        gradient.shape[0],
        gradient.shape[2],
        image_height,
        image_width,
        device=gradient.device,
        dtype=gradient.dtype,
    )


def _fake_render_tiles_forward(
    tiles: torch.Tensor,
    boxes: torch.Tensor,
    assignment: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    return torch.empty(
        tiles.shape[0],
        tiles.shape[2],
        image_height,
        image_width,
        device=tiles.device,
        dtype=torch.float32,
    )


def _fake_render_tiles_backward(
    gradient: torch.Tensor,
    boxes: torch.Tensor,
    assignment: torch.Tensor,
    tile_height: int,
    tile_width: int,
) -> torch.Tensor:
    return torch.empty(
        gradient.shape[0],
        boxes.shape[1],
        gradient.shape[1],
        tile_height,
        tile_width,
        device=gradient.device,
        dtype=torch.float32,
    )


def _fake_universal_codelength_density(
    image: torch.Tensor,
    partition_stride: int,
    scan_directions: int,
    alpha: float,
) -> torch.Tensor:
    grid_height = (image.shape[2] + partition_stride - 1) // partition_stride
    grid_width = (image.shape[3] + partition_stride - 1) // partition_stride
    return torch.empty(
        image.shape[0],
        1,
        grid_height,
        grid_width,
        device=image.device,
        dtype=torch.float32,
    )


def _register_fake_ops() -> None:
    global _fake_ops_registered
    if _fake_ops_registered:
        return
    torch.library.register_fake("fractaltokenizer_v2::quantile_partition")(_fake_quantile_partition)
    torch.library.register_fake("fractaltokenizer_v2::region_sample_forward")(_fake_region_sample_forward)
    torch.library.register_fake("fractaltokenizer_v2::region_sample_backward")(_fake_region_sample_backward)
    torch.library.register_fake("fractaltokenizer_v2::render_tiles_forward")(_fake_render_tiles_forward)
    torch.library.register_fake("fractaltokenizer_v2::render_tiles_backward")(_fake_render_tiles_backward)
    torch.library.register_fake("fractaltokenizer_v2::universal_codelength_density")(
        _fake_universal_codelength_density
    )
    _fake_ops_registered = True


def _resolve_extension(backend: str, *tensors: torch.Tensor):
    global _warning_emitted
    if backend not in {"auto", "cuda", "torch"}:
        raise ValueError("backend must be auto, cuda, or torch")
    if backend == "torch":
        return None
    if not tensors or not all(tensor.is_cuda for tensor in tensors):
        if backend == "cuda":
            raise ValueError("the cuda backend requires CUDA tensors")
        return None
    if backend == "cuda":
        return build_cuda_extension()
    try:
        return build_cuda_extension()
    except Exception as error:
        if not _warning_emitted:
            warnings.warn(
                f"FractalTokenizer CUDA extension unavailable; falling back to PyTorch: {error}",
                RuntimeWarning,
                stacklevel=3,
            )
            _warning_emitted = True
        return None


def cuda_extension_status() -> dict[str, object]:
    return {
        "loaded": _extension is not None,
        "failed": _extension_error is not None,
        "error": None if _extension_error is None else str(_extension_error),
        "cuda_available": torch.cuda.is_available(),
        "sources": _extension_sources(),
    }


def _cached_columns(columns_per_row: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.tensor(columns_per_row, device=device, dtype=torch.int32)
    device_index = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (device_index, columns_per_row)
    with _columns_lock:
        tensor = _columns_cache.get(key)
        if tensor is None:
            tensor = torch.tensor(columns_per_row, device=device, dtype=torch.int32)
            _columns_cache[key] = tensor
    return tensor


def cuda_quantile_partition(
    density: torch.Tensor,
    columns_per_row: list[int] | tuple[int, ...],
    density_floor_ratio: float,
    backend: str = "auto",
):
    extension = _resolve_extension(backend, density)
    if extension is None:
        return None
    if density.ndim != 3:
        raise ValueError("density must have shape [B,H,W]")
    columns = tuple(int(value) for value in columns_per_row)
    if not columns or min(columns) < 1:
        raise ValueError("columns_per_row must contain positive values")
    columns_tensor = _cached_columns(columns, density.device)
    return torch.ops.fractaltokenizer_v2.quantile_partition(
        density.float().contiguous(),
        columns_tensor,
        int(sum(columns)),
        float(density_floor_ratio),
    )


class _RegionSampleFunction(Function):
    @staticmethod
    def forward(
        ctx,
        image: torch.Tensor,
        boxes: torch.Tensor,
        mask: torch.Tensor,
        output_size: int,
        sampling_ratio: int,
    ) -> torch.Tensor:
        image = image.contiguous()
        boxes = boxes.float().contiguous()
        mask = mask.bool().contiguous()
        output = torch.ops.fractaltokenizer_v2.region_sample_forward(
            image,
            boxes,
            mask,
            int(output_size),
            int(sampling_ratio),
        )
        ctx.save_for_backward(boxes, mask)
        ctx.image_shape = tuple(image.shape)
        ctx.sampling_ratio = int(sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient: torch.Tensor):
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("region sampling backward uses nondeterministic CUDA atomics")
        boxes, mask = ctx.saved_tensors
        batch_size, channels, height, width = ctx.image_shape
        image_gradient = torch.ops.fractaltokenizer_v2.region_sample_backward(
            gradient.contiguous(),
            boxes,
            mask,
            int(height),
            int(width),
            int(ctx.sampling_ratio),
        )
        return image_gradient, None, None, None, None


def cuda_sample_region_tiles(
    image: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int,
    mask: torch.Tensor,
    sampling_ratio: int = 2,
    backend: str = "auto",
) -> torch.Tensor | None:
    extension = _resolve_extension(backend, image, boxes, mask)
    if extension is None:
        return None
    if sampling_ratio < 1:
        raise ValueError("sampling_ratio must be positive")
    if image.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("CUDA region sampling supports float16, bfloat16, and float32 images")
    if len({image.device, boxes.device, mask.device}) != 1:
        raise ValueError("image, boxes, and mask must be on the same CUDA device")
    return _RegionSampleFunction.apply(image, boxes, mask, int(output_size), int(sampling_ratio))


class _RenderTokenTilesFunction(Function):
    @staticmethod
    def forward(
        ctx,
        tiles: torch.Tensor,
        boxes: torch.Tensor,
        assignment: torch.Tensor,
        height: int,
        width: int,
        clamp_output: bool,
    ) -> torch.Tensor:
        tiles = tiles.contiguous()
        boxes = boxes.float().contiguous()
        assignment = assignment.long().contiguous()
        output = torch.ops.fractaltokenizer_v2.render_tiles_forward(
            tiles,
            boxes,
            assignment,
            int(height),
            int(width),
        )
        clamp_mask = (output >= 0.0) & (output <= 1.0) if clamp_output else torch.empty(0, device=output.device, dtype=torch.bool)
        ctx.save_for_backward(boxes, assignment, clamp_mask)
        ctx.tile_shape = tuple(tiles.shape)
        ctx.tile_dtype = tiles.dtype
        ctx.image_shape = (int(height), int(width))
        ctx.clamp_output = bool(clamp_output)
        return output.clamp(0.0, 1.0) if clamp_output else output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient: torch.Tensor):
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("tile rendering backward uses nondeterministic CUDA atomics")
        boxes, assignment, clamp_mask = ctx.saved_tensors
        if ctx.clamp_output:
            gradient = gradient * clamp_mask
        _, _, _, tile_height, tile_width = ctx.tile_shape
        tiles_gradient = torch.ops.fractaltokenizer_v2.render_tiles_backward(
            gradient.contiguous(),
            boxes,
            assignment,
            int(tile_height),
            int(tile_width),
        )
        return tiles_gradient.to(ctx.tile_dtype), None, None, None, None, None


def cuda_render_token_tiles(
    tiles: torch.Tensor,
    boxes: torch.Tensor,
    assignment: torch.Tensor,
    image_shape: tuple[int, int],
    clamp_output: bool = False,
    backend: str = "auto",
) -> torch.Tensor | None:
    extension = _resolve_extension(backend, tiles, boxes, assignment)
    if extension is None:
        return None
    if tiles.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("CUDA tile rendering supports float16, bfloat16, and float32 tiles")
    if len({tiles.device, boxes.device, assignment.device}) != 1:
        raise ValueError("tiles, boxes, and assignment must be on the same CUDA device")
    height, width = int(image_shape[0]), int(image_shape[1])
    return _RenderTokenTilesFunction.apply(
        tiles,
        boxes,
        assignment,
        height,
        width,
        bool(clamp_output),
    )


def cuda_universal_codelength_density(
    image: torch.Tensor,
    partition_stride: int,
    scan_directions: int,
    alpha: float,
    backend: str = "auto",
) -> torch.Tensor | None:
    extension = _resolve_extension(backend, image)
    if extension is None:
        return None
    if image.ndim != 4 or image.shape[1] not in {1, 3}:
        raise ValueError("image must have shape [B,1,H,W] or [B,3,H,W]")
    return torch.ops.fractaltokenizer_v2.universal_codelength_density(
        image.detach().float().contiguous(),
        int(partition_stride),
        int(scan_directions),
        float(alpha),
    )
