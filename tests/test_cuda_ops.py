import pytest
import torch

from fractaltokenizer_v2.cuda_ops import build_cuda_extension, cuda_extension_status
from fractaltokenizer_v2.partition import QuantileGridPartitioner
from fractaltokenizer_v2.regions import render_token_tiles, sample_region_tiles
from fractaltokenizer_v2.universal_codelength import UniversalCodelengthDensity


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cuda_required
@pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 1, 63, 79)])
def test_cuda_universal_codelength_matches_torch(shape: tuple[int, int, int, int]):
    torch.manual_seed(9)
    image = torch.rand(*shape, device="cuda")
    reference = UniversalCodelengthDensity(partition_stride=8, backend="torch")(image)
    measured = UniversalCodelengthDensity(partition_stride=8, backend="cuda")(image)
    torch.testing.assert_close(measured, reference, rtol=2e-6, atol=0.004)


@cuda_required
@pytest.mark.parametrize("token_count", [13, 64, 96, 128])
def test_cuda_quantile_partition_matches_torch(token_count: int):
    torch.manual_seed(12 + token_count)
    density = torch.rand(3, 1, 32, 32, device="cuda")
    torch_partition = QuantileGridPartitioner(backend="torch")(density, token_count)
    cuda_partition = QuantileGridPartitioner(backend="cuda")(density, token_count)
    assert torch.equal(cuda_partition.assignment, torch_partition.assignment)
    assert torch.equal(cuda_partition.boxes_grid, torch_partition.boxes_grid)
    assert torch.equal(cuda_partition.mask, torch_partition.mask)
    assert torch.equal(cuda_partition.depth, torch_partition.depth)
    torch.testing.assert_close(cuda_partition.boxes, torch_partition.boxes, rtol=0, atol=0)
    torch.testing.assert_close(cuda_partition.mass, torch_partition.mass, rtol=2e-5, atol=2e-5)


@cuda_required
@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_cuda_quantile_partition_handles_degenerate_density(fill: float):
    density = torch.full((2, 1, 16, 16), fill, device="cuda")
    torch_partition = QuantileGridPartitioner(backend="torch")(density, 37)
    cuda_partition = QuantileGridPartitioner(backend="cuda")(density, 37)
    assert cuda_partition.assignment.amin().item() == 0
    assert cuda_partition.assignment.amax().item() == 36
    assert cuda_partition.mask.all()
    assert ((cuda_partition.boxes_grid[..., 2:] - cuda_partition.boxes_grid[..., :2]) > 0).all()
    assert (cuda_partition.boxes_grid - torch_partition.boxes_grid).abs().amax() <= 1
    torch.testing.assert_close(cuda_partition.mass.sum(dim=1), torch_partition.mass.sum(dim=1), rtol=2e-5, atol=2e-5)


@cuda_required
def test_cuda_region_sampling_matches_roi_align_forward_and_backward():
    torch.manual_seed(21)
    density = torch.rand(2, 1, 16, 16, device="cuda")
    partition = QuantileGridPartitioner(backend="torch")(density, 37)
    torch_image = torch.rand(2, 3, 128, 128, device="cuda", requires_grad=True)
    cuda_image = torch_image.detach().clone().requires_grad_(True)
    torch_tiles = sample_region_tiles(
        torch_image,
        partition.boxes,
        12,
        partition.mask,
        backend="torch",
        sampling_ratio=2,
    )
    cuda_tiles = sample_region_tiles(
        cuda_image,
        partition.boxes,
        12,
        partition.mask,
        backend="cuda",
        sampling_ratio=2,
    )
    torch.testing.assert_close(cuda_tiles, torch_tiles, rtol=2e-5, atol=2e-5)
    gradient = torch.randn_like(torch_tiles)
    torch_tiles.backward(gradient)
    cuda_tiles.backward(gradient)
    torch.testing.assert_close(cuda_image.grad, torch_image.grad, rtol=2e-4, atol=2e-4)


@cuda_required
def test_cuda_render_matches_torch_forward_and_backward():
    torch.manual_seed(30)
    density = torch.rand(2, 1, 16, 16, device="cuda")
    partition = QuantileGridPartitioner(backend="torch")(density, 37)
    torch_tiles = torch.rand(2, 37, 3, 17, 13, device="cuda", requires_grad=True)
    cuda_tiles = torch_tiles.detach().clone().requires_grad_(True)
    torch_image = render_token_tiles(
        torch_tiles,
        partition.boxes,
        partition.assignment,
        (128, 128),
        backend="torch",
    )
    cuda_image = render_token_tiles(
        cuda_tiles,
        partition.boxes,
        partition.assignment,
        (128, 128),
        backend="cuda",
    )
    torch.testing.assert_close(cuda_image, torch_image, rtol=2e-5, atol=2e-5)
    gradient = torch.randn_like(torch_image)
    torch_image.backward(gradient)
    cuda_image.backward(gradient)
    torch.testing.assert_close(cuda_tiles.grad, torch_tiles.grad, rtol=3e-4, atol=3e-4)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_sampling_and_render_support_reduced_precision(dtype: torch.dtype):
    density = torch.rand(1, 1, 8, 8, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 16)
    image = torch.rand(1, 3, 64, 64, device="cuda", dtype=dtype, requires_grad=True)
    tiles = sample_region_tiles(image, partition.boxes, 8, partition.mask, backend="cuda")
    reconstruction = render_token_tiles(tiles, partition.boxes, partition.assignment, (64, 64), backend="cuda")
    reconstruction.float().mean().backward()
    assert reconstruction.dtype == torch.float32
    assert torch.isfinite(reconstruction).all()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


@cuda_required
def test_cuda_render_uses_fp32_geometry_for_bfloat16_tiles():
    torch.manual_seed(44)
    density = torch.rand(2, 1, 16, 16, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 37)
    tiles = torch.rand(2, 37, 3, 17, 13, device="cuda", dtype=torch.bfloat16)
    reference = render_token_tiles(
        tiles.float(),
        partition.boxes,
        partition.assignment,
        (128, 128),
        backend="torch",
    )
    cuda_image = render_token_tiles(
        tiles,
        partition.boxes,
        partition.assignment,
        (128, 128),
        backend="cuda",
    )
    assert cuda_image.dtype == torch.float32
    torch.testing.assert_close(cuda_image, reference, rtol=2e-5, atol=2e-5)


@cuda_required
def test_cuda_bfloat16_render_backward_tracks_fp32_oracle():
    torch.manual_seed(45)
    density = torch.rand(2, 1, 16, 16, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 37)
    reference_tiles = torch.rand(2, 37, 3, 17, 13, device="cuda", requires_grad=True)
    cuda_tiles = reference_tiles.detach().to(torch.bfloat16).requires_grad_(True)
    reference = render_token_tiles(
        reference_tiles,
        partition.boxes,
        partition.assignment,
        (128, 130),
        backend="torch",
    )
    cuda_image = render_token_tiles(
        cuda_tiles,
        partition.boxes,
        partition.assignment,
        (128, 130),
        backend="cuda",
    )
    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    cuda_image.backward(gradient)
    error = (cuda_tiles.grad.float() - reference_tiles.grad).abs()
    assert error.mean() < 0.0015
    assert torch.quantile(error, 0.99) < 0.008
    assert error.max() < 0.025


@cuda_required
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_cuda_render_matches_odd_shape_and_channel_counts(channels: int):
    torch.manual_seed(50 + channels)
    density = torch.rand(1, 1, 7, 11, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 23)
    tiles = torch.rand(1, 23, channels, 9, 13, device="cuda")
    reference = render_token_tiles(
        tiles,
        partition.boxes,
        partition.assignment,
        (61, 97),
        backend="torch",
    )
    cuda_image = render_token_tiles(
        tiles,
        partition.boxes,
        partition.assignment,
        (61, 97),
        backend="cuda",
    )
    torch.testing.assert_close(cuda_image, reference, rtol=2e-5, atol=2e-5)


@cuda_required
def test_cuda_render_clamp_matches_torch_value_and_gradient():
    torch.manual_seed(61)
    density = torch.rand(1, 1, 8, 8, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 16)
    reference_tiles = (torch.rand(1, 16, 3, 8, 8, device="cuda") * 2.0 - 0.5).requires_grad_(True)
    cuda_tiles = reference_tiles.detach().clone().requires_grad_(True)
    reference = render_token_tiles(
        reference_tiles,
        partition.boxes,
        partition.assignment,
        (64, 65),
        backend="torch",
        clamp_output=True,
    )
    cuda_image = render_token_tiles(
        cuda_tiles,
        partition.boxes,
        partition.assignment,
        (64, 65),
        backend="cuda",
        clamp_output=True,
    )
    torch.testing.assert_close(cuda_image, reference, rtol=2e-5, atol=2e-5)
    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    cuda_image.backward(gradient)
    torch.testing.assert_close(cuda_tiles.grad, reference_tiles.grad, rtol=4e-4, atol=4e-4)


@cuda_required
def test_cuda_extension_build_and_status():
    module = build_cuda_extension()
    status = cuda_extension_status()
    assert module is not None
    assert status["loaded"] is True
    assert status["failed"] is False


@cuda_required
def test_cuda_render_composes_with_fullgraph_compile_and_backward():
    build_cuda_extension()
    density = torch.rand(1, 1, 8, 8, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 16)

    def loss_function(tiles: torch.Tensor) -> torch.Tensor:
        image = render_token_tiles(
            tiles,
            partition.boxes,
            partition.assignment,
            (64, 64),
            backend="cuda",
        )
        return image.square().mean()

    eager_tiles = torch.rand(1, 16, 3, 8, 8, device="cuda", requires_grad=True)
    compiled_tiles = eager_tiles.detach().clone().requires_grad_(True)
    eager_loss = loss_function(eager_tiles)
    compiled_loss = torch.compile(loss_function, fullgraph=True, mode="reduce-overhead")(compiled_tiles)
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(compiled_tiles.grad, eager_tiles.grad, rtol=3e-4, atol=3e-4)


@cuda_required
def test_cuda_partition_composes_with_fullgraph_compile():
    build_cuda_extension()
    partitioner = QuantileGridPartitioner(backend="cuda").cuda()

    def partition_boxes(density: torch.Tensor) -> torch.Tensor:
        return partitioner(density, 37).boxes

    density = torch.rand(2, 1, 16, 16, device="cuda")
    eager = partition_boxes(density)
    compiled = torch.compile(partition_boxes, fullgraph=True, mode="reduce-overhead")(density)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


@cuda_required
def test_cuda_region_sampling_composes_with_fullgraph_compile_and_backward():
    build_cuda_extension()
    density = torch.rand(1, 1, 8, 8, device="cuda")
    partition = QuantileGridPartitioner(backend="cuda")(density, 16)

    def loss_function(image: torch.Tensor) -> torch.Tensor:
        tiles = sample_region_tiles(
            image,
            partition.boxes,
            8,
            partition.mask,
            backend="cuda",
        )
        return tiles.square().mean()

    eager_image = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    compiled_image = eager_image.detach().clone().requires_grad_(True)
    eager_loss = loss_function(eager_image)
    compiled_loss = torch.compile(loss_function, fullgraph=True, mode="reduce-overhead")(compiled_image)
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(compiled_image.grad, eager_image.grad, rtol=3e-4, atol=3e-4)
