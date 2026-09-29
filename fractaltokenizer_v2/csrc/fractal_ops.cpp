#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> quantile_partition_cuda(
    torch::Tensor density,
    torch::Tensor columns_per_row,
    int64_t token_count,
    double density_floor_ratio);

torch::Tensor region_sample_forward_cuda(
    torch::Tensor image,
    torch::Tensor boxes,
    torch::Tensor mask,
    int64_t output_size,
    int64_t sampling_ratio);

torch::Tensor region_sample_backward_cuda(
    torch::Tensor gradient,
    torch::Tensor boxes,
    torch::Tensor mask,
    int64_t image_height,
    int64_t image_width,
    int64_t sampling_ratio);

torch::Tensor render_tiles_forward_cuda(
    torch::Tensor tiles,
    torch::Tensor boxes,
    torch::Tensor assignment,
    int64_t image_height,
    int64_t image_width);

torch::Tensor render_tiles_backward_cuda(
    torch::Tensor gradient,
    torch::Tensor boxes,
    torch::Tensor assignment,
    int64_t tile_height,
    int64_t tile_width);

torch::Tensor universal_codelength_density_cuda(
    torch::Tensor image,
    int64_t partition_stride,
    int64_t scan_directions,
    double alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("quantile_partition", &quantile_partition_cuda);
    module.def("region_sample_forward", &region_sample_forward_cuda);
    module.def("region_sample_backward", &region_sample_backward_cuda);
    module.def("render_tiles_forward", &render_tiles_forward_cuda);
    module.def("render_tiles_backward", &render_tiles_backward_cuda);
    module.def("universal_codelength_density", &universal_codelength_density_cuda);
}


TORCH_LIBRARY(fractaltokenizer_v2, module) {
    module.def("quantile_partition(Tensor density, Tensor columns_per_row, int token_count, float density_floor_ratio) -> Tensor[]");
    module.def("region_sample_forward(Tensor image, Tensor boxes, Tensor mask, int output_size, int sampling_ratio) -> Tensor");
    module.def("region_sample_backward(Tensor gradient, Tensor boxes, Tensor mask, int image_height, int image_width, int sampling_ratio) -> Tensor");
    module.def("render_tiles_forward(Tensor tiles, Tensor boxes, Tensor assignment, int image_height, int image_width) -> Tensor");
    module.def("render_tiles_backward(Tensor gradient, Tensor boxes, Tensor assignment, int tile_height, int tile_width) -> Tensor");
    module.def("universal_codelength_density(Tensor image, int partition_stride, int scan_directions, float alpha) -> Tensor");
}


TORCH_LIBRARY_IMPL(fractaltokenizer_v2, CUDA, module) {
    module.impl("quantile_partition", &quantile_partition_cuda);
    module.impl("region_sample_forward", &region_sample_forward_cuda);
    module.impl("region_sample_backward", &region_sample_backward_cuda);
    module.impl("render_tiles_forward", &render_tiles_forward_cuda);
    module.impl("render_tiles_backward", &render_tiles_backward_cuda);
    module.impl("universal_codelength_density", &universal_codelength_density_cuda);
}
