#include <torch/extension.h>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <vector>


namespace {

constexpr int threads_per_block = 256;
constexpr int universal_alphabet_size = 511;


__device__ __forceinline__ float positive_value(float value) {
    return fmaxf(value, 0.0f);
}


__device__ __forceinline__ int quantized_pixel(
    const float* image,
    int batch,
    int channel,
    int y,
    int x,
    int channels,
    int height,
    int width) {
    int64_t index = ((static_cast<int64_t>(batch) * channels + channel) * height + y) * width + x;
    return static_cast<int>(nearbyintf(fminf(fmaxf(image[index], 0.0f), 1.0f) * 255.0f));
}


__device__ __forceinline__ void oriented_to_source(
    int rotation,
    int oriented_y,
    int oriented_x,
    int height,
    int width,
    int& source_y,
    int& source_x) {
    if (rotation == 0) {
        source_y = oriented_y;
        source_x = oriented_x;
    } else if (rotation == 1) {
        source_y = oriented_x;
        source_x = width - 1 - oriented_y;
    } else if (rotation == 2) {
        source_y = height - 1 - oriented_y;
        source_x = width - 1 - oriented_x;
    } else {
        source_y = height - 1 - oriented_x;
        source_x = oriented_y;
    }
}


__device__ __forceinline__ int oriented_pixel(
    const float* image,
    int batch,
    int channel,
    int oriented_y,
    int oriented_x,
    int rotation,
    int channels,
    int height,
    int width) {
    int source_y;
    int source_x;
    oriented_to_source(rotation, oriented_y, oriented_x, height, width, source_y, source_x);
    return quantized_pixel(image, batch, channel, source_y, source_x, channels, height, width);
}


__global__ void universal_codelength_contributions_kernel(
    const float* image,
    int* counts,
    int batch_size,
    int channels,
    int height,
    int width,
    int scan_directions,
    float alpha,
    float* contributions) {
    int block = blockIdx.x;
    int batch = block / scan_directions;
    int rotation = block - batch * scan_directions;
    int channel = threadIdx.x;
    if (batch >= batch_size || channel >= channels) {
        return;
    }
    int oriented_height = rotation % 2 == 0 ? height : width;
    int oriented_width = rotation % 2 == 0 ? width : height;
    int64_t stream_index = (static_cast<int64_t>(block) * channels + channel);
    int* stream_counts = counts + stream_index * 4 * universal_alphabet_size;
    for (int index = 0; index < 4 * universal_alphabet_size; ++index) {
        stream_counts[index] = 0;
    }
    float denominator_prior = universal_alphabet_size * alpha;
    int position = 0;
    for (int y = 0; y < oriented_height; ++y) {
        for (int x = 0; x < oriented_width; ++x, ++position) {
            int current = oriented_pixel(image, batch, channel, y, x, rotation, channels, height, width);
            int left = x == 0
                ? 128
                : oriented_pixel(image, batch, channel, y, x - 1, rotation, channels, height, width);
            int top = y == 0
                ? 128
                : oriented_pixel(image, batch, channel, y - 1, x, rotation, channels, height, width);
            int top_left = y == 0 || x == 0
                ? 128
                : oriented_pixel(image, batch, channel, y - 1, x - 1, rotation, channels, height, width);
            int gradient = min(max(left + top - top_left, 0), 255);
            int lower = min(left, top);
            int upper = max(left, top);
            int median = top_left >= upper ? lower : (top_left <= lower ? upper : gradient);
            int predictors[4] = {left, top, gradient, median};
            float probability_sum = 0.0f;
            float denominator = static_cast<float>(position) + denominator_prior;
            for (int predictor = 0; predictor < 4; ++predictor) {
                int symbol = current - predictors[predictor] + 255;
                int* count = stream_counts + predictor * universal_alphabet_size + symbol;
                probability_sum += (static_cast<float>(*count) + alpha) / denominator;
                *count += 1;
            }
            float bits = -log2f(fmaxf(probability_sum * 0.25f, 1e-20f));
            int source_y;
            int source_x;
            oriented_to_source(rotation, y, x, height, width, source_y, source_x);
            int64_t output_index = (((static_cast<int64_t>(batch) * scan_directions + rotation) * channels
                + channel) * height + source_y) * width + source_x;
            contributions[output_index] = bits;
        }
    }
}


__global__ void universal_codelength_reduce_kernel(
    const float* contributions,
    int batch_size,
    int channels,
    int height,
    int width,
    int scan_directions,
    int stride,
    int grid_height,
    int grid_width,
    float* output) {
    int64_t total = static_cast<int64_t>(batch_size) * grid_height * grid_width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        int grid_x = remaining % grid_width;
        remaining /= grid_width;
        int grid_y = remaining % grid_height;
        int batch = remaining / grid_height;
        int start_y = grid_y * stride;
        int start_x = grid_x * stride;
        int end_y = min(start_y + stride, height);
        int end_x = min(start_x + stride, width);
        float sum = 0.0f;
        for (int rotation = 0; rotation < scan_directions; ++rotation) {
            for (int channel = 0; channel < channels; ++channel) {
                int64_t base = ((static_cast<int64_t>(batch) * scan_directions + rotation) * channels
                    + channel) * height * width;
                for (int y = start_y; y < end_y; ++y) {
                    for (int x = start_x; x < end_x; ++x) {
                        sum += contributions[base + y * width + x];
                    }
                }
            }
        }
        output[index] = sum / static_cast<float>(scan_directions);
    }
}


__global__ void quantile_partition_kernel(
    const float* density,
    const int* columns_per_row,
    int batch_size,
    int height,
    int width,
    int row_count,
    int token_count,
    float density_floor_ratio,
    int64_t* boxes_grid,
    float* boxes,
    bool* mask,
    float* mass,
    int64_t* depth,
    int64_t* assignment) {
    int batch = blockIdx.x;
    int thread = threadIdx.x;
    if (batch >= batch_size) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    float* row_profile = reinterpret_cast<float*>(shared_bytes);
    float* column_profile = row_profile + height;
    int* row_boundaries = reinterpret_cast<int*>(column_profile + width);
    int* column_boundaries = row_boundaries + row_count + 1;
    __shared__ float total_mass;
    __shared__ float density_floor;
    __shared__ int token_offset;

    const float* batch_density = density + static_cast<int64_t>(batch) * height * width;
    for (int y = thread; y < height; y += blockDim.x) {
        float sum = 0.0f;
        for (int x = 0; x < width; ++x) {
            sum += positive_value(batch_density[y * width + x]);
        }
        row_profile[y] = sum;
    }
    __syncthreads();

    if (thread == 0) {
        total_mass = 0.0f;
        for (int y = 0; y < height; ++y) {
            total_mass += row_profile[y];
        }
        density_floor = density_floor_ratio * (total_mass / static_cast<float>(height * width) + 1e-6f);
        for (int y = 0; y < height; ++y) {
            row_profile[y] += density_floor * width;
        }
        total_mass += density_floor * height * width;
        row_boundaries[0] = 0;
        int previous = 0;
        int cumulative_tokens = 0;
        for (int row = 0; row < row_count - 1; ++row) {
            cumulative_tokens += columns_per_row[row];
            float target = total_mass * static_cast<float>(cumulative_tokens) / static_cast<float>(token_count);
            float cumulative = 0.0f;
            int raw = height;
            for (int y = 0; y < height; ++y) {
                cumulative += row_profile[y];
                if (cumulative >= target) {
                    raw = y + 1;
                    break;
                }
            }
            int remaining = row_count - 1 - row;
            int low = previous + 1;
            int high = height - remaining;
            int current = min(max(raw, low), high);
            row_boundaries[row + 1] = current;
            previous = current;
        }
        row_boundaries[row_count] = height;
        token_offset = 0;
    }
    __syncthreads();

    for (int token = thread; token < token_count; token += blockDim.x) {
        mask[static_cast<int64_t>(batch) * token_count + token] = true;
    }
    __syncthreads();

    for (int row = 0; row < row_count; ++row) {
        int y0 = row_boundaries[row];
        int y1 = row_boundaries[row + 1];
        int columns = columns_per_row[row];
        for (int x = thread; x < width; x += blockDim.x) {
            float sum = 0.0f;
            for (int y = y0; y < y1; ++y) {
                sum += positive_value(batch_density[y * width + x]) + density_floor;
            }
            column_profile[x] = sum;
        }
        __syncthreads();

        if (thread == 0) {
            float band_mass = 0.0f;
            for (int x = 0; x < width; ++x) {
                band_mass += column_profile[x];
            }
            column_boundaries[0] = 0;
            int previous = 0;
            for (int column = 0; column < columns - 1; ++column) {
                float target = band_mass * static_cast<float>(column + 1) / static_cast<float>(columns);
                float cumulative = 0.0f;
                int raw = width;
                for (int x = 0; x < width; ++x) {
                    cumulative += column_profile[x];
                    if (cumulative >= target) {
                        raw = x + 1;
                        break;
                    }
                }
                int remaining = columns - 1 - column;
                int low = previous + 1;
                int high = width - remaining;
                int current = min(max(raw, low), high);
                column_boundaries[column + 1] = current;
                previous = current;
            }
            column_boundaries[columns] = width;
        }
        __syncthreads();

        for (int column = thread; column < columns; column += blockDim.x) {
            int x0 = column_boundaries[column];
            int x1 = column_boundaries[column + 1];
            int token = token_offset + column;
            int64_t token_index = static_cast<int64_t>(batch) * token_count + token;
            int64_t box_index = token_index * 4;
            boxes_grid[box_index] = x0;
            boxes_grid[box_index + 1] = y0;
            boxes_grid[box_index + 2] = x1;
            boxes_grid[box_index + 3] = y1;
            boxes[box_index] = static_cast<float>(x0) / static_cast<float>(width);
            boxes[box_index + 1] = static_cast<float>(y0) / static_cast<float>(height);
            boxes[box_index + 2] = static_cast<float>(x1) / static_cast<float>(width);
            boxes[box_index + 3] = static_cast<float>(y1) / static_cast<float>(height);
            float token_mass = 0.0f;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    token_mass += positive_value(batch_density[y * width + x]) + density_floor;
                }
            }
            mass[token_index] = token_mass;
            float area = static_cast<float>((x1 - x0) * (y1 - y0));
            float raw_depth = 0.5f * log2f(static_cast<float>(height * width) / fmaxf(area, 1.0f));
            depth[token_index] = static_cast<int64_t>(max(0.0f, rintf(raw_depth)));
        }

        int band_pixels = (y1 - y0) * width;
        for (int index = thread; index < band_pixels; index += blockDim.x) {
            int local_y = index / width;
            int x = index - local_y * width;
            int column = 0;
            while (column + 1 < columns && x >= column_boundaries[column + 1]) {
                ++column;
            }
            int y = y0 + local_y;
            assignment[(static_cast<int64_t>(batch) * height + y) * width + x] = token_offset + column;
        }
        __syncthreads();
        if (thread == 0) {
            token_offset += columns;
        }
        __syncthreads();
    }
}


__device__ __forceinline__ bool bilinear_parameters(
    float y,
    float x,
    int height,
    int width,
    int& y_low,
    int& x_low,
    int& y_high,
    int& x_high,
    float& ly,
    float& lx) {
    if (y < -1.0f || y > static_cast<float>(height) || x < -1.0f || x > static_cast<float>(width)) {
        return false;
    }
    y = fmaxf(y, 0.0f);
    x = fmaxf(x, 0.0f);
    y_low = static_cast<int>(y);
    x_low = static_cast<int>(x);
    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = static_cast<float>(y_low);
    } else {
        y_high = y_low + 1;
    }
    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = static_cast<float>(x_low);
    } else {
        x_high = x_low + 1;
    }
    ly = y - static_cast<float>(y_low);
    lx = x - static_cast<float>(x_low);
    return true;
}


template <typename scalar_t>
__global__ void region_sample_forward_kernel(
    const scalar_t* image,
    const float* boxes,
    const bool* mask,
    int batch_size,
    int channels,
    int image_height,
    int image_width,
    int token_count,
    int output_size,
    int sampling_ratio,
    scalar_t* output) {
    int64_t total = static_cast<int64_t>(batch_size) * token_count * channels * output_size * output_size;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        int px = remaining % output_size;
        remaining /= output_size;
        int py = remaining % output_size;
        remaining /= output_size;
        int channel = remaining % channels;
        remaining /= channels;
        int token = remaining % token_count;
        int batch = remaining / token_count;
        int64_t token_index = static_cast<int64_t>(batch) * token_count + token;
        if (!mask[token_index]) {
            output[index] = static_cast<scalar_t>(0.0f);
            continue;
        }
        const float* box = boxes + token_index * 4;
        float start_x = box[0] * image_width - 0.5f;
        float start_y = box[1] * image_height - 0.5f;
        float roi_width = (box[2] - box[0]) * image_width;
        float roi_height = (box[3] - box[1]) * image_height;
        float bin_width = roi_width / static_cast<float>(output_size);
        float bin_height = roi_height / static_cast<float>(output_size);
        float value = 0.0f;
        for (int iy = 0; iy < sampling_ratio; ++iy) {
            float y = start_y + py * bin_height + (iy + 0.5f) * bin_height / sampling_ratio;
            for (int ix = 0; ix < sampling_ratio; ++ix) {
                float x = start_x + px * bin_width + (ix + 0.5f) * bin_width / sampling_ratio;
                int y_low;
                int x_low;
                int y_high;
                int x_high;
                float ly;
                float lx;
                if (!bilinear_parameters(y, x, image_height, image_width, y_low, x_low, y_high, x_high, ly, lx)) {
                    continue;
                }
                int64_t image_base = (static_cast<int64_t>(batch) * channels + channel) * image_height * image_width;
                float v00 = static_cast<float>(image[image_base + y_low * image_width + x_low]);
                float v01 = static_cast<float>(image[image_base + y_low * image_width + x_high]);
                float v10 = static_cast<float>(image[image_base + y_high * image_width + x_low]);
                float v11 = static_cast<float>(image[image_base + y_high * image_width + x_high]);
                float hy = 1.0f - ly;
                float hx = 1.0f - lx;
                value += hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
            }
        }
        output[index] = static_cast<scalar_t>(value / static_cast<float>(sampling_ratio * sampling_ratio));
    }
}


template <typename scalar_t>
__global__ void region_sample_backward_kernel(
    const scalar_t* gradient,
    const float* boxes,
    const bool* mask,
    int batch_size,
    int channels,
    int image_height,
    int image_width,
    int token_count,
    int output_size,
    int sampling_ratio,
    scalar_t* image_gradient) {
    int64_t total = static_cast<int64_t>(batch_size) * token_count * channels * output_size * output_size;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        int px = remaining % output_size;
        remaining /= output_size;
        int py = remaining % output_size;
        remaining /= output_size;
        int channel = remaining % channels;
        remaining /= channels;
        int token = remaining % token_count;
        int batch = remaining / token_count;
        int64_t token_index = static_cast<int64_t>(batch) * token_count + token;
        if (!mask[token_index]) {
            continue;
        }
        const float* box = boxes + token_index * 4;
        float start_x = box[0] * image_width - 0.5f;
        float start_y = box[1] * image_height - 0.5f;
        float roi_width = (box[2] - box[0]) * image_width;
        float roi_height = (box[3] - box[1]) * image_height;
        float bin_width = roi_width / static_cast<float>(output_size);
        float bin_height = roi_height / static_cast<float>(output_size);
        float grad = static_cast<float>(gradient[index]) / static_cast<float>(sampling_ratio * sampling_ratio);
        for (int iy = 0; iy < sampling_ratio; ++iy) {
            float y = start_y + py * bin_height + (iy + 0.5f) * bin_height / sampling_ratio;
            for (int ix = 0; ix < sampling_ratio; ++ix) {
                float x = start_x + px * bin_width + (ix + 0.5f) * bin_width / sampling_ratio;
                int y_low;
                int x_low;
                int y_high;
                int x_high;
                float ly;
                float lx;
                if (!bilinear_parameters(y, x, image_height, image_width, y_low, x_low, y_high, x_high, ly, lx)) {
                    continue;
                }
                float hy = 1.0f - ly;
                float hx = 1.0f - lx;
                int64_t image_base = (static_cast<int64_t>(batch) * channels + channel) * image_height * image_width;
                gpuAtomicAdd(image_gradient + image_base + y_low * image_width + x_low, static_cast<scalar_t>(grad * hy * hx));
                gpuAtomicAdd(image_gradient + image_base + y_low * image_width + x_high, static_cast<scalar_t>(grad * hy * lx));
                gpuAtomicAdd(image_gradient + image_base + y_high * image_width + x_low, static_cast<scalar_t>(grad * ly * hx));
                gpuAtomicAdd(image_gradient + image_base + y_high * image_width + x_high, static_cast<scalar_t>(grad * ly * lx));
            }
        }
    }
}


template <typename scalar_t>
__global__ void render_tiles_forward_kernel(
    const scalar_t* tiles,
    const float* boxes,
    const int64_t* assignment,
    int batch_size,
    int token_count,
    int channels,
    int tile_height,
    int tile_width,
    int assignment_height,
    int assignment_width,
    int image_height,
    int image_width,
    float* output) {
    int64_t total = static_cast<int64_t>(batch_size) * image_height * image_width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        int x = remaining % image_width;
        remaining /= image_width;
        int y = remaining % image_height;
        remaining /= image_height;
        int batch = remaining;
        int owner_x = min(x * assignment_width / image_width, assignment_width - 1);
        int owner_y = min(y * assignment_height / image_height, assignment_height - 1);
        int64_t owner_index = (static_cast<int64_t>(batch) * assignment_height + owner_y) * assignment_width + owner_x;
        int token = static_cast<int>(assignment[owner_index]);
        token = min(max(token, 0), token_count - 1);
        const float* box = boxes + (static_cast<int64_t>(batch) * token_count + token) * 4;
        float x_normalized = (static_cast<float>(x) + 0.5f) / static_cast<float>(image_width);
        float y_normalized = (static_cast<float>(y) + 0.5f) / static_cast<float>(image_height);
        float u = fminf(fmaxf((x_normalized - box[0]) / fmaxf(box[2] - box[0], 1e-6f), 0.0f), 1.0f);
        float v = fminf(fmaxf((y_normalized - box[1]) / fmaxf(box[3] - box[1], 1e-6f), 0.0f), 1.0f);
        float tile_x = fminf(fmaxf(u * tile_width - 0.5f, 0.0f), static_cast<float>(tile_width - 1));
        float tile_y = fminf(fmaxf(v * tile_height - 0.5f, 0.0f), static_cast<float>(tile_height - 1));
        int x_low = static_cast<int>(floorf(tile_x));
        int y_low = static_cast<int>(floorf(tile_y));
        int x_high = min(x_low + 1, tile_width - 1);
        int y_high = min(y_low + 1, tile_height - 1);
        float lx = tile_x - x_low;
        float ly = tile_y - y_low;
        for (int channel = 0; channel < channels; ++channel) {
            int64_t tile_base = ((static_cast<int64_t>(batch) * token_count + token) * channels + channel) * tile_height * tile_width;
            float v00 = static_cast<float>(tiles[tile_base + y_low * tile_width + x_low]);
            float v01 = static_cast<float>(tiles[tile_base + y_low * tile_width + x_high]);
            float v10 = static_cast<float>(tiles[tile_base + y_high * tile_width + x_low]);
            float v11 = static_cast<float>(tiles[tile_base + y_high * tile_width + x_high]);
            float value = (1.0f - ly) * ((1.0f - lx) * v00 + lx * v01) + ly * ((1.0f - lx) * v10 + lx * v11);
            int64_t output_index = (static_cast<int64_t>(batch) * channels + channel) * image_height * image_width
                + y * image_width + x;
            output[output_index] = value;
        }
    }
}


__global__ void render_tiles_backward_kernel(
    const float* gradient,
    const float* boxes,
    const int64_t* assignment,
    int batch_size,
    int token_count,
    int channels,
    int tile_height,
    int tile_width,
    int assignment_height,
    int assignment_width,
    int image_height,
    int image_width,
    float* tile_gradient) {
    int64_t total = static_cast<int64_t>(batch_size) * image_height * image_width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        int x = remaining % image_width;
        remaining /= image_width;
        int y = remaining % image_height;
        remaining /= image_height;
        int batch = remaining;
        int owner_x = min(x * assignment_width / image_width, assignment_width - 1);
        int owner_y = min(y * assignment_height / image_height, assignment_height - 1);
        int64_t owner_index = (static_cast<int64_t>(batch) * assignment_height + owner_y) * assignment_width + owner_x;
        int token = static_cast<int>(assignment[owner_index]);
        token = min(max(token, 0), token_count - 1);
        const float* box = boxes + (static_cast<int64_t>(batch) * token_count + token) * 4;
        float x_normalized = (static_cast<float>(x) + 0.5f) / static_cast<float>(image_width);
        float y_normalized = (static_cast<float>(y) + 0.5f) / static_cast<float>(image_height);
        float u = fminf(fmaxf((x_normalized - box[0]) / fmaxf(box[2] - box[0], 1e-6f), 0.0f), 1.0f);
        float v = fminf(fmaxf((y_normalized - box[1]) / fmaxf(box[3] - box[1], 1e-6f), 0.0f), 1.0f);
        float tile_x = fminf(fmaxf(u * tile_width - 0.5f, 0.0f), static_cast<float>(tile_width - 1));
        float tile_y = fminf(fmaxf(v * tile_height - 0.5f, 0.0f), static_cast<float>(tile_height - 1));
        int x_low = static_cast<int>(floorf(tile_x));
        int y_low = static_cast<int>(floorf(tile_y));
        int x_high = min(x_low + 1, tile_width - 1);
        int y_high = min(y_low + 1, tile_height - 1);
        float lx = tile_x - x_low;
        float ly = tile_y - y_low;
        for (int channel = 0; channel < channels; ++channel) {
            int64_t gradient_index = (static_cast<int64_t>(batch) * channels + channel) * image_height * image_width
                + y * image_width + x;
            float grad = gradient[gradient_index];
            int64_t tile_base = ((static_cast<int64_t>(batch) * token_count + token) * channels + channel) * tile_height * tile_width;
            gpuAtomicAdd(tile_gradient + tile_base + y_low * tile_width + x_low, grad * (1.0f - ly) * (1.0f - lx));
            gpuAtomicAdd(tile_gradient + tile_base + y_low * tile_width + x_high, grad * (1.0f - ly) * lx);
            gpuAtomicAdd(tile_gradient + tile_base + y_high * tile_width + x_low, grad * ly * (1.0f - lx));
            gpuAtomicAdd(tile_gradient + tile_base + y_high * tile_width + x_high, grad * ly * lx);
        }
    }
}


int launch_blocks(int64_t elements) {
    return static_cast<int>(std::min<int64_t>((elements + threads_per_block - 1) / threads_per_block, 65535));
}

}


std::vector<torch::Tensor> quantile_partition_cuda(
    torch::Tensor density,
    torch::Tensor columns_per_row,
    int64_t token_count,
    double density_floor_ratio) {
    TORCH_CHECK(density.is_cuda(), "density must be CUDA");
    TORCH_CHECK(columns_per_row.is_cuda(), "columns_per_row must be CUDA");
    TORCH_CHECK(density.device() == columns_per_row.device(), "density and columns_per_row must share a device");
    TORCH_CHECK(density.scalar_type() == torch::kFloat32, "density must be float32");
    TORCH_CHECK(columns_per_row.scalar_type() == torch::kInt32, "columns_per_row must be int32");
    TORCH_CHECK(density.is_contiguous(), "density must be contiguous");
    TORCH_CHECK(columns_per_row.is_contiguous(), "columns_per_row must be contiguous");
    TORCH_CHECK(density.dim() == 3, "density must have shape [B,H,W]");
    TORCH_CHECK(columns_per_row.dim() == 1, "columns_per_row must be one-dimensional");
    TORCH_CHECK(token_count > 0, "token_count must be positive");
    TORCH_CHECK(density_floor_ratio >= 0.0, "density_floor_ratio cannot be negative");
    int batch_size = density.size(0);
    int height = density.size(1);
    int width = density.size(2);
    int row_count = columns_per_row.size(0);
    TORCH_CHECK(row_count > 0 && row_count <= height, "invalid row count");
    TORCH_CHECK(token_count <= static_cast<int64_t>(height) * width, "too many tokens for density grid");
    c10::cuda::CUDAGuard device_guard(density.device());
    auto long_options = density.options().dtype(torch::kInt64);
    auto boxes_grid = torch::empty({batch_size, token_count, 4}, long_options);
    auto boxes = torch::empty({batch_size, token_count, 4}, density.options());
    auto mask = torch::empty({batch_size, token_count}, density.options().dtype(torch::kBool));
    auto mass = torch::empty({batch_size, token_count}, density.options());
    auto depth = torch::empty({batch_size, token_count}, long_options);
    auto assignment = torch::empty({batch_size, height, width}, long_options);
    size_t shared_memory = static_cast<size_t>(height + width) * sizeof(float)
        + static_cast<size_t>(row_count + 1 + width + 1) * sizeof(int);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(density.get_device()).stream();
    quantile_partition_kernel<<<batch_size, threads_per_block, shared_memory, stream>>>(
        density.data_ptr<float>(),
        columns_per_row.data_ptr<int>(),
        batch_size,
        height,
        width,
        row_count,
        static_cast<int>(token_count),
        static_cast<float>(density_floor_ratio),
        boxes_grid.data_ptr<int64_t>(),
        boxes.data_ptr<float>(),
        mask.data_ptr<bool>(),
        mass.data_ptr<float>(),
        depth.data_ptr<int64_t>(),
        assignment.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {boxes_grid, boxes, mask, mass, depth, assignment};
}


torch::Tensor region_sample_forward_cuda(
    torch::Tensor image,
    torch::Tensor boxes,
    torch::Tensor mask,
    int64_t output_size,
    int64_t sampling_ratio) {
    TORCH_CHECK(image.is_cuda() && boxes.is_cuda() && mask.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(image.device() == boxes.device() && image.device() == mask.device(), "all tensors must share a device");
    TORCH_CHECK(
        image.scalar_type() == torch::kFloat16 || image.scalar_type() == torch::kBFloat16 || image.scalar_type() == torch::kFloat32,
        "image must be float16, bfloat16, or float32");
    TORCH_CHECK(image.is_contiguous() && boxes.is_contiguous() && mask.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(image.dim() == 4, "image must have shape [B,C,H,W]");
    TORCH_CHECK(boxes.dim() == 3 && boxes.size(2) == 4, "boxes must have shape [B,N,4]");
    TORCH_CHECK(mask.sizes() == boxes.sizes().slice(0, 2), "mask must have shape [B,N]");
    TORCH_CHECK(boxes.scalar_type() == torch::kFloat32, "boxes must be float32");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    TORCH_CHECK(output_size > 0 && sampling_ratio > 0, "sampling settings must be positive");
    int batch_size = image.size(0);
    int channels = image.size(1);
    int height = image.size(2);
    int width = image.size(3);
    int token_count = boxes.size(1);
    c10::cuda::CUDAGuard device_guard(image.device());
    auto output = torch::empty({batch_size, token_count, channels, output_size, output_size}, image.options());
    int64_t elements = output.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, image.scalar_type(), "region_sample_forward_cuda", [&] {
        region_sample_forward_kernel<scalar_t><<<launch_blocks(elements), threads_per_block, 0, stream>>>(
            image.data_ptr<scalar_t>(), boxes.data_ptr<float>(), mask.data_ptr<bool>(), batch_size, channels,
            height, width, token_count, static_cast<int>(output_size), static_cast<int>(sampling_ratio),
            output.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor region_sample_backward_cuda(
    torch::Tensor gradient,
    torch::Tensor boxes,
    torch::Tensor mask,
    int64_t image_height,
    int64_t image_width,
    int64_t sampling_ratio) {
    TORCH_CHECK(gradient.is_cuda() && boxes.is_cuda() && mask.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(gradient.device() == boxes.device() && gradient.device() == mask.device(), "all tensors must share a device");
    TORCH_CHECK(gradient.is_contiguous() && boxes.is_contiguous() && mask.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(gradient.dim() == 5, "gradient must have shape [B,N,C,P,P]");
    TORCH_CHECK(image_height > 0 && image_width > 0 && sampling_ratio > 0, "invalid dimensions");
    int batch_size = gradient.size(0);
    int token_count = gradient.size(1);
    int channels = gradient.size(2);
    int output_size = gradient.size(3);
    TORCH_CHECK(gradient.size(4) == output_size, "output tiles must be square");
    c10::cuda::CUDAGuard device_guard(gradient.device());
    auto image_gradient = torch::zeros({batch_size, channels, image_height, image_width}, gradient.options());
    int64_t elements = gradient.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(gradient.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, gradient.scalar_type(), "region_sample_backward_cuda", [&] {
        region_sample_backward_kernel<scalar_t><<<launch_blocks(elements), threads_per_block, 0, stream>>>(
            gradient.data_ptr<scalar_t>(), boxes.data_ptr<float>(), mask.data_ptr<bool>(), batch_size, channels,
            static_cast<int>(image_height), static_cast<int>(image_width), token_count, output_size,
            static_cast<int>(sampling_ratio), image_gradient.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return image_gradient;
}


torch::Tensor render_tiles_forward_cuda(
    torch::Tensor tiles,
    torch::Tensor boxes,
    torch::Tensor assignment,
    int64_t image_height,
    int64_t image_width) {
    TORCH_CHECK(tiles.is_cuda() && boxes.is_cuda() && assignment.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(tiles.device() == boxes.device() && tiles.device() == assignment.device(), "all tensors must share a device");
    TORCH_CHECK(
        tiles.scalar_type() == torch::kFloat16 || tiles.scalar_type() == torch::kBFloat16 || tiles.scalar_type() == torch::kFloat32,
        "tiles must be float16, bfloat16, or float32");
    TORCH_CHECK(tiles.is_contiguous() && boxes.is_contiguous() && assignment.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(tiles.dim() == 5, "tiles must have shape [B,N,C,H,W]");
    TORCH_CHECK(boxes.dim() == 3 && boxes.size(2) == 4, "boxes must have shape [B,N,4]");
    TORCH_CHECK(assignment.dim() == 3, "assignment must have shape [B,H,W]");
    TORCH_CHECK(boxes.scalar_type() == torch::kFloat32, "boxes must be float32");
    TORCH_CHECK(assignment.scalar_type() == torch::kInt64, "assignment must be int64");
    TORCH_CHECK(image_height > 0 && image_width > 0, "image dimensions must be positive");
    int batch_size = tiles.size(0);
    int token_count = tiles.size(1);
    int channels = tiles.size(2);
    int tile_height = tiles.size(3);
    int tile_width = tiles.size(4);
    int assignment_height = assignment.size(1);
    int assignment_width = assignment.size(2);
    TORCH_CHECK(boxes.size(0) == batch_size && boxes.size(1) == token_count, "boxes shape mismatch");
    TORCH_CHECK(assignment.size(0) == batch_size, "assignment batch mismatch");
    c10::cuda::CUDAGuard device_guard(tiles.device());
    auto output = torch::empty(
        {batch_size, channels, image_height, image_width},
        tiles.options().dtype(torch::kFloat32));
    int64_t elements = static_cast<int64_t>(batch_size) * image_height * image_width;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tiles.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, tiles.scalar_type(), "render_tiles_forward_cuda", [&] {
        render_tiles_forward_kernel<scalar_t><<<launch_blocks(elements), threads_per_block, 0, stream>>>(
            tiles.data_ptr<scalar_t>(), boxes.data_ptr<float>(), assignment.data_ptr<int64_t>(), batch_size,
            token_count, channels, tile_height, tile_width, assignment_height, assignment_width,
            static_cast<int>(image_height), static_cast<int>(image_width), output.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor render_tiles_backward_cuda(
    torch::Tensor gradient,
    torch::Tensor boxes,
    torch::Tensor assignment,
    int64_t tile_height,
    int64_t tile_width) {
    TORCH_CHECK(gradient.is_cuda() && boxes.is_cuda() && assignment.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(gradient.device() == boxes.device() && gradient.device() == assignment.device(), "all tensors must share a device");
    TORCH_CHECK(gradient.is_contiguous() && boxes.is_contiguous() && assignment.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(gradient.dim() == 4, "gradient must have shape [B,C,H,W]");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    int batch_size = gradient.size(0);
    int channels = gradient.size(1);
    int image_height = gradient.size(2);
    int image_width = gradient.size(3);
    int token_count = boxes.size(1);
    int assignment_height = assignment.size(1);
    int assignment_width = assignment.size(2);
    c10::cuda::CUDAGuard device_guard(gradient.device());
    TORCH_CHECK(gradient.scalar_type() == torch::kFloat32, "render gradient must be float32");
    auto tile_gradient = torch::zeros({batch_size, token_count, channels, tile_height, tile_width}, gradient.options());
    int64_t elements = static_cast<int64_t>(batch_size) * image_height * image_width;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(gradient.get_device()).stream();
    render_tiles_backward_kernel<<<launch_blocks(elements), threads_per_block, 0, stream>>>(
        gradient.data_ptr<float>(), boxes.data_ptr<float>(), assignment.data_ptr<int64_t>(), batch_size,
        token_count, channels, static_cast<int>(tile_height), static_cast<int>(tile_width), assignment_height,
        assignment_width, image_height, image_width, tile_gradient.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return tile_gradient;
}


torch::Tensor universal_codelength_density_cuda(
    torch::Tensor image,
    int64_t partition_stride,
    int64_t scan_directions,
    double alpha) {
    TORCH_CHECK(image.is_cuda(), "image must be CUDA");
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
    TORCH_CHECK(image.dim() == 4, "image must have shape [B,C,H,W]");
    TORCH_CHECK(image.size(1) == 1 || image.size(1) == 3, "image must have one or three channels");
    TORCH_CHECK(partition_stride > 0, "partition_stride must be positive");
    TORCH_CHECK(scan_directions == 1 || scan_directions == 2 || scan_directions == 4, "invalid scan directions");
    TORCH_CHECK(alpha > 0.0, "alpha must be positive");
    int batch_size = image.size(0);
    int channels = image.size(1);
    int height = image.size(2);
    int width = image.size(3);
    int grid_height = (height + partition_stride - 1) / partition_stride;
    int grid_width = (width + partition_stride - 1) / partition_stride;
    c10::cuda::CUDAGuard device_guard(image.device());
    auto float_options = image.options().dtype(torch::kFloat32);
    auto counts = torch::empty(
        {batch_size, scan_directions, channels, 4, universal_alphabet_size},
        image.options().dtype(torch::kInt32));
    auto contributions = torch::empty(
        {batch_size, scan_directions, channels, height, width},
        float_options);
    auto output = torch::empty({batch_size, 1, grid_height, grid_width}, float_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(image.get_device()).stream();
    universal_codelength_contributions_kernel<<<batch_size * scan_directions, 32, 0, stream>>>(
        image.data_ptr<float>(), counts.data_ptr<int>(), batch_size, channels, height, width,
        static_cast<int>(scan_directions), static_cast<float>(alpha), contributions.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t output_elements = static_cast<int64_t>(batch_size) * grid_height * grid_width;
    universal_codelength_reduce_kernel<<<launch_blocks(output_elements), threads_per_block, 0, stream>>>(
        contributions.data_ptr<float>(), batch_size, channels, height, width, static_cast<int>(scan_directions),
        static_cast<int>(partition_stride), grid_height, grid_width, output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
