// An expert's matrix requantized on the device, for the GPU tier's warm pool (vram-warm-pool.h): the weights through
// ggml's dequantize kernel for their type into a float scratch, a chunk of rows at a time, and from there to Q4_0 or
// Q2_0 by the block rules of quantize_row_q4_0_ref and quantize_row_q2_0_ref. Nothing crosses the link and nothing
// synchronizes: the caller orders readers and overwriters by an event after the last chunk.
#include "../common.cuh"
#include "../convert.cuh"

#include <algorithm>

namespace {
__global__ void quantize_blocks_q4_0(const float * __restrict__ x, block_q4_0 * __restrict__ y, int blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blocks) { return; }
    const float * v = x + int64_t(i) * QK4_0;
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        if (amax < fabsf(v[j])) { amax = fabsf(v[j]); max = v[j]; }
    }
    const float d = max / -8;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = __float2half(d);
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int x0 = min(15, int(int8_t(v[j] * id + 8.5f)));
        const int x1 = min(15, int(int8_t(v[QK4_0 / 2 + j] * id + 8.5f)));
        y[i].qs[j] = uint8_t(x0 | x1 << 4);
    }
}

__global__ void quantize_blocks_q2_0(const float * __restrict__ x, block_q2_0 * __restrict__ y, int blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blocks) { return; }
    const float * v = x + int64_t(i) * QK2_0;
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
    const float id = amax > 0.0f ? 1.0f / amax : 0.0f;
    y[i].d = __float2half(amax);
    for (int b = 0; b < QK2_0 / 4; ++b) {
        int packed = 0;
        for (int j = 0; j < 4; ++j) {
            const int q = max(0, min(3, int(roundf(v[b * 4 + j] * id)) + 1));
            packed |= q << (j * 2);
        }
        y[i].qs[b] = uint8_t(packed);
    }
}
} // namespace

extern "C" bool ggml_cuda_moe_stream_requantize_supports(int source, int target) {
    return source != target && (target == GGML_TYPE_Q4_0 || target == GGML_TYPE_Q2_0) && source >= 0 &&
        source < GGML_TYPE_COUNT && ggml_get_to_fp32_cuda(ggml_type(source)) != nullptr;
}

extern "C" bool ggml_cuda_moe_stream_requantize(int source, const void * src, int64_t rows, int64_t cols, int target,
        void * dst, float * scratch, int64_t scratch_elements, cudaStream_t stream) {
    if (!ggml_cuda_moe_stream_requantize_supports(source, target) || !src || !dst || !scratch || rows < 1 || cols < 1) {
        return false;
    }
    const int blocks_per_row = int(cols / (target == GGML_TYPE_Q4_0 ? QK4_0 : QK2_0));
    const int64_t step = scratch_elements / cols;
    if (step < 1 || cols % (target == GGML_TYPE_Q4_0 ? QK4_0 : QK2_0)) { return false; }
    const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(ggml_type(source));
    const size_t source_row = ggml_row_size(ggml_type(source), cols), target_row = ggml_row_size(ggml_type(target), cols);
    for (int64_t row = 0; row < rows; row += step) {
        const int64_t n = std::min(step, rows - row);
        const int blocks = int(n) * blocks_per_row;
        to_fp32(static_cast<const char *>(src) + size_t(row) * source_row, scratch, n * cols, stream);
        char * out = static_cast<char *>(dst) + size_t(row) * target_row;
        if (target == GGML_TYPE_Q4_0) {
            quantize_blocks_q4_0<<<(blocks + 255) / 256, 256, 0, stream>>>(scratch, reinterpret_cast<block_q4_0 *>(out), blocks);
        } else {
            quantize_blocks_q2_0<<<(blocks + 255) / 256, 256, 0, stream>>>(scratch, reinterpret_cast<block_q2_0 *>(out), blocks);
        }
        if (cudaGetLastError() != cudaSuccess) { return false; }
    }
    return true;
}
