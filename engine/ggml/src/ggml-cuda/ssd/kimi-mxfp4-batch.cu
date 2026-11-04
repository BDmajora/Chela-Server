#include "kimi-mxfp4-batch.h"
#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#endif
#include "ggml-common.h"
#include "gpu-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {
constexpr int token_tile = 16;

__device__ inline int fp4_value(int nibble) {
    const int m = nibble & 7;
    const int magnitude = m < 4 ? m : (2 + (m & 1)) << ((m >> 1) - 1);
    return (nibble & 8) ? -magnitude : magnitude;
}

// The signed 8-bit four-way dot product: CUDA's __dp4a, on AMD the same exact integer sum (as ggml_cuda_dp4a).
__device__ inline int batch_dp4a(int a, int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a), vb = reinterpret_cast<const int8x4_t &>(b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
#else
    return __dp4a(a, b, c);
#endif
}

__device__ inline float fp4_scale(uint8_t exponent) {
    const uint32_t bits = exponent < 2 ? (uint32_t(0x00200000) << exponent) : ((uint32_t(exponent) - 1) << 23);
    return __uint_as_float(bits);
}

// A warp cooperates on the K dimension of one output row. Consecutive lanes
// fetch consecutive MXFP4 blocks; the same weight block serves sixteen tokens.
__global__ void batch_rows_warp(const char * up, const char * gate, const char * down,
                                const block_q8_0 * input, int tokens, int columns, int rows,
                                float * output, bool first_stage) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int first = blockIdx.y * token_tile;
    const int role = first_stage ? blockIdx.z : 0;
    const char * weights = first_stage ? (role ? gate : up) : down;
    const int blocks = columns / QK_MXFP4;
    const auto * x = reinterpret_cast<const block_mxfp4 *>(weights) + size_t(row) * blocks;
    float sums[token_tile] = {};
    for (int ib = lane; ib < blocks; ib += 32) {
        const block_mxfp4 w = x[ib];
        const float scale = fp4_scale(w.e);
        int packed[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int low = 4 * j;
            const int v0 = low < 16 ? fp4_value(w.qs[low] & 15) : fp4_value(w.qs[low - 16] >> 4);
            const int v1 = low + 1 < 16 ? fp4_value(w.qs[low + 1] & 15) : fp4_value(w.qs[low - 15] >> 4);
            const int v2 = low + 2 < 16 ? fp4_value(w.qs[low + 2] & 15) : fp4_value(w.qs[low - 14] >> 4);
            const int v3 = low + 3 < 16 ? fp4_value(w.qs[low + 3] & 15) : fp4_value(w.qs[low - 13] >> 4);
            packed[j] = int(uint32_t(uint8_t(v0)) | (uint32_t(uint8_t(v1)) << 8) |
                            (uint32_t(uint8_t(v2)) << 16) | (uint32_t(uint8_t(v3)) << 24));
        }
#pragma unroll
        for (int t = 0; t < token_tile; ++t) {
            if (first + t >= tokens) break;
            const block_q8_0 y = input[size_t(first + t) * blocks + ib];
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int index = 4 * j;
                const uint32_t q = uint32_t(uint8_t(y.qs[index])) |
                    (uint32_t(uint8_t(y.qs[index + 1])) << 8) |
                    (uint32_t(uint8_t(y.qs[index + 2])) << 16) |
                    (uint32_t(uint8_t(y.qs[index + 3])) << 24);
                dot = batch_dp4a(packed[j], int(q), dot);
            }
            sums[t] += (__half2float(y.d) * scale) * float(dot);
        }
    }
#pragma unroll
    for (int t = 0; t < token_tile; ++t) {
        float sum = sums[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset, 32);
        }
        if (lane == 0 && first + t < tokens) {
            const size_t index = first_stage ? size_t(2 * (first + t) + role) * rows + row
                                             : size_t(first + t) * rows + row;
            output[index] = sum;
        }
    }
}

__global__ void activate(const float * up_gate, float * hidden, int tokens, int rows, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * rows) return;
    const int token = i / rows, row = i % rows;
    const float u = up_gate[size_t(2 * token) * rows + row];
    const float g = up_gate[size_t(2 * token + 1) * rows + row];
    const float gc = fminf(g, limit);
    hidden[i] = (gc / (1.0f + expf(-gc))) * fminf(fmaxf(u, -limit), limit);
}

__device__ inline float half_even(float value) {
    const float rounded = roundf(value);
    return fabsf(rounded - value) == 0.5f ? 2.0f * roundf(value * 0.5f) : rounded;
}

__global__ void quantize(const float * input, block_q8_0 * output) {
    const int block = blockIdx.x, lane = threadIdx.x;
    const float * row = input + size_t(block) * QK8_0;
    __shared__ float inverse;
    if (lane == 0) {
        float maximum = 0;
        for (int j = 0; j < QK8_0; ++j) maximum = fmaxf(maximum, fabsf(row[j]));
        inverse = maximum ? 127.0f / maximum : 0;
        output[block].d = __float2half(maximum / 127.0f);
    }
    __syncthreads();
    output[block].qs[lane] = int8_t(int(half_even(row[lane] * inverse)));
}

__global__ void scatter_weighted(const float * source, float * destination,
                                 const int * token_indices, const float * weights,
                                 int routes, int latent) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= routes * latent) return;
    const int route = index / latent;
    const int column = index % latent;
    destination[size_t(token_indices[route]) * latent + column] += source[index] * weights[route];
}

__global__ void gather_rows(const float * source, float * destination,
                            const int * token_indices, int routes, int latent) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= routes * latent) return;
    destination[index] = source[size_t(token_indices[index / latent]) * latent + index % latent];
}

template <typename T>
bool alloc_device(T ** pointer, size_t bytes) { return cudaMalloc(reinterpret_cast<void **>(pointer), bytes) == cudaSuccess; }

void message(char * error, size_t size, const char * stage, cudaError_t rc) {
    if (error && size) std::snprintf(error, size, "%s: %s", stage, cudaGetErrorString(rc));
}
}

struct kimi_mxfp4_batch {
    int device = -1, latent = 0, hidden = 0, capacity = 0;
    cudaStream_t stream = nullptr;
    char * host_input = nullptr, * device_input = nullptr;
    float * host_output = nullptr, * device_output = nullptr;
    float * up_gate = nullptr, * hidden_float = nullptr;
    block_q8_0 * hidden_q8 = nullptr;
    float * accumulated = nullptr;
    int * host_indices = nullptr, * device_indices = nullptr;
    float * host_weights = nullptr, * device_weights = nullptr;
    int accumulated_tokens = 0;
    uint64_t h2d = 0, d2h = 0;
    size_t input_bytes = 0, output_bytes = 0, scratch_bytes = 0, accumulated_bytes = 0;
};

namespace {
int batch_destroy(kimi_mxfp4_batch * batch) {
    if (!batch) return 1;
    int previous = -1;
    cudaGetDevice(&previous);
    if (previous != batch->device) cudaSetDevice(batch->device);
    bool ok = true;
    if (batch->stream) ok &= cudaStreamSynchronize(batch->stream) == cudaSuccess;
    if (batch->host_input) ok &= cudaFreeHost(batch->host_input) == cudaSuccess;
    if (batch->host_output) ok &= cudaFreeHost(batch->host_output) == cudaSuccess;
    if (batch->host_indices) ok &= cudaFreeHost(batch->host_indices) == cudaSuccess;
    if (batch->host_weights) ok &= cudaFreeHost(batch->host_weights) == cudaSuccess;
    if (batch->device_input) ok &= cudaFree(batch->device_input) == cudaSuccess;
    if (batch->device_output) ok &= cudaFree(batch->device_output) == cudaSuccess;
    if (batch->up_gate) ok &= cudaFree(batch->up_gate) == cudaSuccess;
    if (batch->hidden_float) ok &= cudaFree(batch->hidden_float) == cudaSuccess;
    if (batch->hidden_q8) ok &= cudaFree(batch->hidden_q8) == cudaSuccess;
    if (batch->accumulated) ok &= cudaFree(batch->accumulated) == cudaSuccess;
    if (batch->device_indices) ok &= cudaFree(batch->device_indices) == cudaSuccess;
    if (batch->device_weights) ok &= cudaFree(batch->device_weights) == cudaSuccess;
    if (batch->stream) ok &= cudaStreamDestroy(batch->stream) == cudaSuccess;
    if (previous >= 0 && previous != batch->device) ok &= cudaSetDevice(previous) == cudaSuccess;
    delete batch;
    return ok;
}

kimi_mxfp4_batch * batch_create(int device, int latent, int hidden, int capacity, char * error, size_t size) {
    if (latent < 64 || hidden < 64 || latent > 16384 || hidden > 16384 ||
            latent % 64 || hidden % 64 || capacity < 1 || capacity > 256) {
        if (error && size) std::snprintf(error, size, "invalid MXFP4 batch geometry");
        return nullptr;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    cudaError_t rc = cudaSetDevice(device);
    if (rc != cudaSuccess) { message(error, size, "batch device", rc); return nullptr; }
    auto * batch = new (std::nothrow) kimi_mxfp4_batch;
    if (!batch) return nullptr;
    batch->device = device;
    batch->latent = latent;
    batch->hidden = hidden;
    batch->capacity = capacity;
    batch->input_bytes = size_t(capacity) * (latent / QK8_0) * sizeof(block_q8_0);
    batch->output_bytes = size_t(capacity) * latent * sizeof(float);
    const size_t up_gate_bytes = size_t(2) * capacity * hidden * sizeof(float);
    const size_t hidden_float_bytes = size_t(capacity) * hidden * sizeof(float);
    const size_t hidden_q8_bytes = size_t(capacity) * (hidden / QK8_0) * sizeof(block_q8_0);
    const size_t index_bytes = size_t(capacity) * sizeof(int);
    const size_t weight_bytes = size_t(capacity) * sizeof(float);
    batch->scratch_bytes = batch->input_bytes + batch->output_bytes + up_gate_bytes +
        hidden_float_bytes + hidden_q8_bytes + index_bytes + weight_bytes;
    bool ok = cudaStreamCreateWithFlags(&batch->stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaMallocHost(&batch->host_input, batch->input_bytes) == cudaSuccess &&
        cudaMallocHost(&batch->host_output, batch->output_bytes) == cudaSuccess &&
        cudaMallocHost(&batch->host_indices, index_bytes) == cudaSuccess &&
        cudaMallocHost(&batch->host_weights, weight_bytes) == cudaSuccess &&
        alloc_device(&batch->device_input, batch->input_bytes) &&
        alloc_device(&batch->device_output, batch->output_bytes) &&
        alloc_device(&batch->up_gate, up_gate_bytes) &&
        alloc_device(&batch->hidden_float, hidden_float_bytes) &&
        alloc_device(&batch->hidden_q8, hidden_q8_bytes) &&
        alloc_device(&batch->device_indices, index_bytes) &&
        alloc_device(&batch->device_weights, weight_bytes);
    if (!ok) {
        message(error, size, "batch scratch allocation", cudaGetLastError());
        batch_destroy(batch);
        return nullptr;
    }
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
    return batch;
}

cudaError_t launch_chain(kimi_mxfp4_batch * batch, const void * up, const void * gate,
                         const void * down, const void * input, int tokens, float limit) {
    const size_t input_bytes = size_t(tokens) * batch->input_bytes / batch->capacity;
    std::memcpy(batch->host_input, input, input_bytes);
    cudaError_t rc = cudaMemcpyAsync(batch->device_input, batch->host_input, input_bytes,
                                      cudaMemcpyHostToDevice, batch->stream);
    if (rc == cudaSuccess) batch->h2d += input_bytes;
    if (rc == cudaSuccess) {
        batch_rows_warp<<<dim3(batch->hidden, (tokens + token_tile - 1) / token_tile, 2),
                32, 0, batch->stream>>>(static_cast<const char *>(up), static_cast<const char *>(gate),
                static_cast<const char *>(down), reinterpret_cast<const block_q8_0 *>(batch->device_input),
                tokens, batch->latent, batch->hidden, batch->up_gate, true);
        rc = cudaGetLastError();
    }
    if (rc == cudaSuccess) {
        activate<<<(tokens * batch->hidden + 255) / 256, 256, 0, batch->stream>>>(
                batch->up_gate, batch->hidden_float, tokens, batch->hidden, limit);
        rc = cudaGetLastError();
    }
    if (rc == cudaSuccess) {
        quantize<<<tokens * batch->hidden / QK8_0, QK8_0, 0, batch->stream>>>(
                batch->hidden_float, batch->hidden_q8);
        rc = cudaGetLastError();
    }
    if (rc == cudaSuccess) {
        batch_rows_warp<<<dim3(batch->latent, (tokens + token_tile - 1) / token_tile, 1),
                32, 0, batch->stream>>>(static_cast<const char *>(up), static_cast<const char *>(gate),
                static_cast<const char *>(down), batch->hidden_q8, tokens, batch->hidden, batch->latent,
                batch->device_output, false);
        rc = cudaGetLastError();
    }
    return rc;
}

int batch_compute(kimi_mxfp4_batch * batch, const void * up, const void * gate, const void * down,
                  const void * input, int tokens, float limit, float * output, char * error, size_t size) {
    if (!batch || !up || !gate || !down || !input || !output || tokens < 1 ||
            !std::isfinite(limit) || limit <= 0) {
        if (error && size) std::snprintf(error, size, "invalid MXFP4 batch input");
        return 0;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    cudaError_t rc = cudaSetDevice(batch->device);
    if (rc != cudaSuccess) { message(error, size, "batch device", rc); return 0; }
    const size_t input_stride = batch->input_bytes / batch->capacity;
    for (int first = 0; first < tokens && rc == cudaSuccess; first += batch->capacity) {
        const int count = std::min(batch->capacity, tokens - first);
        rc = launch_chain(batch, up, gate, down,
                          static_cast<const char *>(input) + size_t(first) * input_stride, count, limit);
        const size_t bytes = size_t(count) * batch->latent * sizeof(float);
        if (rc == cudaSuccess) rc = cudaMemcpyAsync(batch->host_output, batch->device_output,
                                                     bytes, cudaMemcpyDeviceToHost, batch->stream);
        if (rc == cudaSuccess) {
            rc = cudaStreamSynchronize(batch->stream);
            batch->d2h += bytes;
        }
        if (rc == cudaSuccess) std::memcpy(output + size_t(first) * batch->latent, batch->host_output, bytes);
    }
    if (rc != cudaSuccess) message(error, size, "batch compute", rc);
    if (previous >= 0 && previous != batch->device) cudaSetDevice(previous);
    return rc == cudaSuccess;
}

int batch_begin_accumulate(kimi_mxfp4_batch * batch, int tokens, char * error, size_t size) {
    if (!batch || tokens < 1) {
        if (error && size) std::snprintf(error, size, "invalid MXFP4 accumulation geometry");
        return 0;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    cudaError_t rc = cudaSetDevice(batch->device);
    const size_t bytes = size_t(tokens) * batch->latent * sizeof(float);
    if (rc == cudaSuccess && batch->accumulated_bytes < bytes) {
        if (batch->accumulated) {
            rc = cudaFree(batch->accumulated);
            if (rc == cudaSuccess) { batch->accumulated = nullptr; batch->accumulated_bytes = 0; }
        }
        if (rc == cudaSuccess) {
            rc = cudaMalloc(&batch->accumulated, bytes);
            if (rc == cudaSuccess) batch->accumulated_bytes = bytes;
        }
    }
    if (rc == cudaSuccess) rc = cudaMemsetAsync(batch->accumulated, 0, bytes, batch->stream);
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(batch->stream);
    if (rc == cudaSuccess) { batch->accumulated_tokens = tokens; batch->h2d = batch->d2h = 0; }
    else message(error, size, "batch accumulation begin", rc);
    if (previous >= 0 && previous != batch->device) cudaSetDevice(previous);
    return rc == cudaSuccess;
}

int batch_compute_accumulate(kimi_mxfp4_batch * batch, const void * up, const void * gate, const void * down,
                             const void * input, const int * token_indices, const float * weights,
                             int routes, float limit, char * error, size_t size) {
    if (!batch || !up || !gate || !down || !input || !token_indices || !weights ||
            routes < 1 || routes > batch->accumulated_tokens || !batch->accumulated_tokens ||
            !std::isfinite(limit) || limit <= 0) {
        if (error && size) std::snprintf(error, size, "invalid MXFP4 accumulated batch");
        return 0;
    }
    for (int i = 0; i < routes; ++i) {
        if (token_indices[i] < 0 || token_indices[i] >= batch->accumulated_tokens ||
                (i && token_indices[i] <= token_indices[i - 1]) ||
                !std::isfinite(weights[i]) || weights[i] < 0) {
            if (error && size) std::snprintf(error, size, "invalid MXFP4 route mapping");
            return 0;
        }
    }
    int previous = -1;
    cudaGetDevice(&previous);
    cudaError_t rc = cudaSetDevice(batch->device);
    if (rc != cudaSuccess) { message(error, size, "batch device", rc); return 0; }
    const size_t input_stride = batch->input_bytes / batch->capacity;
    for (int first = 0; first < routes && rc == cudaSuccess; first += batch->capacity) {
        const int count = std::min(batch->capacity, routes - first);
        rc = launch_chain(batch, up, gate, down,
                          static_cast<const char *>(input) + size_t(first) * input_stride, count, limit);
        const size_t indices_bytes = size_t(count) * sizeof(int);
        const size_t weights_bytes = size_t(count) * sizeof(float);
        if (rc == cudaSuccess) {
            std::memcpy(batch->host_indices, token_indices + first, indices_bytes);
            std::memcpy(batch->host_weights, weights + first, weights_bytes);
            rc = cudaMemcpyAsync(batch->device_indices, batch->host_indices, indices_bytes,
                                 cudaMemcpyHostToDevice, batch->stream);
        }
        if (rc == cudaSuccess) rc = cudaMemcpyAsync(batch->device_weights, batch->host_weights,
                                  weights_bytes, cudaMemcpyHostToDevice, batch->stream);
        if (rc == cudaSuccess) {
            batch->h2d += indices_bytes + weights_bytes;
            const int elements = count * batch->latent;
            scatter_weighted<<<(elements + 255) / 256, 256, 0, batch->stream>>>(
                batch->device_output, batch->accumulated, batch->device_indices,
                batch->device_weights, count, batch->latent);
            rc = cudaGetLastError();
        }
        if (rc == cudaSuccess) rc = cudaStreamSynchronize(batch->stream);
    }
    if (rc != cudaSuccess) message(error, size, "batch accumulated compute", rc);
    if (previous >= 0 && previous != batch->device) cudaSetDevice(previous);
    return rc == cudaSuccess;
}

int batch_finish_accumulate(kimi_mxfp4_batch * batch, const int * token_indices, int count,
                            float * output, char * error, size_t size) {
    if (!batch || !token_indices || !output || count < 0 || count > batch->accumulated_tokens ||
            !batch->accumulated_tokens) {
        if (error && size) std::snprintf(error, size, "invalid MXFP4 accumulation output");
        return 0;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    cudaError_t rc = cudaSetDevice(batch->device);
    if (rc != cudaSuccess) { message(error, size, "batch device", rc); return 0; }
    for (int first = 0; first < count && rc == cudaSuccess; first += batch->capacity) {
        const int rows = std::min(batch->capacity, count - first);
        const size_t indices_bytes = size_t(rows) * sizeof(int);
        const size_t output_bytes = size_t(rows) * batch->latent * sizeof(float);
        for (int i = 0; i < rows; ++i) {
            const int token = token_indices[first + i];
            if (token < 0 || token >= batch->accumulated_tokens) {
                if (error && size) std::snprintf(error, size, "invalid MXFP4 output token");
                rc = cudaErrorInvalidValue; break;
            }
            batch->host_indices[i] = token;
        }
        if (rc == cudaSuccess) rc = cudaMemcpyAsync(batch->device_indices, batch->host_indices,
                                                     indices_bytes, cudaMemcpyHostToDevice, batch->stream);
        if (rc == cudaSuccess) {
            batch->h2d += indices_bytes;
            const int elements = rows * batch->latent;
            gather_rows<<<(elements + 255) / 256, 256, 0, batch->stream>>>(
                batch->accumulated, batch->device_output, batch->device_indices, rows, batch->latent);
            rc = cudaGetLastError();
        }
        if (rc == cudaSuccess) rc = cudaMemcpyAsync(batch->host_output, batch->device_output,
                                                     output_bytes, cudaMemcpyDeviceToHost, batch->stream);
        if (rc == cudaSuccess) rc = cudaStreamSynchronize(batch->stream);
        if (rc == cudaSuccess) {
            batch->d2h += output_bytes;
            for (int i = 0; i < rows; ++i) {
                std::memcpy(output + size_t(batch->host_indices[i]) * batch->latent,
                            batch->host_output + size_t(i) * batch->latent,
                            size_t(batch->latent) * sizeof(float));
            }
        }
    }
    batch->accumulated_tokens = 0;
    if (rc != cudaSuccess) message(error, size, "batch accumulation finish", rc);
    if (previous >= 0 && previous != batch->device) cudaSetDevice(previous);
    return rc == cudaSuccess;
}

int batch_release_accumulate(kimi_mxfp4_batch * batch, char * error, size_t size) {
    if (!batch) {
        if (error && size) std::snprintf(error, size, "null MXFP4 batch");
        return 0;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    cudaError_t rc = cudaSetDevice(batch->device);
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(batch->stream);
    if (rc == cudaSuccess && batch->accumulated) rc = cudaFree(batch->accumulated);
    if (rc == cudaSuccess) {
        batch->accumulated = nullptr;
        batch->accumulated_bytes = 0;
        batch->accumulated_tokens = 0;
    } else message(error, size, "batch accumulation release", rc);
    if (previous >= 0 && previous != batch->device) cudaSetDevice(previous);
    return rc == cudaSuccess;
}

size_t batch_device_bytes(kimi_mxfp4_batch * batch) { return batch ? batch->scratch_bytes + batch->accumulated_bytes : 0; }
size_t batch_host_bytes(kimi_mxfp4_batch * batch) { return batch ? batch->input_bytes + batch->output_bytes + size_t(batch->capacity) * (sizeof(int) + sizeof(float)) : 0; }
uint64_t batch_h2d_bytes(kimi_mxfp4_batch * batch) { return batch ? batch->h2d : 0; }
uint64_t batch_d2h_bytes(kimi_mxfp4_batch * batch) { return batch ? batch->d2h : 0; }
const kimi_mxfp4_batch_api batch_api = {3, sizeof(kimi_mxfp4_batch_api),
    batch_create, batch_compute, batch_destroy, batch_device_bytes, batch_host_bytes,
    batch_begin_accumulate, batch_compute_accumulate, batch_finish_accumulate,
    batch_release_accumulate, batch_h2d_bytes, batch_d2h_bytes};
}

extern "C" __attribute__((visibility("default")))
const kimi_mxfp4_batch_api * kimi_mxfp4_batch_get_api() { return &batch_api; }
