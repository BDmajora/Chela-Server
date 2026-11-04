// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-cuda.h"
#include "gpu-runtime.h"

// Private normal-link boundaries inside ggml-cuda. Definitions are backend
// kernels/services in this same library, never runtime-discovered DSOs.
extern "C" {
bool ggml_cuda_moe_stream_mmvq_dev(int, const void *, int64_t, int64_t, size_t,
    const float *, void *, float *, cudaStream_t);
bool ggml_cuda_moe_stream_mmvq_supports(int);
bool ggml_cuda_moe_stream_mm_supports(int);
void * ggml_cuda_moe_stream_mm_create(int);
void ggml_cuda_moe_stream_mm_destroy(void *);
void ggml_cuda_moe_stream_mm_release(void *);
bool ggml_cuda_moe_stream_mm_reserve(void *, int, int64_t, int64_t, int64_t);
size_t ggml_cuda_moe_stream_mm_operand_bytes(void *, int, int64_t, int64_t);
bool ggml_cuda_moe_stream_mm_quantize(void *, int, const float *, const int32_t *, int64_t, int64_t, int64_t, void *, cudaStream_t);
bool ggml_cuda_moe_stream_mm(void *, int, const void *, int64_t, int64_t, const void *, int64_t, float *, cudaStream_t);
size_t ggml_cuda_moe_stream_mm_device_bytes(void *);
bool ggml_cuda_moe_stream_requantize_supports(int, int);
bool ggml_cuda_moe_stream_requantize(int, const void *, int64_t, int64_t, int, void *, float *, int64_t, cudaStream_t);
}
