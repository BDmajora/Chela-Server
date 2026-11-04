// SPDX-License-Identifier: Apache-2.0
// The warm pool's conversion against ggml's CPU quantizers on a real device: an expert's matrix in each source format,
// requantized by the kernels of expert-requantize.cu, block by block against the bytes quantize_q4_0 and quantize_q2_0
// make from the same floats. A step of one in a code is allowed where the device's fast-math reciprocal rounds the other
// way; the scale must agree. The chunked path runs with a scratch smaller than the matrix.
#include "ggml.h"
#include "ggml-cuda.h"
#include "gpu-runtime.h"
#include "backend-functions.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t rc, const char * what) {
    if (rc != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc)); }
}

std::vector<float> matrix(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> out(size_t(rows) * size_t(cols));
    for (auto & value : out) { value = normal(rng) * 0.25f; }
    for (int block = 0; block < cols / 32; block += 7) { // blocks of zeros: both scales' division by zero
        for (int j = 0; j < 32; ++j) { out[size_t(block) * 32 + size_t(j)] = 0.0f; }
    }
    return out;
}

// Decoded bytes of `type` from floats as the source format holds them.
std::vector<float> through(ggml_type type, const std::vector<float> & floats, int rows, int cols) {
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(floats.size());
        ggml_fp32_to_fp16_row(floats.data(), half.data(), int64_t(floats.size()));
        std::vector<float> out(floats.size());
        ggml_fp16_to_fp32_row(half.data(), out.data(), int64_t(floats.size()));
        return out;
    }
    std::vector<uint8_t> packed(ggml_row_size(type, cols) * size_t(rows));
    quantize_q8_0(floats.data(), packed.data(), rows, cols, nullptr);
    std::vector<float> out(floats.size());
    dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(packed.data()), out.data(), int64_t(floats.size()));
    return out;
}

std::vector<uint8_t> source_bytes(ggml_type type, const std::vector<float> & floats, int rows, int cols) {
    std::vector<uint8_t> packed(ggml_row_size(type, cols) * size_t(rows));
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(floats.data(), reinterpret_cast<ggml_fp16_t *>(packed.data()), int64_t(floats.size()));
    } else {
        quantize_q8_0(floats.data(), packed.data(), rows, cols, nullptr);
    }
    return packed;
}

// Codes of a block of each format, in the elements' order
void codes(ggml_type type, const uint8_t * block, int * out) {
    const uint8_t * qs = block + sizeof(ggml_fp16_t);
    for (int j = 0; j < 32; ++j) {
        out[j] = type == GGML_TYPE_Q4_0 ? (j < 16 ? qs[j] & 15 : qs[j - 16] >> 4) : (qs[j / 4] >> ((j % 4) * 2)) & 3;
    }
}

void run(ggml_type source, ggml_type target, int rows, int cols, int64_t scratch_elements, cudaStream_t stream, size_t & blocks) {
    if (!ggml_cuda_moe_stream_requantize_supports(source, target)) {
        throw std::runtime_error(std::string("no conversion from ") + ggml_type_name(source));
    }
    const auto floats = matrix(rows, cols, 20261103u + unsigned(source) * 31 + unsigned(target));
    const auto packed = source_bytes(source, floats, rows, cols);
    const auto decoded = through(source, floats, rows, cols);
    const size_t target_bytes = ggml_row_size(target, cols) * size_t(rows);
    std::vector<uint8_t> expected(target_bytes), got(target_bytes);
    (target == GGML_TYPE_Q4_0 ? quantize_q4_0 : quantize_q2_0)(decoded.data(), expected.data(), rows, cols, nullptr);
    void * device_source = nullptr, * device_target = nullptr;
    float * scratch = nullptr;
    check(cudaMalloc(&device_source, packed.size()), "cudaMalloc source");
    check(cudaMalloc(&device_target, target_bytes), "cudaMalloc target");
    check(cudaMalloc(reinterpret_cast<void **>(&scratch), size_t(scratch_elements) * sizeof(float)), "cudaMalloc scratch");
    check(cudaMemcpy(device_source, packed.data(), packed.size(), cudaMemcpyHostToDevice), "upload");
    if (!ggml_cuda_moe_stream_requantize(source, device_source, rows, cols, target, device_target, scratch, scratch_elements, stream)) {
        throw std::runtime_error("the conversion refused its arguments");
    }
    check(cudaStreamSynchronize(stream), "conversion");
    check(cudaMemcpy(got.data(), device_target, target_bytes, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(scratch), "free"); check(cudaFree(device_target), "free"); check(cudaFree(device_source), "free");
    const size_t block_bytes = ggml_type_size(target);
    int steps = 0;
    for (size_t b = 0; b < target_bytes / block_bytes; ++b, ++blocks) {
        const uint8_t * a = got.data() + b * block_bytes, * c = expected.data() + b * block_bytes;
        if (std::memcmp(a, c, sizeof(ggml_fp16_t))) {
            throw std::runtime_error("block " + std::to_string(b) + ": the scale differs from the CPU's");
        }
        int x[32], y[32];
        codes(target, a, x);
        codes(target, c, y);
        for (int j = 0; j < 32; ++j) {
            if (std::abs(x[j] - y[j]) > 1) { throw std::runtime_error("block " + std::to_string(b) + ": a code differs by more than a step"); }
            steps += x[j] != y[j];
        }
    }
    std::printf("%s to %s rows=%d cols=%d scratch_rows=%lld: codes off by one %d of %zu\n", ggml_type_name(source),
        ggml_type_name(target), rows, cols, (long long) (scratch_elements / cols), steps, target_bytes / block_bytes * 32);
}
} // namespace

int main() {
    try {
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        size_t blocks = 0;
        for (ggml_type source : { GGML_TYPE_Q8_0, GGML_TYPE_F16 }) {
            for (ggml_type target : { GGML_TYPE_Q4_0, GGML_TYPE_Q2_0 }) {
                run(source, target, 512, 2048, int64_t(512) * 2048, stream, blocks); // the matrix in one chunk
                run(source, target, 1024, 256, int64_t(100) * 256, stream, blocks);  // in chunks, a short last one
            }
        }
        check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        std::printf("VRAM_WARM_PASS blocks=%zu\n", blocks);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "VRAM_WARM_FAIL: %s\n", e.what());
        return 1;
    }
}
