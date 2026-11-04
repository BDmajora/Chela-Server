// MXFP4 x Q8_0 matvec component for the CPU sink VRAM tier. It follows the
// accepted Q3_K parity component (kimi-q3k-parity.cu) and is built the same way,
// outside the backend CUDA flags with --fmad=false --ftz=false
// (build_mxfp4_component.sh). Every floating-point operation is explicit.
#include "ggml-kimi-mxfp4-parity.h"
#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#endif
#include "ggml-common.h"
#include "gpu-runtime.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {
struct device_job {
    const char * weights;
    size_t row_stride, q8_offset, output_offset;
    int columns, rows;
};

// kvalues_mxfp4 of ggml-common.h: doubled E2M1 magnitudes, sign in bit 3.
__device__ inline int mxfp4_value(int nibble) {
    int magnitude = 0;
    switch (nibble & 7) {
        case 0: magnitude = 0; break;
        case 1: magnitude = 1; break;
        case 2: magnitude = 2; break;
        case 3: magnitude = 3; break;
        case 4: magnitude = 4; break;
        case 5: magnitude = 6; break;
        case 6: magnitude = 8; break;
        default: magnitude = 12; break;
    }
    return (nibble & 8) ? -magnitude : magnitude;
}

// ggml_e8m0_to_fp32_half: 2^(e-128), with the denormal patterns for e < 2.
__device__ inline float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? (uint32_t(0x00200000) << e) : ((uint32_t(e) - 1) << 23);
    return __uint_as_float(bits);
}

// One thread per (row, lane); lane l covers block elements 4l..4l+3 exactly as
// one int32 lane of VPDPBSSD over the 256-bit q4/q8 byte vectors (low nibbles
// of qs[0..15] are elements 0..15, high nibbles elements 16..31). The AVX2 loop
// fuses scale*lane into accum1 for even and accum2 for odd blocks; the result is
// hsum_float_8(accum1 + accum2). Blocks are paired, so columns are a multiple of
// 64 and no scalar tail exists.
__global__ void mxfp4_avx2(const device_job * jobs, const char * q8, float * output) {
    const device_job job = jobs[blockIdx.y];
    const int lane = threadIdx.x & 7;
    const int row = blockIdx.x * 16 + threadIdx.x / 8;
    if (row >= job.rows) return;
    const auto * x = reinterpret_cast<const block_mxfp4 *>(job.weights + row * job.row_stride);
    const auto * y = reinterpret_cast<const block_q8_0 *>(q8 + job.q8_offset);
    float even = 0.0f, odd = 0.0f;
    const int blocks = job.columns / QK_MXFP4;
    for (int ib = 0; ib < blocks; ++ib) {
        int dot = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int p = 4 * lane + k;
            const int nibble = p < 16 ? (x[ib].qs[p] & 15) : (x[ib].qs[p - 16] >> 4);
            dot += mxfp4_value(nibble) * int(y[ib].qs[p]);
        }
        const float d = __fmul_rn(__half2float(y[ib].d), e8m0_half(x[ib].e));
        if (ib & 1) {
            odd = __fmaf_rn(d, __int2float_rn(dot), odd);
        } else {
            even = __fmaf_rn(d, __int2float_rn(dot), even);
        }
    }
    const float acc = __fadd_rn(even, odd);
    const unsigned mask = 0xffu << (8 * ((threadIdx.x & 31) / 8));
    const float a = __fadd_rn(__shfl_sync(mask, acc, 4, 8), __shfl_sync(mask, acc, 0, 8));
    const float b = __fadd_rn(__shfl_sync(mask, acc, 6, 8), __shfl_sync(mask, acc, 2, 8));
    const float c = __fadd_rn(__shfl_sync(mask, acc, 5, 8), __shfl_sync(mask, acc, 1, 8));
    const float e = __fadd_rn(__shfl_sync(mask, acc, 7, 8), __shfl_sync(mask, acc, 3, 8));
    const float value = __fadd_rn(__fadd_rn(a, b), __fadd_rn(c, e));
    if (lane == 0) output[job.output_offset + row] = value;
}

struct device_scope {
    int previous = -1;
    cudaError_t status;
    explicit device_scope(int target) {
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != target) status = cudaSetDevice(target);
    }
    ~device_scope() { if (previous >= 0) cudaSetDevice(previous); }
};
}

// Weights may sit in pageable host memory when the device reads it in place through the host page
// tables (ATS: GB10, Grace Hopper); otherwise they must be device memory of this handle. HMM also
// reports pageable access, but it migrates the pages to the device on first touch.
static bool kimi_mxfp4_weights_readable(const cudaPointerAttributes & attr, int device) {
    if (attr.type == cudaMemoryTypeDevice) { return attr.device == device; }
    static std::atomic<int> in_place{-1}; // one device per process
    int value = in_place.load(std::memory_order_relaxed);
    if (value < 0) {
        int flag = 0;
        value = cudaDeviceGetAttribute(&flag, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device) == cudaSuccess && flag ? 1 : 0;
        in_place.store(value, std::memory_order_relaxed);
    }
    return value == 1;
}

struct kimi_mxfp4_gpu {
    kimi_mxfp4_gpu_config config{};
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    device_job * h_jobs = nullptr, * d_jobs = nullptr;
    char * h_q8 = nullptr, * d_q8 = nullptr;
    float * h_output = nullptr, * d_output = nullptr;
    size_t job_bytes = 0, q8_bytes = 0, output_bytes = 0;
    int njobs = 0;
    bool busy = false, failed = false;
    char message[256]{};
    // Optional timing marks, enabled only through set_timing of the chain extension
    // (kimi-mxfp4-chain.cu): timing events recorded around the copies and kernels of a
    // submission and summed when it completes. With timing off mark() returns at once
    // and the submission and kernel below are unchanged.
    bool timing = false;
    int timing_stage = 0, nmarks = 0;
    cudaEvent_t marks[8]{};
    int mark_slot[8]{};
    double stage_ms[8]{}, h2d_call_us = 0;
    uint64_t timed_submits = 0;
    void mark(int slot) {
        if (!timing || nmarks >= 8 || !marks[nmarks]) return;
        if (cudaEventRecord(marks[nmarks], stream) == cudaSuccess) mark_slot[nmarks++] = slot;
    }
    void collect() {
        for (int i = 1; timing && i < nmarks; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, marks[i - 1], marks[i]) == cudaSuccess) stage_ms[mark_slot[i]] += ms;
        }
        if (timing && nmarks > 1) ++timed_submits;
        nmarks = 0;
    }
    bool check(cudaError_t e, const char * what) {
        if (e == cudaSuccess) return true;
        std::snprintf(message, sizeof(message), "%s: %s", what, cudaGetErrorString(e));
        failed = true;
        return false;
    }
    int invalid(const char * why) { std::snprintf(message, sizeof(message), "%s", why); return 0; }
};

namespace {
int destroy(kimi_mxfp4_gpu * c) {
    if (!c) return 1;
    device_scope device(c->config.device);
    bool ok = device.status == cudaSuccess;
    if (ok) {
        if (c->stream) ok = cudaStreamSynchronize(c->stream) == cudaSuccess && ok;
        if (c->event) ok = cudaEventDestroy(c->event) == cudaSuccess && ok;
        for (auto & m : c->marks) if (m) ok = cudaEventDestroy(m) == cudaSuccess && ok;
        if (c->d_jobs) ok = cudaFree(c->d_jobs) == cudaSuccess && ok;
        if (c->d_q8) ok = cudaFree(c->d_q8) == cudaSuccess && ok;
        if (c->d_output) ok = cudaFree(c->d_output) == cudaSuccess && ok;
        if (c->h_jobs) ok = cudaFreeHost(c->h_jobs) == cudaSuccess && ok;
        if (c->h_q8) ok = cudaFreeHost(c->h_q8) == cudaSuccess && ok;
        if (c->h_output) ok = cudaFreeHost(c->h_output) == cudaSuccess && ok;
        if (c->stream) ok = cudaStreamDestroy(c->stream) == cudaSuccess && ok;
    }
    delete c;
    return ok;
}

kimi_mxfp4_gpu * create(const kimi_mxfp4_gpu_config * config, char * error, size_t size) {
    auto fail = [&](const char * why) -> kimi_mxfp4_gpu * {
        if (error && size) std::snprintf(error, size, "%s", why);
        return nullptr;
    };
    if (!config || config->device < 0 || config->max_jobs < 1 || config->max_jobs > 64 ||
            config->max_columns < 64 || config->max_columns > 32768 || config->max_columns % 64 ||
            config->max_rows < 1 || config->max_rows > 32768) return fail("invalid MXFP4 parity capacity");
    device_scope device(config->device);
    if (device.status != cudaSuccess) return fail(cudaGetErrorString(device.status));
    auto * c = new (std::nothrow) kimi_mxfp4_gpu;
    if (!c) return fail("MXFP4 parity handle allocation failed");
    c->config = *config;
    c->job_bytes = config->max_jobs * sizeof(device_job);
    c->q8_bytes = size_t(config->max_jobs) * (config->max_columns / QK8_0) * sizeof(block_q8_0);
    c->output_bytes = size_t(config->max_jobs) * config->max_rows * sizeof(float);
    if (!c->check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream") ||
            !c->check(cudaEventCreateWithFlags(&c->event, cudaEventDisableTiming), "event") ||
            !c->check(cudaMallocHost(&c->h_jobs, c->job_bytes), "host jobs") ||
            !c->check(cudaMallocHost(&c->h_q8, c->q8_bytes), "host Q8_0") ||
            !c->check(cudaMallocHost(&c->h_output, c->output_bytes), "host output") ||
            !c->check(cudaMalloc(&c->d_jobs, c->job_bytes), "device jobs") ||
            !c->check(cudaMalloc(&c->d_q8, c->q8_bytes), "device Q8_0") ||
            !c->check(cudaMalloc(&c->d_output, c->output_bytes), "device output")) {
        fail(c->message); destroy(c); return nullptr;
    }
    if (error && size) error[0] = 0;
    return c;
}

int query(kimi_mxfp4_gpu * c) {
    if (!c || c->failed) return -1;
    if (!c->busy) return 1;
    device_scope device(c->config.device);
    if (!c->check(device.status, "query device")) return -1;
    const auto rc = cudaEventQuery(c->event);
    if (rc == cudaErrorNotReady) return 0;
    if (!c->check(rc, "query event")) return -1;
    c->busy = false;
    c->collect();
    return 1;
}

int wait(kimi_mxfp4_gpu * c) {
    if (!c || c->failed) return 0;
    if (!c->busy) return 1;
    device_scope device(c->config.device);
    if (!c->check(device.status, "wait device") || !c->check(cudaEventSynchronize(c->event), "wait event")) return 0;
    c->busy = false;
    c->collect();
    return 1;
}

int submit(kimi_mxfp4_gpu * c, const kimi_mxfp4_gpu_job * jobs, int n, const void * q8, size_t bytes) {
    if (!c || c->failed) return 0;
    if (c->busy) return c->invalid("consume the previous event before submitting");
    if (!jobs || !q8 || n < 1 || n > c->config.max_jobs || bytes > c->q8_bytes)
        return c->invalid("invalid jobs or Q8_0 capacity");
    device_scope device(c->config.device);
    if (!c->check(device.status, "submit device")) return 0;
    size_t out = 0;
    int max_rows = 0;
    // Validate the entire batch before overwriting the previous result layout.
    for (int i = 0; i < n; ++i) {
        const auto & j = jobs[i];
        const size_t row_bytes = size_t(j.columns / QK_MXFP4) * sizeof(block_mxfp4);
        const size_t q8_row = size_t(j.columns / QK8_0) * sizeof(block_q8_0);
        if (!j.weights || j.columns < 64 || j.columns > c->config.max_columns || j.columns % 64 ||
                j.rows < 1 || j.rows > c->config.max_rows || j.q8_offset % alignof(block_q8_0) ||
                j.q8_offset > bytes || q8_row > bytes - j.q8_offset ||
                j.row_stride < row_bytes || j.weight_bytes < row_bytes ||
                (j.rows > 1 && j.row_stride > (j.weight_bytes - row_bytes) / (j.rows - 1)))
            return c->invalid("unsupported MXFP4 parity job layout/range");
        cudaPointerAttributes attr{};
        if (!c->check(cudaPointerGetAttributes(&attr, j.weights), "weight pointer")) return 0;
        if (!kimi_mxfp4_weights_readable(attr, c->config.device))
            return c->invalid("weights must reside on the handle device or in host memory it reads");
    }
    for (int i = 0; i < n; ++i) {
        const auto & j = jobs[i];
        c->h_jobs[i] = {static_cast<const char *>(j.weights), j.row_stride, j.q8_offset, out, j.columns, j.rows};
        out += j.rows;
        max_rows = std::max(max_rows, j.rows);
    }
    c->njobs = n;
    std::memcpy(c->h_q8, q8, bytes);
    c->nmarks = 0;
    c->mark(0);
    const auto h2d_start = c->timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!c->check(cudaMemcpyAsync(c->d_jobs, c->h_jobs, n * sizeof(device_job), cudaMemcpyHostToDevice, c->stream), "jobs H2D") ||
            !c->check(cudaMemcpyAsync(c->d_q8, c->h_q8, bytes, cudaMemcpyHostToDevice, c->stream), "Q8_0 H2D")) return 0;
    if (c->timing) c->h2d_call_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h2d_start).count();
    c->mark(1);
    mxfp4_avx2<<<dim3((max_rows + 15) / 16, n), 128, 0, c->stream>>>(c->d_jobs, c->d_q8, c->d_output);
    if (!c->check(cudaGetLastError(), "MXFP4 kernel")) return 0;
    c->mark(c->timing_stage == 2 ? 5 : 2);
    if (!c->check(cudaMemcpyAsync(c->h_output, c->d_output, out * sizeof(float), cudaMemcpyDeviceToHost, c->stream), "result D2H")) return 0;
    c->mark(6);
    if (!c->check(cudaEventRecord(c->event, c->stream), "completion event")) return 0;
    c->busy = true;
    return 1;
}

const float * result(kimi_mxfp4_gpu * c, int job, int * rows) {
    if (query(c) != 1 || job < 0 || job >= c->njobs) return nullptr;
    if (rows) *rows = c->h_jobs[job].rows;
    return c->h_output + c->h_jobs[job].output_offset;
}
const char * error(kimi_mxfp4_gpu * c) { return c ? c->message : "null MXFP4 parity handle"; }
size_t host_bytes(kimi_mxfp4_gpu * c) { return c ? c->job_bytes + c->q8_bytes + c->output_bytes : 0; }
size_t device_bytes(kimi_mxfp4_gpu * c) { return host_bytes(c); }
int upload(kimi_mxfp4_gpu * c, const void * src, size_t bytes, void ** dst) {
    if (!c || c->failed) return 0;
    if (c->busy || !src || !dst || !bytes) return c->invalid("invalid upload or handle busy");
    device_scope device(c->config.device);
    if (!c->check(device.status, "upload device")) return 0;
    *dst = nullptr;
    if (!c->check(cudaMalloc(dst, bytes), "weight allocation")) return 0;
    if (!c->check(cudaMemcpyAsync(*dst, src, bytes, cudaMemcpyHostToDevice, c->stream), "weight upload") ||
            !c->check(cudaStreamSynchronize(c->stream), "weight upload completion")) {
        cudaFree(*dst); *dst = nullptr; return 0;
    }
    return 1;
}
int release_weights(kimi_mxfp4_gpu * c, void * ptr) {
    if (!c || c->busy) return 0;
    device_scope device(c->config.device);
    return c->check(device.status, "release device") && c->check(cudaFree(ptr), "release weights");
}
const kimi_mxfp4_gpu_api_v1 api = {1, sizeof(kimi_mxfp4_gpu_job), create, submit, query, wait, result, error,
    destroy, host_bytes, device_bytes, upload, release_weights};
}

extern "C" KIMI_MXFP4_EXPORT const kimi_mxfp4_gpu_api_v1 * kimi_mxfp4_gpu_get_api_v1() { return &api; }
