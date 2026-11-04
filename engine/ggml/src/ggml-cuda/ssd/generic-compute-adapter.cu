#include "vram-compute-api.h"
#include "vram-activation.cuh"
#include "ggml.h"
#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#endif
#include "../../ggml-common.h"

#include "gpu-runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backend-functions.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using mmvq_fn = bool (*)(int, const void *, int64_t, int64_t, size_t,
                         const float *, void *, float *, cudaStream_t);
using clock_type = std::chrono::steady_clock;

void cuda_check(cudaError_t rc, const char * operation) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
    }
}

size_t q8_1_bytes(int columns) {
    constexpr int row_padding = 512;
    const size_t padded = (size_t(columns) + row_padding - 1) / row_padding * row_padding;
    return padded / QK8_1 * sizeof(block_q8_1);
}

__global__ void generic_activate(const float * up_gate, float * hidden, int hits, int rows,
        int kind, float situ_beta, float situ_linear_beta, float swiglu_limit) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= hits * rows) {
        return;
    }
    const int hit = index / rows;
    const int row = index - hit * rows;
    hidden[index] = kimi_vram_activate(up_gate[(2 * hit) * rows + row], up_gate[(2 * hit + 1) * rows + row], kind,
        situ_beta, situ_linear_beta, swiglu_limit);
}

struct generic_engine {
    int device = -1;
    int latent = 0;
    int hidden = 0;
    int max_hits = 0;
    int up_type = GGML_TYPE_COUNT;
    int down_type_ = GGML_TYPE_COUNT;
    std::vector<std::pair<int, int>> formats; // the up/down types it computes, the created one first
    size_t q8_bytes = 0;
    cudaStream_t stream = nullptr;
    float * device_input = nullptr;
    float * device_up_gate = nullptr;
    float * device_hidden = nullptr;
    float * device_down = nullptr;
    void * device_q8 = nullptr;
    float * host_input = nullptr;
    float * host_down = nullptr;
    mmvq_fn mmvq = nullptr;
    // The warm pool's copies: their format and what a conversion uses.
    int warm_type = GGML_TYPE_COUNT;
    size_t warm_role[3] = {};
    cudaStream_t warm_stream = nullptr;
    cudaEvent_t warm_done = nullptr;
    float * warm_scratch = nullptr;
    int64_t warm_scratch_elements = 0;
    uint64_t calls = 0;
    uint64_t hits = 0;
    double total_us = 0.0;
    bool poisoned = false;
    bool closed = false;

    generic_engine(int device_, int latent_, int hidden_, int max_hits_, int weight_type, int down_type) :
        device(device_), latent(latent_), hidden(hidden_), max_hits(max_hits_) {
        if (latent <= 0 || hidden <= 0 || latent > 16384 || hidden > 16384 || max_hits < 1 || max_hits > 16) {
            throw std::runtime_error("unsupported generic VRAM expert geometry");
        }
        mmvq = ggml_cuda_moe_stream_mmvq_dev;
        add_format(weight_type, down_type);
        up_type = weight_type; down_type_ = down_type;
        q8_bytes = std::max(q8_1_bytes(latent), q8_1_bytes(hidden));
        try {
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_input), size_t(latent) * sizeof(float)), "cudaMalloc input");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_up_gate), size_t(2 * max_hits) * hidden * sizeof(float)), "cudaMalloc up/gate");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_hidden), size_t(max_hits) * hidden * sizeof(float)), "cudaMalloc hidden");
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_down), size_t(max_hits) * latent * sizeof(float)), "cudaMalloc down");
            if (q8_bytes) { cuda_check(cudaMalloc(&device_q8, q8_bytes), "cudaMalloc Q8_1"); }
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_input), size_t(latent) * sizeof(float)), "cudaMallocHost input");
            cuda_check(cudaMallocHost(reinterpret_cast<void **>(&host_down), size_t(max_hits) * latent * sizeof(float)), "cudaMallocHost down");
        } catch (...) {
            close();
            throw;
        }
        auto is_float = [](int type) { return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16; };
        std::fprintf(stderr,
            "[kimi_vram_bridge] format=%s/%s latent=%d hidden=%d input=f32 hidden_quantizer=%s bitwise=0\n",
            ggml_type_name(ggml_type(up_type)), ggml_type_name(ggml_type(down_type_)), latent, hidden,
            is_float(up_type) && is_float(down_type_) ? "none_f32" : "cuda_q8_1");
    }

    // A row's padded Q8_1 input is whole blocks at any length, so a type asks only for whole blocks of its own.
    void add_format(int weight_type, int down_type) {
        if (weight_type == GGML_TYPE_IQ2_XXS && down_type == GGML_TYPE_Q2_K) {
            throw std::runtime_error("IQ2_XXS/Q2_K expert format is retired");
        }
        const auto supports = ggml_cuda_moe_stream_mmvq_supports;
        if (!supports(weight_type) || !supports(down_type)) {
            throw std::runtime_error(std::string("GPU expert layer: no CUDA matrix-vector kernel for ") +
                ggml_type_name(ggml_type(weight_type)) + "/" + ggml_type_name(ggml_type(down_type)) + " experts");
        }
        if (latent % ggml_blck_size(ggml_type(weight_type)) || hidden % ggml_blck_size(ggml_type(down_type))) {
            throw std::runtime_error("unsupported generic VRAM expert geometry");
        }
        formats.emplace_back(weight_type, down_type);
    }

    void close() noexcept {
        if (closed) { return; }
        closed = true;
        const auto checked = [](cudaError_t rc, const char * operation) noexcept {
            if (rc != cudaSuccess) {
                std::fprintf(stderr, "fatal: generic VRAM engine cannot release ownership: %s: %s\n",
                    operation, cudaGetErrorString(rc));
                std::abort();
            }
        };
        if (stream || device_input || device_up_gate || device_hidden || device_down || device_q8 || host_input || host_down) {
            checked(cudaSetDevice(device), "cudaSetDevice during close");
        }
        if (stream) { checked(cudaStreamSynchronize(stream), "compute drain during close"); }
        if (warm_stream) { checked(cudaStreamSynchronize(warm_stream), "warm conversion drain during close"); }
        if (warm_scratch) { checked(cudaFree(warm_scratch), "free warm scratch"); warm_scratch = nullptr; }
        if (warm_stream) { checked(cudaStreamDestroy(warm_stream), "destroy warm stream"); warm_stream = nullptr; }
        if (host_down)  { checked(cudaFreeHost(host_down), "free host down"); host_down = nullptr; }
        if (host_input) { checked(cudaFreeHost(host_input), "free host input"); host_input = nullptr; }
        if (device_q8)      { checked(cudaFree(device_q8), "free Q8 scratch"); device_q8 = nullptr; }
        if (device_down)    { checked(cudaFree(device_down), "free device down"); device_down = nullptr; }
        if (device_hidden)  { checked(cudaFree(device_hidden), "free device hidden"); device_hidden = nullptr; }
        if (device_up_gate) { checked(cudaFree(device_up_gate), "free device up/gate"); device_up_gate = nullptr; }
        if (device_input)   { checked(cudaFree(device_input), "free device input"); device_input = nullptr; }
        if (stream) { checked(cudaStreamDestroy(stream), "destroy compute stream"); stream = nullptr; }

    }

    ~generic_engine() {
        std::fprintf(stderr, "[kimi_vram_generic_timing] calls=%llu hits=%llu total_us=%.0f\n",
            (unsigned long long) calls, (unsigned long long) hits, total_us);
        close();
    }

    size_t device_bytes() const {
        return (device_input ? size_t(latent) * sizeof(float) : 0) +
            (device_up_gate ? size_t(2 * max_hits) * hidden * sizeof(float) : 0) +
            (device_hidden ? size_t(max_hits) * hidden * sizeof(float) : 0) +
            (device_down ? size_t(max_hits) * latent * sizeof(float) : 0) + (device_q8 ? q8_bytes : 0) +
            (warm_scratch ? size_t(warm_scratch_elements) * sizeof(float) : 0);
    }

    size_t host_bytes() const {
        return (host_input ? size_t(latent) * sizeof(float) : 0) +
            (host_down ? size_t(max_hits) * latent * sizeof(float) : 0);
    }

    // A conversion dequantizes 256 rows at a time; the scratch is small against the slots it serves.
    void configure_warm(int bits, cudaEvent_t done, size_t * role_bytes) {
        if (closed || poisoned || warm_stream) { throw std::runtime_error("generic VRAM engine warm configuration phase"); }
        if ((bits != 2 && bits != 4) || !done) { throw std::runtime_error("VRAM warm pool precision"); }
        const int type = bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q2_0;
        std::fill(role_bytes, role_bytes + 3, size_t(0));
        if (!ggml_cuda_moe_stream_mmvq_supports(type)) { return; }
        const auto page = [](size_t bytes) { return (bytes + 4095) & ~size_t(4095); };
        const size_t up_row = ggml_row_size(ggml_type(type), latent), down_row = ggml_row_size(ggml_type(type), hidden);
        const size_t up_bytes = page(up_row * hidden), down_bytes = page(down_row * latent);
        // measured against a slot, which holds the largest format
        size_t native = 0;
        for (const auto & format : formats) {
            if (!ggml_cuda_moe_stream_requantize_supports(format.first, type) ||
                    !ggml_cuda_moe_stream_requantize_supports(format.second, type)) {
                return;
            }
            native = std::max(native, 2 * ggml_row_size(ggml_type(format.first), latent) * hidden +
                ggml_row_size(ggml_type(format.second), hidden) * latent);
        }
        if ((2 * up_bytes + down_bytes) * 4 > native * 3) { return; }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        warm_scratch_elements = int64_t(256) * std::max(latent, hidden);
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&warm_scratch), size_t(warm_scratch_elements) * sizeof(float)), "cudaMalloc warm scratch");
        cuda_check(cudaStreamCreateWithFlags(&warm_stream, cudaStreamNonBlocking), "cudaStreamCreate warm");
        warm_done = done;
        warm_type = type;
        warm_role[0] = role_bytes[0] = up_bytes;
        warm_role[1] = role_bytes[1] = up_bytes;
        warm_role[2] = role_bytes[2] = down_bytes;
    }

    void demote(const vram_compute_expert & native, void * warm) {
        if (closed || poisoned || !warm_stream) { throw std::runtime_error("generic VRAM engine has no warm pool"); }
        if (!native.up || !native.gate || !native.down || !warm) { throw std::runtime_error("null warm conversion"); }
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        char * target = static_cast<char *>(warm);
        const int up = native.up_type >= 0 ? native.up_type : up_type, down = native.down_type >= 0 ? native.down_type : down_type_;
        const bool queued =
            ggml_cuda_moe_stream_requantize(up, native.up, hidden, latent, warm_type, target, warm_scratch,
                warm_scratch_elements, warm_stream) &&
            ggml_cuda_moe_stream_requantize(up, native.gate, hidden, latent, warm_type, target + warm_role[0],
                warm_scratch, warm_scratch_elements, warm_stream) &&
            ggml_cuda_moe_stream_requantize(down, native.down, latent, hidden, warm_type,
                target + warm_role[0] + warm_role[1], warm_scratch, warm_scratch_elements, warm_stream);
        if (!queued) { throw std::runtime_error("VRAM warm conversion launch failed"); }
        cuda_check(cudaEventRecord(warm_done, warm_stream), "warm conversion event");
    }

    void run(const vram_compute_expert * experts, int count, const void * input, size_t input_bytes,
            const kimi_vram_activation & activation, float * output) {
        if (closed || poisoned) { throw std::runtime_error("generic VRAM engine is closed"); }
        if (!experts || !input || !output || count < 1 || count > max_hits ||
                input_bytes != size_t(latent) * sizeof(float)) {
            throw std::runtime_error("generic VRAM engine input shape");
        }
        if (activation.kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) {
            if (std::isnan(activation.swiglu_limit) || activation.swiglu_limit <= 1e-6f) {
                throw std::runtime_error("generic VRAM engine invalid SwiGLU clamp");
            }
        } else if (activation.kind == KIMI_VRAM_ACTIVATION_SITU) {
            if (!std::isfinite(activation.situ_beta) || activation.situ_beta <= 0.0f ||
                    !std::isfinite(activation.situ_linear_beta)) {
                throw std::runtime_error("generic VRAM engine invalid SiTU parameters");
            }
        } else if (activation.kind != KIMI_VRAM_ACTIVATION_SWIGLU) {
            throw std::runtime_error("generic VRAM engine activation kind");
        }

        const auto start = clock_type::now();
        std::memcpy(host_input, input, input_bytes);
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        cuda_check(cudaMemcpyAsync(device_input, host_input, input_bytes, cudaMemcpyHostToDevice, stream), "generic input H2D");
        // a warm copy may still be converting: its slot was given to this batch at prepare
        if (std::any_of(experts, experts + count, [](const vram_compute_expert & e) { return e.warm != 0; })) {
            if (!warm_stream) { throw std::runtime_error("generic VRAM engine has no warm pool"); }
            cuda_check(cudaStreamWaitEvent(stream, warm_done, 0), "wait for warm conversions");
        }
        for (int i = 0; i < count; ++i) {
            if (!experts[i].up || !experts[i].gate || !experts[i].down) {
                throw std::runtime_error("generic VRAM engine null resident weight");
            }
            const int type = experts[i].warm ? warm_type : experts[i].up_type >= 0 ? experts[i].up_type : up_type;
            const size_t stride = ggml_row_size(ggml_type(type), latent);
            if (!mmvq(type, experts[i].up, hidden, latent, stride,
                    device_input, device_q8, device_up_gate + size_t(2 * i) * hidden, stream) ||
                !mmvq(type, experts[i].gate, hidden, latent, stride,
                    device_input, device_q8, device_up_gate + size_t(2 * i + 1) * hidden, stream)) {
                throw std::runtime_error("VRAM expert up/gate MMVQ launch failed");
            }
        }
        const int rows = count * hidden;
        generic_activate<<<(rows + 255) / 256, 256, 0, stream>>>(device_up_gate, device_hidden,
            count, hidden, activation.kind, activation.situ_beta, activation.situ_linear_beta,
            activation.swiglu_limit);
        cuda_check(cudaGetLastError(), "generic activation launch");
        for (int i = 0; i < count; ++i) {
            const int type = experts[i].warm ? warm_type : experts[i].down_type >= 0 ? experts[i].down_type : down_type_;
            if (!mmvq(type, experts[i].down, latent, hidden, ggml_row_size(ggml_type(type), hidden),
                    device_hidden + size_t(i) * hidden, device_q8, device_down + size_t(i) * latent, stream)) {
                throw std::runtime_error("VRAM expert down MMVQ launch failed");
            }
        }
        const size_t output_bytes = size_t(count) * latent * sizeof(float);
        cuda_check(cudaMemcpyAsync(host_down, device_down, output_bytes, cudaMemcpyDeviceToHost, stream), "generic output D2H");
        cuda_check(cudaStreamSynchronize(stream), "generic compute synchronize");
        std::memcpy(output, host_down, output_bytes);
        ++calls;
        hits += uint64_t(count);
        total_us += std::chrono::duration<double, std::micro>(clock_type::now() - start).count();
    }
};

void * create(int device, int latent, int hidden, int max_hits, int weight_type, int down_type,
        const ssd::gpu::cpu_kernels_v1 *, char * error, size_t error_size) {
    try {
        return new generic_engine(device, latent, hidden, max_hits, weight_type, down_type);
    } catch (const std::exception & e) {
        if (error && error_size) { std::snprintf(error, error_size, "%s", e.what()); }
        return nullptr;
    } catch (...) {
        if (error && error_size) { std::snprintf(error, error_size, "unexpected generic VRAM create failure"); }
        return nullptr;
    }
}

void destroy(void * context) {
    delete static_cast<generic_engine *>(context);
}

int compute(void * context, const vram_compute_expert * experts, int hits,
        const void * input, size_t input_bytes, const kimi_vram_activation * activation,
        float * output, char * error, size_t error_size) {
    try {
        if (!context || !activation) { throw std::runtime_error("null generic VRAM compute argument"); }
        static_cast<generic_engine *>(context)->run(experts, hits, input, input_bytes, *activation, output);
        return 0;
    } catch (const std::exception & e) {
        if (context) {
            auto * engine = static_cast<generic_engine *>(context);
            engine->poisoned = true;
            engine->close();
        }
        if (error && error_size) { std::snprintf(error, error_size, "%s", e.what()); }
        return 1;
    } catch (...) {
        if (context) {
            auto * engine = static_cast<generic_engine *>(context);
            engine->poisoned = true;
            engine->close();
        }
        if (error && error_size) { std::snprintf(error, error_size, "unexpected generic VRAM compute failure"); }
        return 1;
    }
}

size_t device_bytes(void * context) {
    return context ? static_cast<generic_engine *>(context)->device_bytes() : 0;
}

size_t host_bytes(void * context) {
    return context ? static_cast<generic_engine *>(context)->host_bytes() : 0;
}

template <class F> int warm_call(void * context, char * error, size_t error_size, F && call) {
    try {
        if (!context) { throw std::runtime_error("null generic VRAM engine argument"); }
        call(*static_cast<generic_engine *>(context));
        return 0;
    } catch (const std::exception & e) {
        if (error && error_size) { std::snprintf(error, error_size, "%s", e.what()); }
        return 1;
    }
}

int warm_configure(void * context, int bits, vram_gpu_event done, size_t * role_bytes, char * error, size_t error_size) {
    return warm_call(context, error, error_size, [&](generic_engine & e) {
        e.configure_warm(bits, reinterpret_cast<cudaEvent_t>(done), role_bytes);
    });
}

int warm_demote(void * context, const vram_compute_expert * native, void * warm, char * error, size_t error_size) {
    return warm_call(context, error, error_size, [&](generic_engine & e) { e.demote(*native, warm); });
}

int warm_drain(void * context, char * error, size_t error_size) {
    return warm_call(context, error, error_size, [&](generic_engine & e) {
        if (e.warm_stream) { cuda_check(cudaStreamSynchronize(e.warm_stream), "warm conversion drain"); }
    });
}

int add_format(void * context, int weight_type, int down_type, char * error, size_t error_size) {
    return warm_call(context, error, error_size, [&](generic_engine & e) { e.add_format(weight_type, down_type); });
}
} // namespace

extern "C" const vram_format_compute_api * kimi_vram_generic_format_compute_get_api() {
    static const vram_format_compute_api api = {
        4, sizeof(vram_format_compute_api), create, destroy, compute, device_bytes, host_bytes,
        warm_configure, warm_demote, warm_drain, add_format
    };
    return &api;
}
