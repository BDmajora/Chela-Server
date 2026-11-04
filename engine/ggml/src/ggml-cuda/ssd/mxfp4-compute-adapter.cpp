#include "vram-compute-api.h"
#include "ggml-kimi-mxfp4-parity.h"
#include "ggml-kimi-mxfp4-chain.h"
#include "ggml.h"
#include "ggml-ssd/gpu/kernels.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// Test hook (kimi_vram_mxfp4_selfcheck_reference_for_testing): replaces the CPU
// reference of later engine self-checks. Nothing in llama.cpp sets it.
std::atomic<ssd::gpu::vec_dot_t> selfcheck_reference{nullptr};

// splitmix64, fully specified, so the self-check rows are the same in every
// build and on every host.
struct selfcheck_rng {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float unit() { return float(int32_t(uint32_t(next() >> 32))) * (1.0f / 2147483648.0f); } // [-1, 1)
};
struct selfcheck_totals { int checks = 0, jobs = 0; size_t rows = 0; };

using clock_type = std::chrono::steady_clock;
double elapsed_us(clock_type::time_point from, clock_type::time_point to) {
    return std::chrono::duration<double, std::micro>(to - from).count();
}

const char * chain_kernel_name(int kernel) {
    switch (kernel) {
        case KIMI_MXFP4_CHAIN_PARITY_Q8: return "parity_q8";
        case KIMI_MXFP4_CHAIN_ROWS_Q8: return "rows_q8";
        case KIMI_MXFP4_CHAIN_ROWS_FLOAT: return "rows_float";
        default: return "none";
    }
}

// Host side of the optional timing split (GGML_KIMI_CPU_SINK_VRAM_TIMING=1), all on
// the tier's worker thread.
struct host_timing {
    uint64_t calls = 0, chain_calls = 0, hits = 0;
    double enqueue_us = 0, wait_us = 0, activation_us = 0, quantize_us = 0, copy_us = 0;
};

// Explicit-format engine for MXFP4 experts. By default the GPU runs both matvec
// stages through the MXFP4 parity component; between them this host bridge applies
// the activation with the CPU sink's own expressions (kimi-cpu-sink.cpp
// compute_rows) and quantizes the hidden rows with the CPU backend's Q8_0
// from_float, so GPU rows equal CPU MUL_MAT_ID rows bit for bit. With
// GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN=1, SWIGLU_CLAMP layers instead run the
// component's GPU hidden chain in one submission (not bitwise); SiTU stays here.
struct format_bridge {
    static constexpr int selfcheck_rows = 37;   // two full 16-row CUDA blocks and a partial one
    static constexpr int selfcheck_inputs = 4;
    const kimi_mxfp4_gpu_api_v1 * api = nullptr;
    kimi_mxfp4_gpu * gpu = nullptr;
    const kimi_mxfp4_gpu_chain_api_v1 * chain_api = nullptr;
    kimi_mxfp4_gpu_chain * chain = nullptr;
    ssd::gpu::cpu_kernels_v1 cpu{};
    int latent, hidden, max_hits, chain_kernel = -1;
    bool timing = false;
    size_t up_stride = 0, down_stride = 0, input_bytes = 0, hidden_bytes = 0;
    ggml_from_float_t quantize = nullptr;
    std::vector<float> activation;
    std::vector<unsigned char> quant_hidden;
    std::vector<kimi_mxfp4_gpu_job> jobs;
    std::vector<kimi_mxfp4_chain_expert> experts;
    host_timing host;

    format_bridge(int device, int l, int h, int n, int weight_type, int down_type,
            const ssd::gpu::cpu_kernels_v1 * kernels, bool gpu_hidden = true, bool profile = false,
            const kimi_mxfp4_gpu_api_v1 * component = nullptr,
            const kimi_mxfp4_gpu_chain_api_v1 * extension = nullptr) : latent(l), hidden(h), max_hits(n), timing(profile) {
        if (weight_type != GGML_TYPE_MXFP4 || down_type != GGML_TYPE_MXFP4) {
            throw std::runtime_error("format VRAM engine supports MXFP4 up/gate/down experts only");
        }
        if (l <= 0 || h <= 0 || l > 16384 || h > 16384 || l % 64 || h % 64 || n < 1 || n > 16) {
            throw std::runtime_error("unsupported MXFP4 VRAM expert geometry");
        }
        if (!ssd::gpu::valid(kernels)) { throw std::runtime_error("CPU kernels provider missing or incompatible"); }
        cpu = *kernels;
        api = component ? component : kimi_mxfp4_gpu_get_api_v1();
        chain_api = extension;
        if (!api || api->version != 1 || api->job_size != sizeof(kimi_mxfp4_gpu_job) || !api->create ||
                !api->submit || !api->wait || !api->result || !api->error || !api->destroy ||
                !api->device_bytes || !api->host_bytes || !api->upload || !api->release_weights) {
            throw std::runtime_error("MXFP4 parity component ABI mismatch");
        }
        if (ggml_type_size(GGML_TYPE_MXFP4) != 17 || ggml_blck_size(GGML_TYPE_MXFP4) != 32) {
            throw std::runtime_error("unexpected MXFP4 block layout");
        }
        quantize = cpu.quantize_q8_0;
        up_stride = ggml_row_size(GGML_TYPE_MXFP4, l);
        down_stride = ggml_row_size(GGML_TYPE_MXFP4, h);
        input_bytes = ggml_row_size(GGML_TYPE_Q8_0, l);
        hidden_bytes = ggml_row_size(GGML_TYPE_Q8_0, h);
        activation.resize(size_t(h));
        quant_hidden.resize(size_t(n) * hidden_bytes);
        jobs.reserve(size_t(2 * n));
        kimi_mxfp4_gpu_config config{device, 2 * n, std::max(l, h), std::max(l, h)};
        char error[512] = {};
        gpu = api->create(&config, error, sizeof(error));
        if (!gpu) { throw std::runtime_error(error); }
        std::fprintf(stderr, "[kimi_vram_bridge] format=mxfp4 latent=%d hidden=%d hidden_quantizer=injected_ggml_cpu\n", l, h);
        try {
            // The parity self-check guards bitwise equality with the CPU sink only. The GPU hidden
            // chain makes no such claim, so it starts without the check and also runs on hosts
            // whose CPU vec_dot takes another path.
            if (gpu_hidden) {
                std::fprintf(stderr, "[kimi_vram_bridge] mxfp4_selfcheck=skipped reason=gpu_hidden_chain\n");
            } else {
                self_check();
            }
        } catch (...) { close(); throw; }
        try { configure_extensions(gpu_hidden); } catch (...) { close(); throw; }
    }
    // The GPU hidden chain and the timing split live in the component's chain
    // extension, which must come from the same DSO as the base API.
    void configure_extensions(bool gpu_hidden) {
        if (gpu_hidden || timing) {
            if (!chain_api) { chain_api = kimi_mxfp4_gpu_get_chain_api_v1(); }
            if (!chain_api || chain_api->version != 1 || chain_api->size != sizeof(*chain_api) || !chain_api->create ||
                    !chain_api->submit || !chain_api->destroy || !chain_api->host_bytes || !chain_api->device_bytes ||
                    !chain_api->set_timing || !chain_api->set_timing_stage || !chain_api->timing) {
                throw std::runtime_error("MXFP4 chain extension ABI mismatch");
            }
        }
        if (timing && !chain_api->set_timing(gpu, 1)) { throw std::runtime_error(api->error(gpu)); }
        if (gpu_hidden) {
            chain_kernel = KIMI_MXFP4_CHAIN_ROWS_Q8;
            chain = chain_api->create(gpu, latent, hidden, max_hits, chain_kernel, 0);
            if (!chain) { throw std::runtime_error(api->error(gpu)); }
            experts.reserve(size_t(max_hits));
        }
        std::fprintf(stderr, "[kimi_vram_bridge] mxfp4_gpu_hidden=%d kernel=%s bitwise_swiglu=%d timing=%d\n",
            int(gpu_hidden), chain_kernel_name(chain_kernel), int(!gpu_hidden), int(timing));
    }
    void close() noexcept {
        if (chain && !chain_api->destroy(chain)) {
            std::fprintf(stderr, "[kimi_vram_bridge] failed to drain the MXFP4 chain at destroy\n");
            std::abort();
        }
        chain = nullptr;
        if (gpu && !api->destroy(gpu)) {
            std::fprintf(stderr, "[kimi_vram_bridge] failed to drain the MXFP4 component at destroy\n");
            std::abort();
        }
        gpu = nullptr;
    }
    void report_timing() noexcept {
        kimi_mxfp4_gpu_timing t{};
        if (!timing || !gpu || !chain_api || !chain_api->timing(gpu, &t)) { return; }
        std::fprintf(stderr,
            "[kimi_vram_timing] path=%s kernel=%s calls=%llu chain_calls=%llu hits=%llu gpu_submits=%llu "
            "gpu_h2d_us=%.0f gpu_stage1_us=%.0f gpu_activation_us=%.0f gpu_quantize_us=%.0f gpu_stage2_us=%.0f gpu_d2h_us=%.0f "
            "host_h2d_call_us=%.0f host_enqueue_us=%.0f host_wait_us=%.0f host_activation_us=%.0f host_quantize_us=%.0f host_copy_us=%.0f\n",
            chain ? "chain" : "bridge", chain ? chain_kernel_name(chain_kernel) : "parity_lanes",
            (unsigned long long) host.calls, (unsigned long long) host.chain_calls, (unsigned long long) host.hits,
            (unsigned long long) t.submits, t.h2d_ms * 1e3, t.stage1_ms * 1e3, t.activation_ms * 1e3, t.quantize_ms * 1e3,
            t.stage2_ms * 1e3, t.d2h_ms * 1e3, t.h2d_call_us, host.enqueue_us, host.wait_us, host.activation_us,
            host.quantize_us, host.copy_us);
    }
    ~format_bridge() { report_timing(); close(); }
    void submit(const void * q8, size_t bytes) {
        const auto enqueue_start = timing ? clock_type::now() : clock_type::time_point{};
        if (!api->submit(gpu, jobs.data(), int(jobs.size()), q8, bytes)) { throw std::runtime_error(api->error(gpu)); }
        const auto wait_start = timing ? clock_type::now() : clock_type::time_point{};
        if (!api->wait(gpu)) { throw std::runtime_error(api->error(gpu)); }
        if (timing) {
            host.enqueue_us += elapsed_us(enqueue_start, wait_start);
            host.wait_us += elapsed_us(wait_start, clock_type::now());
        }
    }
    const float * result(int job, int expected) {
        int rows = 0;
        const float * p = api->result(gpu, job, &rows);
        if (!p || rows != expected) { throw std::runtime_error("MXFP4 component result shape mismatch"); }
        return p;
    }
    // Runtime parity gate, once per engine. The kernel reproduces one CPU path:
    // the AVX2 (AVX-VNNI-INT8) ggml_vec_dot_mxfp4_q8_0 lanes with FMA
    // accumulators and the hsum_float_8 order. Any other path gives rows that
    // are close but not bitwise equal. Deterministic MXFP4 rows at both of the
    // engine's column counts, with realistic and edge E8M0 exponents and Q8_0
    // inputs from the linked CPU quantizer, must equal the linked CPU vec_dot
    // bit for bit; otherwise engine creation fails closed. No override exists.
    void self_check() {
        const auto start = std::chrono::steady_clock::now();
        const ssd::gpu::vec_dot_t replaced = selfcheck_reference.load();
        const ssd::gpu::vec_dot_t reference = replaced ? replaced : cpu.reference_mxfp4;
        const char * reference_name = replaced ? "test_override" : "linked_cpu_vec_dot";
        selfcheck_totals totals;
        for (int stage = 0; stage < 2; ++stage) {
            for (int family = 0; family < 2; ++family) { run_check(reference, reference_name, stage, family, totals); }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "[kimi_vram_bridge] mxfp4_selfcheck=pass checks=%d jobs=%d rows=%zu columns=%d,%d reference=%s us=%.0f\n",
            totals.checks, totals.jobs, totals.rows, latent, hidden, reference_name, us);
    }
    void run_check(ssd::gpu::vec_dot_t reference, const char * reference_name, int stage, int family, selfcheck_totals & totals) {
        const int columns = stage ? hidden : latent, blocks = columns / 32, rows = selfcheck_rows;
        char name[64];
        std::snprintf(name, sizeof(name), "%s_c%d/%s", stage ? "down" : "up_gate", columns, family ? "e8m0_edges" : "scales");
        selfcheck_rng rng{0x4d58465034ull ^ (uint64_t(columns) << 8) ^ (uint64_t(stage) << 4) ^ uint64_t(family)};
        const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, columns), q8_row = ggml_row_size(GGML_TYPE_Q8_0, columns);
        static const uint8_t edges[] = {0, 1, 2, 60, 127, 128, 140, 170};
        std::vector<uint8_t> weights(stride * size_t(rows));
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < blocks; ++b) {
                uint8_t * block = weights.data() + size_t(r) * stride + size_t(b) * 17;
                block[0] = family ? edges[(r + b) % 8] : uint8_t(110 + rng.next() % 30); // E8M0 scale
                for (int k = 1; k < 17; ++k) { block[k] = uint8_t(rng.next()); }         // all nibbles, -0 included
            }
        }
        // Q8_0 inputs through the linked CPU quantizer: realistic, large, sparse
        // (zero blocks) and alternating signs, or 1..1e6 and tiny magnitudes on
        // the edge exponents (subnormal scales).
        static const float magnitude[] = {1.0f, 1e3f, 1e6f, 1e-3f};
        std::vector<float> x(static_cast<size_t>(columns), 0.0f);
        std::vector<uint8_t> q8(q8_row * size_t(selfcheck_inputs));
        for (int i = 0; i < selfcheck_inputs; ++i) {
            for (int c = 0; c < columns; ++c) {
                const float v = rng.unit();
                x[size_t(c)] = family ? v * magnitude[i] :
                    i == 0 ? v : i == 1 ? 64.0f * v : i == 2 ? (c / 32 % 3 ? v : 0.0f) : (c % 2 ? 1.0f : -1.0f);
            }
            quantize(x.data(), q8.data() + size_t(i) * q8_row, columns);
        }
        std::vector<float> expected(size_t(rows) * selfcheck_inputs);
        for (int i = 0; i < selfcheck_inputs; ++i) {
            for (int r = 0; r < rows; ++r) {
                float & y = expected[size_t(i) * size_t(rows) + size_t(r)];
                reference(columns, &y, 0, weights.data() + size_t(r) * stride, 0, q8.data() + size_t(i) * q8_row, 0, 1);
                if (!std::isfinite(y)) { throw std::runtime_error(std::string("MXFP4 parity self-check: nonfinite CPU row, check=") + name); }
            }
        }
        void * device = nullptr;
        if (!api->upload(gpu, weights.data(), weights.size(), &device)) {
            throw std::runtime_error(std::string("MXFP4 parity self-check upload: ") + api->error(gpu));
        }
        struct held_weights {
            const kimi_mxfp4_gpu_api_v1 * api; kimi_mxfp4_gpu * gpu; void * ptr;
            ~held_weights() { api->release_weights(gpu, ptr); }
        } held{api, gpu, device};
        const int batch = std::min(selfcheck_inputs, 2 * max_hits);
        for (int first = 0; first < selfcheck_inputs; first += batch) {
            const int n = std::min(batch, selfcheck_inputs - first);
            jobs.clear();
            for (int i = 0; i < n; ++i) { jobs.push_back({device, weights.size(), stride, size_t(i) * q8_row, columns, rows}); }
            submit(q8.data() + size_t(first) * q8_row, size_t(n) * q8_row);
            for (int i = 0; i < n; ++i) {
                const float * got = result(i, rows);
                for (int r = 0; r < rows; ++r) {
                    const float want = expected[size_t(first + i) * size_t(rows) + size_t(r)];
                    if (!std::memcmp(&want, got + r, sizeof(float))) { continue; }
                    char detail[192];
                    std::snprintf(detail, sizeof(detail), "check=%s input=%d row=%d cpu=%a gpu=%a",
                        name, first + i, r, double(want), double(got[r]));
                    std::fprintf(stderr, "[kimi_vram_bridge] mxfp4_selfcheck=FAIL %s reference=%s\n", detail, reference_name);
                    throw std::runtime_error(std::string("MXFP4 VRAM parity self-check failed: ") + detail +
                        "; the GPU kernel does not reproduce this host's ggml_vec_dot_mxfp4_q8_0 path; MXFP4 VRAM tier refused, no fallback");
                }
                ++totals.jobs;
                totals.rows += size_t(rows);
            }
        }
        jobs.clear();
        ++totals.checks;
    }
    // One chain submission for the layer: the component's stage-1 matvec, device
    // SwiGLU, device Q8_0 and stage-2 matvec, then one copy of the down rows.
    void compute_chain(const vram_compute_expert * weights, int hits, const void * q8, size_t bytes, float limit, float * down) {
        experts.clear();
        for (int i = 0; i < hits; ++i) { experts.push_back({weights[i].up, weights[i].gate, weights[i].down}); }
        const auto enqueue_start = timing ? clock_type::now() : clock_type::time_point{};
        if (!chain_api->submit(chain, experts.data(), hits, q8, bytes, limit)) { throw std::runtime_error(api->error(gpu)); }
        const auto wait_start = timing ? clock_type::now() : clock_type::time_point{};
        if (!api->wait(gpu)) { throw std::runtime_error(api->error(gpu)); }
        const auto copy_start = timing ? clock_type::now() : clock_type::time_point{};
        for (int i = 0; i < hits; ++i) {
            std::memcpy(down + size_t(i) * latent, result(i, latent), size_t(latent) * sizeof(float));
        }
        if (timing) {
            ++host.chain_calls;
            host.enqueue_us += elapsed_us(enqueue_start, wait_start);
            host.wait_us += elapsed_us(wait_start, copy_start);
            host.copy_us += elapsed_us(copy_start, clock_type::now());
        }
    }
    void compute(const vram_compute_expert * weights, int hits, const void * q8, size_t bytes,
                 const kimi_vram_activation & act, float * down) {
        if (!gpu) { throw std::runtime_error("MXFP4 bridge closed after component error"); }
        if (hits < 1 || hits > max_hits || !weights || !down || !q8 || bytes != input_bytes) {
            throw std::runtime_error("MXFP4 bridge input shape");
        }
        const bool clamp = act.kind == KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP;
        if (clamp ? !(std::isfinite(act.swiglu_limit) && act.swiglu_limit > 1e-6f) :
                (act.kind != KIMI_VRAM_ACTIVATION_SITU || !std::isfinite(act.situ_beta) || act.situ_beta <= 0.0f ||
                 !std::isfinite(act.situ_linear_beta))) {
            throw std::runtime_error("MXFP4 bridge activation parameters");
        }
        if (timing) {
            ++host.calls;
            host.hits += uint64_t(hits);
        }
        if (chain && clamp) {
            compute_chain(weights, hits, q8, bytes, act.swiglu_limit, down);
            return;
        }
        jobs.clear();
        for (int i = 0; i < hits; ++i) {
            jobs.push_back({weights[i].up, up_stride * size_t(hidden), up_stride, 0, latent, hidden});
            jobs.push_back({weights[i].gate, up_stride * size_t(hidden), up_stride, 0, latent, hidden});
        }
        if (timing) { chain_api->set_timing_stage(gpu, 1); }
        submit(q8, bytes);
        for (int i = 0; i < hits; ++i) {
            const float * up = result(2 * i, hidden);
            const float * gate = result(2 * i + 1, hidden);
            const auto activation_start = timing ? clock_type::now() : clock_type::time_point{};
            if (clamp) {
                const float limit = act.swiglu_limit;
                for (int row = 0; row < hidden; ++row) {
                    const float u = up[row], g = gate[row];
                    // kimi-cpu-sink.cpp compute_rows, SWIGLU_CLAMP branch.
                    const float gate_value = std::min(g, limit);
                    const float up_value = std::clamp(u, -limit, limit);
                    activation[row] = gate_value / (1.f + expf(-gate_value)) * up_value;
                }
            } else {
                const float beta = act.situ_beta, linear = act.situ_linear_beta;
                for (int row = 0; row < hidden; ++row) {
                    float u = up[row];
                    const float g = gate[row];
                    // kimi-cpu-sink.cpp compute_rows, SiTU branch.
                    const float a = (beta * std::tanh(g * (1.0f / beta))) * (1.0f / (1.0f + std::exp(-g)));
                    if (linear > 0.0f) { u = linear * std::tanh(u * (1.0f / linear)); }
                    activation[row] = a * u;
                }
            }
            const auto quantize_start = timing ? clock_type::now() : clock_type::time_point{};
            quantize(activation.data(), quant_hidden.data() + size_t(i) * hidden_bytes, hidden);
            if (timing) {
                host.activation_us += elapsed_us(activation_start, quantize_start);
                host.quantize_us += elapsed_us(quantize_start, clock_type::now());
            }
        }
        jobs.clear();
        for (int i = 0; i < hits; ++i) {
            jobs.push_back({weights[i].down, down_stride * size_t(latent), down_stride, size_t(i) * hidden_bytes, hidden, latent});
        }
        if (timing) { chain_api->set_timing_stage(gpu, 2); }
        submit(quant_hidden.data(), size_t(hits) * hidden_bytes);
        const auto copy_start = timing ? clock_type::now() : clock_type::time_point{};
        for (int i = 0; i < hits; ++i) {
            std::memcpy(down + size_t(i) * latent, result(i, latent), size_t(latent) * sizeof(float));
        }
        if (timing) { host.copy_us += elapsed_us(copy_start, clock_type::now()); }
    }
};

void * create(int device, int latent, int hidden, int max_hits, int weight_type, int down_type, const ssd::gpu::cpu_kernels_v1 * kernels, char * error, size_t size) {
    try { return new format_bridge(device, latent, hidden, max_hits, weight_type, down_type, kernels); }
    catch (const std::exception & e) { if (size) { std::snprintf(error, size, "%s", e.what()); } return nullptr; }
}
void destroy(void * p) { delete static_cast<format_bridge *>(p); }
int compute(void * p, const vram_compute_expert * weights, int hits, const void * input, size_t bytes,
            const kimi_vram_activation * activation, float * down, char * error, size_t size) {
    try {
        if (!activation) { throw std::runtime_error("null VRAM activation"); }
        static_cast<format_bridge *>(p)->compute(weights, hits, input, bytes, *activation, down);
        return 0;
    } catch (const std::exception & e) {
        if (size) { std::snprintf(error, size, "%s", e.what()); }
        // A failed submit may already have queued work; destroy drains it while
        // the cache still pins the weights, and the bridge stays closed.
        static_cast<format_bridge *>(p)->close();
        return 1;
    }
}
size_t device_bytes(void * p) {
    auto * b = static_cast<format_bridge *>(p);
    return (b->gpu ? b->api->device_bytes(b->gpu) : 0) + (b->chain ? b->chain_api->device_bytes(b->chain) : 0);
}
size_t host_bytes(void * p) {
    auto * b = static_cast<format_bridge *>(p);
    return (b->gpu ? b->api->host_bytes(b->gpu) : 0) + (b->chain ? b->chain_api->host_bytes(b->chain) : 0) +
        b->activation.capacity() * sizeof(float) + b->quant_hidden.capacity() + b->jobs.capacity() * sizeof(kimi_mxfp4_gpu_job) +
        b->experts.capacity() * sizeof(kimi_mxfp4_chain_expert);
}
}

extern "C" const vram_format_compute_api * kimi_vram_format_compute_get_api() {
    static const vram_format_compute_api api{4, sizeof(vram_format_compute_api), create, destroy, compute, device_bytes, host_bytes,
        nullptr, nullptr, nullptr, nullptr};
    return &api;
}

// Test hook only: replace the CPU reference of later MXFP4 engine self-checks
// (nullptr restores the linked ggml_vec_dot_mxfp4_q8_0). The self-check log line
// reports reference=test_override while it is set.
extern "C" __attribute__((visibility("default"))) void kimi_vram_mxfp4_selfcheck_reference_for_testing(ssd::gpu::vec_dot_t reference) {
    selfcheck_reference.store(reference);
}

// Explicit fixture factory: retained host-bridge parity and queued-error tests
// exercise the real component DSO without a production environment switch.
extern "C" __attribute__((visibility("default"))) void * ggml_ssd_mxfp4_create_for_testing(
        int device, int latent, int hidden, int max_hits, int weight_type, int down_type,
        bool gpu_hidden, bool profile, const ssd::gpu::cpu_kernels_v1 * kernels, char * error, size_t size) noexcept {
    try { return new format_bridge(device, latent, hidden, max_hits, weight_type, down_type, kernels, gpu_hidden, profile); }
    catch (const std::exception & e) {
        if (error && size) { std::snprintf(error, size, "%s", e.what()); }
        return nullptr;
    }
}

extern "C" __attribute__((visibility("default"))) int ggml_backend_cuda_ssd_mxfp4_set_profile(
        void * value, bool enabled, char * error, size_t size) noexcept {
    try {
        auto * bridge = static_cast<format_bridge *>(value);
        if (!bridge || !bridge->gpu || !bridge->chain_api || !bridge->chain_api->set_timing(bridge->gpu, int(enabled))) {
            throw std::runtime_error("cannot configure MXFP4 profile channel");
        }
        bridge->timing = enabled;
        return 0;
    } catch (const std::exception & e) {
        if (error && size) { std::snprintf(error, size, "%s", e.what()); }
        return 1;
    }
}

// Explicit dependency injection for queued-error fixtures. Product callers use
// only the directly linked component above; no symbol interposition is needed.
extern "C" void * ggml_ssd_mxfp4_create_with_components_for_testing(
        int device, int latent, int hidden, int max_hits, int weight_type, int down_type,
        bool gpu_hidden, bool profile, const ssd::gpu::cpu_kernels_v1 * kernels,
        const kimi_mxfp4_gpu_api_v1 * component, const kimi_mxfp4_gpu_chain_api_v1 * extension,
        char * error, size_t size) noexcept {
    try {
        if (!component || !extension) { throw std::runtime_error("missing explicit fixture component"); }
        return new format_bridge(device, latent, hidden, max_hits, weight_type, down_type,
            kernels, gpu_hidden, profile, component, extension);
    } catch (const std::exception & e) {
        if (error && size) { std::snprintf(error, size, "%s", e.what()); }
        return nullptr;
    } catch (...) {
        if (error && size) { std::snprintf(error, size, "unexpected MXFP4 fixture failure"); }
        return nullptr;
    }
}
