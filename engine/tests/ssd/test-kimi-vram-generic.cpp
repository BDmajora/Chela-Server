#include "gpu-kernels.h"
#include "explicit-fixture-inputs.h"
// End-to-end generic-format CPU-sink test with the real VRAM tier: Q4_K/Q5_K
// as in Muse Spark UD-Q4_K_XL, and Q8_0. Synthetic
// packed weights cover non-uniform padded role sizes and mixed CPU/GPU expert
// reduction without requiring the full model. The CUDA path intentionally uses
// Q8_1 activations and is checked numerically, not bitwise.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include "kimi-host-jobs-test.inc"
#include "gpu-runtime.h"

namespace {
double relative_l2(const std::vector<float> & got, const std::vector<float> & expected) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double delta = double(got[i]) - double(expected[i]);
        error += delta * delta;
        norm += double(expected[i]) * double(expected[i]);
    }
    return norm > 0.0 ? std::sqrt(error / norm) : std::sqrt(error);
}

double max_abs_error(const std::vector<float> & got, const std::vector<float> & expected) {
    double result = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        result = std::max(result, std::abs(double(got[i]) - double(expected[i])));
    }
    return result;
}

struct mode {
    bool production;
    const char * name;
};

void set_switch(fixture_inputs & inputs, const char * name, bool enabled) {
    inputs.set(name, enabled ? "1" : "0");
}

void run_shape(fixture_inputs & inputs, int latent, int hidden, int experts, mode selected, size_t & checks,
        double & worst_rel, double & worst_abs,
        ggml_type up_type, ggml_type down_type) {
    fixture f;
    f.type = up_type;
    f.down_type = down_type;
    f.latent = latent;
    f.hidden = hidden;
    f.experts = experts;
    make_synthetic(f, uint32_t(20260920 + latent + int(selected.production)));
    const uint64_t unit = triplet_bytes(f);

    inputs.set("vram_budget", "0");
    set_switch(inputs, "tier.global", false);
    set_switch(inputs, "vram_exclusive", false);
    set_switch(inputs, "vram_async", false);
    set_switch(inputs, "vram_deferred", false);
    set_switch(inputs, "early_up_gate", false);
    set_switch(inputs, "role_major", false);
    set_switch(inputs, "nonblocking_down_sched", false);
    auto cpu = make_sink(inputs, f, 9, 8);

    const uint64_t vram_bytes = selected.production ? uint64_t(2 * experts) * unit : 2 * unit;
    inputs.set("vram_budget", std::to_string(vram_bytes).c_str());
    set_switch(inputs, "tier.global", selected.production);
    set_switch(inputs, "vram_exclusive", selected.production);
    set_switch(inputs, "vram_async", selected.production);
    set_switch(inputs, "vram_deferred", selected.production);
    set_switch(inputs, "tier.keep_across_requests", true);
    set_switch(inputs, "early_up_gate", selected.production);
    set_switch(inputs, "role_major", selected.production);
    set_switch(inputs, "nonblocking_down_sched", selected.production);
    // production: the request-level prior from the expert predictor, persisted across sinks. It changes which experts
    // the tier holds, never a result: the outputs below are checked against the CPU sink as without it.
    const std::string trace_file = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
        "/kimi-vram-generic-trace-" + std::to_string(::getpid()) + "-" + std::to_string(latent) + ".bin";
    if (selected.production) {
        std::remove(trace_file.c_str());
        inputs.set("trace_capacity", "8");
        set_switch(inputs, "trace_prior", true);
        inputs.set("trace_file", trace_file.c_str());
    }
    auto mixed = make_sink(inputs, f, 9, 8);
    inputs.set("trace_capacity", "0");
    set_switch(inputs, "trace_prior", false);
    inputs.set("trace_file", "");

    const uint64_t ram_bytes = selected.production ? 2 * unit : 0;
    configure_cache(cpu.get(), ram_bytes, {1, 2}, experts);
    configure_cache(mixed.get(), ram_bytes, {1, 2}, experts);
    uint64_t cpu_reads = 0;
    uint64_t mixed_reads = 0;
    const kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f
    };

    for (int token = 1; token <= 4; ++token) {
        begin_cache(cpu.get(), 91, uint64_t(token));
        begin_cache(mixed.get(), 91, uint64_t(token));
        for (int layer : {1, 2}) {
            const int used = std::min(4, experts);
            std::vector<int32_t> ids;
            std::vector<float> weights;
            for (int i = 0; i < used; ++i) {
                ids.push_back((i + layer - 1) % experts);
                weights.push_back(i == 0 ? 0.4f : 0.6f / float(used - 1));
            }
            std::vector<float> input(static_cast<size_t>(latent));
            for (int i = 0; i < latent; ++i) {
                input[size_t(i)] = 1.7f * std::sin(float(i * 11 + token * 23 + layer * 5));
            }
            std::vector<float> expected(static_cast<size_t>(latent));
            std::vector<float> got(static_cast<size_t>(latent));
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                char error[512] = {};
                const bool ok = kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(),
                    meta.role[0], meta.role[1], meta.role[2], input.data(), ids.data(), weights.data(),
                    int(ids.size()), &activation, (arm ? got : expected).data(), &stats, error, sizeof(error));
                require(ok, std::string("generic sink run failed mode=") + selected.name +
                    " token=" + std::to_string(token) + " layer=" + std::to_string(layer) +
                    " arm=" + std::to_string(arm) + ": " + error);
                require(stats.read_errors == 0 && stats.pending_reads == 0 && stats.fallback_count == 0,
                    "generic VRAM work did not drain");
                (arm ? mixed_reads : cpu_reads) += stats.read_requests;
                ++checks;
            }
            for (float value : got) {
                require(std::isfinite(value), "generic VRAM output is non-finite");
            }
            const double rel = relative_l2(got, expected);
            const double abs = max_abs_error(got, expected);
            worst_rel = std::max(worst_rel, rel);
            worst_abs = std::max(worst_abs, abs);
            require(rel <= 2e-2, "generic VRAM output exceeds 2e-2 relative L2");
            ++checks;
        }
    }
    require(mixed_reads < cpu_reads, "generic VRAM tier served no resident expert hits");
    ++checks;
    if (selected.production) {
        // destroying the sink writes the predictor's state; a sink of the same experts reads it back
        mixed.reset();
        std::FILE * written = std::fopen(trace_file.c_str(), "rb");
        require(written != nullptr, "the expert trace file was not written");
        std::fseek(written, 0, SEEK_END);
        const long bytes = std::ftell(written);
        std::fclose(written);
        require(bytes > 16, "the expert trace file is empty");
        std::remove(trace_file.c_str());
        ++checks;
    }
    std::printf("GENERIC_VRAM_SHAPE_PASS mode=%s types=%s/%s latent=%d hidden=%d cpu_reads=%llu mixed_reads=%llu worst_rel_l2=%.6g worst_abs=%.6g\n",
        selected.name, ggml_type_name(up_type), ggml_type_name(down_type), latent, hidden, (unsigned long long) cpu_reads,
        (unsigned long long) mixed_reads, worst_rel, worst_abs);
}

// The tier's lookahead (vram.lookahead) and prior through the sink: experts large enough that the GPU saves time,
// selections that move every other token, in a tier smaller than the experts they visit and a RAM cache that holds
// them all, each layer's selection hinted before the layer before it runs. Which experts go up ahead changes where they
// are computed, never a result. Runs its checks of the two counters only with a finite cgroup, which the RAM cache
// needs (test_resources); without one the tier is reached only through its reads.
void run_lookahead(fixture_inputs & inputs, size_t & checks, double & worst_rel, double & worst_abs) {
    fixture f;
    f.type = GGML_TYPE_Q8_0;
    f.down_type = GGML_TYPE_Q8_0;
    f.latent = 4096;
    f.hidden = 2048;
    f.experts = 8;
    make_synthetic(f, 20261103u);
    const uint64_t unit = triplet_bytes(f);

    inputs.set("vram_budget", "0");
    set_switch(inputs, "tier.global", false);
    set_switch(inputs, "vram_exclusive", false);
    set_switch(inputs, "vram_async", false);
    set_switch(inputs, "vram_deferred", false);
    set_switch(inputs, "early_up_gate", false);
    set_switch(inputs, "role_major", false);
    set_switch(inputs, "nonblocking_down_sched", false);
    auto cpu = make_sink(inputs, f, 9, 8);

    inputs.set("vram_budget", std::to_string(10 * unit).c_str());
    inputs.set("tier.safety_mib", "32");
    set_switch(inputs, "tier.global", true);
    set_switch(inputs, "vram_exclusive", true);
    set_switch(inputs, "vram_async", true);
    set_switch(inputs, "vram_deferred", true);
    set_switch(inputs, "tier.keep_across_requests", true);
    set_switch(inputs, "early_up_gate", true);
    set_switch(inputs, "role_major", true);
    set_switch(inputs, "nonblocking_down_sched", true);
    // the fixed admission rule: this machine's measured upload would refuse these short-lived experts, which is the
    // cost rule's business; the path through the sink is what runs here
    inputs.set("tier.policy", "count_decay2");
    inputs.set("tier.half_life", "16");
    set_switch(inputs, "tier.adaptive_admission", false);
    inputs.set("trace_capacity", "8");
    set_switch(inputs, "trace_prior", true);
    inputs.set("vram_lookahead", "2");
    auto mixed = make_sink(inputs, f, 9, 8);
    inputs.set("trace_capacity", "0");
    set_switch(inputs, "trace_prior", false);
    inputs.set("vram_lookahead", "0");
    set_switch(inputs, "tier.adaptive_admission", true);
    inputs.set("tier.policy", "lru");
    inputs.set("tier.half_life", "2");

    const uint64_t ram_bytes = 2 * uint64_t(f.experts) * unit;
    configure_cache(cpu.get(), ram_bytes, {1, 2}, f.experts);
    configure_cache(mixed.get(), ram_bytes, {1, 2}, f.experts);
    const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f };
    const auto selection = [&](int token, int layer) {
        std::vector<int32_t> ids;
        for (int i = 0; i < 4; ++i) { ids.push_back(int32_t((token / 2 + 3 * i + layer) % f.experts)); }
        return ids;
    };
    for (int token = 1; token <= 40; ++token) {
        begin_cache(cpu.get(), 92, uint64_t(token));
        begin_cache(mixed.get(), 92, uint64_t(token));
        for (int layer : {1, 2}) {
            const auto ids = selection(token, layer);
            const std::vector<float> weights(4, 0.25f);
            std::vector<float> input(static_cast<size_t>(f.latent)), expected(input.size()), got(input.size());
            for (int i = 0; i < f.latent; ++i) { input[size_t(i)] = 1.3f * std::sin(float(i * 7 + token * 13 + layer * 3)); }
            if (layer == 1) {
                const auto next = selection(token, 2);
                kimi_cpu_sink_prefetch_hint(mixed.get(), 2, next.data(), int(next.size()));
            }
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                char error[512] = {};
                require(kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(), meta.role[0], meta.role[1], meta.role[2],
                    input.data(), ids.data(), weights.data(), int(ids.size()), &activation, (arm ? got : expected).data(),
                    &stats, error, sizeof(error)), std::string("lookahead sink run failed: ") + error);
                require(stats.read_errors == 0 && stats.pending_reads == 0, "lookahead VRAM work did not drain");
                ++checks;
            }
            for (float value : got) { require(std::isfinite(value), "lookahead output is non-finite"); }
            const double rel = relative_l2(got, expected);
            worst_rel = std::max(worst_rel, rel);
            worst_abs = std::max(worst_abs, max_abs_error(got, expected));
            require(rel <= 2e-2, "lookahead output exceeds 2e-2 relative L2");
            ++checks;
        }
    }
    kimi_cpu_sink_cache_stats stats = {};
    kimi_cpu_sink_cache_get_stats(mixed.get(), &stats);
    require(stats.prior_layers > 0, "the tier took no prior");
    if (stats.capacity_bytes) { require(stats.lookahead_uploads > 0, "no predicted expert went up ahead of its selection"); }
    std::printf("GENERIC_VRAM_LOOKAHEAD_PASS priors=%llu lookahead_uploads=%llu ram_cache=%d worst_rel_l2=%.6g\n",
        (unsigned long long) stats.prior_layers, (unsigned long long) stats.lookahead_uploads, int(stats.capacity_bytes != 0), worst_rel);
}

// Layers of different formats through the sink, as a quantizer leaves them (Qwen3.8 IQ1_M: 7 up/down pairs over 48
// layers, at a hidden width of 640): the smaller layer runs first, so that its slots must already fit the larger one,
// which only the owner's table can tell the tier. Every expert of both layers goes up and comes back from its own layer's
// slot layout; a role read at another layer's offset or kernel would fail the comparison with the CPU sink.
void run_mixed(fixture_inputs & inputs, bool elastic, size_t & checks, double & worst_rel, double & worst_abs) {
    fixture f;
    f.type = GGML_TYPE_IQ2_S;
    f.down_type = GGML_TYPE_Q2_0;
    f.layer2_type = GGML_TYPE_IQ3_S;
    f.layer2_down_type = GGML_TYPE_IQ4_NL;
    f.latent = 2560;
    f.hidden = 640;
    f.experts = 12; // an elastic tier's second block comes after its 16 slots of the larger size
    make_synthetic(f, 20261103u);
    uint64_t unit = 0;
    const kimi_vram_layer_format formats[2] = {
        { 1, int32_t(f.role_type(1, 0)), int32_t(f.role_type(1, 2)), {} },
        { 2, int32_t(f.role_type(2, 0)), int32_t(f.role_type(2, 2)), {} } };
    std::vector<kimi_vram_layer_format> table(formats, formats + 2);
    for (auto & format : table) {
        for (int role = 0; role < 3; ++role) {
            format.role_bytes[role] = f.entries.at({ tensor_name(format.layer, role), 0 }).read_bytes; // as the pack stores it
        }
        unit = std::max(unit, format.role_bytes[0] + format.role_bytes[1] + format.role_bytes[2]);
    }

    inputs.set("vram_budget", "0");
    set_switch(inputs, "tier.global", false);
    set_switch(inputs, "vram_exclusive", false);
    set_switch(inputs, "vram_async", false);
    set_switch(inputs, "vram_deferred", false);
    auto cpu = make_sink(inputs, f, 9, 8);

    inputs.set("vram_budget", std::to_string(uint64_t(2 * f.experts) * unit).c_str());
    set_switch(inputs, "tier.global", true);
    set_switch(inputs, "vram_exclusive", true);
    set_switch(inputs, "vram_async", true);
    set_switch(inputs, "vram_deferred", true);
    set_switch(inputs, "tier.keep_across_requests", true);
    set_switch(inputs, "tier.elastic", elastic);
    auto mixed = make_sink(inputs, f, 9, 8);
    set_switch(inputs, "tier.elastic", false);

    configure_cache(cpu.get(), 0, {1, 2}, f.experts);
    configure_cache(mixed.get(), 0, {1, 2}, f.experts);
    char error[512] = {};
    require(kimi_cpu_sink_expert_formats(mixed.get(), table.data(), int(table.size()), error, sizeof(error)), error);

    uint64_t cpu_reads = 0, mixed_reads = 0;
    const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f };
    const std::vector<float> weights = { 0.7f, 0.3f };
    for (int token = 1; token <= 6; ++token) {
        begin_cache(cpu.get(), 94, uint64_t(token));
        begin_cache(mixed.get(), 94, uint64_t(token));
        for (int layer : {1, 2}) {
            const std::vector<int32_t> ids = { int32_t((token + layer) % f.experts), int32_t((token + layer + 1) % f.experts) };
            std::vector<float> input(static_cast<size_t>(f.latent)), expected(input.size()), got(input.size());
            for (int i = 0; i < f.latent; ++i) { input[size_t(i)] = 1.5f * std::sin(float(i * 13 + token * 19 + layer * 3)); }
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                const bool ok = kimi_cpu_sink_run_activation(arm ? mixed.get() : cpu.get(), meta.role[0], meta.role[1], meta.role[2],
                    input.data(), ids.data(), weights.data(), int(ids.size()), &activation, (arm ? got : expected).data(),
                    &stats, error, sizeof(error));
                require(ok, std::string("mixed sink run failed: ") + error);
                require(stats.read_errors == 0 && stats.pending_reads == 0 && stats.fallback_count == 0,
                    "mixed VRAM work did not drain");
                (arm ? mixed_reads : cpu_reads) += stats.read_requests;
                ++checks;
            }
            for (float value : got) { require(std::isfinite(value), "mixed VRAM output is non-finite"); }
            const double rel = relative_l2(got, expected);
            worst_rel = std::max(worst_rel, rel);
            worst_abs = std::max(worst_abs, max_abs_error(got, expected));
            require(rel <= 2e-2, "mixed VRAM output exceeds 2e-2 relative L2 layer=" + std::to_string(layer) +
                " rel=" + std::to_string(rel));
            ++checks;
        }
    }
    require(mixed_reads < cpu_reads, "the tier served neither layer from its slots");
    std::printf("GENERIC_VRAM_MIXED_PASS elastic=%d types=%s/%s,%s/%s latent=%d hidden=%d cpu_reads=%llu mixed_reads=%llu worst_rel_l2=%.6g\n",
        int(elastic), ggml_type_name(f.role_type(1, 0)), ggml_type_name(f.role_type(1, 2)), ggml_type_name(f.role_type(2, 0)),
        ggml_type_name(f.role_type(2, 2)), f.latent, f.hidden, (unsigned long long) cpu_reads, (unsigned long long) mixed_reads, worst_rel);
}

// The warm pool through the sink: a tier of few slots, experts that move on faster than they stay, so that evictions are
// many and a selection comes back to a copy. Copies are 4-bit: the output is near the exact one, not equal to it.
void run_warm(fixture_inputs & inputs, size_t & checks) {
    fixture f;
    f.type = GGML_TYPE_Q8_0;
    f.down_type = GGML_TYPE_Q8_0;
    f.latent = 4096;
    f.hidden = 2048;
    f.experts = 8;
    make_synthetic(f, 20261104u);
    const uint64_t unit = triplet_bytes(f);

    inputs.set("vram_budget", "0");
    set_switch(inputs, "tier.global", false);
    set_switch(inputs, "vram_exclusive", false);
    set_switch(inputs, "vram_async", false);
    set_switch(inputs, "vram_deferred", false);
    auto cpu = make_sink(inputs, f, 9, 8);

    inputs.set("vram_budget", std::to_string(5 * unit).c_str());
    inputs.set("tier.safety_mib", "32");
    set_switch(inputs, "tier.global", true);
    set_switch(inputs, "vram_exclusive", true);
    set_switch(inputs, "vram_async", true);
    set_switch(inputs, "vram_deferred", true);
    set_switch(inputs, "tier.keep_across_requests", true);
    inputs.set("tier.policy", "count_decay2");
    inputs.set("tier.half_life", "16");
    set_switch(inputs, "tier.adaptive_admission", false);
    inputs.set("tier.warm_mib", "64");
    inputs.set("tier.warm_bits", "4");
    auto warm = make_sink(inputs, f, 9, 8);
    inputs.set("tier.warm_mib", "0");
    set_switch(inputs, "tier.adaptive_admission", true);
    inputs.set("tier.policy", "lru");
    inputs.set("tier.half_life", "2");

    const uint64_t ram_bytes = 2 * uint64_t(f.experts) * unit;
    configure_cache(cpu.get(), ram_bytes, {1, 2}, f.experts);
    configure_cache(warm.get(), ram_bytes, {1, 2}, f.experts);
    const kimi_cpu_sink_activation activation = { KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 0.75f };
    const std::vector<float> weights(2, 0.5f);
    double worst = 0.0;
    uint64_t previous_hits = 0;
    for (int token = 1; token <= 80; ++token) {
        begin_cache(cpu.get(), 93, uint64_t(token));
        begin_cache(warm.get(), 93, uint64_t(token));
        for (int layer : {1, 2}) {
            const int phase = token / 6;
            const std::vector<int32_t> ids = { int32_t((phase + layer) % f.experts), int32_t((phase * 3 + layer + 1) % f.experts) };
            if (ids[0] == ids[1]) { continue; }
            std::vector<float> input(static_cast<size_t>(f.latent)), expected(input.size()), got(input.size());
            for (int i = 0; i < f.latent; ++i) { input[size_t(i)] = 1.1f * std::sin(float(i * 5 + token * 11 + layer * 7)); }
            tensors meta(f, layer);
            for (int arm = 0; arm < 2; ++arm) {
                kimi_cpu_sink_stats stats = {};
                char error[512] = {};
                require(kimi_cpu_sink_run_activation(arm ? warm.get() : cpu.get(), meta.role[0], meta.role[1], meta.role[2],
                    input.data(), ids.data(), weights.data(), int(ids.size()), &activation, (arm ? got : expected).data(),
                    &stats, error, sizeof(error)), std::string("warm sink run failed: ") + error);
                require(stats.read_errors == 0 && stats.pending_reads == 0, "warm VRAM work did not drain");
                ++checks;
            }
            for (float value : got) { require(std::isfinite(value), "warm output is non-finite"); }
            kimi_cpu_sink_cache_stats now = {};
            kimi_cpu_sink_cache_get_stats(warm.get(), &now);
            const double rel = relative_l2(got, expected);
            if (now.warm_hits != previous_hits) { worst = std::max(worst, rel); previous_hits = now.warm_hits; }
            require(rel <= 0.25, "warm output exceeds 0.25 relative L2");
            ++checks;
        }
    }
    kimi_cpu_sink_cache_stats stats = {};
    kimi_cpu_sink_cache_get_stats(warm.get(), &stats);
    if (stats.capacity_bytes) { require(stats.warm_hits > 0, "the warm pool served nothing"); }
    std::printf("GENERIC_VRAM_WARM_PASS warm_hits=%llu worst_rel_l2_with_a_copy=%.4g\n", (unsigned long long) stats.warm_hits, worst);
}

} // namespace

int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services());
    try {
        ggml_cpu_init();
        inputs.set("ready_first", "1");
        inputs.set("cache_policy", "lru");
        inputs.set("cache_allocation", "uniform");
        inputs.set("cache_keep", "1");
        inputs.set("cache_reserve_bytes", "536870912");
        inputs.set("tier.policy", "lru");
        inputs.set("tier.min_count", "0");
        inputs.set("tier.safety_mib", "32");
        inputs.set("tier.elastic", "0");
        size_t checks = 0;
        double worst_rel = 0.0;
        double worst_abs = 0.0;
        run_shape(inputs, 512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K);
        run_shape(inputs, 512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        // the production tier's count policy, which takes the predictor's prior (trace.prior) at each request's decode
        inputs.set("tier.policy", "count_decay2");
        inputs.set("tier.half_life", "16");
        run_shape(inputs, 512, 256, 8, {true, "production-generic"}, checks, worst_rel, worst_abs, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K);
        inputs.set("tier.policy", "lru");
        inputs.set("tier.half_life", "2");
        run_lookahead(inputs, checks, worst_rel, worst_abs);
        run_mixed(inputs, false, checks, worst_rel, worst_abs);
        run_mixed(inputs, true, checks, worst_rel, worst_abs);
        run_warm(inputs, checks);
        run_host_jobs(inputs, GGML_TYPE_Q8_0, checks);
        std::printf("REAL_GPU_GENERIC_TIER_PASS checks=%zu worst_rel_l2=%.6g worst_abs=%.6g\n",
            checks, worst_rel, worst_abs);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "REAL_GPU_GENERIC_TIER_FAIL: %s\n", e.what());
        return 1;
    }
}
