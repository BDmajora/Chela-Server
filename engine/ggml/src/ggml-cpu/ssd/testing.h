// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "runtime.h"
#include "../../ggml-ssd/testing/tier.h"
#include <map>
#include <string>
#include <vector>

namespace ssd {
namespace cpu {
namespace testing {
// Explicit construction policies for numerical and resource lifecycle fixtures. The
// production configured factory always builds its canonical policy and never
// consumes this structure or any ambient policy input.
struct execution_policy {
    bool ready_first = false, early_up_gate = false, role_major = false;
    bool nonblocking_down_sched = false, dynamic_rows = false, read_fixed = false;
    int poll_us = 0, dynamic_chunk_rows = 64;
    std::vector<double> row_weights; // one per participant of each run
    std::vector<int> io_cpus;
    int decode_participants = 0, prefetch_depth = 0;
    uint64_t prefetch_fail_every = 0, prefetch_throw_drain = 0, prefetch_throw_schedule = 0;
    std::string cache_policy = "window", cache_allocation = "priority";
    bool cache_keep = false, cache_prefill_read = false, cache_elastic = false, cache_partial_reclaim = false;
    bool cache_reclaimable_file = false;
    uint64_t cache_reserve_bytes = 2ull << 30;
    size_t cache_prefill_slots = 0;
    std::map<int32_t, size_t> cache_layer_quotas;
    uint64_t vram_budget = 0;
    bool vram_async = false, vram_exclusive = false, vram_deferred = false, vram_prefill = false;
    bool vram_locations = false;
    ssd::gpu::testing::tier_options tier;
    std::map<std::string, double> place_costs;
    // Request-level expert prediction (ssd::place::expert_predictor): the requests it traces, whether the expert caches
    // start each request's decode from its prediction, and the file that keeps it across runs. Off unless one is set.
    size_t trace_capacity = 0;
    bool trace_prior = false;
    std::string trace_file;
    int vram_lookahead = 0; // the next layer's predicted experts that go to the GPU tier ahead of their selection
    int drop_mass = 0;      // routing weight, percent, decode may drop among experts no cache holds (decode.drop_mass)
};

// Test callers own explicit policy inputs until this call returns; the executor
// copies them before starting its threads. No process-wide mutation is involved.
[[nodiscard]] GGML_BACKEND_API kimi_cpu_sink * create(
    int ring_experts, const execution_policy & policy,
    const runtime_services * services, kimi_cpu_sink_resolver resolver, void * userdata,
    char * error, size_t error_size) noexcept;
}  // namespace testing
}  // namespace cpu
}  // namespace ssd
