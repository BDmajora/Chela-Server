// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>
// Shared plain records preserve the existing private callback ABI.
struct kimi_cpu_sink_stats {
    uint64_t experts;
    uint64_t read_requests;
    uint64_t read_bytes;
    uint64_t read_errors;
    uint64_t fallback_count; // always zero; errors never use tensor->data
    uint64_t ring_bytes;
    double total_us;
    double io_us;           // I/O job wall time, overlaps compute_us
    double compute_us;      // conversion + CPU phases, excluding I/O waits
    double wait_io_us;
    double input_quantize_us;
    double hidden_quantize_us;
    double up_gate_us;
    double down_reduce_us;
    uint64_t demand_bytes;       // all live slots, including cache hits
    uint64_t cache_hits;         // only entries resident before this token
    uint64_t cache_hit_bytes;
    uint64_t cache_misses;
    uint64_t cache_admitted;
    uint64_t cache_admitted_bytes;
    double cache_copy_us;        // zero: admissions read directly into cache slots
    uint64_t cache_fill_read_requests; // submitted SSD reads targeting cache payload
    uint64_t cache_fill_read_bytes;    // positive CQE bytes, including a possible short read
    uint64_t ring_read_requests;
    uint64_t ring_read_bytes;
    uint64_t fixed_read_requests;      // READ_FIXED SQEs; only ring targets are eligible
    uint64_t read_completions;         // consumed CQEs, including error-path drain
    uint64_t pending_reads;            // must be zero when run returns, including errors
    uint64_t cache_new_slot_bytes;     // previously untouched slots reserved as read targets
    // PROFILE-only samples. The first three are thread CPU time, not wall time.
    // compute_cpu_us includes all worker kernels and decode quantization; caller
    // overlaps worker 0 and quantization, so those counters must not be added.
    double compute_cpu_us;
    double caller_cpu_us;
    double io_cpu_us;
    double io_wait_us;        // io_uring GETEVENTS wall time, overlaps compute
    double io_credit_wait_us; // I/O thread waiting for a compute-owned ring slot, nothing in flight (always set)
    double other_us;          // total_us - compute_us - wait_io_us (caller wall)
    uint64_t ready_scratch_bytes; // retained vector capacities, zero when disabled
    uint64_t ready_out_of_order_experts; // executed at a different ordinal
    double ordered_reduce_us; // ready-first final slot-order sum; inside compute_us
};

struct kimi_cpu_sink_cache_stats {
    uint64_t budget_bytes;       // hard payload mapping limit; metadata/ring are separate
    uint64_t capacity_bytes;
    uint64_t mapped_bytes;
    uint64_t committed_bytes;    // upper bound: resident slots; MADV_DONTNEED removes them
    uint64_t enabled_bytes;      // slots eligible for hits/admission in the current phase
    uint64_t valid_bytes;
    uint64_t peak_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t admitted_bytes;
    uint64_t evictions;
    uint64_t admission_skips;
    uint64_t clears;
    uint64_t pressure_stops;
    double copy_us;
    uint64_t page_table_bytes;   // mapping-lifetime upper bound for touched slots; not measured RSS
    uint64_t last_guard_extra_bytes; // most recent incremental cgroup check, including allocation overhead
    uint64_t peak_guard_extra_bytes;
    uint64_t shrink_ops;             // successful phase/pressure shrink operations
    uint64_t phase_grows;            // phase-elastic decode logical grows (no prefault)
    uint64_t released_bytes;         // cumulative resident payload released
    uint64_t refaulted_bytes;        // cumulative released payload reserved again
    uint64_t guard_reclaims;         // shrink operations initiated by the guard
    uint64_t shrink_us;              // cumulative synchronous shrink wall time
    uint64_t prefill_hits;            // read-only prefill borrows; excluded from decode heat
    uint64_t prefill_misses;
    uint64_t prefill_hit_bytes;
    uint64_t prefill_leases;          // successfully acquired read-only leases
    uint64_t prior_layers;            // GPU tier layers that took a request's predicted selections (trace.prior)
    uint64_t lookahead_uploads;       // predicted experts uploaded ahead of their selection (vram.lookahead)
    uint64_t warm_hits;               // requests the GPU tier served from a low-precision copy (vram.warm_mib)
};
