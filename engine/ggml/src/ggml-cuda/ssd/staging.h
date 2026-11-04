// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-ssd/store/object-identity.h"
#include "ggml-ssd/io/object-uring.h"
#include <vector>
#if defined(GGML_USE_HIP)
#include "gpu-runtime.h" // cudaEvent_t and cudaStream_t name HIP's handles
#else
struct CUevent_st;
using cudaEvent_t = CUevent_st *;
struct CUstream_st;
using cudaStream_t = CUstream_st *;
#endif
namespace ggml_moe_common {
using namespace ssd::io;
struct object_stage_slot {
    void *host = nullptr;
    size_t host_capacity = 0;
    cudaEvent_t done = nullptr;
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_done = nullptr;
    bool pending = false;
    bool timing_pending = false;
    bool object_ram_hit_timing_pending = false;
    int object_ram_hit_layer = -1;
    uint64_t object_ram_hit_bytes = 0;
    object_key object;
    uint64_t object_used = 0;
    uint64_t object_read_us = 0;
};

struct object_stage_ring : ssd::io::object_io_ring {
    std::vector<object_stage_slot> slots;
    size_t slot_sz = 0;
    size_t next = 0;
    bool failed = false;
    bool report_registered = false;
    uint64_t copies = 0;
    uint64_t waits = 0;
    uint64_t fallbacks = 0;
    uint64_t h2d_timed = 0;
    double slot_wait_ms = 0.0;
    double host_stage_ms = 0.0;
    double enqueue_ms = 0.0;
    double h2d_ms = 0.0;
};
bool object_stage_ring_allocate(
        object_stage_ring &ring,
        size_t slot_size,
        size_t slot_count,
        bool create_timing_events);
bool object_stage_slot_allocate(
        object_stage_slot &slot,
        size_t slot_size,
        bool create_timing_events);
bool object_stage_slot_ensure_capacity(
        object_stage_slot &slot,
        size_t slot_size,
        bool create_timing_events);
bool object_stage_slot_collect_timing(
        object_stage_ring &ring,
        object_stage_slot &slot,
        float *h2d_ms_out);
bool object_stage_slot_synchronize(
        object_stage_ring &ring,
        object_stage_slot &slot);

bool object_stage_slot_record(
        object_stage_slot &slot,
        cudaStream_t stream);
void object_stage_ring_release(object_stage_ring &ring);
inline void object_stage_slot_publish_ready(object_stage_slot &slot) { slot.pending = false; }
} // namespace ggml_moe_common
