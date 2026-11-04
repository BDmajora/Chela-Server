// SPDX-License-Identifier: Apache-2.0
#include "staging.h"
#include "ggml-cuda.h"
#include "gpu-runtime.h"
#include <cstdio>
#include <cstdlib>
namespace ggml_moe_common {
namespace {
void require_release(cudaError_t result) {
    if (result != cudaSuccess) { std::fprintf(stderr, "[ssd.transport] staging release failed: %s\n", cudaGetErrorString(result)); std::abort(); }
}
}

bool object_stage_slot_collect_timing(
        object_stage_ring &ring,
        object_stage_slot &slot,
        float *h2d_ms_out) {
    if (h2d_ms_out) *h2d_ms_out = 0.0f;
    if (!slot.timing_pending || !slot.copy_start || !slot.copy_done) return false;

    bool collected = false;
    if (cudaEventSynchronize(slot.copy_done) == cudaSuccess) {
        float h2d_ms = 0.0f;
        if (cudaEventElapsedTime(&h2d_ms, slot.copy_start, slot.copy_done) == cudaSuccess) {
            ring.h2d_ms += h2d_ms;
            ++ring.h2d_timed;
            if (h2d_ms_out) *h2d_ms_out = h2d_ms;
            collected = true;
        }
    }
    slot.timing_pending = false;
    return collected;
}

bool object_stage_slot_synchronize(
        object_stage_ring &ring,
        object_stage_slot &slot) {
    if (!slot.pending) return true;
    if (!slot.done || cudaEventSynchronize(slot.done) != cudaSuccess) return false;
    object_stage_slot_publish_ready(slot);
    ++ring.waits;
    return true;
}

bool object_stage_slot_record(
        object_stage_slot &slot,
        cudaStream_t stream) {
    if (!slot.done || !stream || slot.pending ||
            cudaEventRecord(slot.done, stream) != cudaSuccess) {
        return false;
    }
    slot.pending = true;
    return true;
}

void object_stage_ring_release(object_stage_ring &ring) {
    if (!object_io_ring_release(ring)) { std::fprintf(stderr, "[ssd.transport] cannot drain staging reads\n"); std::abort(); }
    for (object_stage_slot &slot : ring.slots) {
        if (slot.pending && slot.done) {
            require_release(cudaEventSynchronize(slot.done));
            slot.pending = false;
        }
        (void)object_stage_slot_collect_timing(ring, slot, nullptr);
        if (slot.host) {
            require_release(cudaFreeHost(slot.host));
            slot.host = nullptr;
        }
        slot.host_capacity = 0;
        if (slot.done) {
            require_release(cudaEventDestroy(slot.done));
            slot.done = nullptr;
        }
        if (slot.copy_start) {
            require_release(cudaEventDestroy(slot.copy_start));
            slot.copy_start = nullptr;
        }
        if (slot.copy_done) {
            require_release(cudaEventDestroy(slot.copy_done));
            slot.copy_done = nullptr;
        }
    }
    ring.slots.clear();
    ring.slot_sz = 0;
    ring.next = 0;
}

bool object_stage_slot_allocate(
        object_stage_slot &slot,
        size_t slot_size,
        bool create_timing_events) {
    if (slot_size == 0 || slot.host || slot.done || slot.copy_start || slot.copy_done) return false;
    (void) ggml_cuda_admit_host_alloc(slot_size); // unified memory: the reserve first, as for device memory
    if (cudaHostAlloc(&slot.host, slot_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming) != cudaSuccess) {
        if (slot.host) cudaFreeHost(slot.host);
        if (slot.done) cudaEventDestroy(slot.done);
        slot.host = nullptr;
        slot.done = nullptr;
        return false;
    }
    slot.host_capacity = slot_size;
    if (create_timing_events &&
            (cudaEventCreate(&slot.copy_start) != cudaSuccess ||
             cudaEventCreate(&slot.copy_done) != cudaSuccess)) {
        if (slot.copy_start) cudaEventDestroy(slot.copy_start);
        if (slot.copy_done) cudaEventDestroy(slot.copy_done);
        slot.copy_start = nullptr;
        slot.copy_done = nullptr;
    }
    return true;
}

bool object_stage_slot_ensure_capacity(
        object_stage_slot &slot,
        size_t slot_size,
        bool create_timing_events) {
    if (slot_size == 0 || slot.pending) return false;
    const bool has_timing = slot.copy_start && slot.copy_done;
    if (slot.host && slot.done && slot.host_capacity >= slot_size &&
            (!create_timing_events || has_timing)) {
        return true;
    }
    if (slot.host) cudaFreeHost(slot.host);
    if (slot.done) cudaEventDestroy(slot.done);
    if (slot.copy_start) cudaEventDestroy(slot.copy_start);
    if (slot.copy_done) cudaEventDestroy(slot.copy_done);
    slot.host = nullptr;
    slot.host_capacity = 0;
    slot.done = nullptr;
    slot.copy_start = nullptr;
    slot.copy_done = nullptr;
    slot.timing_pending = false;
    return object_stage_slot_allocate(slot, slot_size, create_timing_events);
}

bool object_stage_ring_allocate(
        object_stage_ring &ring,
        size_t slot_size,
        size_t slot_count,
        bool create_timing_events) {
    if (ring.failed || slot_size == 0 || slot_count == 0 || !ring.slots.empty()) return false;

    ring.slot_sz = slot_size;
    ring.slots.resize(slot_count);
    for (object_stage_slot &slot : ring.slots) {
        if (!object_stage_slot_allocate(slot, ring.slot_sz, create_timing_events)) {
            object_stage_ring_release(ring);
            ring.failed = true;
            return false;
        }
    }
    return true;
}

} // namespace ggml_moe_common
