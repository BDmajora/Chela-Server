// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/gpu/transport.h"
#include "ggml-ssd/store/source.h"
#include "ggml-ssd/host/arbiter.h"
#include "staging.h"
#include "ggml-ssd/io/submit-window.h"
#include "gpu-runtime.h"
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace {
using namespace ssd::io;
using ring_type = ggml_moe_common::object_stage_ring;
using slot_type = ggml_moe_common::object_stage_slot;
constexpr size_t alignment = 4096, stage_slots = 16, io_depth = 32, scheduler_refill = 16;
[[noreturn]] void fatal(const char * reason) noexcept {
    std::fprintf(stderr, "[ssd.transport] cannot drain owned work: %s\n", reason); std::abort();
}
uint64_t align_up_u64(uint64_t value, uint64_t a) { return (value + a - 1) / a * a; }
struct extent { int fd = -1; uint64_t offset = 0, file_size = 0; size_t bytes = 0; };
struct transport_owner;
struct dense_stage_half {
    std::string tensor;
    size_t n_blocks = 0, block_bytes = 0;
    std::vector<size_t> prefix, expect;
    std::vector<uint8_t> done;
    size_t in_flight = 0;
    bool failed = false;
};
struct dense_stage_state {
    transport_owner * owner = nullptr;
    ring_type ring;
    size_t blocks_per_half = 0, need_blocks = 0, need_slot = 0;
    bool failed = false;
    dense_stage_half half[2];
    int last = 1;
};
struct transport_owner {
    ssd::store::source_view source;
    ssd::host::context_client * host;
    int device;
    std::mutex mutex, seed_mutex;
    std::atomic<bool> closed{false};
    ring_type ring;
    dense_stage_state dense;
    std::unordered_map<int, uint64_t> source_sizes;
    uint64_t reads = 0, bytes = 0;
    ssd::gpu::seed_offer_fn seed = nullptr;
    void * seed_data = nullptr;
    cudaEvent_t ready = nullptr, pending = nullptr, last_seed_done = nullptr;
    transport_owner(const ssd::runtime_services & services, int ordinal) : source(*services.source), host(services.host), device(ordinal) { dense.owner = this; }
    bool resolve(const char * name, int expert, size_t bytes, int64_t experts, extent & out) {
        size_t logical = 0;
        if (!source.resolve(source.userdata, name, expert, bytes, experts, &out.fd, &out.offset, &logical) || logical != bytes || out.fd < 0) return false;
        auto found = source_sizes.find(out.fd);
        if (found == source_sizes.end()) {
            struct stat info{};
            const int flags = fcntl(out.fd, F_GETFL);
            if (fstat(out.fd, &info) || info.st_size < 0 || !S_ISREG(info.st_mode) || flags < 0 || !(flags & O_DIRECT)) return false;
            found = source_sizes.emplace(out.fd, uint64_t(info.st_size)).first;
        }
        out.file_size = found->second; out.bytes = bytes;
        return out.offset <= out.file_size && bytes <= out.file_size - out.offset;
    }
    void release(ring_type & value) noexcept {
        bool allocated = !value.slots.empty();
        if (!object_io_ring_drain(value)) fatal("io_uring");
        for (auto & slot : value.slots) {
            if (!ggml_moe_common::object_stage_slot_synchronize(value, slot)) fatal("staging copy");
        }
        ggml_moe_common::object_stage_ring_release(value);
        if (allocated && host) { (void)host->note_known_memory_change(); }
    }
    bool allocate(ring_type & value, size_t slot_size, size_t slots) {
        if (!slot_size || !slots || slots > SIZE_MAX / slot_size || value.failed) return false;
        if (value.slot_sz >= slot_size && value.slots.size() == slots) return true;
        release(value);
        auto ticket = host->take(slot_size * slots);
        if (!ticket) return false;
        if (!ggml_moe_common::object_stage_ring_allocate(value, slot_size, slots, false)) return false;
        // Pinning and the first writes are both inside the reservation interval.
        for (auto & slot : value.slots) std::memset(slot.host, 0, slot.host_capacity);
        ticket.value.commit();
        return true;
    }
};
struct copy_job { void * dst; extent source; };
struct read_plan { size_t job; uint64_t offset; size_t read_sz, prefix, expect; };

// Fixed production profile of the previous scheduler path: no experimental
// coalescing/sorting/cache producer; initial batch and CQE refill order retained.
bool copy_jobs(transport_owner & owner, const std::vector<copy_job> & jobs, cudaStream_t stream) {
    if (jobs.empty()) return true;
    std::vector<read_plan> plans; plans.reserve(jobs.size());
    size_t maximum = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto & source = jobs[i].source;
        const uint64_t offset = source.offset / alignment * alignment;
        const size_t prefix = size_t(source.offset - offset);
        if (source.bytes > UINT_MAX - prefix - alignment + 1) return false;
        const size_t read = size_t(align_up_u64(prefix + source.bytes, alignment));
        const size_t expected = size_t(std::min<uint64_t>(read, source.file_size - offset));
        if (expected < prefix + source.bytes || read > INT_MAX) return false;
        plans.push_back({i, offset, read, prefix, expected}); maximum = std::max(maximum, read);
    }
    auto & ring = owner.ring;
    if (!owner.allocate(ring, maximum, stage_slots)) return false;
    const size_t depth = std::min(io_depth, ring.slots.size());
    if (!object_io_ring_ensure(ring, depth, 0, nullptr)) return false;
    struct drain_guard {
        ring_type & ring; cudaStream_t stream; bool success = false;
        ~drain_guard() {
            if (!object_io_ring_drain(ring)) fatal("copy read queue");
            if (!success && cudaStreamSynchronize(stream) != cudaSuccess) fatal("failed copy stream");
        }
    } drain{ring, stream};
    struct pending_job { size_t group_idx = 0, slot_idx = 0; };
    std::vector<pending_job> pending(depth);
    std::vector<object_io_window_slot> window(depth);
    std::vector<size_t> free_pending;
    for (size_t i = 0; i < depth; ++i) free_pending.push_back(depth - 1 - i);
    struct reusable_pending { size_t pending_idx, slot_idx; };
    std::vector<reusable_pending> reusable; reusable.reserve(depth);
    size_t next = 0, inflight = 0, completed = 0;
    auto submit_one = [&](size_t group, size_t slot_idx, size_t pending_idx) {
        const auto & plan = plans[group]; const auto & job = jobs[plan.job];
        auto & slot = ring.slots[slot_idx];
        if (!ggml_moe_common::object_stage_slot_synchronize(ring, slot)) return false;
        pending[pending_idx] = {group, slot_idx};
        const object_io_window_request request{job.source.fd, slot.host, plan.read_sz, plan.offset, group, false, plan.expect};
        return object_io_window_prepare_read(ring, window.data(), window.size(), pending_idx, request);
    };
    auto complete = [&](const object_io_completion & completion) {
        object_io_window_completion resolved;
        const auto status = object_io_window_resolve(ring, window.data(), window.size(), completion, resolved);
        if (status != object_io_completion_status::complete || resolved.window_slot >= pending.size()) return false;
        const auto done = pending[resolved.window_slot];
        if (resolved.request_id != done.group_idx) return false;
        const auto & plan = plans[done.group_idx]; const auto & job = jobs[plan.job]; auto & slot = ring.slots[done.slot_idx];
        if (cudaMemcpyAsync(job.dst, static_cast<const char *>(slot.host) + plan.prefix, job.source.bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        if (!ggml_moe_common::object_stage_slot_record(slot, stream)) return false;
        ++owner.reads; owner.bytes += job.source.bytes; ++ring.copies;
        ++completed; --inflight; reusable.push_back({resolved.window_slot, done.slot_idx});
        return true;
    };
    const bool initial = ggml_moe_common::object_io_submit_initial(plans.size(), depth, 0,
        [&](size_t group) {
            const size_t id = free_pending.back(); free_pending.pop_back();
            const size_t slot = ring.next++ % ring.slots.size();
            if (!submit_one(group, slot, id)) return false;
            ++next; ++inflight; return true;
        }, [&](size_t) { return object_io_ring_submit_count(ring); }, [] { return true; });
    if (!initial) return false;
    while (completed < plans.size()) {
        object_io_completion completion;
        if (!object_io_ring_wait(ring, completion) || !complete(completion)) return false;
        while (completed < plans.size() && inflight > 0) {
            object_io_completion extra;
            const auto ready = object_io_ring_peek(ring, extra);
            if (ready == object_io_poll_result::empty) break;
            if (ready == object_io_poll_result::error || !complete(extra)) return false;
        }
        size_t submitted = 0;
        while (next < plans.size() && inflight < depth && !reusable.empty() && submitted < std::min(scheduler_refill, depth)) {
            const auto item = reusable.back(); reusable.pop_back();
            if (!submit_one(next, item.slot_idx, item.pending_idx)) return false;
            ++next; ++inflight; ++submitted;
        }
        if (submitted && !object_io_ring_submit(ring)) return false;
    }
    object_io_window_reset(window.data(), window.size()); drain.success = true;
    return true;
}

static size_t dense_slot_bytes(size_t block_bytes) {
    const size_t a = alignment;
    return (size_t)align_up_u64((uint64_t)block_bytes + a, a);
}

// the blocks of a matrix the index holds as n_blocks x block_bytes == nbytes, n >= 1; 0 when it holds none
static size_t dense_index_blocks(transport_owner & owner, const char * name, size_t bytes, size_t * block) {
    return owner.source.dense_blocks(owner.source.userdata, name, bytes, block);
}

#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
// the ring failed: every read in flight is reaped unmatched (no read lands in a slot later) and its half failed
static void dense_stage_abort(dense_stage_state &s) {
    (void)ggml_moe_common::object_io_ring_drain(s.ring);
    for (dense_stage_half &half : s.half) {
        half.failed = half.failed || half.in_flight > 0;
        half.in_flight = 0;
    }
}

// takes one completion of the stage's ring and marks its block (a failed or short read fails its half); false when
// the ring gives none (dense_stage_abort)
static bool dense_stage_reap_one(dense_stage_state &s) {
    ggml_moe_common::object_io_completion c;
    if (!ggml_moe_common::object_io_ring_wait(s.ring, c)) {
        dense_stage_abort(s);
        return false;
    }
    const size_t slot = (size_t)c.user_data - 1;
    if (c.user_data == 0 || s.blocks_per_half == 0 || slot >= 2 * s.blocks_per_half) return true;
    dense_stage_half &half = s.half[slot / s.blocks_per_half];
    const size_t i = slot % s.blocks_per_half;
    if (half.in_flight > 0) --half.in_flight;
    if (i >= half.n_blocks || c.result < 0 || (size_t)c.result != half.expect[i]) {
        half.failed = true;
    } else {
        half.done[i] = 1;
    }
    return true;
}

// no read in flight in half h; false when one of its reads failed
static bool dense_stage_drain(dense_stage_state &s, int h) {
    while (s.half[h].in_flight > 0 && dense_stage_reap_one(s)) {
    }
    return !s.half[h].failed;
}

// the stage at the size of the largest matrix asked about: every read finished, every slot's copy done; a stage that
// could not be allocated is not tried again
static bool dense_stage_ensure(dense_stage_state &s, size_t n_blocks, size_t slot_bytes) {
    s.need_blocks = std::max(s.need_blocks, n_blocks);
    s.need_slot = std::max(s.need_slot, slot_bytes);
    if (s.blocks_per_half >= n_blocks && s.ring.slot_sz >= slot_bytes) {
        return true;
    }
    if (s.failed) {
        return false;
    }
    for (int h = 0; h < 2; ++h) {
        if (!s.ring.slots.empty()) (void)dense_stage_drain(s, h);
        s.half[h] = dense_stage_half();
    }
    s.owner->release(s.ring);
    s.ring = ring_type();
    if (!s.owner->allocate(s.ring, s.need_slot, 2 * s.need_blocks) ||
            !ggml_moe_common::object_io_ring_ensure(s.ring, 2 * s.need_blocks, 0, nullptr)) {
        std::fprintf(stderr, "[moe_stream_batch] dense stage: init failed for 2 x %zu slots of %.2f MiB\n",
                     s.need_blocks, s.need_slot / (1024.0 * 1024.0));
        s.owner->release(s.ring);
        s.ring = ring_type();
        s.blocks_per_half = 0;
        s.failed = true;
        return false;
    }
    s.blocks_per_half = s.need_blocks;
    std::fprintf(stderr, "[moe_stream_batch] dense stage: 2 x %zu slots of %.2f MiB pinned\n",
                 s.blocks_per_half, s.ring.slot_sz / (1024.0 * 1024.0));
    return true;
}

// submits the reads of a matrix into half h, its slots' earlier copies finished first
static bool dense_stage_submit(dense_stage_state &s, int h, const char *tensor_name, size_t n_blocks,
                               size_t block_bytes) {
    dense_stage_half &half = s.half[h];
    (void)dense_stage_drain(s, h);
    half = dense_stage_half();
    half.tensor = tensor_name;
    half.n_blocks = n_blocks;
    half.block_bytes = block_bytes;
    half.prefix.assign(n_blocks, 0);
    half.expect.assign(n_blocks, 0);
    half.done.assign(n_blocks, 0);
    const size_t a = alignment;
    for (size_t i = 0; i < n_blocks; ++i) {
        extent resolved;
        const bool located = s.owner->resolve(tensor_name, int(i), block_bytes, int64_t(n_blocks), resolved);
        const extent * src = located ? &resolved : nullptr;
        slot_type &slot = s.ring.slots[(size_t)h * s.blocks_per_half + i];
        if (!src || src->fd < 0 || !ggml_moe_common::object_stage_slot_synchronize(s.ring, slot)) {
            half.failed = true;
            break;
        }
        const uint64_t off = src->offset / a * a;
        half.prefix[i] = (size_t)(src->offset - off);
        const size_t read_sz = (size_t)align_up_u64(half.prefix[i] + block_bytes, a);
        half.expect[i] = src->file_size > off && src->file_size - off < read_sz ? (size_t)(src->file_size - off) : read_sz;
        if (read_sz > slot.host_capacity || half.expect[i] < half.prefix[i] + block_bytes ||
                !ggml_moe_common::object_io_ring_prepare_read(s.ring, src->fd, slot.host, read_sz, off,
                    (uint64_t)h * s.blocks_per_half + i + 1)) {
            half.failed = true;
            break;
        }
        ++half.in_flight;
    }
    if (!ggml_moe_common::object_io_ring_submit(s.ring)) {
        dense_stage_abort(s);
    }
    if (half.failed) {
        (void)dense_stage_drain(s, h);
        half = dense_stage_half();
        return false;
    }
    return true;
}
#endif

bool dense_has_impl(transport_owner & owner, const char *tensor_name, size_t nbytes, size_t *block_bytes) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    size_t bb = 0;
    const size_t n = dense_index_blocks(owner, tensor_name, nbytes, &bb);
    if (n == 0) return false;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    owner.dense.need_blocks = std::max(owner.dense.need_blocks, n);
    owner.dense.need_slot = std::max(owner.dense.need_slot, dense_slot_bytes(bb));
    if (block_bytes) *block_bytes = bb;
    return true;
#else
    (void)tensor_name;
    (void)nbytes;
    (void)block_bytes;
    return false;
#endif
}

// the stage for the largest matrix asked about so far, allocated now (the load receipt, once every weight homed on the
// drive was asked about): its pinned bytes; false when it cannot be
bool dense_prepare_impl(transport_owner & owner, size_t *pinned_bytes) {
    if (pinned_bytes) *pinned_bytes = 0;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    dense_stage_state &s = owner.dense;
    if (s.need_blocks == 0 || !dense_stage_ensure(s, s.need_blocks, s.need_slot)) {
        return false;
    }
    if (pinned_bytes) *pinned_bytes = 2 * s.blocks_per_half * s.ring.slot_sz;
    return true;
#else
    return false;
#endif
}

bool dense_read_ahead_impl(transport_owner & owner, const char *tensor_name, size_t nbytes) {
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    size_t bb = 0;
    const size_t n = dense_index_blocks(owner, tensor_name, nbytes, &bb);
    if (n == 0) return false;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    dense_stage_state &s = owner.dense;
    for (const dense_stage_half &h : s.half) {
        if (h.tensor == tensor_name) return true;
    }
    if (!dense_stage_ensure(s, n, dense_slot_bytes(bb))) return false;
    return dense_stage_submit(s, 1 - s.last, tensor_name, n, bb);
#else
    (void)tensor_name;
    (void)nbytes;
    return false;
#endif
}

bool dense_copy_impl(transport_owner & owner,
        const char *tensor_name,
        void *device_dst,
        size_t nbytes,
        void *stream_ptr,
        bool *read_ahead_hit) {
    if (read_ahead_hit) *read_ahead_hit = false;
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING) && !defined(_WIN32)
    size_t bb = 0;
    const size_t n = dense_index_blocks(owner, tensor_name, nbytes, &bb);
    if (n == 0 || !device_dst || !stream_ptr) return false;
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
    dense_stage_state &s = owner.dense;
    if (!dense_stage_ensure(s, n, dense_slot_bytes(bb))) return false;
    int h = s.half[0].tensor == tensor_name ? 0 : s.half[1].tensor == tensor_name ? 1 : -1;
    if (h >= 0 && (s.half[h].n_blocks != n || s.half[h].block_bytes != bb || s.half[h].failed)) {
        (void)dense_stage_drain(s, h);
        s.half[h] = dense_stage_half();
        h = -1;
    }
    if (read_ahead_hit) *read_ahead_hit = h >= 0;
    if (h < 0) {
        h = 1 - s.last;
        if (!dense_stage_submit(s, h, tensor_name, n, bb)) return false;
    }
    dense_stage_half &half = s.half[h];
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bool ok = true;
    for (size_t i = 0; i < n && ok; ++i) {
        while (!half.done[i] && !half.failed && dense_stage_reap_one(s)) {
        }
        slot_type &slot = s.ring.slots[(size_t)h * s.blocks_per_half + i];
        ok = half.done[i] &&
            cudaMemcpyAsync((char *)device_dst + i * bb, (const char *)slot.host + half.prefix[i], bb,
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            ggml_moe_common::object_stage_slot_record(slot, stream);
    }
    (void)dense_stage_drain(s, h);
    half = dense_stage_half();
    s.last = h;
    if (!ok && cudaStreamSynchronize(stream) != cudaSuccess) fatal("dense failure stream");
    return ok;
#else
    (void)tensor_name;
    (void)device_dst;
    (void)nbytes;
    (void)stream_ptr;
    return false;
#endif
}

void quiesce(void * value) noexcept {
    if (!value) return;
    auto & owner = *static_cast<transport_owner *>(value);
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.closed) return;
    owner.closed = true;
    if (cudaSetDevice(owner.device) != cudaSuccess) fatal("device selection");
    owner.release(owner.ring);
#if defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
    dense_stage_abort(owner.dense);
#endif
    owner.release(owner.dense.ring);
    for (auto & half : owner.dense.half) half = dense_stage_half();
    std::lock_guard<std::mutex> seed_lock(owner.seed_mutex);
    if (owner.last_seed_done && cudaEventSynchronize(owner.last_seed_done) != cudaSuccess) fatal("last seed copy");
    if (owner.ready && cudaEventSynchronize(owner.ready) != cudaSuccess) fatal("seed source");
    if (owner.ready && cudaEventDestroy(owner.ready) != cudaSuccess) fatal("seed event");
    owner.pending = owner.ready = owner.last_seed_done = nullptr;
}
void destroy(void * value) noexcept {
    if (!value) return;
    quiesce(value);
    auto * owner = static_cast<transport_owner *>(value);
    if (owner->seed) fatal("tier still registered during owner destruction");
    delete owner;
}
void * create(const ssd::runtime_services * services, int device, char * error, size_t size) noexcept {
    try {
#if !defined(GGML_MOE_OBJECT_TRANSPORT_HAS_LIBURING)
        (void)services; (void)device;
        throw std::runtime_error("SSD CUDA transport requires Linux liburing; no fallback");
#else
        if (!services || !services->host || !services->source || !services->source->resolve ||
                !services->source->dense_blocks || device < 0) throw std::runtime_error("invalid context transport services");
        return new transport_owner(*services, device);
#endif
    } catch (const std::exception & failure) {
        if (error && size) std::snprintf(error, size, "%s", failure.what());
        return nullptr;
    } catch (...) {
        if (error && size) std::snprintf(error, size, "transport construction failed");
        return nullptr;
    }
}
bool copy_ranges(void * value, const char * name, const int32_t * first, const int32_t * last, size_t count,
        int64_t experts, size_t expert_bytes, size_t padding, bool cache_bypass, void * destination, size_t capacity, void * stream_ptr,
        uint64_t * copied, uint64_t * reads, uint64_t * bytes, uint64_t * lookups, uint64_t * hits) noexcept {
    if (copied) *copied = 0;
    if (reads) *reads = 0;
    if (bytes) *bytes = 0;
    if (lookups) *lookups = 0;
    if (hits) *hits = 0;
    (void)cache_bypass; // The retired process-global cache producer is never consulted.
    if (!value || !name || !name[0] || !first || !last || !count || !expert_bytes || experts <= 0 || experts > INT32_MAX ||
            !destination || !stream_ptr || uint64_t(experts) > SIZE_MAX / expert_bytes || size_t(experts) * expert_bytes > capacity) return false;
    auto & owner = *static_cast<transport_owner *>(value);
    try {
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (owner.closed || cudaSetDevice(owner.device) != cudaSuccess) return false;
        std::vector<copy_job> jobs;
        int32_t previous = -1;
        for (size_t range = 0; range < count; ++range) {
            if (first[range] < 0 || last[range] < first[range] || last[range] >= experts || first[range] <= previous) return false;
            previous = last[range];
            for (int32_t expert = first[range]; expert <= last[range]; ++expert) {
                extent source;
                if (!owner.resolve(name, expert, expert_bytes, experts, source)) return false;
                jobs.push_back({static_cast<char *>(destination) + size_t(expert) * expert_bytes, source});
            }
        }
        const uint64_t before_reads = owner.reads, before_bytes = owner.bytes;
        const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        bool ok = copy_jobs(owner, jobs, stream);
        if (ok && padding) {
            for (size_t range = 0; range < count; ++range) {
                if (last[range] >= experts - 1) continue;
                const size_t offset = (size_t(last[range]) + 1) * expert_bytes;
                if (offset > capacity || padding > capacity - offset ||
                        cudaMemsetAsync(static_cast<char *>(destination) + offset, 0, padding, stream) != cudaSuccess) { ok = false; break; }
            }
        }
        if (!ok && cudaStreamSynchronize(stream) != cudaSuccess) fatal("copy failure stream");
        if (copied && ok) *copied = jobs.size();
        if (reads) *reads = owner.reads - before_reads;
        if (bytes) *bytes = owner.bytes - before_bytes;
        return ok;
    } catch (...) { quiesce(value); return false; }
}
bool dense_has(void * p, const char * name, size_t bytes, size_t * block) noexcept {
    if (!p) return false;
    try { return dense_has_impl(*static_cast<transport_owner *>(p), name, bytes, block); }
    catch (...) { quiesce(p); return false; }
}
bool dense_prepare(void * p, size_t * bytes) noexcept {
    if (!p) return false;
    try { return dense_prepare_impl(*static_cast<transport_owner *>(p), bytes); }
    catch (...) { quiesce(p); return false; }
}
bool dense_read_ahead(void * p, const char * name, size_t bytes) noexcept {
    if (!p) return false;
    try { return dense_read_ahead_impl(*static_cast<transport_owner *>(p), name, bytes); }
    catch (...) { quiesce(p); return false; }
}
bool dense_copy(void * p, const char * name, void * dst, size_t bytes, void * stream, bool * hit) noexcept {
    if (!p) return false;
    try { return dense_copy_impl(*static_cast<transport_owner *>(p), name, dst, bytes, stream, hit); }
    catch (...) { quiesce(p); return false; }
}
bool seed_register(void * p, ssd::gpu::seed_offer_fn callback, void * data) noexcept {
    if (!p || !callback || !data) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.closed) return false;
    if (owner.seed) return owner.seed == callback && owner.seed_data == data;
    owner.seed = callback; owner.seed_data = data; return true;
}
bool seed_unregister(void * p, ssd::gpu::seed_offer_fn callback, void * data) noexcept {
    if (!p) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.seed != callback || owner.seed_data != data) return false;
    if (owner.last_seed_done && cudaEventSynchronize(owner.last_seed_done) != cudaSuccess) fatal("seed unregister");
    owner.seed = nullptr; owner.seed_data = nullptr; owner.pending = owner.last_seed_done = nullptr; return true;
}
bool seed_offer(void * p, const char * name, const void * data, size_t bytes, int64_t experts,
        const int32_t * counts, void * stream_ptr) noexcept {
    if (!p || !name || !data || !bytes || experts <= 0 || !counts || !stream_ptr) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.closed) return false;
    if (!owner.seed) return true;
    if (cudaSetDevice(owner.device) != cudaSuccess) return false;
    if (!owner.ready && cudaEventCreateWithFlags(&owner.ready, cudaEventDisableTiming) != cudaSuccess) return false;
    if (cudaEventRecord(owner.ready, reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) return false;
    void * done = nullptr;
    const int result = owner.seed(owner.seed_data, name, data, bytes, experts, counts, owner.ready, &done);
    if (result < 0) return false;
    if (result > 0 && done) owner.pending = owner.last_seed_done = static_cast<cudaEvent_t>(done);
    return true;
}
bool seed_fence(void * p, void * stream_ptr) noexcept {
    if (!p || !stream_ptr) return false;
    auto & owner = *static_cast<transport_owner *>(p);
    std::lock_guard<std::mutex> lock(owner.seed_mutex);
    if (owner.closed) return false;
    if (owner.pending) {
        if (cudaSetDevice(owner.device) != cudaSuccess || cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_ptr), owner.pending, 0) != cudaSuccess) return false;
        owner.pending = nullptr;
    }
    return true;
}
} // namespace
extern "C" const ssd::gpu::transport_v1 * ggml_backend_cuda_ssd_transport_v1() {
    static const ssd::gpu::transport_v1 api{1, sizeof(ssd::gpu::transport_v1), sizeof(ssd::runtime_services),
        create, quiesce, destroy, copy_ranges, dense_has, dense_prepare, dense_read_ahead, dense_copy,
        seed_register, seed_unregister, seed_offer, seed_fence};
    return &api;
}
