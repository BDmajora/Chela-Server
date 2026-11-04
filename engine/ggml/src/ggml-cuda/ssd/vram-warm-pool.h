#pragma once
#include "vram-slot-table.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

// The GPU tier's warm pool (vram.warm_mib): low-precision copies of experts the slot table evicted, kept on the device
// and served until the table admits them again. Where the table holds the native bytes and prices each upload, this
// holds what eviction would have lost, without a byte over the link (the device requantizes the victim in place), so it
// takes no admission decision: the most recently evicted experts stay, recency being what predicts an expert's return
// (MoE-Infinity's traces of Qwen3). A pool entry is not removed by a promotion: the next eviction of that expert then
// needs no conversion. Decisions only, like the slot table; the tier owns the bytes and their ordering.
class vram_warm_pool {
public:
    explicit vram_warm_pool(size_t capacity) : entries(capacity) {
        if (!capacity) { throw std::runtime_error("VRAM warm pool capacity"); }
    }
    size_t capacity() const { return entries.size(); }
    // The pool slot holding this payload, marked as this batch's (a reservation does not take it until release), or -1.
    // A key of the same layer and expert with another identity is not this payload.
    int find(const kimi_vram_request & key) {
        const auto it = index.find(id(key));
        if (it == index.end() || !vram_slot_table::same(entries[size_t(it->second)].key, key)) { return -1; }
        auto & e = entries[size_t(it->second)];
        e.touched = ++clock;
        e.pinned = true;
        return it->second;
    }
    // A slot for this payload: the free one or the least recently evicted or served that this batch does not read.
    // -1 when the pool holds it already (nothing to convert) or every slot is this batch's. The caller converts into it.
    int reserve(const kimi_vram_request & key) {
        const auto it = index.find(id(key));
        if (it != index.end() && vram_slot_table::same(entries[size_t(it->second)].key, key)) {
            entries[size_t(it->second)].touched = ++clock;
            return -1;
        }
        int victim = -1;
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto & e = entries[i];
            if (e.pinned) { continue; }
            if (!e.valid) { victim = int(i); break; }
            if (victim < 0 || e.touched < entries[size_t(victim)].touched) { victim = int(i); }
        }
        if (victim < 0) { return -1; }
        auto & e = entries[size_t(victim)];
        if (e.valid) { index.erase(id(e.key)); ++replaced; }
        // the same layer and expert under another identity leaves with its index entry
        if (it != index.end()) { entries[size_t(it->second)] = {}; index.erase(id(key)); }
        e.key = key;
        e.valid = true;
        e.touched = ++clock;
        index[id(key)] = victim;
        return victim;
    }
    void release() { for (auto & e : entries) { e.pinned = false; } }
    uint64_t replaced = 0;
private:
    struct entry { kimi_vram_request key = {}; uint64_t touched = 0; bool valid = false, pinned = false; };
    static uint64_t id(const kimi_vram_request & key) { return uint64_t(uint32_t(key.layer)) << 32 | uint32_t(key.expert); }
    std::vector<entry> entries;
    std::unordered_map<uint64_t, int> index;
    uint64_t clock = 0;
};
