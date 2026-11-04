#pragma once
#include "ggml-ssd/gpu/kimi-vram-tier-api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Deterministic cache decisions, independent of I/O completion order. The
// transport owns bytes; this table publishes a slot only after copy completion.
class vram_slot_table {
public:
    struct entry {
        kimi_vram_request key = {};
        uint64_t touched = 0;
        uint32_t native_refs = 0;
        // Selection count when admitted and hits since: what an admission earned, read at its eviction.
        double admit_count = 0;
        uint32_t hits_since_admit = 0;
        // uploading: reserved by a deferred upload that outlives its layer; never a victim until published
        bool valid = false, locked = false, uploading = false;
        // lookahead: admitted ahead of its selection (reserve_lookahead), until a decode selects it or it is evicted
        bool lookahead = false;
        // seed: put into an empty slot by a prefill (reserve_cold_prefill), not admitted by decode. Its selection
        // count is zero until decode requests it, and it stays out of the earned statistics, which price decode's
        // own admissions. The first decode request makes it an ordinary entry, admitted at that request's count.
        bool seed = false;
    };
    struct layer {
        size_t begin = 0, count = 0;
        uint64_t observed = 0;
        std::vector<double> counts; // one entry per routed expert of the layer
        std::vector<uint16_t> held; // per routed expert: the slots that hold it (see holds)
        mutable uint64_t score_step = 0;
        mutable double score_factor = 1;
    };
    std::vector<entry> slots;
    // What each slot holds where slots differ in size (layers of different formats); empty: any slot holds any expert. An
    // expert takes a slot of its size, or a larger one that is empty or holds an expert of its size; a slot's own size
    // evicts whatever is there, so that an expert in a larger slot gives way when that size needs it.
    std::vector<size_t> slot_bytes;
    std::vector<layer> layers;
    std::vector<int> pool_layers;
    std::vector<int> hits, admissions;
    std::vector<kimi_vram_request> requests;
    bool count_decay = true;
    double minimum_count = 0;
    bool global_pool = false;
    double half_life = 2;
    // Kimi-K3 defaults: layers 1..92 with 896 experts. The explicit-format tier
    // passes the model's layer range and expert count.
    int first_layer = 1, last_layer = 92, experts = 896;
    bool in_flight = false;
    uint64_t step = 0, clock = 0, evictions = 0, admitted = 0;
    uint64_t origin = 0; // the step of the first decode selections: the counts have seen step - origin + 1 tokens
    // What an admission earns: its hits until eviction, by its selection count at admission, in steps of 1/4
    // (with half-life 2 a count stays below 1/(1-2^-1/2), about 3.4). Evicted entries weigh more the more recent
    // they are; entries still resident count with the hits they have so far. The prior is one hit per unit of
    // count, worth 4 samples per step. The same estimate prices an eviction: a resident entry whose count has
    // decayed to c is expected to earn what an admission at count c earns. Counts are taken over the table's
    // memory there (mature).
    static constexpr int earned_steps = 17;
    std::array<double, earned_steps> earned_hits{}, earned_weight{}, expected{};
    uint64_t earned_samples = 0;
    // Hits an admission must earn beyond what its victim would to pay for its upload (the tier measures the
    // costs); below 0 the fixed minimum_count rule applies.
    double admission_hits = -1;
    static int earned_step(double count) { return std::min(earned_steps - 1, std::max(0, int(count * 4))); }
    double expected_hits(double count) const { return expected[earned_step(count)]; }
    void refresh_expected() {
        auto hits = earned_hits, weight = earned_weight;
        for (const auto & e : slots) { if (e.valid && !e.seed) { const int s = earned_step(e.admit_count); hits[s] += e.hits_since_admit; weight[s] += 1; } }
        double floor = 0;
        for (int s = 0; s < earned_steps; ++s) { // more selections never earn fewer hits
            expected[s] = floor = std::max(floor, (hits[s] + 4 * (s + 0.5) / 4) / (weight[s] + 4));
        }
    }
    // A count sums an expert's selections, each decayed by its age in half-lives. In the table's first tokens it
    // covers fewer tokens than that memory and reads low: after two tokens an expert has at most 1.96, where one
    // selected every token reaches about 23 with half-life 16. Where a count meets what admissions earned (an
    // admission's or an eviction's price, an admission's record) it is taken over the memory, count / (1 -
    // 2^(-age/half-life)) for a steady selection rate. Taken as they were, a young table's admissions were priced
    // as cold experts and recorded as such while they were hot (RTX 5060 Ti: the count-1 admissions of the second
    // decode token earned 20 hits each and raised the expected hits at count 1 from 1.1 to 2.2 by the tenth token,
    // 27.5 after two requests, as high as at any count; a candidate then gained nothing over a resident, and the
    // tier replaced a third fewer entries in tokens 33-128). Nothing changes once the table is a few half-lives
    // old. The fixed minimum_count, the rule before the costs are measured, and a candidate's comparison with its
    // victim keep the counts as they are.
    double mature(double count) const {
        if (!count_decay || primed || !origin || step < origin) return count;
        return count / (1 - std::exp2(-double(step - origin + 1) / half_life));
    }
    // A request's predicted selections per decode token (ssd::place::expert_predictor) as the counts they would have
    // left: an expert selected at rate r every token has the decayed count r / (1 - 2^(-1/half_life)). The prior
    // replaces the layer's counts, the previous request's heat, which the prediction already weighs (its last source),
    // and decays from here like any count, so the request's own selections take over within a few half-lives. The
    // counts are then over the memory already (primed): mature() leaves them. Only between layers, with no lease.
    bool primed = false;
    uint64_t priors = 0, lookaheads = 0, lookahead_used = 0, lookahead_unused = 0;
    void set_prior(int id, const float * rates, size_t n) {
        if(!count_decay)return;
        if(in_flight || lease_generation || !rates)throw std::runtime_error("invalid VRAM prior phase");
        if(id<first_layer || size_t(id)>=layers.size() || n!=size_t(experts))throw std::runtime_error("VRAM prior layer or expert count");
        auto & l=layers[id];
        const double memory=1.0/(1.0-std::exp2(-1.0/half_life));
        for(size_t e=0;e<n;++e){
            const double r=rates[e];
            if(!std::isfinite(r) || r<0)throw std::runtime_error("VRAM prior rate");
            l.counts[e]=r*memory;
        }
        // the history up to the token before this one: the layer's prepare at this token decays it by one token and
        // adds this token's selections, as for any count
        if(!origin)origin=step?step:1;
        l.observed=step?step-1:0;l.score_step=step;l.score_factor=std::exp2(-double(step-l.observed)/half_life);
        primed=true;++priors;
    }
    // Smallest count, over the table's memory, an admission into a free slot needs (diagnostics).
    double admission_count() const {
        for (int s = 0; s < earned_steps; ++s) { if (expected[s] >= admission_hits) { return s / 4.0; } }
        return earned_steps / 4.0;
    }
    uint64_t lease_generation = 0, next_generation = 0;
    // Decode's evictions, while track_evicted, for the tier's warm pool to take before anything overwrites their slots.
    struct evicted_entry { int slot; kimi_vram_request key; };
    bool track_evicted = false;
    std::vector<evicted_entry> evicted;

    explicit vram_slot_table(size_t capacity, const std::vector<int> & priority, bool decay, double minimum=0,
            bool global=false, double half=2, int first=1, int last=92, int experts_per_layer=896) :
            count_decay(decay), minimum_count(minimum), global_pool(global), half_life(half),
            first_layer(first), last_layer(last), experts(experts_per_layer) {
        if (!std::isfinite(minimum) || minimum < 0 || minimum > 4 || (!decay && minimum != 0)) { throw std::runtime_error("VRAM minimum selection count"); }
        if (!std::isfinite(half) || half < 1 || half > 64 || (!decay && half != 2)) { throw std::runtime_error("VRAM decay half-life"); }
        if (first < 0 || last < first || last > 1023 || experts_per_layer < 1 || experts_per_layer > 65536) { throw std::runtime_error("VRAM layer/expert geometry"); }
        if (priority.empty() || capacity > priority.size()*size_t(experts)) { throw std::runtime_error("VRAM cache capacity"); }
        pool_layers = priority;
        for(int id:priority)if(id<first_layer || id>last_layer)throw std::runtime_error("VRAM layer priority range");
        layers.resize(*std::max_element(priority.begin(),priority.end())+1);
        for(auto & l:layers){l.counts.assign(size_t(experts),0.0);l.held.assign(size_t(experts),0);}
        std::vector<bool> seen(layers.size());
        size_t assigned=0;
        for(size_t i=0;i<priority.size();++i) {
            int id=priority[i];
            if(id<first_layer || seen[id]) { throw std::runtime_error("VRAM cache layer priority"); }
            seen[id]=true;auto & l=layers[id];l.begin=assigned;
            l.count=capacity/priority.size()+(i<capacity%priority.size());assigned+=l.count;
        }
        slots.resize(capacity);
        // One payload arena, shared by all layers. Source keys retain the
        // layer identity; live compute/readiness leases still protect slots.
        if (global_pool) for (int id : priority) { layers[id].begin=0;layers[id].count=capacity; }
    }
    // Elastic storage is deliberately limited to the global pool. Its logical
    // slot IDs remain stable while tail blocks are added or removed; per-layer
    // heat survives a resize even when every payload slot is reclaimed.
    void reserve_global(size_t capacity) {
        if (!global_pool || capacity > pool_layers.size()*size_t(experts)) {
            throw std::runtime_error("invalid VRAM global reserve");
        }
        slots.reserve(capacity);
    }
    void resize_global(size_t capacity) {
        if (!global_pool || capacity > pool_layers.size()*size_t(experts) ||
                in_flight || lease_generation || native_pinned()) {
            throw std::runtime_error("invalid VRAM global resize phase");
        }
        if (capacity < slots.size() && std::any_of(
                slots.begin() + capacity, slots.end(), [](const entry & e) { return e.uploading || e.locked || e.native_refs; })) {
            throw std::runtime_error("VRAM global shrink with a live tail slot");
        }
        for (size_t i = capacity; i < slots.size(); ++i) change(slots[i], [](entry & x) { x = {}; });
        slots.resize(capacity);
        if (slot_bytes.size() > capacity) { slot_bytes.resize(capacity); }
        for (int id : pool_layers) {
            layers[id].begin = 0;
            layers[id].count = capacity;
        }
    }
    static bool same_source(const kimi_vram_source & a,const kimi_vram_source & b) {
        return a.device==b.device && a.inode==b.inode && a.inode_hi==b.inode_hi && a.size==b.size && a.offset==b.offset &&
            a.bytes==b.bytes && a.mtime_s==b.mtime_s && a.mtime_ns==b.mtime_ns &&
            a.ctime_s==b.ctime_s && a.ctime_ns==b.ctime_ns;
    }
    static bool same(const kimi_vram_request & a,const kimi_vram_request & b) {
        if(a.layer!=b.layer || a.expert!=b.expert)return false;
        for(int r=0;r<3;++r)if(!same_source(a.role[r],b.role[r]))return false;
        return true; // Routing weight is not part of immutable payload identity.
    }
    static size_t expert_bytes(const kimi_vram_request & r) { return r.role[0].bytes + r.role[1].bytes + r.role[2].bytes; }
    bool fits(size_t i, size_t bytes) const {
        if (slot_bytes.empty() || slot_bytes[i] == bytes) { return true; }
        return slot_bytes[i] > bytes && (!slots[i].valid || expert_bytes(slots[i].key) == bytes);
    }
    bool native_pinned() const {
        return std::any_of(slots.begin(),slots.end(),[](const entry & e){return e.native_refs!=0;});
    }
    // Whether the tier holds an expert now: a slot is valid with it, or reserved for a deferred upload of it that has
    // not completed (the layer's next prepare collects it first). A seed (reserve_cold_prefill) counts from its
    // publication (publish_cold_prefill), which no prepare does: a prefill's planned seeds are published when the
    // decode phase completes the prefill (phase_complete), Q2's cold-prefill admission right after its own copy. The
    // next-layer prefetch asks before it reads an expert into RAM. (layer, expert) stands for the key: one model's
    // experts have one source each. Every change of a slot's key, valid, uploading or seed goes through change(),
    // which keeps the count.
    bool holds(int layer,int expert) const {
        return layer>=0 && size_t(layer)<layers.size() && expert>=0 && size_t(expert)<layers[layer].held.size() &&
            layers[layer].held[size_t(expert)]>0;
    }
    static bool held_by(const entry & e) { return e.valid || (e.uploading && !e.seed); }
    template<class F> void change(entry & e,F && update) {
        if(held_by(e))--layers[e.key.layer].held[size_t(e.key.expert)];
        update(e);
        if(held_by(e))++layers[e.key.layer].held[size_t(e.key.expert)];
    }
    // The native consumer cancelled: no payload is valid any more; heat stays.
    void invalidate() {
        for(auto & e:slots)change(e,[](entry & x){x.valid=false;x.locked=false;});
    }
    void clear() {
        if(in_flight || lease_generation || native_pinned()) { throw std::runtime_error("clear with live VRAM slots"); }
        for(auto & e:slots)e={};
        for(auto & l:layers){std::fill(l.counts.begin(),l.counts.end(),0.0);std::fill(l.held.begin(),l.held.end(),0);l.observed=l.score_step=0;l.score_factor=1;}
        hits.clear();admissions.clear();requests.clear();evicted.clear();step=clock=origin=0;primed=false;
    }
    void next_token() { if(in_flight || lease_generation || native_pinned())throw std::runtime_error("token with live VRAM slots");++step; }
    double score(const entry & e) const {
        if (!count_decay) return 0;
        const auto & l=layers[e.key.layer];
        if (!global_pool) return l.counts[e.key.expert];
        // A victim can belong to a layer not visited yet in this token. Bring
        // its score to this token without training its selection history.
        if (l.score_step!=step) {
            l.score_factor=std::exp2(-double(step-l.observed)/half_life);
            l.score_step=step;
        }
        return l.counts[e.key.expert]*l.score_factor;
    }
    void prepare(const kimi_vram_request * input,size_t n,uint32_t demand_mask=UINT32_MAX) {
        if(in_flight || lease_generation || !step || n>16)throw std::runtime_error("invalid VRAM prepare phase");
        if(n)requests.assign(input,input+n);else requests.clear();
        hits.assign(n,-1);admissions.assign(n,-1);
        if(n==0){in_flight=true;return;}
        const int id=requests[0].layer;
        if(id<first_layer || size_t(id)>=layers.size())throw std::runtime_error("VRAM layer out of range");
        auto & l=layers[id];
        for(const auto & r:requests)if(r.layer!=id || r.expert<0 || r.expert>=experts)throw std::runtime_error("VRAM expert out of range");
        if(!origin)origin=step;
        if(l.observed!=step) {
            const double factor=std::exp2(-double(step-l.observed)/half_life);
            for(auto & value:l.counts)value*=factor;
            std::vector<bool> seen(static_cast<size_t>(experts), false);
            for(size_t j=0;j<n;++j){const auto & r=requests[j];if((demand_mask&(1u<<j))&&!seen[r.expert]){l.counts[r.expert]+=1;seen[r.expert]=true;}}
            l.observed=step;
            l.score_step=step;l.score_factor=1;
        }
        // Pin all selected hits before choosing any victim, including a hit
        // that appears after a miss in the original router order. A seed's
        // first hit stands where an admission's miss would: from it on the
        // entry earns, and is accounted, as one admitted at this count.
        for(size_t j=0;j<n;++j)for(size_t i=l.begin;i<l.begin+l.count;++i) {
            if(!(demand_mask&(1u<<j)))continue;
            auto & e=slots[i];
            if(e.valid && same(e.key,requests[j])){
                e.locked=true;e.touched=++clock;hits[j]=int(i);
                if(e.seed){e.seed=false;e.admit_count=count_decay?mature(l.counts[requests[j].expert]):1;e.hits_since_admit=0;}
                else ++e.hits_since_admit;
                if(e.lookahead){e.lookahead=false;++lookahead_used;}
                break;
            }
        }
        std::vector<size_t> order;
        for(size_t i=0;i<n;++i)if((demand_mask&(1u<<i))&&hits[i]<0)order.push_back(i);
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){
            return count_decay && l.counts[requests[a].expert]>l.counts[requests[b].expert];
        });
        for(size_t j:order) {
            const double count=l.counts[requests[j].expert];
            if(count_decay && admission_hits<0 && count<minimum_count)continue;
            if(std::any_of(slots.begin()+l.begin,slots.begin()+l.begin+l.count,[&](const entry & e){return (e.locked || e.native_refs || e.uploading) && same(e.key,requests[j]);}))continue;
            int victim=-1;
            const size_t bytes=expert_bytes(requests[j]);
            for(size_t i=l.begin;i<l.begin+l.count;++i) {
                const auto & e=slots[i];if(e.locked || e.native_refs || e.uploading || !fits(i,bytes))continue;
                if(!e.valid){victim=int(i);break;}
                if(victim<0 || score(e)<score(slots[victim]) ||
                   (score(e)==score(slots[victim]) && e.touched<slots[victim].touched))victim=int(i);
            }
            if(victim<0)continue;
            auto & e=slots[victim];
            if(e.valid && count_decay && count<score(e))continue;
            if(count_decay && admission_hits>=0 && expected_hits(mature(count))-(e.valid?expected_hits(mature(score(e))):0)<admission_hits)continue;
            if(e.valid && !e.seed)record_earned(e);
            if(track_evicted && e.valid && !e.seed)evicted.push_back({victim,e.key});
            lookahead_unused+=e.lookahead;
            change(e,[&](entry & x){evictions+=x.valid;x.valid=false;x.seed=false;x.lookahead=false;x.locked=true;x.key=requests[j];x.touched=++clock;});
            admissions[j]=victim;
            e.admit_count=count_decay?mature(count):1;e.hits_since_admit=0;
        }
        in_flight=true;
    }
    // What an evicted entry earned, for the admissions after it.
    void record_earned(const entry & e) {
        for(int s=0;s<earned_steps;++s){earned_hits[s]*=0.999;earned_weight[s]*=0.999;}
        const int s=earned_step(e.admit_count);earned_hits[s]+=e.hits_since_admit;earned_weight[s]+=1;++earned_samples;
    }
    // An expert of a later layer that the router lookahead predicts, admitted ahead of its selection on prepare's terms:
    // its count with that selection, its expected hits weighed by the probability that the selection comes, against
    // the victim's and the upload's price; before the costs are measured (admission_hits below 0), the fixed minimum
    // count. The selection itself is not counted: prepare counts what decode selects. Returns the slot, reserved for a
    // deferred upload (publish_upload or cancel_upload), or -1 (held, no victim, not worth it). Any phase but a lease: a
    // live layer's slots are locked, and nothing locked is a victim.
    int reserve_lookahead(const kimi_vram_request & request, double probability) {
        if(lease_generation || request.layer<first_layer || size_t(request.layer)>=layers.size() ||
                request.expert<0 || request.expert>=experts || !(probability>=0 && probability<=1)) {
            throw std::runtime_error("invalid VRAM lookahead reservation");
        }
        if(!count_decay)return -1;
        const auto & l=layers[request.layer];
        if(l.held[size_t(request.expert)])return -1;
        const double count=l.counts[size_t(request.expert)]*std::exp2(-double(step-l.observed)/half_life)+1;
        int victim=-1;
        const size_t bytes=expert_bytes(request);
        for(size_t i=l.begin;i<l.begin+l.count;++i) {
            const auto & e=slots[i];if(e.locked || e.native_refs || e.uploading || !fits(i,bytes))continue;
            if(!e.valid){victim=int(i);break;}
            if(victim<0 || score(e)<score(slots[victim]) ||
               (score(e)==score(slots[victim]) && e.touched<slots[victim].touched))victim=int(i);
        }
        if(victim<0)return -1;
        auto & e=slots[victim];
        if(e.valid && count<score(e))return -1;
        if(admission_hits<0 ? count<minimum_count :
                probability*expected_hits(mature(count))-(e.valid?expected_hits(mature(score(e))):0)<admission_hits)return -1;
        if(e.valid && !e.seed)record_earned(e);
        if(track_evicted && e.valid && !e.seed)evicted.push_back({victim,e.key});
        lookahead_unused+=e.lookahead;
        change(e,[&](entry & x){evictions+=x.valid;x.valid=false;x.seed=false;x.lookahead=true;x.uploading=true;x.key=request;x.touched=++clock;});
        e.admit_count=mature(count);e.hits_since_admit=0;
        ++lookaheads;
        return victim;
    }
    void publish(size_t request_index) {
        if(!in_flight || request_index>=admissions.size() || admissions[request_index]<0)throw std::runtime_error("VRAM publication without reservation");
        auto & e=slots[admissions[request_index]];
        if(!e.locked || e.valid || e.uploading)throw std::runtime_error("duplicate VRAM publication");
        change(e,[](entry & x){x.valid=true;});++admitted;
    }
    // A deferred upload keeps its reservation after finish() unlocks the layer, until publish_upload.
    void reserve_upload(size_t request_index) {
        if(!in_flight || request_index>=admissions.size() || admissions[request_index]<0)throw std::runtime_error("deferred VRAM upload without reservation");
        auto & e=slots[admissions[request_index]];
        if(!e.locked || e.valid || e.uploading)throw std::runtime_error("duplicate deferred VRAM upload");
        change(e,[](entry & x){x.uploading=true;});
    }
    void publish_upload(size_t slot,const kimi_vram_request & key) {
        if(in_flight || lease_generation || slot>=slots.size())throw std::runtime_error("invalid deferred VRAM publication phase");
        auto & e=slots[slot];
        if(!e.uploading || e.valid || !same(e.key,key))throw std::runtime_error("deferred VRAM publication lost its reservation");
        change(e,[](entry & x){x.uploading=false;x.valid=true;});++admitted;
    }
    // Cold prefill may fill only physically empty slots. It deliberately does
    // not call prepare(), increment counts, or evict an existing decode entry.
    // A seeded entry therefore has score zero until decode actually requests
    // it, making unused prompt heat the first victim of later decode misses.
    int reserve_cold_prefill(const kimi_vram_request & request) {
        if(in_flight || lease_generation || native_pinned() || request.layer<first_layer ||
                size_t(request.layer)>=layers.size() || request.expert<0 || request.expert>=experts) {
            throw std::runtime_error("invalid cold-prefill VRAM reservation phase");
        }
        const auto & l=layers[request.layer];
        for(size_t i=l.begin;i<l.begin+l.count;++i) {
            const auto & e=slots[i];
            if((e.valid || e.uploading) && same(e.key,request))return -1;
        }
        const size_t bytes=expert_bytes(request);
        for(size_t i=l.begin;i<l.begin+l.count;++i) {
            auto & e=slots[i];
            if(e.valid || e.locked || e.native_refs || e.uploading || !fits(i,bytes))continue;
            change(e,[&](entry & x){x.key=request;x.touched=++clock;x.uploading=true;x.seed=true;x.admit_count=0;x.hits_since_admit=0;});
            return int(i);
        }
        return -1;
    }
    // Empty slots a prefill may seed in layer `id` (all of them in a global pool).
    size_t free_slots(int id) const {
        if(id<first_layer || size_t(id)>=layers.size())return 0;
        const auto & l=layers[id];
        return size_t(std::count_if(slots.begin()+l.begin,slots.begin()+l.begin+l.count,[](const entry & e){
            return !e.valid && !e.locked && !e.native_refs && !e.uploading;}));
    }
    // Seeds the decode after their prefill never selected: prompt heat of a request that is over. Freed before the
    // next prefill seeds, so that its own prompt plans with them. Returns how many.
    size_t free_unused_seeds() {
        if(in_flight || lease_generation || native_pinned())throw std::runtime_error("free seeds with live VRAM slots");
        size_t n=0;
        for(auto & e:slots)if(e.valid && e.seed && !e.locked){change(e,[](entry & x){x={};});++n;}
        return n;
    }
    void publish_cold_prefill(size_t slot,const kimi_vram_request & request) {
        if(in_flight || lease_generation || slot>=slots.size())
            throw std::runtime_error("invalid cold-prefill VRAM publication phase");
        auto & e=slots[slot];
        if(!e.uploading || e.valid || !same(e.key,request))
            throw std::runtime_error("cold-prefill VRAM publication lost its reservation");
        change(e,[](entry & x){x.uploading=false;x.valid=true;});++admitted;
    }
    // Drops a reservation whose copy did not complete: a seed's, or a deferred upload's in a block given back.
    void cancel_upload(size_t slot,const kimi_vram_request & request) {
        if(slot>=slots.size())return;
        auto & e=slots[slot];
        if(e.uploading && !e.valid && same(e.key,request))change(e,[](entry & x){x={};});
    }
    void cancel_cold_prefill(size_t slot,const kimi_vram_request & request) { cancel_upload(slot,request); }
    void finish() {
        if(!in_flight)throw std::runtime_error("VRAM finish without prepare");
        for(auto & e:slots)e.locked=false;
        in_flight=false;
    }
    uint64_t acquire_ready(const kimi_vram_request * input, size_t n, int * ready) {
        if (in_flight || lease_generation || native_pinned() || n > 16 || (n && (!input || !ready)) || next_generation == UINT64_MAX)
            throw std::runtime_error("invalid VRAM ready lease phase");
        std::vector<int> found(n, -1);
        const uint64_t generation = acquire_slots(input, n, found.data());
        for (size_t j = 0; j < n; ++j) { ready[j] = found[j] >= 0; }
        return generation;
    }
    // Read-only grouped-prefill lease. Unlike prepare(), this accepts the full
    // distinct-expert union, does not update heat/admission state, and returns
    // physical slot IDs so a later worker can reuse each resident triplet.
    uint64_t acquire_slots(const kimi_vram_request * input, size_t n, int * found) {
        if (in_flight || lease_generation || native_pinned() || n > size_t(experts) ||
                (n && (!input || !found)) || next_generation == UINT64_MAX) {
            throw std::runtime_error("invalid VRAM grouped-prefill lease phase");
        }
        for (size_t j = 0; j < n; ++j) {
            found[j] = -1;
            if (input[j].layer < first_layer || size_t(input[j].layer) >= layers.size() || input[j].expert < 0 || input[j].expert >= experts)
                throw std::runtime_error("invalid VRAM residency key");
            const auto & l = layers[input[j].layer];
            for (size_t i = l.begin; i < l.begin + l.count; ++i) {
                auto & e = slots[i];
                if (e.valid && same(e.key, input[j])) { e.locked = true; found[j] = int(i); break; }
            }
        }
        return lease_generation = ++next_generation;
    }
    void validate_ready(uint64_t generation) const {
        if (!generation || generation != lease_generation || in_flight)
            throw std::runtime_error("stale VRAM ready lease");
    }
    void release_ready(uint64_t generation) {
        validate_ready(generation);
        for (auto & e : slots) e.locked = false;
        lease_generation = 0;
    }
    void abort() {
        if(native_pinned())throw std::runtime_error("abort with native VRAM leases");
        for(auto & e:slots){if(e.locked && !e.valid)e={};e.locked=false;}
        in_flight=false;lease_generation=0;
        clear(); // Error paths never preserve partially validated payload.
    }
};
