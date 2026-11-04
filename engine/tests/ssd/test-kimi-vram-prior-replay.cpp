// SPDX-License-Identifier: Apache-2.0
// The GPU tier's slot table over a synthetic request stream, with and without the request-level prior
// (expert_predictor -> vram_slot_table::set_prior). Requests belong to domains with their own expert popularity per
// layer (a Zipf law over a domain's order of the experts, mixed with a popularity all domains share); a request's
// prompt routes from its domain, its decode tokens from its domain with temporal reuse of the token before; a request
// continues its predecessor's domain with probability 0.4. The tier sees only decode, as the production tier's counts
// do. A model-free check of the mechanism and its direction; the gain on a real model is a device measurement.
// The warm arm spends the same memory otherwise: a quarter of the slots' bytes, at half an expert's bytes a copy, buys the
// warm pool as many entries as half of them; what it serves is not an upload and not a CPU miss, but at low precision.
#include "vram-slot-table.h"
#include "vram-warm-pool.h"
#include "ggml-ssd/place/expert-trace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int layers = 16, experts = 64, k = 4, domains = 8, prompt_tokens = 48, decode_tokens = 40;

kimi_vram_request request(int layer, int expert) {
    kimi_vram_request r = {};
    r.layer = layer; r.expert = expert; r.weight = 1;
    for (int i = 0; i < 3; ++i) { r.role[i] = { 1, 7, 1u << 30, uint64_t((layer * experts + expert) * 3 + i) * 4096, 4096, 3, 4, 5, 6, 0 }; }
    return r;
}

struct workload {
    std::mt19937 rng{20261103};
    std::vector<std::vector<std::vector<double>>> popularity; // [domain][layer][expert], cumulative
    workload() {
        std::vector<double> shared(experts);
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (int i = 0; i < experts; ++i) { shared[size_t(order[size_t(i)])] = 1.0 / std::pow(i + 1, 1.1); }
        popularity.resize(domains);
        for (int d = 0; d < domains; ++d) {
            popularity[size_t(d)].resize(layers);
            for (int l = 0; l < layers; ++l) {
                std::shuffle(order.begin(), order.end(), rng);
                std::vector<double> p(experts);
                for (int i = 0; i < experts; ++i) { p[size_t(order[size_t(i)])] = 1.0 / std::pow(i + 1, 1.1); }
                const double sp = std::accumulate(p.begin(), p.end(), 0.0), ss = std::accumulate(shared.begin(), shared.end(), 0.0);
                double c = 0;
                auto & cum = popularity[size_t(d)][size_t(l)];
                cum.resize(experts);
                for (int e = 0; e < experts; ++e) { c += 0.7 * p[size_t(e)] / sp + 0.3 * shared[size_t(e)] / ss; cum[size_t(e)] = c; }
            }
        }
    }
    int sample_with(int d, int l, std::mt19937 & generator) const {
        const auto & cum = popularity[size_t(d)][size_t(l)];
        const double u = std::uniform_real_distribution<double>(0.0, cum.back())(generator);
        return int(std::lower_bound(cum.begin(), cum.end(), u) - cum.begin());
    }
    int sample(int d, int l) { return sample_with(d, l, rng); }
    // k distinct experts; with probability `reuse` a pick repeats one of the previous token's
    std::vector<int32_t> token(int d, int l, const std::vector<int32_t> & previous, double reuse) {
        std::vector<int32_t> ids;
        while (int(ids.size()) < k) {
            int e = !previous.empty() && std::uniform_real_distribution<double>(0, 1)(rng) < reuse ?
                previous[rng() % previous.size()] : sample(d, l);
            if (std::find(ids.begin(), ids.end(), e) == ids.end()) { ids.push_back(e); }
        }
        return ids;
    }
};

struct result { uint64_t hits = 0, warm = 0, accesses = 0, early_hits = 0, early = 0, uploads = 0, tokens = 0; };

// The upload's price in hits at which the tier admits, as its measured costs give it (kimi_vram_policy_api): about what
// a discrete GPU's link and CPU make of it. And the router lookahead: each selection of the next layer is in the hint
// with this probability (68 % of the picks on Flash-Next, moe-cache's measurement).
constexpr double admission_cost_hits = 2.0, hint_recall = 0.68;
constexpr int lookahead_depth = 2;

// Replays the same request stream (same seed) through a table of `capacity` slots; with `prior` the predictor's rates
// start each request's decode, with `lookahead` the next layer's hint goes up ahead of its selection. The hint draws from
// its own generator: the arms replay the same selections.
result replay(size_t capacity, bool prior, bool lookahead, int requests, size_t warm_slots = 0) {
    workload w;
    std::mt19937 hints(4242);
    std::vector<int> priority(layers);
    std::iota(priority.begin(), priority.end(), 0);
    vram_slot_table table(capacity, priority, true, 0, true, 16, 0, layers - 1, experts);
    table.admission_hits = admission_cost_hits;
    table.track_evicted = warm_slots != 0;
    std::vector<vram_warm_pool> warm;
    if (warm_slots) { warm.emplace_back(warm_slots); }
    ssd::place::expert_predictor predictor;
    if (!predictor.configure(layers, experts, 32)) { throw std::runtime_error("predictor geometry"); }
    result out;
    int domain = 0;
    for (int r = 0; r < requests; ++r) {
        if (r && std::uniform_real_distribution<double>(0, 1)(w.rng) >= 0.4) { domain = int(w.rng() % domains); }
        for (int l = 0; l < layers; ++l) {
            std::vector<int32_t> ids;
            std::vector<int32_t> routes;
            for (int t = 0; t < prompt_tokens; ++t) {
                for (int32_t e : w.token(domain, l, {}, 0.0)) {
                    const auto it = std::find(ids.begin(), ids.end(), e);
                    if (it == ids.end()) { ids.push_back(e); routes.push_back(1); } else { ++routes[size_t(it - ids.begin())]; }
                }
            }
            predictor.observe_prompt(l, ids.data(), routes.data(), int(ids.size()));
            predictor.observe_prompt_tokens(l, prompt_tokens);
        }
        std::vector<std::vector<int32_t>> previous(layers);
        for (int t = 0; t < decode_tokens; ++t) {
            table.next_token();
            table.refresh_expected();
            ++out.tokens;
            std::vector<std::vector<int32_t>> selected(layers);
            for (int l = 0; l < layers; ++l) {
                selected[size_t(l)] = w.token(domain, l, previous[size_t(l)], 0.5);
                previous[size_t(l)] = selected[size_t(l)];
            }
            for (int l = 0; l < layers; ++l) {
                const auto & ids = selected[size_t(l)];
                predictor.observe_decode(l, ids.data(), k);
                if (prior && t == 0 && l == 0) {
                    for (int x = 0; x < layers; ++x) { table.set_prior(x, predictor.rates(x), experts); }
                }
                std::vector<kimi_vram_request> batch;
                for (int32_t e : ids) { batch.push_back(request(l, e)); }
                table.prepare(batch.data(), batch.size());
                for (size_t j = 0; j < batch.size(); ++j) {
                    const bool hit = table.hits[j] >= 0;
                    out.hits += hit; ++out.accesses;
                    if (t < 8) { out.early_hits += hit; ++out.early; }
                    if (!warm.empty() && !hit && table.admissions[j] < 0) { out.warm += warm[0].find(batch[j]) >= 0; }
                }
                if (!warm.empty()) {
                    for (const auto & v : table.evicted) { warm[0].reserve(v.key); }
                    table.evicted.clear();
                    warm[0].release();
                }
                for (size_t j = 0; j < table.admissions.size(); ++j) {
                    if (table.admissions[j] >= 0) { table.publish(j); ++out.uploads; }
                }
                table.finish();
                if (lookahead && l + 1 < layers) {
                    // the hint: the next layer's picks it catches, its other ranks drawn from the domain
                    std::vector<int32_t> hint;
                    for (int32_t e : selected[size_t(l + 1)]) {
                        if (std::uniform_real_distribution<double>(0, 1)(hints) < hint_recall) { hint.push_back(e); }
                    }
                    while (int(hint.size()) < k) {
                        const int e = w.sample_with(domain, l + 1, hints);
                        if (std::find(hint.begin(), hint.end(), e) == hint.end()) { hint.push_back(e); }
                    }
                    std::shuffle(hint.begin(), hint.end(), hints);
                    int taken = 0;
                    for (int32_t e : hint) {
                        if (taken == lookahead_depth) { break; }
                        const auto r = request(l + 1, e);
                        const int slot = table.reserve_lookahead(r, hint_recall);
                        if (slot >= 0) { table.publish_upload(size_t(slot), r); ++out.uploads; ++taken; }
                        if (!warm.empty()) {
                            for (const auto & v : table.evicted) { warm[0].reserve(v.key); }
                            table.evicted.clear();
                        }
                    }
                }
            }
        }
    }
    return out;
}
} // namespace

int main() {
    try {
        const int requests = 240;
        bool worse = false;
        for (size_t capacity : { size_t(64), size_t(128), size_t(256) }) {
            const result base = replay(capacity, false, false, requests), with = replay(capacity, true, false, requests);
            const result ahead = replay(capacity, true, true, requests);
            const auto rate = [](const result & r) { return double(r.hits) / double(r.accesses); };
            const auto early = [](const result & r) { return double(r.early_hits) / double(r.early); };
            const auto uploads = [](const result & r) { return double(r.uploads) / double(r.tokens); };
            std::printf("capacity=%zu (%.0f%% of %d) hit_rate base=%.4f prior=%.4f lookahead=%.4f "
                "first8_tokens base=%.4f prior=%.4f lookahead=%.4f uploads_per_token base=%.1f prior=%.1f lookahead=%.1f\n",
                capacity, 100.0 * double(capacity) / (layers * experts), layers * experts,
                rate(base), rate(with), rate(ahead), early(base), early(with), early(ahead),
                uploads(base), uploads(with), uploads(ahead));
            worse |= rate(with) + 0.005 < rate(base) || rate(ahead) + 0.005 < rate(with);
            // the same bytes: 3/4 of the slots, and as many copies as half of them
            const result warm = replay(capacity * 3 / 4, true, true, requests, capacity / 2);
            const double warm_rate = double(warm.warm) / double(warm.accesses);
            std::printf("capacity=%zu warm_arm slots=%zu copies=%zu exact=%.4f warm=%.4f on_device=%.4f (slots alone %.4f) "
                "uploads_per_token=%.1f\n", capacity, capacity * 3 / 4, capacity / 2, rate(warm), warm_rate,
                rate(warm) + warm_rate, rate(ahead), uploads(warm));
            if (warm.warm == 0) { throw std::runtime_error("the warm pool served nothing"); }
            if (early(with) <= early(base)) { throw std::runtime_error("the prior did not raise the early-decode hit rate"); }
        }
        if (worse) { throw std::runtime_error("the prior or the lookahead lowered the decode hit rate"); }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "VRAM_PRIOR_REPLAY_FAIL %s\n", e.what());
        return 1;
    }
    std::printf("VRAM_PRIOR_REPLAY_PASS\n");
    return 0;
}
