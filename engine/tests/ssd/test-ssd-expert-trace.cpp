// SPDX-License-Identifier: Apache-2.0
// The request-level expert predictor (ggml-ssd/place/expert-trace.h): its sources, the online weights, the trace
// library and its persistence, model-free.
#include "ggml-ssd/place/expert-trace.h"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using ssd::place::expert_predictor;

static int checks = 0;
static void require(bool ok, const char * what) {
    ++checks;
    if (!ok) { throw std::runtime_error(what); }
}

// A prompt of `tokens` tokens in every layer, each token routed to the k experts of `set` (cycled).
static void prompt(expert_predictor & p, int layers, const std::vector<int32_t> & set, int tokens, int k) {
    for (int l = 0; l < layers; ++l) {
        std::vector<int32_t> routes(set.size(), 0);
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < k; ++j) { ++routes[size_t(t * k + j) % set.size()]; }
        }
        p.observe_prompt(l, set.data(), routes.data(), int(set.size()));
        p.observe_prompt_tokens(l, uint64_t(tokens));
    }
}
// `tokens` decode tokens selecting k experts of `set` in every layer (cycled).
static void decode(expert_predictor & p, int layers, const std::vector<int32_t> & set, int tokens, int k) {
    std::vector<int32_t> ids(static_cast<size_t>(k));
    for (int t = 0; t < tokens; ++t) {
        for (int l = 0; l < layers; ++l) {
            for (int j = 0; j < k; ++j) { ids[size_t(j)] = set[size_t(t * k + j) % set.size()]; }
            p.observe_decode(l, ids.data(), k);
        }
    }
}
static double mass(const float * rates, const std::vector<int32_t> & set) {
    double m = 0;
    for (int32_t e : set) { m += rates[e]; }
    return m;
}

static void test_geometry() {
    expert_predictor p;
    require(!p.configure(0, 8, 4), "zero layers accepted");
    require(!p.configure(4, 0, 4), "zero experts accepted");
    require(!p.configure(4096, 4096, 4), "oversized geometry accepted");
    require(p.predict().empty(), "an unconfigured predictor predicted");
    p.observe_decode(0, nullptr, 0); // no-op when off
    require(p.configure(4, 32, 4), "valid geometry refused");
}

// The prompt's experts carry the prediction of a request nothing was traced for, and each layer's rates sum to the
// prompt's routes per token.
static void test_prompt_source() {
    const int layers = 3, experts = 32, k = 4;
    expert_predictor p;
    require(p.configure(layers, experts, 8), "configure");
    const std::vector<int32_t> hot = { 3, 7, 11, 19, 23, 29 };
    prompt(p, layers, hot, 40, k);
    const auto & rates = p.predict();
    require(rates.size() == size_t(layers * experts), "rate matrix size");
    for (int l = 0; l < layers; ++l) {
        const float * r = p.rates(l);
        const double total = std::accumulate(r, r + experts, 0.0);
        require(std::fabs(total - k) < 1e-3, "a layer's rates do not sum to its selections per token");
        require(mass(r, hot) > 0.5 * k, "the prompt's experts do not carry the prediction");
        for (int e = 0; e < experts; ++e) { require(r[e] > 0, "an expert was predicted never to be selected"); }
    }
}

// A request whose prompt routes like a traced one is predicted to decode like it; the traced source's weight grows
// while decode follows it.
static void test_traced_source() {
    const int layers = 4, experts = 64, k = 4;
    expert_predictor p;
    require(p.configure(layers, experts, 16), "configure");
    const std::vector<int32_t> prompt_a = { 1, 2, 3, 4, 5, 6, 7, 8 }, decode_a = { 40, 41, 42, 43, 44, 45 };
    const std::vector<int32_t> prompt_b = { 20, 21, 22, 23, 24, 25, 26, 27 }, decode_b = { 50, 51, 52, 53, 54, 55 };
    for (int round = 0; round < 3; ++round) {
        prompt(p, layers, prompt_a, 32, k); decode(p, layers, decode_a, 24, k);
        prompt(p, layers, prompt_b, 32, k); decode(p, layers, decode_b, 24, k);
    }
    // a request is traced when the next one's prompt arrives: the sixth is still decoding
    require(p.traced() == 5, "requests were not traced");
    const double before = p.weights()[expert_predictor::traced_source];
    prompt(p, layers, prompt_a, 32, k);
    require(p.traced() == 6, "the next prompt did not end the request before it");
    p.predict();
    require(p.best_similarity() > 0.9, "a like prompt was not matched");
    for (int l = 0; l < layers; ++l) {
        const float * r = p.rates(l);
        require(mass(r, decode_a) > mass(r, decode_b) * 3, "the traced decode was not predicted");
    }
    decode(p, layers, decode_a, 24, k);
    require(p.weights()[expert_predictor::traced_source] > before, "the traced source did not gain weight");
    require(p.recall() > 0.3, "the predicted top-k missed the traced decode");
}

// Consecutive turns of one conversation: the last request's decode predicts the next one's without any prompt.
static void test_last_source() {
    const int layers = 2, experts = 32, k = 2;
    expert_predictor p;
    require(p.configure(layers, experts, 0), "configure");
    const std::vector<int32_t> turn = { 9, 10, 11, 12 };
    for (int round = 0; round < 5; ++round) { decode(p, layers, turn, 16, k); p.finish_request(); }
    p.predict();
    for (int l = 0; l < layers; ++l) { require(mass(p.rates(l), turn) > 0.6 * k, "the last request was not predicted"); }
    require(p.weights()[expert_predictor::last_source] > p.weights()[expert_predictor::uniform_source],
            "the last source lost to uniform on a repeating conversation");
}

// MoE-Infinity's replacement: the least matched trace goes first.
static void test_replacement() {
    const int layers = 1, experts = 16, k = 2;
    expert_predictor p;
    require(p.configure(layers, experts, 2), "configure");
    const std::vector<int32_t> a = { 0, 1, 2 }, b = { 5, 6, 7 }, c = { 10, 11, 12 };
    prompt(p, layers, a, 8, k); decode(p, layers, a, 8, k);
    prompt(p, layers, b, 8, k); decode(p, layers, b, 8, k);
    prompt(p, layers, a, 8, k); decode(p, layers, a, 8, k); // matches a (its access grows)
    p.finish_request();
    require(p.traced() == 2, "capacity not kept");
    prompt(p, layers, c, 8, k); decode(p, layers, c, 8, k);
    p.finish_request();
    require(p.traced() == 2, "capacity exceeded");
    prompt(p, layers, b, 8, k);
    p.predict();
    require(p.best_similarity() < 0.5, "the least matched trace was kept");
    p.finish_request();
    prompt(p, layers, a, 8, k);
    p.predict();
    require(p.best_similarity() > 0.9, "the most matched trace was replaced");
}

// The persisted state predicts what the original predicts; another model's or damaged bytes change nothing.
static void test_persistence() {
    const int layers = 3, experts = 48, k = 3;
    expert_predictor p;
    require(p.configure(layers, experts, 8), "configure");
    std::mt19937 rng(7);
    for (int round = 0; round < 6; ++round) {
        std::vector<int32_t> set;
        for (int i = 0; i < 6; ++i) { set.push_back(int32_t(rng() % experts)); }
        prompt(p, layers, set, 20, k); decode(p, layers, set, 12, k);
    }
    p.finish_request();
    const std::string bytes = p.serialize("model-a");
    expert_predictor q;
    require(q.configure(layers, experts, 8), "configure copy");
    std::string why;
    require(q.deserialize(bytes, "model-a", &why), why.c_str());
    require(q.traced() == p.traced() && q.requests() == p.requests(), "traces or requests not restored");
    const std::vector<int32_t> next = { 4, 8, 15, 16, 23, 42 };
    prompt(p, layers, next, 20, k); prompt(q, layers, next, 20, k);
    const auto & a = p.predict();
    const auto & b = q.predict();
    for (size_t i = 0; i < a.size(); ++i) { require(std::fabs(a[i] - b[i]) < 1e-6f, "a restored prediction differs"); }
    expert_predictor r;
    require(r.configure(layers, experts, 8), "configure r");
    require(!r.deserialize(bytes, "model-b", &why) && why == "another model", "another model's trace accepted");
    require(!r.deserialize(bytes.substr(0, bytes.size() - 5), "model-a", &why), "a truncated trace accepted");
    require(!r.deserialize(bytes + "x", "model-a", &why), "trailing bytes accepted");
    std::string damaged = bytes;
    damaged[3] ^= 0x40;
    require(!r.deserialize(damaged, "model-a", &why), "a damaged magic accepted");
    expert_predictor other;
    require(other.configure(layers, experts + 1, 8), "configure other");
    require(!other.deserialize(bytes, "model-a", &why) && why == "another expert geometry", "another geometry accepted");
    require(r.traced() == 0 && r.requests() == 0, "a refused trace changed the predictor");
}

int main() {
    try {
        test_geometry();
        test_prompt_source();
        test_traced_source();
        test_last_source();
        test_replacement();
        test_persistence();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "EXPERT_TRACE_FAIL after %d checks: %s\n", checks, e.what());
        return 1;
    }
    std::printf("EXPERT_TRACE_PASS checks=%d\n", checks);
    return 0;
}
