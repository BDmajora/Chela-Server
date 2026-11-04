// SPDX-License-Identifier: Apache-2.0
#pragma once
// Request-level expert prediction: which experts a request's decode tokens will select, from what is known when its
// decode starts, as a rate per decode token for every (layer, expert). The expert caches start a request's decode from
// these rates instead of the previous request's heat (kimi_vram_prior_api, expert_cache::set_prior). Five sources:
//   prompt   the experts the request's own prefill routed: a request's decode keeps to the domain of its prompt
//            (DoMoE's prefill-stage prediction, IJCAI 2026);
//   traced   the decode of earlier requests whose prompts routed alike: MoE-Infinity's expert activation matrices and
//            their trace collection (arXiv:2401.14361), matched by the mean per-layer cosine similarity of the prompts;
//   last     the previous request's decode, for consecutive turns of one conversation;
//   profile  the long-run decode usage of every request, a decayed profile kept across runs;
//   uniform  every expert alike, the floor that keeps every rate positive.
// The rate of an expert is the layer's selections per token times the mixture of the sources' distributions over the
// layer's experts. The mixture weights are fitted online to the selections decode makes (online EM for the weights
// of a mixture: each selection moves every weight toward its posterior share of that selection), so a source that
// predicts badly loses its weight and the mixture never does much worse than its best source.
// Model-free and numeric. The executor feeds it routes and selections at its synchronization points and asks it for
// rates; the bytes it persists go through the executor's own file handles. Not thread safe: one executor owns it.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace ssd { namespace place {

// Counts over layers x experts, each layer's row with its total.
struct activation_matrix {
    int32_t layers = 0, experts = 0;
    std::vector<float> counts;
    std::vector<double> totals;
    void reset(int32_t l, int32_t e) {
        layers = l; experts = e;
        counts.assign(size_t(l) * size_t(e), 0.0f);
        totals.assign(size_t(l), 0.0);
    }
    void clear() { std::fill(counts.begin(), counts.end(), 0.0f); std::fill(totals.begin(), totals.end(), 0.0); }
    bool contains(int32_t layer, int32_t expert) const {
        return layer >= 0 && layer < layers && expert >= 0 && expert < experts;
    }
    void add(int32_t layer, int32_t expert, float weight) {
        if (!contains(layer, expert) || !(weight > 0.0f)) { return; }
        counts[size_t(layer) * size_t(experts) + size_t(expert)] += weight;
        totals[size_t(layer)] += weight;
    }
    const float * row(int32_t layer) const { return counts.data() + size_t(layer) * size_t(experts); }
    double total() const { double t = 0; for (double v : totals) { t += v; } return t; }
};

// One traced request: its prompt's and its decode's distribution over each layer's experts (rows summing to 1 where the
// layer routed, 0 elsewhere), and its decode selections per token of each layer.
struct expert_trace {
    std::vector<float> prompt, decode, per_token;
    uint64_t tokens = 0, sequence = 0;
    double access = 1.0; // how often it was matched, decayed at every insertion (MoE-Infinity's collection_access)
};

// Row-normalized copy of a matrix: each routed layer's row sums to 1.
inline std::vector<float> normalized_rows(const activation_matrix & m) {
    std::vector<float> out(m.counts.size(), 0.0f);
    for (int32_t l = 0; l < m.layers; ++l) {
        const double t = m.totals[size_t(l)];
        if (!(t > 0.0)) { continue; }
        const float * r = m.row(l);
        float * o = out.data() + size_t(l) * size_t(m.experts);
        for (int32_t e = 0; e < m.experts; ++e) { o[e] = float(double(r[e]) / t); }
    }
    return out;
}

// The mean over the layers routed in both of the cosine similarity of two row-normalized matrices' rows; 0 when no
// layer is routed in both.
inline double row_similarity(const std::vector<float> & a, const std::vector<float> & b, int32_t layers, int32_t experts) {
    double sum = 0; int32_t n = 0;
    for (int32_t l = 0; l < layers; ++l) {
        const float * x = a.data() + size_t(l) * size_t(experts);
        const float * y = b.data() + size_t(l) * size_t(experts);
        double xy = 0, xx = 0, yy = 0;
        for (int32_t e = 0; e < experts; ++e) { xy += double(x[e]) * y[e]; xx += double(x[e]) * x[e]; yy += double(y[e]) * y[e]; }
        if (xx > 0 && yy > 0) { sum += xy / std::sqrt(xx * yy); ++n; }
    }
    return n ? sum / n : 0.0;
}

class expert_predictor {
public:
    enum source { prompt_source, traced_source, last_source, profile_source, uniform_source, source_count };
    static const char * source_name(int s) {
        static const char * names[source_count] = { "prompt", "traced", "last", "profile", "uniform" };
        return s >= 0 && s < source_count ? names[s] : "?";
    }
    // Requests need this many decode tokens to be traced, and a trace this much prompt similarity to be matched.
    static constexpr uint64_t min_trace_tokens = 4;
    static constexpr double min_similarity = 0.05;
    // The weights' step per selection (online EM), the profile's keep per request and the traces' access decay.
    static constexpr double weight_step = 0.002, profile_keep = 0.9, access_keep = 0.97;
    static constexpr size_t max_cells = size_t(1) << 22; // layers x experts beyond which prediction is off

    // capacity: traced requests kept (0: no tracing, the other sources still predict); matches: traces mixed into the
    // traced source. False when the geometry is too large to predict: every call is then a no-op.
    bool configure(int32_t layers, int32_t experts, size_t capacity, size_t matches = 4) {
        enabled_ = layers > 0 && experts > 0 && size_t(layers) * size_t(experts) <= max_cells;
        if (!enabled_) { return false; }
        layers_ = layers; experts_ = experts; capacity_ = capacity; matches_ = std::max<size_t>(1, matches);
        prompt_.reset(layers, experts); decode_.reset(layers, experts);
        profile_.assign(cells(), 0.0f); last_.assign(cells(), 0.0f);
        profile_per_token_.assign(size_t(layers), 0.0f); last_per_token_.assign(size_t(layers), 0.0f);
        layer_tokens_.assign(size_t(layers), 0);
        library_.clear(); library_.reserve(capacity);
        weights_ = { 0.35, 0.2, 0.25, 0.1, 0.1 };
        rates_.clear(); dist_.clear(); predicted_ = false;
        return true;
    }
    bool enabled() const { return enabled_; }
    int32_t layers() const { return layers_; }
    int32_t experts() const { return experts_; }

    // A prompt chunk's routes in a layer: `routes[i]` tokens routed to `ids[i]`. A chunk after decode tokens starts a new
    // request (the previous one is traced first).
    void observe_prompt(int32_t layer, const int32_t * ids, const int32_t * routes, int n) {
        if (!enabled_) { return; }
        if (decoding_) { finish_request(); }
        prefilled_ = true;
        for (int i = 0; i < n; ++i) { prompt_.add(layer, ids[i], float(routes ? routes[i] : 1)); }
    }
    // The prompt's tokens of a chunk, for its selections per token (once per chunk; layers count their own routes).
    void observe_prompt_tokens(int32_t layer, uint64_t tokens) {
        if (!enabled_ || layer < 0 || layer >= layers_) { return; }
        if (decoding_) { finish_request(); }
        prompt_tokens_.resize(size_t(layers_), 0);
        prompt_tokens_[size_t(layer)] += tokens;
    }
    // A decode token's selections in a layer (ids < 0 skipped, a repeated id counted once). The first one of a request
    // fixes its prediction (predict); every one scores the sources against the mixture and moves its weights.
    void observe_decode(int32_t layer, const int32_t * ids, int n) {
        if (!enabled_ || layer < 0 || layer >= layers_) { return; }
        if (!decoding_) { typical_selections_ = n; predict(); decoding_ = true; }
        for (int i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= experts_ || std::find(ids, ids + i, e) != ids + i) { continue; }
            decode_.add(layer, e, 1.0f);
            learn(layer, e);
        }
        ++layer_tokens_[size_t(layer)];
        tokens_ = std::max(tokens_, layer_tokens_[size_t(layer)]);
    }
    // Ends the current request: traces it (with its prompt) when it decoded enough tokens, and folds its decode into the
    // last-request and profile sources. Called by the next prompt, or explicitly at shutdown before serialize.
    void finish_request() {
        if (!enabled_) { return; }
        if (decode_.total() > 0.0) {
            const auto decode = normalized_rows(decode_);
            std::vector<float> per_token(size_t(layers_), 0.0f);
            for (int32_t l = 0; l < layers_; ++l) {
                const uint64_t t = layer_tokens_[size_t(l)];
                per_token[size_t(l)] = t ? float(decode_.totals[size_t(l)] / double(t)) : 0.0f;
            }
            for (size_t i = 0; i < profile_.size(); ++i) {
                profile_[i] = float(profile_keep * profile_[i] + (1.0 - profile_keep) * decode[i]);
            }
            for (int32_t l = 0; l < layers_; ++l) {
                if (per_token[size_t(l)] > 0) {
                    float & p = profile_per_token_[size_t(l)];
                    p = p > 0 ? float(profile_keep * p + (1.0 - profile_keep) * per_token[size_t(l)]) : per_token[size_t(l)];
                }
            }
            if (capacity_ && tokens_ >= min_trace_tokens && prompt_.total() > 0.0) {
                expert_trace t;
                t.prompt = normalized_rows(prompt_); t.decode = decode; t.per_token = per_token;
                t.tokens = tokens_; t.sequence = ++sequence_;
                insert(std::move(t));
            }
            last_ = decode; last_per_token_ = per_token;
            ++requests_;
        }
        prompt_.clear(); decode_.clear(); prompt_tokens_.clear();
        std::fill(layer_tokens_.begin(), layer_tokens_.end(), 0);
        tokens_ = 0; decoding_ = false; prefilled_ = false; predicted_ = false;
    }

    // The predicted selections per decode token of every (layer, expert), [layers][experts]; computed at the first decode
    // selection of a request (or here, when asked before it) from the prompt seen so far.
    const std::vector<float> & predict() {
        if (!enabled_) { return rates_; }
        if (predicted_) { return rates_; }
        predicted_ = true;
        build_sources();
        rates_.assign(cells(), 0.0f); dist_.assign(cells(), 0.0f); top_.assign(size_t(layers_), 0.0f);
        std::vector<float> sorted(static_cast<size_t>(experts_));
        for (int32_t l = 0; l < layers_; ++l) {
            const float k = layer_rate(l);
            double total = 0;
            for (int s = 0; s < source_count; ++s) { total += available(s, l) ? weights_[size_t(s)] : 0.0; }
            for (int32_t e = 0; e < experts_; ++e) {
                const size_t c = cell(l, e);
                double p = 0;
                for (int s = 0; s < source_count; ++s) {
                    if (available(s, l)) { p += weights_[size_t(s)] * source_dist(s, l, e); }
                }
                dist_[c] = float(p / total);
                rates_[c] = float(p / total * k);
            }
            // the k-th largest probability of the layer: a selection at or above it is in the predicted top-k
            const size_t top = size_t(std::min<int32_t>(experts_, std::max<int32_t>(1, int32_t(std::ceil(k - 1e-6f)))));
            std::copy(dist_.begin() + std::ptrdiff_t(cell(l, 0)), dist_.begin() + std::ptrdiff_t(cell(l, 0) + size_t(experts_)), sorted.begin());
            std::nth_element(sorted.begin(), sorted.begin() + std::ptrdiff_t(top - 1), sorted.end(), std::greater<float>());
            top_[size_t(l)] = sorted[top - 1];
        }
        ++predictions_;
        return rates_;
    }
    const float * rates(int32_t layer) const {
        return predicted_ && layer >= 0 && layer < layers_ ? rates_.data() + size_t(layer) * size_t(experts_) : nullptr;
    }
    bool predicted() const { return predicted_; }
    // Predictions made: one a request that decodes. A new value means new rates.
    uint64_t predictions() const { return predictions_; }
    bool decoding() const { return decoding_; }

    // Diagnostics: the mixture weights, the matched traces' best similarity, the traced requests, and how well the
    // predictions did: per decode selection, the mean probability the mixture and each source gave it.
    const std::array<double, source_count> & weights() const { return weights_; }
    double best_similarity() const { return best_similarity_; }
    size_t traced() const { return library_.size(); }
    uint64_t requests() const { return requests_; }
    uint64_t scored() const { return scored_; }
    double mean_probability() const { return scored_ ? mixture_probability_ / double(scored_) : 0.0; }
    double mean_source_probability(int s) const { return scored_ ? source_probability_[size_t(s)] / double(scored_) : 0.0; }
    // The share of a request's decode selections that fell in the predicted top-k of their layer (k: the layer's
    // selections per token, rounded up), over all scored selections: the recall a cache of k slots a layer would have.
    double recall() const { return scored_ ? double(top_hits_) / double(scored_) : 0.0; }

    // Persistence: the traces, the last request and the profile, with the mixture weights, for this model only.
    std::string serialize(const std::string & fingerprint) const {
        std::string out;
        put(out, magic, sizeof(magic)); put_u32(out, version);
        put_u32(out, uint32_t(fingerprint.size())); out += fingerprint;
        put_u32(out, uint32_t(layers_)); put_u32(out, uint32_t(experts_));
        for (double w : weights_) { put_f64(out, w); }
        put_u64(out, requests_); put_u64(out, sequence_);
        put_rows(out, profile_); put_rows(out, profile_per_token_);
        put_rows(out, last_); put_rows(out, last_per_token_);
        put_u32(out, uint32_t(library_.size()));
        for (const auto & t : library_) {
            put_u64(out, t.tokens); put_u64(out, t.sequence); put_f64(out, t.access);
            put_rows(out, t.prompt); put_rows(out, t.decode); put_rows(out, t.per_token);
        }
        return out;
    }
    // False, with the reason, for another model's or version's bytes or a damaged file; nothing changes then.
    bool deserialize(const std::string & in, const std::string & fingerprint, std::string * why = nullptr) {
        auto fail = [&](const char * reason) { if (why) { *why = reason; } return false; };
        if (!enabled_) { return fail("prediction is off"); }
        size_t at = 0;
        char m[sizeof(magic)] = {};
        uint32_t v = 0, n = 0, l = 0, e = 0;
        if (!get(in, at, m, sizeof(m)) || std::memcmp(m, magic, sizeof(magic)) || !get_u32(in, at, v)) { return fail("not an expert trace"); }
        if (v != version) { return fail("another trace version"); }
        if (!get_u32(in, at, n) || n > in.size() - at || in.compare(at, n, fingerprint)) { return fail("another model"); }
        at += n;
        if (!get_u32(in, at, l) || !get_u32(in, at, e) || int32_t(l) != layers_ || int32_t(e) != experts_) {
            return fail("another expert geometry");
        }
        std::array<double, source_count> w = {};
        uint64_t requests = 0, sequence = 0;
        for (double & x : w) { if (!get_f64(in, at, x) || !(x >= 0.0) || !std::isfinite(x)) { return fail("damaged weights"); } }
        std::vector<float> profile, profile_per_token, last, last_per_token;
        if (!get_u64(in, at, requests) || !get_u64(in, at, sequence) ||
                !get_rows(in, at, profile, cells()) || !get_rows(in, at, profile_per_token, size_t(layers_)) ||
                !get_rows(in, at, last, cells()) || !get_rows(in, at, last_per_token, size_t(layers_)) || !get_u32(in, at, n)) {
            return fail("damaged profile");
        }
        std::vector<expert_trace> library;
        for (uint32_t i = 0; i < n; ++i) {
            expert_trace t;
            if (!get_u64(in, at, t.tokens) || !get_u64(in, at, t.sequence) || !get_f64(in, at, t.access) ||
                    !get_rows(in, at, t.prompt, cells()) || !get_rows(in, at, t.decode, cells()) ||
                    !get_rows(in, at, t.per_token, size_t(layers_)) || !(t.access >= 0.0)) {
                return fail("damaged trace");
            }
            if (library.size() < capacity_) { library.push_back(std::move(t)); }
        }
        if (at != in.size()) { return fail("trailing bytes"); }
        double sum = 0; for (double x : w) { sum += x; }
        if (!(sum > 0)) { return fail("damaged weights"); }
        for (double & x : w) { x /= sum; }
        weights_ = w; requests_ = requests; sequence_ = sequence;
        profile_ = std::move(profile); profile_per_token_ = std::move(profile_per_token);
        last_ = std::move(last); last_per_token_ = std::move(last_per_token);
        library_ = std::move(library);
        predicted_ = false;
        return true;
    }

private:
    static constexpr char magic[8] = { 'I', 'S', 'X', 'T', 'R', 'A', 'C', 'E' };
    static constexpr uint32_t version = 1;
    bool enabled_ = false, decoding_ = false, prefilled_ = false, predicted_ = false;
    int32_t layers_ = 0, experts_ = 0;
    size_t capacity_ = 0, matches_ = 4;
    activation_matrix prompt_, decode_;
    std::vector<uint64_t> prompt_tokens_, layer_tokens_;
    uint64_t tokens_ = 0, sequence_ = 0, requests_ = 0, predictions_ = 0;
    int typical_selections_ = 0;
    std::vector<float> profile_, profile_per_token_, last_, last_per_token_;
    std::vector<expert_trace> library_;
    // the current request's sources (built at its prediction), the mixture and the rates
    std::vector<float> prompt_dist_, traced_dist_, traced_per_token_;
    std::array<double, source_count> weights_ = {};
    std::vector<float> rates_, dist_, top_;
    double best_similarity_ = 0;
    // scoring
    uint64_t scored_ = 0, top_hits_ = 0;
    double mixture_probability_ = 0;
    std::array<double, source_count> source_probability_ = {};

    size_t cells() const { return size_t(layers_) * size_t(experts_); }
    size_t cell(int32_t l, int32_t e) const { return size_t(l) * size_t(experts_) + size_t(e); }
    static bool has_row(const std::vector<float> & m, size_t begin, size_t n) {
        if (m.size() < begin + n) { return false; }
        for (size_t i = begin; i < begin + n; ++i) { if (m[i] > 0.0f) { return true; } }
        return false;
    }
    // Whether a source has anything for a layer: uniform always; the others where their row has mass. A source with
    // nothing takes no part in that layer's mixture (the others share its weight) and no part in its learning.
    bool available(int s, int32_t l) const {
        const std::vector<float> * m = source_matrix(s);
        return !m || has_row(*m, size_t(l) * size_t(experts_), size_t(experts_));
    }
    const std::vector<float> * source_matrix(int s) const {
        switch (s) {
            case prompt_source: return &prompt_dist_;
            case traced_source: return &traced_dist_;
            case last_source:   return &last_;
            case profile_source: return &profile_;
            default: return nullptr;
        }
    }
    // A source's distribution over a layer's experts (uniform for the uniform source, and for a source with nothing).
    double source_dist(int s, int32_t l, int32_t e) const {
        const std::vector<float> * m = source_matrix(s);
        const size_t begin = size_t(l) * size_t(experts_);
        if (!m || !has_row(*m, begin, size_t(experts_))) { return 1.0 / double(experts_); }
        return (*m)[begin + size_t(e)];
    }
    // A layer's selections per decode token: the prompt's routes per token, else the traced, last or profile value,
    // else what the request's first decode layer selected (a graph prefill reports routes without its tokens).
    float layer_rate(int32_t l) const {
        const size_t i = size_t(l);
        if (i < prompt_tokens_.size() && prompt_tokens_[i] && prompt_.totals[i] > 0) {
            return float(prompt_.totals[i] / double(prompt_tokens_[i]));
        }
        for (const auto * v : { &traced_per_token_, &last_per_token_, &profile_per_token_ }) {
            if (i < v->size() && (*v)[i] > 0) { return (*v)[i]; }
        }
        return float(std::max(1, typical_selections_));
    }
    void build_sources() {
        prompt_dist_ = normalized_rows(prompt_);
        traced_dist_.assign(cells(), 0.0f); traced_per_token_.assign(size_t(layers_), 0.0f);
        best_similarity_ = 0;
        if (library_.empty() || !(prompt_.total() > 0)) { return; }
        // the most similar prompts, mixed by similarity
        std::vector<std::pair<double, size_t>> ranked;
        for (size_t i = 0; i < library_.size(); ++i) {
            const double s = row_similarity(prompt_dist_, library_[i].prompt, layers_, experts_);
            if (s >= min_similarity) { ranked.push_back({ s, i }); }
        }
        std::sort(ranked.begin(), ranked.end(), [&](const auto & a, const auto & b) {
            return a.first != b.first ? a.first > b.first : library_[a.second].sequence > library_[b.second].sequence;
        });
        if (ranked.size() > matches_) { ranked.resize(matches_); }
        if (ranked.empty()) { return; }
        best_similarity_ = ranked.front().first;
        double total = 0;
        for (const auto & r : ranked) { total += r.first; }
        for (const auto & r : ranked) {
            auto & t = library_[r.second];
            t.access += 1.0;
            const double w = r.first / total;
            for (size_t c = 0; c < traced_dist_.size(); ++c) { traced_dist_[c] += float(w * t.decode[c]); }
            for (int32_t l = 0; l < layers_; ++l) { traced_per_token_[size_t(l)] += float(w * t.per_token[size_t(l)]); }
        }
    }
    // One decode selection: each source's posterior share of it moves that source's weight (online EM); the mixture's
    // and each source's probability of it, and whether it was in the layer's predicted top-k, are scored.
    void learn(int32_t l, int32_t e) {
        if (!predicted_) { return; }
        std::array<double, source_count> p = {};
        std::array<bool, source_count> in = {};
        double mixture = 0, present = 0;
        for (int s = 0; s < source_count; ++s) {
            in[size_t(s)] = available(s, l);
            p[size_t(s)] = source_dist(s, l, e);
            if (in[size_t(s)]) { mixture += weights_[size_t(s)] * p[size_t(s)]; present += weights_[size_t(s)]; }
        }
        ++scored_;
        mixture_probability_ += present > 0 ? mixture / present : 0.0;
        for (int s = 0; s < source_count; ++s) { source_probability_[size_t(s)] += p[size_t(s)]; }
        top_hits_ += dist_[cell(l, e)] >= top_[size_t(l)]; // in its layer's predicted top-k (ties count in)
        if (!(mixture > 0) || !(present > 0)) { return; }
        // each present source toward its posterior share of the selection, the present sources' total weight kept: an
        // absent source neither gains nor loses
        double sum = 0;
        for (int s = 0; s < source_count; ++s) {
            if (!in[size_t(s)]) { continue; }
            const double posterior = weights_[size_t(s)] * p[size_t(s)] / mixture;
            weights_[size_t(s)] = (1.0 - weight_step) * weights_[size_t(s)] + weight_step * present * posterior;
        }
        for (double w : weights_) { sum += w; }
        for (double & w : weights_) { w /= sum; }
    }
    // MoE-Infinity's replacement: a full library replaces its least matched trace (the oldest among equals); every
    // insertion decays the access counts so that a domain no request visits any more gives way.
    void insert(expert_trace && t) {
        for (auto & x : library_) { x.access *= access_keep; }
        if (library_.size() < capacity_) { library_.push_back(std::move(t)); return; }
        size_t victim = 0;
        for (size_t i = 1; i < library_.size(); ++i) {
            const auto & a = library_[i];
            const auto & b = library_[victim];
            if (a.access < b.access || (a.access == b.access && a.sequence < b.sequence)) { victim = i; }
        }
        library_[victim] = std::move(t);
    }
    // little-endian encoding; rows as (count, then index/value pairs of the nonzero cells)
    static void put(std::string & o, const void * p, size_t n) { o.append(static_cast<const char *>(p), n); }
    static void put_u32(std::string & o, uint32_t v) { for (int i = 0; i < 4; ++i) { o.push_back(char(v >> (8 * i))); } }
    static void put_u64(std::string & o, uint64_t v) { for (int i = 0; i < 8; ++i) { o.push_back(char(v >> (8 * i))); } }
    static void put_f64(std::string & o, double v) { uint64_t u; std::memcpy(&u, &v, 8); put_u64(o, u); }
    static void put_f32(std::string & o, float v) { uint32_t u; std::memcpy(&u, &v, 4); put_u32(o, u); }
    static void put_rows(std::string & o, const std::vector<float> & m) {
        uint32_t n = 0;
        for (float v : m) { n += v != 0.0f; }
        put_u32(o, uint32_t(m.size())); put_u32(o, n);
        for (size_t i = 0; i < m.size(); ++i) { if (m[i] != 0.0f) { put_u32(o, uint32_t(i)); put_f32(o, m[i]); } }
    }
    static bool get(const std::string & in, size_t & at, void * p, size_t n) {
        if (n > in.size() - at) { return false; }
        std::memcpy(p, in.data() + at, n); at += n; return true;
    }
    static bool get_u32(const std::string & in, size_t & at, uint32_t & v) {
        unsigned char b[4]; if (!get(in, at, b, 4)) { return false; }
        v = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24; return true;
    }
    static bool get_u64(const std::string & in, size_t & at, uint64_t & v) {
        uint32_t lo = 0, hi = 0; if (!get_u32(in, at, lo) || !get_u32(in, at, hi)) { return false; }
        v = uint64_t(lo) | uint64_t(hi) << 32; return true;
    }
    static bool get_f64(const std::string & in, size_t & at, double & v) {
        uint64_t u = 0; if (!get_u64(in, at, u)) { return false; } std::memcpy(&v, &u, 8); return true;
    }
    static bool get_rows(const std::string & in, size_t & at, std::vector<float> & m, size_t expected) {
        uint32_t size = 0, n = 0;
        if (!get_u32(in, at, size) || !get_u32(in, at, n)) { return false; }
        // an empty row set (a source never filled) reads back as zeros of the expected size
        if ((size != expected && size != 0) || n > size || size_t(n) * 8 > in.size() - at) { return false; }
        m.assign(expected, 0.0f);
        for (uint32_t k = 0; k < n; ++k) {
            uint32_t i = 0, u = 0;
            if (!get_u32(in, at, i) || !get_u32(in, at, u) || i >= size) { return false; }
            float v; std::memcpy(&v, &u, 4);
            if (!std::isfinite(v) || v < 0.0f) { return false; }
            m[i] = v;
        }
        return true;
    }
};

}} // namespace ssd::place
