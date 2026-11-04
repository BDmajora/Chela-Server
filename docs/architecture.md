<!-- SPDX-License-Identifier: Apache-2.0 -->
# How it works

[Back to README](../README.md)

The model router chooses the experts needed by each token. InferSSD manages how those expert weights are indexed, read from SSD, cached and executed.

```mermaid
flowchart LR
    S[GGUF or matching expert pack on SSD] --> I[Expert source index]
    M[Model router] --> E[Selected experts]
    I --> R[RAM expert cache]
    E --> R
    R --> C[CPU expert compute]
    R --> G[Linux CUDA expert tier]
    C --> O[Model output]
    G --> O
    D[Dense compute: Metal or CUDA] --> O
```

## Expert sources and caching

Source indexes locate expert byte ranges in the original GGUF files or a matching v1 expert pack. Indexing GGUF headers does not copy the model payload. Packs group expert slices by layer and expert with **4 KiB alignment**, preserving the existing quantized bytes. The v1 format has no checkpoint fingerprint, so keep each pack paired with its source checkpoint.

The runtime owns immutable source metadata, readers, host-budget accounting and telemetry for each context. Executors borrow those services. Linux, macOS and Windows run configured experts on the same CPU sink, which reads through the OS read ring (`io_uring` on Linux, IoRing on Windows, persistent reader threads with `F_NOCACHE` on macOS) into its RAM cache; with CUDA, or with HIP on AMD GPUs (the `linux-rocm` preset builds the same tier sources through ggml's CUDA-to-HIP names), Linux adds the GPU cache tier. A GPU backend without that tier, such as Metal, runs the dense layers, and the sink computes every routed expert on the CPU. Device and source validation remains necessary for the selected configuration.

## Expert prediction

Each context's sink keeps a request-level expert predictor (`ggml-ssd/place/expert-trace.h`, typed `trace.*` settings). It sees the routes of every prompt through grouped prefill and the selections of every decode token, and when a request's decode starts it predicts the rate at which each routed expert of each layer will be selected per token. Five sources are mixed:

| Source | What it predicts from | After |
| --- | --- | --- |
| prompt | the experts the request's own prefill routed: decode keeps to the domain of its prompt | DoMoE (IJCAI 2026), prefill-stage domain prediction |
| traced | the decodes of earlier requests whose prompts routed alike: expert activation matrices kept in a bounded collection, matched by mean per-layer cosine similarity, the least matched replaced first | MoE-Infinity (arXiv:2401.14361), request-level activation tracing |
| last | the previous request's decode: consecutive turns of one conversation | |
| profile | a decayed usage profile of every request, kept across runs in `trace.file` | moe-cache usage profiles (warm start) |
| uniform | every expert alike, so that no rate is zero | |

The mixture weights are fitted online to the selections decode makes (online EM for mixture weights); a source with no data for a layer drops out of that layer's mixture. With `trace.prior` the GPU tier starts the request's decode from the predicted rates, turned into the decayed selection counts they would have left (`vram_slot_table::set_prior`), instead of the previous request's heat. Its admissions and evictions then price the experts the request is expected to use as if it had already used them, under the same cost rule as before, and the request's own selections take over within a few half-lives. A prior changes which experts the tier holds, never a result. The RAM cache takes the same prediction for the experts above each layer's mean rate. A replay of a synthetic multi-domain request stream through the slot table (`test-kimi-vram-prior-replay`) is the model-free check of the mechanism; gains on a real model are device measurements.

Within a token, the next layer's router is known before the layer runs: its gate takes the current layer's output, and the sink reads ahead what the hint predicts. With `vram.lookahead` the highest-ranked of those experts also go from RAM to the GPU tier ahead of their selection, as far as the measured uploads fit in the measured time to the next layer (`vram_slot_table::reserve_lookahead`). The admission's expected hits are weighed by how often that rank of the hint proves right, measured per rank, so a poor hint admits nothing.

Two optional mechanisms are approximate, and `decode.drop_mass` and `vram.warm_mib` keep them off by default. Expert drop skips, per token and layer, the cold experts of least routing weight, within a budget of that weight, and scales the rest back to the original sum (MoE-Infinity's measured tolerance of low-mass experts, moe-cache's expert skip). The warm pool (`vram-warm-pool.h`) follows DynaExq's hot-high, cold-low precision without its online retraining: when the slot table evicts an expert the device requantizes it in place to Q4_0 or Q2_0 and keeps serving it at that precision, the most recently evicted staying, until the ordinary admission rule promotes it. Each conversion's readers and the upload that overwrites its slot wait for it on an event, so nothing stalls the sink's thread.

Grouped prefill moves to the GPU the experts whose computation saves the most CPU time, and its GPU lane runs them in Johnson's order for the two-stage flow shop of their copies and computations where the pipeline model ends sooner (ES-MoE's expert-level scheduling, PMLR 235). With one expert size every copy is equal and that is the order by routes in which they are chosen.

The tier's upload streams and compute stream already overlap: a deferred upload runs on its own stream and thread while the layer computes, and a layer's next prepare waits only for what it needs (DeepSpeed-FastGen's overlapped streams and ZeRO-Inference's host offload, in the tier's own form).

## Threads

CPU compute runs on one pool, ggml's threadpool, for dense and expert work alike. A sink node is an ordinary multi-threaded ggml op: every CPU-backend thread of the graph enters it, thread 0 opens a call for the node's thread count and runs the layer, and the other threads join the call as its compute participants until it closes. Each compute phase splits its rows over the participants, by atomic row chunks or explicit row weights; a decode phase uses the first N of them, where N is all of them or the number that fits the fastest CPU cluster, whichever measures faster, or `cpu.decode_participants`. Grouped prefill and the cache executor dispatch through the same phases. The participants take no lock: thread 0 publishes each phase in one atomic word, and the others park on it through the OS's wait-on-address primitive between phases, so one wake reaches all of them.

Placement is ggml's. `-t` and `-tb` set the decode and batch thread counts, `-C` and `-Cb` their CPU masks, and `--cpu-strict` and `--cpu-strict-batch` place thread i on the i-th CPU of its mask in increasing order; `--prio` and `--poll` set priority and polling. The sink keeps only its own I/O thread, its prefetch readers and, on unified memory, a watch thread, all placed by `cpu.io_cpus`.

The CUDA cache tier's worker and upload threads run with thread 0, which waits for them: the sink gives the tier thread 0's CPUs at its first layer on the tier and again whenever they change, so with `--cpu-strict` they share thread 0's core.

## Batching

Prompt batching lets tokens share weight access. `inferssd launch --batch` and `--ubatch` set the native batch and micro-batch capacities. The Linux controller consumes native telemetry to inform placement and prefill decisions; see [controller observations](architecture/controller-observations.md).

On macOS the launcher supports static settings only; it does not include an online adaptive controller. The settings used in the recorded Mac benchmark runs are listed in [benchmark methodology](benchmark.md#engine-configurations).

## Modules

The runtime extends the bundled llama.cpp snapshot. Its integration points are listed in [upstream patches](development/upstream-patches.md).

| Layer | Directory | Owns |
| --- | --- | --- |
| Core | `engine/ggml/src/ggml-ssd/store` | Immutable per-context source indexes and explicit pack/alias parsing |
| Core | `engine/ggml/src/ggml-ssd/io` | Independent syscall and liburing readers; portable pread |
| Core | `engine/ggml/src/ggml-ssd/host` | Process host arbiter, per-context leases, expert cache and page accounting |
| Core | `engine/ggml/src/ggml-ssd/place` | Shared placement cost evaluation, learning state, prefetch decisions and request-level expert prediction |
| Core | `engine/ggml/src/ggml-ssd/stream` | Expert ranges, dense lookahead, seed ordering and copy observations |
| Core | `engine/ggml/src/ggml-ssd/settings`, `telemetry` | Immutable validated settings, source receipts and owned JSONL writers |
| Executors | `engine/ggml/src/ggml-cpu/ssd`, `engine/ggml/src/ggml-cuda/ssd` | CPU scheduling and compute; GPU tier (CUDA and HIP), staging, events and format kernels |
| Runtime | `engine/src/ssd` | Context assembly, backend registry binding, model sources, model adaptation, graph admission and diagnostics |
| llama extensions | `engine/src/inferssd` | KV state codecs and positional state serialization |
| Common | `engine/common/inferssd` | CLI glue, model chat parsing, reasoning budgets |
| Server | `engine/tools/server/inferssd` | Prompt cache, checkpoints and request lifecycle |
| Framework | `framework/src/inferssd` | Standard-library Python package: GGUF and pack metadata preparation (`source/`), probes, plans, typed settings (`configuration/`) and owned launch (`launch/`) |

## Dependency direction

Core code depends only on ggml-base and platform facilities. A CPU executor never links CUDA; a CUDA executor never links CPU. The runtime obtains immutable, size-checked provider tables from the same build through the ordinary backend registries and supplies context-owned services. The scheduler calls a borrowed hook table, so ggml-base has no dependency on the core implementation. Kernels stay in their backend.

Model-specific SSD requirements are described in `engine/src/ssd/model-adapter.h`: expert layout and quantization relationships, and activation and normalization requirements, are validated before the shared MoE bridge creates an executor node. The context uses the same descriptor for its early support check. Executors consume validated operations rather than model identities; the ordinary shared MoE implementation remains in place.

`engine/ggml/src/ggml-ssd/place/runtime-policy.h` owns numeric learning state and placement and prefetch decisions. Executors supply workload, resource and measured cost inputs, execute the returned request and rank plans using their own caches, readers and workers, then publish feedback at fixed update points. Policy code holds no sink, device, file or worker handles. All executors share one placement evaluator, so they apply the same cost arithmetic and selection thresholds.

Run `python3 scripts/inferssd/check-layering.py` to check these boundaries.

## Runtime ownership

The context owns sources, host registration, the telemetry writer, transport and diagnostics; executors borrow them. Graph failure fences outstanding seed consumers before reset. A precompute transaction restores touched and purged KV metadata without allocating, and it ends before compute submission. Stateful memory that cannot roll back a submitted failure clears the affected sequences for checkpoint restore or replay. Backend destruction quiesces transport before destroying CUDA streams. The writer and source descriptors close after all borrowers stop.

Allocation admission precedes allocation and first touch; committed allocations invalidate learning intervals when they are later freed or replaced. An admission that does not fit first has the context's expert caches give room back: the executor registers their reclaimer with the context's host client, so every allocation, including the CPU backend's work buffer, gets that room without asking for it. Linux uses finite cgroup constraints; macOS and Windows use the process envelope described in [configuration](configuration.md).

## Settings and deployment

Production configuration uses the typed `inferssd-settings 1` format described in [configuration](configuration.md). Retired environment names are rejected by name without reporting their values. Only `INFERSSD_DEBUG` and `INFERSSD_TRACE_DIR` are observation channels, and trace or profile output requires an owned telemetry writer. Options inherited from llama.cpp keep their upstream behavior.

Deployment paths and capabilities are declared as explicit [engine resources](engine-resources.md), so the installed framework needs no source checkout. Native probes live in `engine/tools/inferssd-probe` and run only when requested.

The DeepSeek and Muse benchmark configurations retain native expert routing. Throughput alone does not establish model quality; record numerical checks, exact settings and workload evidence alongside each experiment.
