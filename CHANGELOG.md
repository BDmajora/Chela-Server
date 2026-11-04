# Changelog

## Unreleased

### New

- **AMD GPUs:** the GPU expert tier now builds through ROCm's HIP (`linux-rocm` preset), alongside the CPU sink. It has run on an RX 6800 (gfx1030), and there `--fit on` was faster than the sink. The launcher still reads NVIDIA devices only.
- **ROCm build script:** `./build.sh` checks the prerequisites, detects the GPU, builds the server and confirms that it can see the GPU.
- **Q2_0 weights:** quantization, CPU kernels and GPU kernels for the 2-bit format.
- **Request-level expert prediction:** each request's decode starts from a prediction of its experts, made from its prompt, from traced earlier requests, from the previous request and from a usage profile that `trace.file` keeps across runs. The GPU tier and the RAM cache take it as a prior. It changes what is cached, not results.
- **Router lookahead for the GPU tier:** `vram.lookahead` sends the next layer's predicted experts up ahead of their selection. Off by default.
- **Qwen 3.8 router lookahead:** `qwen4exp` models now give the sink a hint for the next layer.
- **Layer ordering:** grouped prefill's GPU lane runs in Johnson's order where the pipeline model ends sooner.
- **Mixed expert formats:** the GPU tier now runs models whose layers use different expert types, in decode and in grouped prefill, with blocks of slots in each expert size.
- **Approximate decode, off by default:** `decode.drop_mass` skips cold experts of little routing weight, and `vram.warm_mib` keeps low-precision copies of evicted experts on the GPU. Both change results.

### Fixes

- **RAM copies of GPU experts:** the RAM cache gave up its copy of every expert that went up to the GPU tier, even with slots free, so each eviction from the tier read that expert from the drive again. It now keeps the copy while the layer has a free slot.

## 0.1.1 - 2026-10-31

This release fixes the Linux source install with recent CUDA toolkits.

### Fixes

- **Build with CCCL 3.4 and later:** the installer failed to compile with CCCL 3.4.0 and later, including the CCCL 3.4.3 bundled with CUDA 13.4.3 and CCCL 3.5.0 on Arch Linux. Thanks to @rokam for the report ([#95](https://github.com/wici-ai/inferssd/issues/95)).
- **`inferssd profile` after a one-command install:** the command now finds the installed probes instead of reporting them as missing. Thanks to @HamedShams for the report ([#97](https://github.com/wici-ai/inferssd/issues/97)).

### Install or upgrade

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.1/install.sh | sh
```

Run the same command to upgrade from v0.1.0. Downloaded models are not touched. The [known limitations of v0.1.0](https://github.com/wici-ai/inferssd/releases/tag/v0.1.0#known-limitations) still apply.

## 0.1.0 - 2026-10-30

First public release of InferSSD, which runs mixture-of-experts models larger than your RAM or VRAM from SSD. InferSSD builds on llama.cpp, reads GGUF weights directly, and serves browser chat and an OpenAI-compatible API.

### Highlights

- **Models:** DeepSeek-V4.1-Flash-1017 (Q8_0) and Muse-Spark-Open (UD-Q4_K_XL), with routed experts streamed from SSD through bounded RAM and VRAM caches. See the [benchmarks](https://github.com/wici-ai/inferssd/blob/v0.1.0/README.md#performance-benchmarks).
- **Platforms:** macOS on Apple Silicon with Metal, and Linux with NVIDIA CUDA on x86_64 and ARM64. Windows support is in progress.
- **One-command install and launch:** the installer sets up a private environment, and `inferssd cli` and `inferssd serve` download a tested model and start it.
- **Automatic setup:** the first launch measures the machine, and the engine sizes its caches from the memory available. There is no memory setting to choose.

### Install

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.0/install.sh | sh
```

This release installs from source, so the engine is compiled for your machine's CPU and GPU. Prebuilt packages are not available yet. Building takes about 2 to 12 minutes, depending on the machine.

The installer needs Git, Python 3.9 or later, and a C/C++ compiler. macOS also needs the Xcode Command Line Tools; Linux also needs the CUDA toolkit, `liburing` headers, and systemd with cgroup v2.

Then start a model. The first run downloads about 470 GiB of model files.

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf unsloth/Muse-Spark-Open-816B-A42B-GGUF
```

### Known limitations

- `-hf` accepts only the two tested repositories above. Use `-m` for local GGUF files.
- If the CUDA runtime is outside the system library path, the first launch cannot measure the GPU, and prefill runs on the CPU.

### License

InferSSD is licensed under Apache-2.0. Bundled llama.cpp/ggml and other third-party components keep their original licenses and notices.
