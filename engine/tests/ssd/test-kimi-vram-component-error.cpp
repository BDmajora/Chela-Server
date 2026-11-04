#include "vram-compute-api.h"
#include "ggml.h"
#include "ggml-kimi-mxfp4-parity.h"
#include "ggml-kimi-mxfp4-chain.h"
#include "gpu-runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "gpu-kernels.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const kimi_mxfp4_gpu_api_v1 * real_base;
const kimi_mxfp4_gpu_chain_api_v1 * real_chain;
kimi_mxfp4_gpu_api_v1 proxy_base;
kimi_mxfp4_gpu_chain_api_v1 proxy_chain;
enum class fault { none, submit, wait } inject = fault::none;
bool injected = false;
int submissions = 0, assertions = 0;
std::string destroyed;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
    ++assertions;
}
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int submitted(int status) {
    if (status) ++submissions;
    // The real implementation has already enqueued work using cache weights.
    if (status && inject == fault::submit) { injected = true; return 0; }
    return status;
}
int submit(kimi_mxfp4_gpu * p, const kimi_mxfp4_gpu_job * jobs, int n, const void * q8, size_t bytes) {
    return submitted(real_base->submit(p, jobs, n, q8, bytes));
}
int chain_submit(kimi_mxfp4_gpu_chain * p, const kimi_mxfp4_chain_expert * experts,
        int n, const void * q8, size_t bytes, float limit) {
    return submitted(real_chain->submit(p, experts, n, q8, bytes, limit));
}
int wait(kimi_mxfp4_gpu * p) {
    if (inject == fault::wait) { injected = true; return 0; }
    return real_base->wait(p);
}
const char * error(kimi_mxfp4_gpu * p) {
    return injected ? "injected error after real GPU submission" : real_base->error(p);
}
int destroy(kimi_mxfp4_gpu * p) {
    int status = real_base->destroy(p); // unconditional real stream synchronize
    if (status) destroyed += 'B';
    return status;
}
int chain_destroy(kimi_mxfp4_gpu_chain * p) {
    int status = real_chain->destroy(p);
    if (status) destroyed += 'C';
    return status;
}
struct device_payload {
    void * data = nullptr;
    ~device_payload() { if (data) cudaFree(data); }
};
struct context {
    const vram_format_compute_api * api;
    void * value;
    ~context() { if (value) api->destroy(value); }
};
}

int main(int argc, char ** argv) { try {
    require(argc == 1, "usage: test-kimi-vram-component-error");
    real_base = kimi_mxfp4_gpu_get_api_v1(); real_chain = kimi_mxfp4_gpu_get_chain_api_v1();
    proxy_base = *real_base; proxy_chain = *real_chain;
    proxy_base.submit = submit; proxy_base.wait = wait; proxy_base.error = error; proxy_base.destroy = destroy;
    proxy_chain.submit = chain_submit; proxy_chain.destroy = chain_destroy;
    const auto * api = kimi_vram_format_compute_get_api();
    require(api && api->version == 4 && api->size == sizeof(*api), "compute ABI");
    require(!api->warm_configure && !api->warm_demote && !api->warm_drain, "MXFP4 offered a warm pool");
    constexpr int latent = 256, hidden = 256;
    const size_t role = ggml_row_size(GGML_TYPE_MXFP4, latent) * hidden;
    device_payload weights;
    cuda_check(cudaMalloc(&weights.data, 3 * role));
    cuda_check(cudaMemset(weights.data, 0, 3 * role));
    cuda_check(cudaDeviceSynchronize());
    auto * bytes = static_cast<unsigned char *>(weights.data);
    const vram_compute_expert expert{bytes, bytes + role, bytes + 2 * role};
    std::vector<unsigned char> q8(ggml_row_size(GGML_TYPE_Q8_0, latent), 0);
    const kimi_vram_activation activation = {KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0, 0, 3.0f};
    std::vector<float> output(latent, 123.0f);
    for (int gpu_hidden : {0, 1}) {
        for (fault point : {fault::submit, fault::wait}) {
            char message[512] = {};
            // Engine creation runs the retained parity self-check before any injection.
            inject = fault::none; injected = false; submissions = 0; destroyed.clear();
            std::fill(output.begin(), output.end(), 123.0f);
            context failed{api, ggml_ssd_mxfp4_create_with_components_for_testing(0, latent, hidden, 1, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, bool(gpu_hidden), false, test_cpu_kernels(), &proxy_base, &proxy_chain, message, sizeof(message))};
            require(failed.value != nullptr, message);
            inject = point; injected = false; submissions = 0; destroyed.clear();
            const int status = api->compute(failed.value, &expert, 1, q8.data(), q8.size(), &activation,
                    output.data(), message, sizeof(message));
            require(status != 0 && injected && submissions == 1 && std::strstr(message, "injected error"),
                    "real queued work must propagate the injected failure");
            require(destroyed == (gpu_hidden ? "CB" : "B"), "component was not drained before compute returned error");
            require(api->device_scratch_bytes(failed.value) == 0, "failed component retained device scratch");
            require(std::all_of(output.begin(), output.end(), [](float x) { return x == 123.0f; }), "failure published output");
            require(api->compute(failed.value, &expert, 1, q8.data(), q8.size(), &activation,
                    output.data(), message, sizeof(message)) != 0 && submissions == 1 && std::strstr(message, "closed"),
                    "failed bridge silently reused or recreated component");
            api->destroy(failed.value); failed.value = nullptr;
            require(destroyed == (gpu_hidden ? "CB" : "B"), "component destroyed twice");
            std::printf("COMPONENT_ERROR_CASE_PASS gpu_hidden=%d point=%s\n", gpu_hidden, point == fault::submit ? "submit" : "wait");
        }
        // Only a separately created context may run after the failure.
        inject = fault::none; injected = false; destroyed.clear();
        char message[512] = {};
        context healthy{api, ggml_ssd_mxfp4_create_with_components_for_testing(0, latent, hidden, 1, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, bool(gpu_hidden), false, test_cpu_kernels(), &proxy_base, &proxy_chain, message, sizeof(message))};
        require(healthy.value != nullptr, message);
        require(api->compute(healthy.value, &expert, 1, q8.data(), q8.size(), &activation,
                output.data(), message, sizeof(message)) == 0, message);
        require(std::all_of(output.begin(), output.end(), [](float x) { return x == 0.0f; }), "new context numerical result");
    }
    std::printf("REAL_CUDA_COMPONENT_ERROR_PASS cases=4 assertions=%d\n", assertions);
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "COMPONENT_ERROR_FAIL: %s\n", e.what());
    return 1;
} }
