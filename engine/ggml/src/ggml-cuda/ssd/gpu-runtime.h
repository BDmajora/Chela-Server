// SPDX-License-Identifier: Apache-2.0
#pragma once
// The GPU runtime of the expert tier: CUDA's, or HIP's through ggml's CUDA-to-HIP names (vendors/hip.h), so that one
// set of tier sources builds into ggml-cuda and ggml-hip. Under HIP every file that includes this is HIP language.
#if defined(GGML_USE_HIP)
#include "../vendors/hip.h"
#else
#include <cuda_runtime.h>
#endif
