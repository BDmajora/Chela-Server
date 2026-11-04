find_package(Threads REQUIRED)

if (GGML_HIP)
    # The same tier through ggml's CUDA-to-HIP names (vendors/hip.h), linked into ggml-hip. Every source is HIP
    # language, the C++ ones too: the vendor header needs the HIP compiler. The strict parity target keeps the
    # contract of --fmad=false --ftz=false: no contraction into fused multiply-adds, denormals kept.
    set(SSD_PARITY_SOURCES kimi-mxfp4-chain.cu kimi-mxfp4-batch.cu)
    set(SSD_TIER_SOURCES vram-tier.cpp mxfp4-compute-adapter.cpp
        generic-compute-adapter.cu group-compute-adapter.cu provider.cpp transport.cpp)
    set_source_files_properties(${SSD_PARITY_SOURCES} ${SSD_TIER_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(kimi-mxfp4-parity OBJECT ${SSD_PARITY_SOURCES})
    set_target_properties(kimi-mxfp4-parity PROPERTIES
        POSITION_INDEPENDENT_CODE ON HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    target_include_directories(kimi-mxfp4-parity PRIVATE ../../../include ../..)
    target_compile_options(kimi-mxfp4-parity PRIVATE
        -O3 -fno-fast-math -ffp-contract=off -fno-gpu-flush-denormals-to-zero -Wall -Wextra)
    target_link_libraries(kimi-mxfp4-parity PRIVATE hip::host)

    add_library(kimi-vram-tier OBJECT ${SSD_TIER_SOURCES})
    set_target_properties(kimi-vram-tier PROPERTIES POSITION_INDEPENDENT_CODE ON
        HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    target_include_directories(kimi-vram-tier PRIVATE ../../../include ../..)
    target_compile_options(kimi-vram-tier PRIVATE -fno-fast-math -ffp-contract=off)
    target_link_libraries(kimi-vram-tier PRIVATE ggml-base ggml-ssd ggml-ssd-object-io hip::host Threads::Threads)
    target_link_libraries(ggml-hip PRIVATE kimi-vram-tier kimi-mxfp4-parity ggml-ssd ggml-ssd-object-io)
    target_compile_definitions(ggml-hip PRIVATE GGML_SSD_CUDA_EXECUTOR)
    return()
endif()

# Separate object targets keep the strict parity compiler contract independent
# of ggml-cuda's -use_fast_math. Both are normally linked into that backend.
add_library(kimi-mxfp4-parity OBJECT kimi-mxfp4-chain.cu kimi-mxfp4-batch.cu)
set_target_properties(kimi-mxfp4-parity PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(kimi-mxfp4-parity PRIVATE ../../../include ../..)
target_compile_options(kimi-mxfp4-parity PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--ftz=false;-Xcompiler=-fPIC,-Wall,-Wextra>)
# Preserve the standalone component's single architecture choice, including the
# existing explicit override, without an external shell compiler or DSO.
if (DEFINED ENV{PARITY_CUDA_ARCH})
    string(REGEX REPLACE "^sm_" "" SSD_PARITY_ARCH "$ENV{PARITY_CUDA_ARCH}")
elseif (CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 SSD_PARITY_ARCH)
    string(REGEX REPLACE "-(real|virtual)$" "" SSD_PARITY_ARCH "${SSD_PARITY_ARCH}")
else()
    set(SSD_PARITY_ARCH 120a)
endif()
if (NOT SSD_PARITY_ARCH MATCHES "^[0-9]+[af]?$")
    message(FATAL_ERROR "Invalid strict MXFP4 architecture: ${SSD_PARITY_ARCH}")
endif()
set_property(TARGET kimi-mxfp4-parity PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_options(kimi-mxfp4-parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${SSD_PARITY_ARCH}>)

add_library(kimi-vram-tier OBJECT vram-tier.cpp mxfp4-compute-adapter.cpp
    generic-compute-adapter.cu group-compute-adapter.cu provider.cpp transport.cpp)
set_target_properties(kimi-vram-tier PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(kimi-vram-tier PRIVATE ../../../include ../..)
target_compile_features(kimi-vram-tier PRIVATE cxx_std_17)
target_compile_options(kimi-vram-tier PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
target_link_libraries(kimi-vram-tier PRIVATE ggml-base ggml-ssd ggml-ssd-object-io CUDA::cudart Threads::Threads)
target_link_libraries(ggml-cuda PRIVATE kimi-vram-tier kimi-mxfp4-parity ggml-ssd ggml-ssd-object-io CUDA::cudart)
target_compile_definitions(ggml-cuda PRIVATE GGML_SSD_CUDA_EXECUTOR)
