#!/usr/bin/env bash
# Build inferssd-server for AMD GPUs through ROCm. See ./build.sh --help.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=$repo_dir/build-linux-rocm
binary=$build_dir/bin/inferssd-server

usage() {
    cat <<'EOF'
Usage: ./build.sh [options] [-- extra cmake arguments]

Checks the prerequisites, finds your AMD GPU, builds build-linux-rocm/bin/inferssd-server
and confirms that the server can see the GPU.

Options:
  --gpu TARGET   Build for this GPU instead of the detected one, for example gfx1030.
                 Separate several targets with semicolons.
  --jobs N       Parallel compile jobs. By default one per CPU, limited by available RAM.
  --clean        Delete build-linux-rocm first. Needed after changing GPU.
  --check        Check the prerequisites and the GPU, then stop without building.
  -h, --help     Show this help.

ROCM_PATH selects a ROCm install. Otherwise /opt/rocm is used if it exists, else /usr.
EOF
}

die() {
    printf 'build.sh: %s\n' "$*" >&2
    exit 1
}

gpu=${GPU_TARGETS:-}
jobs=${JOBS:-}
clean=0
check_only=0
cmake_args=()
while [ $# -gt 0 ]; do
    case $1 in
        --gpu)
            [ $# -ge 2 ] || die '--gpu needs a target such as gfx1030'
            gpu=$2
            shift 2
            ;;
        --jobs)
            [ $# -ge 2 ] || die '--jobs needs a number'
            jobs=$2
            shift 2
            ;;
        --clean)
            clean=1
            shift
            ;;
        --check)
            check_only=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            cmake_args=("$@")
            break
            ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

[ "$(uname -s)" = Linux ] || die 'ROCm builds need Linux.'
if [ -n "$jobs" ] && [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    die '--jobs must be a positive number.'
fi

# The same search order as engine/ggml/src/ggml-hip/CMakeLists.txt.
rocm_root=${ROCM_PATH:-}
if [ -z "$rocm_root" ]; then
    if [ -d /opt/rocm ]; then rocm_root=/opt/rocm; else rocm_root=/usr; fi
fi
PATH=$rocm_root/bin:$PATH

# rocminfo names the GPU. The kernel's topology still does when ROCm is not installed yet.
detect_gpus() {
    local found='' file version
    if command -v rocminfo >/dev/null 2>&1; then
        found=$(rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ { print $2 }' | sort -u)
    fi
    if [ -z "$found" ]; then
        for file in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            [ -r "$file" ] || continue
            version=$(awk '$1 == "gfx_target_version" { print $2 }' "$file")
            [ "${version:-0}" -gt 0 ] || continue
            found+=$(printf 'gfx%d%d%x\n' $((version / 10000)) $((version / 100 % 100)) $((version % 100)))$'\n'
        done
        found=$(printf '%s' "$found" | sort -u)
    fi
    printf '%s' "$found" | paste -sd ';' -
}

has_cmake_package() {
    [ -n "$(find "$rocm_root/lib" "$rocm_root/lib64" "$rocm_root/share" -maxdepth 5 \
        -name "$1-config.cmake" -print -quit 2>/dev/null)" ]
}

missing=()
packages=()
require() {
    missing+=("$1")
    packages+=("$2")
}

command -v c++ >/dev/null 2>&1 || require 'a C++ compiler' g++
command -v ninja >/dev/null 2>&1 || require 'Ninja' ninja-build
if command -v cmake >/dev/null 2>&1; then
    # CMakePresets.json has schema version 4.
    cmake_version=$(cmake --version | awk 'NR == 1 { print $3 }')
    if [ "$(printf '3.23\n%s\n' "$cmake_version" | sort -V | head -n 1)" != 3.23 ]; then
        require "CMake 3.23 or later (found $cmake_version)" cmake
    fi
else
    require 'CMake 3.23 or later' cmake
fi
command -v hipconfig >/dev/null 2>&1 || require 'the HIP compiler driver (hipconfig)' hipcc
has_cmake_package hip || require 'the HIP runtime headers' libamdhip64-dev
has_cmake_package hipblas || require 'hipBLAS' libhipblas-dev
has_cmake_package rocblas || require 'rocBLAS' librocblas-dev
if ! printf '#include <liburing.h>\n' | c++ -E -x c++ - >/dev/null 2>&1; then
    require 'liburing headers' liburing-dev
fi

if [ ${#missing[@]} -gt 0 ]; then
    printf 'build.sh: missing prerequisites:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    if command -v apt-get >/dev/null 2>&1; then
        printf 'Install them with:\n  sudo apt install %s\n' "${packages[*]}" >&2
    fi
    exit 1
fi

if [ -z "$gpu" ]; then
    gpu=$(detect_gpus)
    [ -n "$gpu" ] || die 'no AMD GPU found. Use --gpu to build for one on another machine.'
fi

if [ -e /dev/kfd ] && { [ ! -r /dev/kfd ] || [ ! -w /dev/kfd ]; }; then
    printf 'Warning: you cannot open /dev/kfd, so the server will not see the GPU.\n' >&2
    printf 'Fix it with: sudo usermod -aG render,video %s\nThen log in again.\n' "$(id -un)" >&2
fi

printf 'GPU target: %s\nROCm: %s\n' "$gpu" "$rocm_root"
if [ "$check_only" = 1 ]; then
    printf 'Prerequisites are in place.\n'
    exit 0
fi

if [ "$clean" = 1 ]; then
    rm -rf -- "$build_dir"
elif [ -f "$build_dir/CMakeCache.txt" ]; then
    previous=$(sed -n 's/^GPU_TARGETS:[^=]*=//p' "$build_dir/CMakeCache.txt")
    if [ -n "$previous" ] && [ "$previous" != "$gpu" ]; then
        die "build-linux-rocm is configured for $previous. Rerun with --clean to build for $gpu."
    fi
fi

if [ -z "$jobs" ]; then
    # HIP objects are heavy to compile, so leave each job about 2 GiB of RAM.
    jobs=$(awk -v cpus="$(getconf _NPROCESSORS_ONLN)" '/^MemAvailable:/ {
        by_memory = int($2 / 2097152)
        print (by_memory < 1 ? 1 : (by_memory < cpus ? by_memory : cpus))
    }' /proc/meminfo)
fi

cd "$repo_dir"
cmake --preset linux-rocm "-DGPU_TARGETS=$gpu" "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "$jobs" --target inferssd-server

devices=''
if [ -e /dev/kfd ]; then
    list=$("$binary" --list-devices 2>&1 || true)
    devices=$(printf '%s\n' "$list" | grep -E '^ *ROCm[0-9]+:' || true)
    if [ -z "$devices" ]; then
        printf '%s\n' "$list" >&2
        die "built $binary, but it sees no ROCm device."
    fi
fi

printf '\nBuilt %s\n' "$binary"
if [ -n "$devices" ]; then
    printf '%s\n' "$devices"
else
    printf 'This machine has no /dev/kfd, so the GPU check was skipped.\n'
fi
printf 'To use it from Chela, set Server executable to that path in the connection settings.\n'
