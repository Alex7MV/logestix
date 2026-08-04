#!/usr/bin/env bash
#
# build-aocc.sh — one-shot AOCC (AMD clang/LLVM) build for Linux.
#
# Configures a fresh `build-aocc` directory with the AOCC toolchain,
# Release + tests, then builds. Passes any extra args through to CMake
# (configure stage). Uses LLVM-LTO through lld; the AOCC `bin` (for
# `ld.lld`) is prepended to PATH.
#
# Override points:
#   AOCC_ROOT  path to the AOCC install dir (auto-detected by default)
#   BUILD_DIR  build dir (default: <repo>/build-aocc)
#   CONFIG     CMAKE_BUILD_TYPE (default: Release)
#   JOBS       parallel jobs (default: nproc)
#
# Usage:
#   ./build-aocc.sh
#   ./build-aocc.sh -DBUILD_TESTING=OFF
#   AOCC_ROOT=/opt/AMD/aocc ./build-aocc.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build-aocc}"
CONFIG="${CONFIG:-Release}"
JOBS="${JOBS:-$(nproc)}"

# ── Locate AOCC ─────────────────────────────────────────────────────────────
if [[ -z "${AOCC_ROOT:-}" ]]; then
  for cand in \
    /home/alexmv2025/projects/amd/aocc-compiler-5.2.0 \
    /opt/AMD/aocc \
    /opt/aocc; do
    if [[ -x "$cand/bin/clang++" ]]; then
      AOCC_ROOT="$cand"
      break
    fi
  done
fi
if [[ -z "${AOCC_ROOT:-}" ]]; then
  echo "error: AOCC not found. Install it or set AOCC_ROOT=/path/to/aocc" >&2
  exit 1
fi
if [[ ! -x "$AOCC_ROOT/bin/ld.lld" ]]; then
  echo "error: ld.lld not found in $AOCC_ROOT/bin (LLVM-LTO needs lld)" >&2
  exit 1
fi
export PATH="$AOCC_ROOT/bin:$PATH"

echo "AOCC:      $AOCC_ROOT ($("$AOCC_ROOT/bin/clang++" --version | sed 's/^/            /' | sed -n '1p'))"
echo "Build dir: $BUILD_DIR"
echo "Config:    $CONFIG"
echo "Jobs:      $JOBS"
echo

cmake -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$CONFIG" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DBUILD_TESTING=ON \
  "$@"

cmake --build "$BUILD_DIR" -j"$JOBS"