#pragma once

// Perf auto-tuning calibration (gpu_perf.h policy layer). The .cu backend
// benchmarks the GEMM paths on synthetic NVFP4 buffers of representative
// shapes; the orchestration decides when a first-Load calibration runs and
// persists the result into the vault cache.

#include <functional>
#include <vector>

#include "lsxcommon/gpu_perf.h"

namespace lsxcommon {
namespace perf {

struct CalibrationShape {
    int K;
    int N;
};

// Fixed representative shape classes (attn qkv, mlp gate/up, expert w2,
// LM head) spanning the GEMV 32 MiB decision boundary from both sides.
const std::vector<CalibrationShape> &RepresentativeShapes();

// Device micro-benchmark: per shape, time the streaming GEMV against the TC
// path (seq==1) on synthetic NVFP4 buffers and return the largest
// GEMV-winning byte size in MiB (clamped 8..64). Saves/restores the process
// routing toggles. Returns false on CUDA failure.
bool CalibrateGemvLimitMib(int &out_mib);

// Orchestration (host-only, unit-testable): returns true when a calibration
// ran (unknown GPU, no cache, perf auto enabled) and persists the result.
// `run_bench` injects the device benchmark (tests pass fakes; production
// passes CalibrateGemvLimitMib). With run_bench == nullptr this is a
// query-only call: true means "calibration pending", no side effects.
bool EnsureCalibrated(const GpuPerfProfile &gpu,
                      const std::function<bool(int &out_mib)> &run_bench =
                          nullptr);

} // namespace perf
} // namespace lsxcommon
