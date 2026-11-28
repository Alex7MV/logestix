#pragma once

// Model-neutral, DEVICE-INDEPENDENT matrix-vector product for decode shapes.
//
// Replaces cuBLAS GEMV for the C-exact (R3) paths: cuBLAS heuristics select
// their algorithm by SM count / clocks / workspace, so two GPUs of the same
// supported set can produce different bits (numeric-contract §1.1.1). This
// kernel uses a fixed geometry (one warp per output row, lane-strided K,
// deterministic fp32 warp-shuffle reduction) that does not depend on any
// device property, and never allocates a workspace.

#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon {
namespace gemv {

enum class LxDtype { kF16, kBF16, kF32 };

// y[s][n] = alpha * sum_k W[n][k] * x[s][k] + beta * y[s][n]
//   W: [N, K] row-major, dtype wdt
//   x: [seq, K] row-major, dtype xdt
//   y: [seq, N] row-major, dtype ydt
// Stream-ordered; no workspace. K is walked with a fixed warp-strided order.
cudaError_t RunLxGemmVec(cudaStream_t stream, const void *w, const void *x,
                         int seq, int N, int K, LxDtype wdt, LxDtype xdt,
                         float alpha, void *y, LxDtype ydt, float beta);

// ── A weight kept in its high-precision storage encoding ──────────────────
//
// The bf16 codec profile stores weights as the BF16 planes the checkpoint held
// (design §2.6), so a loader must be able to hand the GEMV above a device buffer
// of `LxDtype` samples instead of NVFP4 nibbles + scales. This is that handle:
// shape and device pointer only — no dequantization, no scales, nothing to
// reconstruct, because the value is stored.
//
// POD on purpose: it lives in layer-weight structs that .cu TUs include.
struct DenseWeight {
    const void *d_data = nullptr;
    int rows = 0;   // N: output features
    int cols = 0;   // K: input features
    LxDtype dtype = LxDtype::kBF16;

    // A projection is present only when all three are meaningful. A default
    // DenseWeight is "absent", which is how a loader says "this tree has the
    // other encoding" without a separate flag.
    bool valid() const { return d_data != nullptr && rows > 0 && cols > 0; }
};

// Same result, but W is stored [K, N] (K-major): y[s][n] = sum_k x[s][k]*W[k][n].
cudaError_t RunLxGemmVecWT(cudaStream_t stream, const void *w, const void *x,
                           int seq, int N, int K, LxDtype wdt, LxDtype xdt,
                           float alpha, void *y, LxDtype ydt, float beta);

} // namespace gemv
} // namespace lsxcommon
