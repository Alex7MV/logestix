#pragma once

// ── One GEMM entry point for either storage encoding ──────────────────────
//
// A family's inference pipeline is written against one weight representation.
// That was NVFP4, unconditionally, which is why a `bf16` artifact gated and then
// loaded empty (defect A3.13): there was no call a bf16-resident projection could
// be fed to.
//
// Rather than put an `if (dense) … else …` at forty call sites, the dispatch
// lives here once. Each site passes the pair it already has — the NVFP4 handle
// and the dense handle, one of which is empty — and gets the same
// `y = acc ? y + x·Wᵀ : x·Wᵀ` contract either way:
//
//   * NVFP4  → RunLsxNvfp4Gemm (unchanged: nibbles + per-16 E4M3 scales + a
//              per-tensor global scale; the path every existing artifact uses)
//   * dense  → gemv::RunLxGemmVec (fixed-geometry GEMV: one warp per row,
//              lane-strided K, deterministic fp32 shuffle reduction)
//
// Both are device-property-independent by construction (numeric-contract §1.1),
// so switching encoding does not switch numerics CLASS: within a target group the
// result is still bit-reproducible across runs and across models. Across target
// groups it is not — and must not be, because the storage encoding differs
// (numeric-contract §1.1 item 4 as amended).
//
// This header is included from `.cu` TUs, so it pulls in nothing but the two
// weight types and `<cuda_runtime.h>`.

#include <cuda_runtime.h>

#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {

// The axes of a linear projection, in the order the encodings need them.
//
// `N` rows of K weights: y[seq, N] = x[seq, K] · W[N, K]ᵀ. Named rather than
// positional because the two backends take them in different orders
// (RunLsxNvfp4Gemm is (K, N); RunLxGemmVec is (N, K)) and a swap there is a
// silent transposed read, not a compile error.
struct WeightGemmShape {
    int seq = 0;
    int k = 0;   // input features
    int n = 0;   // output features
};

// y = (acc ? 1 : 0)·y + x·Wᵀ.
//
// Returns cudaErrorNotSupported when NEITHER encoding is present, so a caller
// that forgot to load something fails loudly instead of reading a null plane.
cudaError_t RunWeightGemm(cudaStream_t stream, const __half *x,
                          const lsx_nvfp4::LsxNvfp4Weight &nv,
                          const gemv::DenseWeight &dense, __half *y,
                          const WeightGemmShape &shape, bool acc = false);

} // namespace lsxcommon
