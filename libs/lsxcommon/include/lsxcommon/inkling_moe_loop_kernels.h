#pragma once

// Inkling-2-Mini numeric kernels.
//
// The inkling2 attention/MoE arithmetic (rel-bias bank, L2 clips, sink
// attention, short convs, AttnRes depth mix, softcapped SwiGLU, sigmoid router,
// MXFP8 expert GEMV) is the SAME contract as Inkling-2-Small's and is reached
// through the family's existing entry points where the geometry matches; this
// header owns the LOOP deltas plus the routed-slot type the pipeline uses:
//
//  1. MessageGate: `(1 - tanh(forget(down)))*h + tanh(gain(down))*msg`, where
//     down sees concat(unweighted-RMSnorm(h), RMSnorm(msg)). FP32 gates.
//  2. Message retrieval: single-head softmax attention of a learned query over
//     the previous pass's block stack + final hidden (unweighted RMSnorm on
//     the sources, scale hidden**-0.5) — the same arithmetic as the AttnRes
//     depth mix, exposed here for the loop's four messages per layer.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon::inkling {

// ── routed slot (router output) ────────────────────────────────────────────
struct InklingLoopRoutedSlot {
    int eid;
    float coeff;
};

// ── 1. MessageGate ─────────────────────────────────────────────────────────
// h, msg: [hidden] FP32. down: [rank, 2*hidden] FP16; forget/gain:
// [hidden, rank] FP32. out: [hidden] FP32.
//   g = down @ concat(rmsnorm(h), rmsnorm(msg))      (FP32 accumulate)
//   out = (1 - tanh(forget @ g)) * h + tanh(gain @ g) * msg
// The norms are unweighted (eps 1e-6) and computed in FP32; the down input is
// cast to FP16 (the reference casts to the down weight's dtype), the gate
// projections and the blend stay FP32.
cudaError_t RunInklingLoopMessageGate(cudaStream_t stream, const float *h,
                                      const float *msg, const __half *down,
                                      const float *forget, const float *gain,
                                      float *out, int hidden, int rank,
                                      float eps,
                                      // scratch: [2*hidden] F32 concat,
                                      // [rank] F32 gate vector
                                      float *concat_scratch, float *rank_scratch);

// ── 2. message retrieval (previous-pass depth mix) ─────────────────────────
// Identical contract to the AttnRes depth mix: sources [n_src, hidden] F32
// (blocks first, previous pass's final hidden last), queries [4, hidden] F32,
// out [4, hidden] F32; unweighted RMSNorm on every source row, softmax over
// sources with scale hidden**-0.5.
cudaError_t RunInklingLoopMessageMix(cudaStream_t stream, const float *sources,
                                     int n_src, const float *queries,
                                     float *out, int hidden, float eps);

}  // namespace lsxcommon::inkling