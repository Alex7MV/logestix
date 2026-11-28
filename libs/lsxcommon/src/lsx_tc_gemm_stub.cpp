// Stub for the `lsxcommon::lsx_tc::*` entry points in a build that targets no
// 12x arch (see cmake/LsxArch.cmake, lsx_tc_object_lib()).
//
// WHY A STUB, AND NOT #ifdef AT THE CALL SITES
//
// The public header is used from nine host and device files: the NVFP4
// dispatcher (lsx_inference_nvfp4.cu), the Kimi batched chain, the MoE pipeline,
// the attention path, the GLM flash inference, the perf knobs (gpu_perf.cpp) and
// ModelBase's weight epoch. Some of those references sit behind a perf toggle
// rather than a device probe — `lsx_inference_nvfp4.cu` calls RunLsxTcNvfp4Gemm
// when `g_lsx_tc_fast_math` is set, and the major==12 check lives INSIDE the
// callee. So dropping the object library without providing the symbols is a link
// error, and #ifdef-ing forty references would put an arch condition inside
// numerics-adjacent code and make the built code depend on the arch list.
//
// One translation unit that answers "not supported" is the SAME CONTRACT the
// real kernels already give on a non-12x device: every device entry point in
// lsx_tc_gemm.cu checks `prop.major != 12` first and returns cudaErrorNotSupported.
// Every caller in the tree treats that as "fall back to the scalar path".
//
// WHAT IS NOT STUBBED
//
// The host-side plumbing keeps real behaviour, because callers use it outside any
// device call: the split-K knob round-trips (gpu_perf.cpp applies it at policy
// resolution), and the weight epoch increments (ModelBase bumps it on every
// Load/Unload, and cached entries compare it). Returning a constant there would be
// a fake that changes host control flow, which is worse than a call that reports
// the device path missing.
//
// NOT a software emulation (invariant I1): nothing here computes a GEMM. The
// tolerance-class W4A4 path simply does not exist in this binary, and the model
// that needs it cannot be loaded on such a device anyway — the codec gate refuses
// an NVFP4 artifact on sm_86 before any kernel runs.

#include "lsxcommon/lsx_tc_gemm.h"

namespace lsxcommon {
namespace lsx_tc {

namespace {

// The two knobs that must round-trip. Process-lifetime, like the real ones
// (both are consulted while resolving the per-GPU perf policy).
int g_splitk_grid_min = 96;

// Weight epoch: bumped by ModelBase whenever weights become resident or are
// released. Monotonic, never reset.
unsigned long long g_weight_epoch = 0;

} // namespace

cudaError_t LsxTcQuantizeActsNvfp4(cudaStream_t, const __half *, int, int,
                                   uint8_t *, uint8_t *)
{
    return cudaErrorNotSupported;
}

cudaError_t LsxTcQuantizeActsNvfp4Bf16(cudaStream_t, const __nv_bfloat16 *, int,
                                       int, uint8_t *, uint8_t *)
{
    return cudaErrorNotSupported;
}

cudaError_t RunLsxTcNvfp4Gemm(cudaStream_t, const __half *, const uint8_t *,
                              const uint8_t *, float, __half *, int, int, int,
                              bool, const float *, const int *)
{
    return cudaErrorNotSupported;
}

cudaError_t RunLsxTcGemmPreQuant(cudaStream_t, const uint8_t *, const uint8_t *,
                                 const uint8_t *, const uint8_t *, float,
                                 __half *, int, int, int, bool, const float *,
                                 const int *)
{
    return cudaErrorNotSupported;
}

cudaError_t LsxTcBuildFusedGemmWeights(cudaStream_t, const uint8_t *,
                                       const uint8_t *, int, const uint8_t *,
                                       const uint8_t *, int, const uint8_t *,
                                       const uint8_t *, int, const uint8_t *,
                                       const uint8_t *, int, int, int, float,
                                       float, float, float, const uint8_t **,
                                       const uint8_t **, const float **,
                                       const int **, int *)
{
    return cudaErrorNotSupported;
}

cudaError_t LsxProbeTmaBulkCopy(cudaStream_t) { return cudaErrorNotSupported; }

int LsxTcBatchSplitK(int, int)
{
    // 1 = "no split". The value sizes the split-K partials buffer at the call
    // sites, and nothing ever writes it here: RunLsxTcGemmBatch13/BatchDown below
    // return NotSupported, so the caller falls back to the scalar chain.
    return 1;
}

void SetLsxSplitKGridMin(int v) { g_splitk_grid_min = v; }

int LsxSplitKGridMin() { return g_splitk_grid_min; }

bool LsxTcRuntimeSupported()
{
    // The honest answer: this binary contains no mxf4nvf4 code, so the path is
    // unavailable on EVERY device, including a 12x one. Callers use this to skip
    // the TC branch entirely, which is exactly what they should do here.
    return false;
}

unsigned long long LsxTcWeightEpoch() { return g_weight_epoch; }

void BumpLsxTcWeightEpoch() { ++g_weight_epoch; }

__half *LsxTcFusedOutputScratch(const void *, int)
{
    // nullptr = "no scratch", which callers already handle as a failed
    // allocation and fall back from.
    return nullptr;
}

cudaError_t RunLsxTcGemmBatch13(cudaStream_t, const uint8_t *, const uint8_t *,
                                const uint8_t *, const uint8_t *, int64_t,
                                int64_t, const LsxTcBatchSlot *, int, int, int,
                                __half *, float *)
{
    return cudaErrorNotSupported;
}

cudaError_t RunLsxTcGemmBatchDown(cudaStream_t, const uint8_t *, const uint8_t *,
                                  const uint8_t *, const uint8_t *, int64_t,
                                  int64_t, const LsxTcBatchSlot *, int, int64_t,
                                  int64_t, int, int, __half *, float *)
{
    return cudaErrorNotSupported;
}

} // namespace lsx_tc
} // namespace lsxcommon
