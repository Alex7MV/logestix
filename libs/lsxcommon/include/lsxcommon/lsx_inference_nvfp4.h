#pragma once

// ── Model-neutral NVFP4 inference layer ─────────────────────────────────────
// Shared GEMM/codec primitives of the NVFP4 quantization stack. Multiple
// model families (Qwen dense/MoE, GLM-5.3-Flash, DeepSeek TC epilogue) build
// on this layer; the names carry no model family on purpose (AGENTS.md
// model-independence rule: shared infrastructure must not be owned by one
// family's namespace/files).
//
// Contents:
//   LsxNvfp4Weight        — device NVFP4 weight container ([N, K] row-major
//                           E2M1 nibbles + per-16 E4M3 scales + global scale)
//   RunLsxNvfp4Gemm       — NVFP4 dequant GEMM dispatcher: seq==1 weight-
//                           streaming GEMV (decode), TC mxf4nvf4 W4A4 path
//                           (fast-math), scalar tile kernel (bit-exact)
//   RunLsxNvfp4GemmSeq1GemmV — the streaming GEMV kernel entry (extern-linkable
//                           for tests)
//   kernel-routing toggles + LsxFastMathScope — per-model fast-math policy
//                           RAII (see the model-independence note below)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon {
namespace lsx_nvfp4 {

// Device NVFP4 weight ([N, K] row-major; N = output dim, K = input dim).
struct LsxNvfp4Weight {
    const uint8_t *d_packed = nullptr;
    const uint8_t *d_scales = nullptr;
    float global_scale = 1.0f;
    int rows = 0;   // output dim (N for GEMM: W is [N, K] row-major)
    int cols = 0;   // input dim (K)
    int64_t packed_bytes = 0;
    int64_t scale_bytes = 0;
    // True when d_packed/d_scales alias another weight's allocation
    // (gate/up views into a fused gate_up buffer) and Free() must not
    // release them.
    bool view_only = false;

    void Free() {
        if (view_only) {
            d_packed = nullptr; d_scales = nullptr;
            view_only = false;
            return;
        }
        cudaFree(const_cast<uint8_t*>(d_packed));
        cudaFree(const_cast<uint8_t*>(d_scales));
        d_packed = nullptr; d_scales = nullptr;
    }
};

// ── Kernel-routing toggles ──────────────────────────────────────────────────
// Both toggles are process-global state (CUDA kernels are selected per
// launch, there is no per-model device context). The model-independence
// contract: every model asserts its OWN policy via LsxFastMathScope for the
// duration of Infer/SelfCheck — the last-constructed model's config must
// never leak into another model's kernel selection.

// Blackwell tensor-core fast path: when enabled, RunLsxNvfp4Gemm dispatches
// to the sm_120a mxf4nvf4 W4A4 kernel for supported shapes and falls back to
// the scalar path otherwise. Default off.
void SetLsxTcFastMath(bool on);
bool LsxTcFastMath();

// seq==1 decode-path weight-streaming GEMV toggle: when enabled together
// with fast-math, RunLsxNvfp4Gemm routes eligible seq==1 shapes (K % 32 == 0,
// K >= 256, <= 32 MiB packed+scales, no fused segments) to the streaming
// GEMV kernel. Bound env-tunable via LSX_GEMV_MAX_MIB.
void SetLsxSeq1Gemv(bool on);
bool LsxSeq1Gemv();

// Weight-size bound (MiB) for the seq==1 streaming GEMV eligibility.
// Default 32; applied from the resolved perf policy (gpu_perf.h) at engine
// init. The LSX_GEMV_MAX_MIB env read lives in perf::ResolvePerfKnobs.
void SetLsxGemmMaxMib(int mib);
int LsxGemmMaxMib();

// Dual-stream expert-staging toggle (H2D/compute overlap). Default true;
// narrow-bus GPUs resolve this to false and keep single-stream staging.
void SetLsxDualStreamStaging(bool on);
bool LsxDualStreamStaging();

// RAII: apply one model's fast-math policy for the scope of an Infer/SelfCheck
// call and restore the previous process state afterwards. The orchestrator
// dispatches sequentially (a single Infer at a time), so the scope cannot
// straddle two models' inferences.
class LsxFastMathScope {
public:
    LsxFastMathScope(bool tc_fast_math, bool seq1_gemv)
        : prev_tc_(LsxTcFastMath()), prev_gemv_(LsxSeq1Gemv())
    {
        SetLsxTcFastMath(tc_fast_math);
        SetLsxSeq1Gemv(seq1_gemv);
    }
    ~LsxFastMathScope()
    {
        SetLsxTcFastMath(prev_tc_);
        SetLsxSeq1Gemv(prev_gemv_);
    }
    LsxFastMathScope(const LsxFastMathScope &) = delete;
    LsxFastMathScope &operator=(const LsxFastMathScope &) = delete;

private:
    bool prev_tc_;
    bool prev_gemv_;
};

// ── NVFP4 dense GEMM ────────────────────────────────────────────────────────
// y[t][n] = sum_k x[t*K+k] * dequant(packed[n*K+k], scales, gs).
// x: [seq, K] FP16; packed: [N*K/2] nibble-packed E2M1 (row-major [N, K]);
// scales: [N*K/16] E4M3 per-16 block; gs: global scale. y: [seq, N] FP16
// (FP32 accumulate). Dispatch (seq==1, fast-math opt-in): weight-streaming
// GEMV for eligible shapes, TC W4A4 mxf4nvf4 otherwise (tolerance class);
// bit-exact scalar tile kernel as the fallback path. k-order free.
cudaError_t RunLsxNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N, bool acc = false,
    // Optional per-segment global scales for the TC epilogue: two ranges
    // split at N/2 when bounds is null, else four bounds-driven ranges.
    // Ignored by the scalar fallback path.
    const float *seg_wgs_dev = nullptr, const int *seg_bounds_dev = nullptr);

// seq==1 decode-path weight-streaming NVFP4 GEMV: y[n] =
// sum_k x[k] * dequant(packed[n*K+k]) with FP32 accumulate. Streams packed
// E2M1 weights straight from global memory (coalesced uint4 loads) — no
// shared-memory slab staging. Requires K % 32 == 0; k-order free (tolerance
// class), run-to-run deterministic. External linkage for direct testing.
cudaError_t RunLsxNvfp4GemmSeq1GemmV(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int K, int N, bool acc = false);

// Convenience launcher over an LsxNvfp4Weight container: y[t][n] =
// sum_k x[t*K+k] * dequant(w). Same dispatch and numeric contract as
// RunLsxNvfp4Gemm (acc=false).
cudaError_t LaunchNvfp4Gemm(const LsxNvfp4Weight &w, const __half *x, int seq,
                            __half *y, cudaStream_t stream);

} // namespace lsx_nvfp4
} // namespace lsxcommon
