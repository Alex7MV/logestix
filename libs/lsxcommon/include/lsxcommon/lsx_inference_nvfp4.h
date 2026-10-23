#pragma once

#include <atomic>
#include <cstdio>

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

#include "lsxcommon/fast_math_mode.h"

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

// Large-N warp-per-row streaming GEMV (seq==1 decode): routes big dense
// shapes (K >= 512, N >= 4096) to the warp-per-row kernel, which sustains
// ~2x the TC-tile bandwidth at M == 1 (27B decode profile, 2026-09).
// Applied from the resolved perf policy (gemv_large_default).
void SetLsxGemmLarge(bool on);
bool LsxGemmLarge();

// Effective large-N GEMV state with the LSX_GEMV_LARGE tri-state env applied
// (1=forced on, 0=forced off, unset=policy knob). Side gates that must stay
// consistent with the dispatcher's routing read this, not LsxGemmLarge().
bool LsxGemmLargeActive();

// True while an explicit FastMathMode::kOff scope is active. This is the
// canonical strict mode: it forces the large-N streaming GEMV route ON while
// keeping TC/seq1-gemv OFF, so the strict path is fast AND independent of the
// LSX_GEMV_LARGE environment override (audit X4).
void SetLsxBitExactMode(bool on);
bool LsxBitExactMode();

// Dual-stream expert-staging toggle (H2D/compute overlap). Default true;
// narrow-bus GPUs resolve this to false and keep single-stream staging.
void SetLsxDualStreamStaging(bool on);
bool LsxDualStreamStaging();

// RAII: apply one model's fast-math policy for the scope of an Infer/SelfCheck
// call and restore the previous process state afterwards. The orchestrator
// dispatches sequentially (a single Infer at a time), so the scope cannot
// straddle two models' inferences.
// Sequential-dispatch invariant guard (OPEN_TASKS E1).
//
// The two kernel-routing toggles are process-global because CUDA selects kernels
// per launch and there is no per-model device context. That is safe ONLY while
// dispatch is sequential: one Infer at a time, so each model asserts its policy
// via LsxFastMathScope for a window no other model is inside. Concurrent
// multi-model dispatch would have two scopes overlapping, and the inner one's
// save/restore would silently repoint the other's kernel routing mid-decode —
// wrong numerics, no crash. There is no parallel dispatch planned, so instead of
// moving the state per-stream we make the assumption MACHINE-CHECKED: a
// process-global depth counter tracks live scopes and trips an invariant (logged
// loudly, and pinned by a guard test) the moment two are ever open at once.
//
// EnterFastMathScope/LeaveFastMathScope and the depth accessor are defined in the
// header so they are visible to the guard test without linking lsxcommon.
namespace Detail {
// Inline in this header (definitions below) so the guard stays host-only and
// needs no lsxcommon link.
inline void EnterFastMathScope();
inline void LeaveFastMathScope();
inline int ActiveFastMathScopeDepth();
} // namespace Detail

class LsxFastMathScope {
public:
    // Legacy bool pair (unchanged semantics).
    LsxFastMathScope(bool tc_fast_math, bool seq1_gemv)
        : prev_tc_(LsxTcFastMath()), prev_gemv_(LsxSeq1Gemv()),
          prev_large_(LsxGemmLarge()), prev_bit_exact_(LsxBitExactMode()),
          active_(true)
    {
        SetLsxTcFastMath(tc_fast_math);
        SetLsxSeq1Gemv(seq1_gemv);
        Detail::EnterFastMathScope();
    }
    // Tri-state: kAuto leaves the process globals untouched (the engine-init
    // perf policy stands); kOn/kOff force the pair for the scope and restore
    // afterwards. Numeric-class tests pin kOff; production uses kAuto.
    explicit LsxFastMathScope(FastMathMode mode)
        : prev_tc_(LsxTcFastMath()), prev_gemv_(LsxSeq1Gemv()),
          prev_large_(LsxGemmLarge()), prev_bit_exact_(LsxBitExactMode()),
          active_(mode != FastMathMode::kAuto)
    {
        if (mode == FastMathMode::kOn) {
            SetLsxTcFastMath(true);
            SetLsxSeq1Gemv(true);
            SetLsxBitExactMode(false);
        } else if (mode == FastMathMode::kOff) {
            SetLsxTcFastMath(false);
            SetLsxSeq1Gemv(false);
            // kOff keeps the sliced GEMV as its canonical route (fast and
            // deterministic) but the dispatcher ignores LSX_GEMV_LARGE while
            // this flag is set, so the strict hash cannot depend on the env.
            SetLsxGemmLarge(false);
            SetLsxBitExactMode(true);
        }
        if (active_) Detail::EnterFastMathScope();
    }
    ~LsxFastMathScope()
    {
        if (!active_) return;
        SetLsxTcFastMath(prev_tc_);
        SetLsxSeq1Gemv(prev_gemv_);
        SetLsxGemmLarge(prev_large_);
        SetLsxBitExactMode(prev_bit_exact_);
        Detail::LeaveFastMathScope();
    }
    LsxFastMathScope(const LsxFastMathScope &) = delete;
    LsxFastMathScope &operator=(const LsxFastMathScope &) = delete;

private:
    bool prev_tc_;
    bool prev_gemv_;
    bool prev_large_;
    bool prev_bit_exact_;
    bool active_;
};

// ── Sequential-dispatch guard implementation ───────────────────────────────
namespace Detail {
// One PROCESS-GLOBAL depth (an inline variable, external linkage) so every
// translation unit shares it. It used to be an anonymous-namespace static in
// this header: each TU got a private counter, Enter and Leave could land on
// different copies, and real FSI/DET runs printed false VIOLATION lines while
// the depth leaked. Pinned by test_fast_math_guard_cross_tu (two TUs).
inline std::atomic<int> g_fast_math_depth{0};
// Nesting on one thread is legitimate sequential dispatch. Track it per
// thread: a violation is only a SECOND THREAD entering while one is already
// inside, not a nested scope on the same thread.
inline thread_local int g_fast_math_tls_depth = 0;

inline void EnterFastMathScope()
{
    const int d = g_fast_math_depth.fetch_add(1, std::memory_order_relaxed) + 1;
    const int tls = ++g_fast_math_tls_depth;
    if (tls == 1 && d > 1)
    {
        // E1: this is the first moment the sequential-dispatch assumption is
        // observably false. Two THREADS are inside the kernel-routing scope at
        // once, so the inner save/restore will repoint the outer's routing
        // mid-decode. Loud by design: it must not be a silent wrong-numerics bug.
        std::fprintf(stderr,
                     "[lsx_inference_nvfp4] VIOLATION: %d concurrent "
                     "LsxFastMathScope instances (E1 invariant: dispatch is "
                     "sequential). Kernel routing is process-global; concurrent "
                     "multi-model Infer produces wrong numerics.\n",
                     d);
    }
}

inline void LeaveFastMathScope()
{
    g_fast_math_depth.fetch_sub(1, std::memory_order_relaxed);
    --g_fast_math_tls_depth;
}

inline int ActiveFastMathScopeDepth() { return g_fast_math_depth.load(std::memory_order_relaxed); }
} // namespace Detail

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
// Device-table expert batch: `d_ids` (top-k routed ids, already on the device)
// select experts out of `table_rows`, a flat [layer * num_experts] device upload
// of LsxExpertPtrRow. No D2H and no stream synchronize on the streaming path, so
// a caller can capture a decode step into a CUDA graph. `table_offset` is the
// layer's slice; the rows must be BIT-IDENTICAL to the by-value path.
struct LsxExpertPtrRow {
    const uint8_t *packed;
    const uint8_t *scales;
    float gs;
};

// True when RunLsxNvfp4GemmSeq1GemmVExpertBatchIds will dispatch the streaming
// ids kernel for this K (i.e. need no host round trip). A caller that has a
// cheaper host-resolved path for the other case asks this instead of repeating
// the dispatch condition.
bool LsxExpertBatchIdsStreaming(int K);

cudaError_t RunLsxNvfp4GemmSeq1GemmVExpertBatchIds(
    cudaStream_t stream, const __half *x, const int32_t *d_ids,
    const LsxExpertPtrRow *table_rows, int table_offset, int num_experts,
    int count, int x_stride, __half *y, int K, int N, bool acc = false);

cudaError_t RunLsxNvfp4GemmSeq1GemmV(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int K, int N, bool acc = false);

// Same streaming GEMV writing F32 instead of __half, so a caller whose next
// kernel consumes F32 does not pay an elementwise conversion launch per GEMV
// (1.84 us of launch floor per call on an RTX 5090; 244 such conversions per
// Xing decode token). The per-row math is identical; the result skips the F16
// rounding of the store, so it is a TOLERANCE-CLASS variant - same relationship
// to the F16 form as the TC fast-math routes have.
// Two projections sharing ONE input row but living in separate weight tensors
// (an MLA q_a + kv_a pair), writing F32. One launch instead of two; every output
// row is the same dot product in the same k-order as the separate calls, so this
// is bit-identical rather than tolerance-class. Shapes that would route to
// warp-per-row fall back to two ordinary launches.
cudaError_t RunLsxNvfp4GemmSeq1GemmVTwoViewsF32(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    float *y, int K, int N0,
    const uint8_t *packed2, const uint8_t *scales2, float global_scale2,
    float *y2, int N1);

cudaError_t RunLsxNvfp4GemmSeq1GemmVF32(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    float *y, int K, int N, bool acc = false);

// Batched seq==1 streaming NVFP4 GEMV: `count` independent weights (routed
// experts), one block-column per expert on gridDim.y. `x_stride == 0` applies
// the same input row to every expert (up projection); `x_stride > 0` reads
// expert e's own input slice x + e*x_stride (down projection over the
// contiguous relu²(up) outputs). Same per-row math and k-order as
// RunLsxNvfp4GemmSeq1GemmV, so each expert's row is bit-identical to the
// single call. count <= 8.
cudaError_t RunLsxNvfp4GemmSeq1GemmVExpertBatch(
    cudaStream_t stream, const __half *x, const uint8_t *const *packed,
    const uint8_t *const *scales, const float *gs, int count, int x_stride,
    __half *y, int K, int N, bool acc = false);

// Fused gate/up streaming GEMV for seq==1: W is the contiguous
// [2*n_half, K] NVFP4 pack ([n_half, K] gate rows followed by [n_half, K]
// up rows; E4M3 scale rows follow the same split). Rows < n_half scale by
// gs_gate, rows >= n_half by gs_up — bit-identical to two independent
// RunLsxNvfp4GemmSeq1GemmV calls over the sub-ranges. Non-eligible shapes
// fall back to exactly those two calls (bit-exact tail).
cudaError_t RunLsxNvfp4GemmGateUpPair(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float gs_gate,
    float gs_up, __half *y, int K, int n_half, bool acc = false);

// Convenience launcher over an LsxNvfp4Weight container: y[t][n] =
// sum_k x[t*K+k] * dequant(w). Same dispatch and numeric contract as
// RunLsxNvfp4Gemm (acc=false).
cudaError_t LaunchNvfp4Gemm(const LsxNvfp4Weight &w, const __half *x, int seq,
                            __half *y, cudaStream_t stream);

} // namespace lsx_nvfp4
} // namespace lsxcommon
