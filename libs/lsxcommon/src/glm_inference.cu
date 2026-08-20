#include "lsxcommon/glm_inference.h"
#include "lsxcommon/log.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glm_device_math.h"
#include "glm_expert_kernels.cuh"

namespace lsxcommon {
namespace glm {

namespace {

using device::GlmExpfGlibc;
using device::GlmRsqrtfGlibc;

constexpr float kGlmSwigluLimit = 10.0f;
constexpr int kGlmRmsBlock = 256;

// GlmRmsNormKernel / GlmGemmFp64Kernel / GlmRealSwigluKernel: single textual
// source in glm_expert_kernels.cuh (bit-exact, -fmad=false), included above.

__global__ void GlmSwigluClampKernel(__half *__restrict__ gate,
                                     const __half *__restrict__ up,
                                     int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    if (g > limit) g = limit;
    if (u > limit) u = limit;
    if (u < -limit) u = -limit;
    float s = g / (1.0f + GlmExpfGlibc(-g));
    gate[i] = __float2half(s * u);
}

__global__ void GlmAddResidualKernel(__half *__restrict__ dst,
                                     const __half *__restrict__ base,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(base[i]));
}

// FP64-accumulate FP16 GEMM: out[t][i] = H((float)Σ_q F(w[i][q])·F(x[t][q]))
// over ascending q — a bit-exact mirror of the host MatMulFp16 reference
// (which accumulates in FP64 and rounds once to FP16 via (float) then
// __float2half_rn). w is row-major [m, k]; x row-major [k, n]; the FP16 out
// is column-major [m, n] (out[i + t*m]), identical to the cuBLAS output
// GlmGemmFp64Kernel: single textual source in glm_expert_kernels.cuh.

cudaError_t RunGemmFp64(cudaStream_t stream, const __half *w, const __half *x,
                        __half *out, int m, int k, int n)
{
    const int total = static_cast<int>(static_cast<int64_t>(m) * n);
    const int blocks = (total + kGlmRmsBlock - 1) / kGlmRmsBlock;
    GlmGemmFp64Kernel<<<blocks, kGlmRmsBlock, 0, stream>>>(w, x, out, m, k, n);
    return cudaGetLastError();
}

cudaError_t RunGlmRmsNorm(cudaStream_t stream, const __half *x,
                          const __half *w, __half *y, int rows, int dim,
                          float eps)
{
    GlmRmsNormKernel<<<rows, kGlmRmsBlock, 0, stream>>>(x, w, y, dim, eps);
    return cudaGetLastError();
}

cudaError_t RunDenseMLPLayer(cublasHandle_t cublas_h, cudaStream_t stream,
                             const GlmLayerWeights &lw,
                             int seq, int hidden, int inter,
                             const __half *d_x, __half *d_y,
                             __half *d_normed, __half *d_gate, __half *d_up)
{
    (void)cublas_h;
    if (!lw.d_post_attn_norm ||
        !lw.d_mlp_gate || !lw.d_mlp_up || !lw.d_mlp_down)
        return cudaErrorInvalidValue;

    cudaError_t e = RunGlmRmsNorm(stream, d_x, lw.d_post_attn_norm, d_normed,
                                  seq, hidden, kGlmRmsEps);
    if (e != cudaSuccess) return e;
    e = RunGemmFp64(stream, lw.d_mlp_gate, d_normed, d_gate,
                    inter, hidden, seq);
    if (e != cudaSuccess) return e;
    e = RunGemmFp64(stream, lw.d_mlp_up, d_normed, d_up,
                    inter, hidden, seq);
    if (e != cudaSuccess) return e;
    const int n_act = seq * inter;
    GlmSwigluClampKernel<<<(n_act + 255) / 256, 256, 0, stream>>>(
        d_gate, d_up, n_act, kGlmSwigluLimit);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    e = RunGemmFp64(stream, lw.d_mlp_down, d_gate, d_y,
                    hidden, inter, seq);
    if (e != cudaSuccess) return e;
    const int n_out = seq * hidden;
    GlmAddResidualKernel<<<(n_out + 255) / 256, 256, 0, stream>>>(d_y, d_x, n_out);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunGlmDenseMLPStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, int seq,
    __half *d_act, __half *d_alt, __half *d_result, __half *d_scratch)
{
    const int hidden = kGlmHidden;
    const int inter  = kGlmMoeIntermediate;
    if (pw.layers.size() < 3 || seq <= 0) return cudaErrorInvalidValue;
    __half *d_normed = d_scratch;
    __half *d_gate   = d_scratch + static_cast<size_t>(seq) * hidden;
    __half *d_up     = d_gate  + static_cast<size_t>(seq) * inter;
    __half *cur = d_act;
    for (int l = 0; l < 3; ++l) {
        __half *nxt = (l == 2) ? d_result
                     : (cur == d_act ? d_alt : d_act);
        cudaError_t e = RunDenseMLPLayer(cublas_h, stream, pw.layers[l],
                                         seq, hidden, inter,
                                         cur, nxt, d_normed, d_gate, d_up);
        if (e != cudaSuccess) return e;
        cur = nxt;
    }
    return cudaSuccess;
}

// ── TDD Cycle 6: full 79-layer generation pass ──────────────────────────────
namespace {

constexpr int kGlmLoopBlock = 256;
// Real-model EOS ids (config.json eos_token_id), mirrored by the test.
constexpr int kGlmEosIds[3] = {154820, 154827, 154829};


__global__ void GlmEmbedKernel(const __half *__restrict__ embed,
                               const int32_t *__restrict__ ids,
                               __half *__restrict__ out, int seq, int hidden)
{
    const int64_t total = static_cast<int64_t>(seq) * hidden;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= total) return;
    const int t = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<int64_t>(t) * hidden);
    out[i] = embed[static_cast<int64_t>(ids[t]) * hidden + h];
}

__global__ void GlmCopyKernel(const __half *__restrict__ src,
                              __half *__restrict__ dst, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Copies row 0 of a [seq, hidden] attention output into a single [hidden]
// buffer — the frozen row-0 collapse the host reference performs on the
// prefill block at the first loaded branch.
__global__ void GlmCopyRow0Kernel(const __half *__restrict__ src,
                                  __half *__restrict__ dst, int hidden)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < hidden) dst[h] = src[h];
}

// out[i] = fp16(a[i] + b[i]): elementwise residual blend (MoE / MTP).
__global__ void GlmResidualKernel(__half *__restrict__ out,
                                  const __half *__restrict__ a,
                                  const __half *__restrict__ b, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i < n)
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// Greedy argmax over a [vocab] FP32 logits row, matching the host baseline,
// which stores the lm_head row as FP16 and argmaxes over the de-quantized
// values. We therefore round each logit to FP16 (RNE) before comparing —
// otherwise the FP32 margin can flip vs the reference's rounded logits.
// Ascending scan, strict >, so ties resolve to the smallest index. Single
// block, warp-shuffle + shared-memory reduction.
__global__ void GlmArgmaxKernel(const float *__restrict__ logits, int vocab,
                                GlmArgmaxSlot *__restrict__ out)
{
    constexpr int kBlock = kGlmLoopBlock;
    __shared__ int   s_best[kBlock / 32];
    __shared__ float s_maxv[kBlock / 32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int   best = -1;
    float bv   = -FLT_MAX;
    for (int v = tid; v < vocab; v += kBlock) {
        const float lv = __half2float(__float2half(logits[v]));
        if (lv > bv) { bv = lv; best = v; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int   oi = __shfl_down_sync(0xffffffffu, best, off);
        if (ov > bv || (ov == bv && oi >= 0 && oi < best)) { bv = ov; best = oi; }
    }
    if (lane == 0) { s_maxv[warp] = bv; s_best[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        int   b = (tid < (kBlock / 32)) ? s_best[tid] : -1;
        float v = (tid < (kBlock / 32)) ? s_maxv[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, off);
            const int   oi = __shfl_down_sync(0xffffffffu, b, off);
            if (ov > v || (ov == v && oi >= 0 && oi < b)) { v = ov; b = oi; }
        }
        if (lane == 0) { out->best = b; out->maxv = v; }
    }
}

// lm_head projection + the baseline's double-rounded FP16 logit store.
// logits[v] = F(H((float)Σ_h F(head[v][h])·F(hn[h]))), accumulated in FP64 in
// ascending h order — a bit-exact mirror of the FP64 host reference. Replaces
// the cuBLAS fp32-accumulate path, whose ~1e-3 error crosses FP16 rounding
// boundaries and flips the argmax on near-ties.
__global__ void GlmHeadLogitsKernel(const __half *__restrict__ head,
                                    const __half *__restrict__ hn,
                                    float *__restrict__ out,
                                    int vocab, int hidden)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vocab) return;
    const __half *hr = head + static_cast<size_t>(v) * hidden;
    double acc = 0.0;
    for (int h = 0; h < hidden; ++h)
        acc += static_cast<double>(__half2float(hr[h])) *
               static_cast<double>(__half2float(hn[h]));
    out[v] = __half2float(__float2half(static_cast<float>(acc)));
}

// Per-pass scratch + persistent state of the 79-layer generation loop.
struct GlmLoopCtx {
    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const GlmPinnedWeights *pw = nullptr;
    const GlmExpertTables *experts = nullptr;   // device-table path
    GlmExpertDb *db = nullptr;                  // on-demand mmap path
    uint8_t *d_pack_stage = nullptr;            // on-demand staging buffers
    uint8_t *d_scale_stage = nullptr;
    int hidden = 0;
    int inter  = 0;
    int vocab  = 0;
    GlmIndexShareState *d_state = nullptr;
    __half *d_attn = nullptr;        // [seq_max, hidden]   attention output
    __half *d_normed = nullptr;      // [seq_max, hidden]
    __half *d_moe = nullptr;         // [seq_max, hidden]   experts output
    __half *d_gate = nullptr;        // [seq_max, inter]
    __half *d_up = nullptr;          // [seq_max, inter]
    float  *d_logits = nullptr;      // [seq_max, kGlmNumRouted]
    GlmRoutedSlot *d_routing = nullptr;  // [seq_max, kGlmMoETopK]
    __half *d_scratch_e = nullptr;   // experts stack scratch
    __half *d_hidrow = nullptr;      // [hidden]  row-0 collapse buffer
    __half *d_mtp = nullptr;         // [hidden]  eh_proj output
    __half *d_hmtp = nullptr;        // [hidden]  enorm-normed
    __half *d_h78 = nullptr;         // [hidden]  hmtp + moe
    __half *d_hn = nullptr;          // [hidden]  shared_head.norm output
    float  *d_logits_head = nullptr; // [vocab]   raw logits row
    float  *d_gate_bias = nullptr;   // [kGlmNumRouted] staged router bias
    GlmArgmaxSlot *d_argmax = nullptr;
    GlmRoutedSlot *h_routing = nullptr;  // host staging for the routing grid
};

cudaError_t GlmCopyAsync(cudaStream_t st, const __half *src, __half *dst,
                         int64_t n)
{
    const int blocks = static_cast<int>((n + kGlmLoopBlock - 1) / kGlmLoopBlock);
    GlmCopyKernel<<<blocks, kGlmLoopBlock, 0, st>>>(src, dst, n);
    return cudaGetLastError();
}

// Runs the routed-expert stack of one layer through either the resident device
// descriptor table (c.experts, layer-3..78 tables) or the on-demand mmap index
// (c.db) with the caller-provided staging buffers. The layer's routing grid
// (c.d_routing) holds 8 slots for the single decode row.
cudaError_t RunLoopExpertStack(GlmLoopCtx &c, int l)
{
    if (c.db) {
        return RunGlmExpertsStackOnDemand(
            c.ch, c.st, c.d_routing, 1, c.d_normed,
            *c.db, l, kGlmNumRouted, c.d_moe, c.d_scratch_e,
            c.d_pack_stage, c.d_scale_stage, c.h_routing);
    }
    return RunGlmExpertsStack(
        c.ch, c.st, c.d_routing, 1, c.d_normed,
        c.experts->tables[static_cast<size_t>(l)].data(),
        kGlmNumRouted, c.d_moe, c.d_scratch_e, c.h_routing);
}

// One layer of the frozen 79-layer loop. The attention pass always consumes
// the full [seq, hidden] block (turn-ordered KV append + mod-4 refresh inside
// RunGlmIndexShareAttention); the dense/MoE branches then operate on ROW 0
// only and emit a single [hidden] token into d_out — mirroring the host
// reference, which collapses the prefill block at the first loaded branch.
// Layer 78 writes its greedy argmax into ctx->d_argmax instead of d_out.
cudaError_t RunGlmLoopLayer(GlmLoopCtx &c, int l, int turn, int seq,
                            const __half *d_cur, __half *d_out)
{
    const GlmLayerWeights &lw = c.pw->layers[static_cast<size_t>(l)];
    cudaError_t e = RunGlmIndexShareAttention(c.st, l, turn, lw, seq,
                                              d_cur, c.d_attn, c.d_state);
    if (e != cudaSuccess) return e;
    const int hidden = c.hidden;
    const int inter  = c.inter;
    const int row0_blocks = (hidden + kGlmLoopBlock - 1) / kGlmLoopBlock;

    // Dense layers 0-2: post-attn RMSNorm -> gate/up -> swiglu(clamp) -> down.
    if (l < 3) {
        if (!lw.d_post_attn_norm || !lw.d_mlp_gate ||
            !lw.d_mlp_up || !lw.d_mlp_down)
            return GlmCopyAsync(c.st, c.d_attn, d_out,
                                static_cast<int64_t>(seq) * hidden);
        GlmCopyRow0Kernel<<<row0_blocks, kGlmLoopBlock, 0, c.st>>>(
            c.d_attn, c.d_hidrow, hidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        return RunDenseMLPLayer(c.ch, c.st, lw, 1, hidden, inter,
                                c.d_hidrow, d_out, c.d_normed,
                                c.d_gate, c.d_up);
    }

    // Layer-78 MTP epilogue: eh_proj -> enorm -> router -> experts -> residual
    // -> shared_head.norm -> lm_head^T raw logits row, then greedy argmax.
    if (l == 78) {
        if (!lw.d_eh_proj || !lw.d_enorm || !lw.d_shared_head_norm ||
            !lw.d_post_attn_norm || !lw.d_router_gate || !c.pw->d_head)
            return cudaErrorInvalidValue;
        GlmCopyRow0Kernel<<<row0_blocks, kGlmLoopBlock, 0, c.st>>>(
            c.d_attn, c.d_hidrow, hidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = RunGemmFp64(c.st, lw.d_eh_proj, c.d_hidrow, c.d_mtp,
                        hidden, hidden, 1);
        if (e != cudaSuccess) return e;
        e = RunGlmRmsNorm(c.st, c.d_mtp, lw.d_enorm, c.d_hmtp,
                          1, hidden, kGlmRmsEps);
        if (e != cudaSuccess) return e;
        const float *d_bias = nullptr;
        if (lw.h_gate_bias && lw.gate_bias_n == kGlmNumRouted) {
            e = cudaMemcpyAsync(c.d_gate_bias, lw.h_gate_bias,
                                static_cast<size_t>(kGlmNumRouted) *
                                    sizeof(float),
                                cudaMemcpyHostToDevice, c.st);
            if (e != cudaSuccess) return e;
            d_bias = c.d_gate_bias;
        }
        e = RunGlmMoERouter(c.ch, c.st, lw, 1, c.d_hmtp, d_bias,
                            c.d_normed, c.d_logits, c.d_routing);
        if (e != cudaSuccess) return e;
        e = RunLoopExpertStack(c, 78);
        if (e != cudaSuccess) return e;
        GlmResidualKernel<<<row0_blocks, kGlmLoopBlock, 0, c.st>>>(
            c.d_h78, c.d_hmtp, c.d_moe, hidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = RunGlmRmsNorm(c.st, c.d_h78, lw.d_shared_head_norm, c.d_hn,
                          1, hidden, kGlmRmsEps);
        if (e != cudaSuccess) return e;
        const int head_blocks =
            (c.vocab + kGlmLoopBlock - 1) / kGlmLoopBlock;
        GlmHeadLogitsKernel<<<head_blocks, kGlmLoopBlock, 0, c.st>>>(
            c.pw->d_head, c.d_hn, c.d_logits_head, c.vocab, hidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        GlmArgmaxKernel<<<1, kGlmLoopBlock, 0, c.st>>>(
            c.d_logits_head, c.vocab, c.d_argmax);
        return cudaGetLastError();
    }

    // MoE layers 3-77: post-attn RMSNorm -> router top_k=8 -> expert stack.
    if (!lw.d_post_attn_norm || !lw.d_router_gate)
        return GlmCopyAsync(c.st, c.d_attn, d_out,
                            static_cast<int64_t>(seq) * hidden);
    GlmCopyRow0Kernel<<<row0_blocks, kGlmLoopBlock, 0, c.st>>>(
        c.d_attn, c.d_hidrow, hidden);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    const float *d_bias = nullptr;
    if (lw.h_gate_bias && lw.gate_bias_n == kGlmNumRouted) {
        e = cudaMemcpyAsync(c.d_gate_bias, lw.h_gate_bias,
                            static_cast<size_t>(kGlmNumRouted) *
                                sizeof(float),
                            cudaMemcpyHostToDevice, c.st);
        if (e != cudaSuccess) return e;
        d_bias = c.d_gate_bias;
    }
    e = RunGlmMoERouter(c.ch, c.st, lw, 1, c.d_hidrow, d_bias,
                        c.d_normed, c.d_logits, c.d_routing);
    if (e != cudaSuccess) return e;
    e = RunLoopExpertStack(c, l);
    if (e != cudaSuccess) return e;
    GlmResidualKernel<<<row0_blocks, kGlmLoopBlock, 0, c.st>>>(
        d_out, c.d_moe, c.d_hidrow, hidden);
    return cudaGetLastError();
}

} // anonymous namespace

cudaError_t RunGlmEmbed(cudaStream_t stream, const __half *d_embed,
                        const int32_t *d_ids, __half *d_out, int seq,
                        int hidden)
{
    if (!d_embed || !d_ids || !d_out || seq <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    const int64_t total = static_cast<int64_t>(seq) * hidden;
    const int blocks = static_cast<int>((total + kGlmLoopBlock - 1) /
                                        kGlmLoopBlock);
    GlmEmbedKernel<<<blocks, kGlmLoopBlock, 0, stream>>>(
        d_embed, d_ids, d_out, seq, hidden);
    return cudaGetLastError();
}

// Shared generation-loop body used by both public RunGlmGeneration overloads.
// Exactly one of `experts` / `expert_db` must be non-null:
//   experts    device-resident GlmExpertTables (all layers resident in VRAM)
//   expert_db  on-demand mmap index + staging buffers (one expert at a time)
static cudaError_t RunGlmGenerationImpl(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw,
    const GlmExpertTables *experts, GlmExpertDb *expert_db,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler())
{
    result = GlmGenerationResult{};
    const char *g_stage = "validate";
    const cudaError_t g_pending = cudaPeekAtLastError();
    if (g_pending != cudaSuccess) {
        log::warn("[glm_gen] stale cuda error before entry: " +
                  std::string(cudaGetErrorString(g_pending)));
        cudaGetLastError();  // consume, so the caller's stale state can't derail us
    }
    const int hidden = kGlmHidden;
    const int inter  = kGlmMoeIntermediate;
    const int P  = static_cast<int>(prompt.size());
    const int vocab = pw.vocab_size;
    const bool on_demand = (expert_db != nullptr);
    log::debug("[glm_gen] validate: ch=" + std::to_string(cublas_h != nullptr) +
               " st=" + std::to_string(stream != nullptr) +
               " P=" + std::to_string(P) +
               " max_new=" + std::to_string(max_new) +
               " vocab=" + std::to_string(vocab) +
               " layers=" + std::to_string(pw.layers.size()) +
               " tables=" + std::to_string(experts ? experts->tables.size() : 0) +
               " on_demand=" + std::to_string(on_demand));
    if (!cublas_h || !stream || P <= 0 || max_new <= 0 || vocab <= 0 ||
        !pw.d_embed || !pw.d_head ||
        pw.layers.size() != static_cast<size_t>(kGlmNumLayers))
        return cudaErrorInvalidValue;
    if (on_demand) {
        if (expert_db->num_layers != kGlmNumLayers ||
            expert_db->experts_per_layer != kGlmNumRouted ||
            !d_pack_stage || !d_scale_stage)
            return cudaErrorInvalidValue;
    } else {
        if (!experts ||
            experts->tables.size() != static_cast<size_t>(kGlmNumLayers))
            return cudaErrorInvalidValue;
    }

    // Memory safety: the decoder KV staging is fixed at kGlmIndexMaxSlots
    // (1024) slots. prompt + generated tokens must fit, and every prompt id
    // must index the embed table. Reject explicitly here (deterministic)
    // rather than risk silent device-side out-of-bounds access.
    if (P > kGlmIndexMaxSlots ||
        max_new > kGlmIndexMaxSlots - P) {
        log::error("[glm_gen] prompt+" + std::to_string(max_new) +
                   " tokens exceed the decoder KV capacity (" +
                   std::to_string(kGlmIndexMaxSlots) + " slots)");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < P; ++i) {
        const int32_t id = prompt[static_cast<size_t>(i)];
        if (id < 0 || id >= vocab) {
            log::error("[glm_gen] prompt id out of range: " +
                       std::to_string(id) + " (vocab=" +
                       std::to_string(vocab) + ")");
            return cudaErrorInvalidValue;
        }
    }

    const int seq_max = std::max(P, 1);

    // ── Cycle 10: ONE unified arena for the whole 79-layer loop footprint ────
    // The complete IndexShare K-V cache slots (GlmIndexShareState), the
    // attention/dense/MoE activation workspaces and the Layer-78 MTP execution
    // spaces (eh_proj output, enorm/hmtp, h78, shared-head normalized hidden
    // and the [vocab] logits row) live together in a single contiguous device
    // allocation, plus one combined pinned host staging block (routing grid +
    // argmax slot). Mid-flight runtime heap fragmentation is structurally
    // outlawed; the whole run performs exactly one device + one host cudaAlloc.
    GlmMoEArena arena;
    cudaError_t e = arena.AllocateGeneration(seq_max, hidden, kGlmNumRouted,
                                             vocab);
    if (e != cudaSuccess) {
        result.ok = false;
        result.error_msg = cudaGetErrorString(e);
        return e;
    }
    e = GlmIndexShareReset(arena.d_state, stream);
    if (e != cudaSuccess) {
        result.ok = false;
        result.error_msg = cudaGetErrorString(e);
        log::error("[glm_gen] FAIL (stage=kv-reset): " +
                   std::string(cudaGetErrorString(e)));
        arena.Free();
        return e;
    }

    auto cleanup = [&]() { arena.Free(); };
    auto fail = [&](cudaError_t err) -> cudaError_t {
        result.ok = false;
        result.error_msg = cudaGetErrorString(err);
        log::error("[glm_gen] FAIL (stage=" + std::string(g_stage) + "): " +
                   std::string(cudaGetErrorString(err)));
        cleanup();
        return err;
    };

    GlmLoopCtx ctx;
    ctx.ch = cublas_h;
    ctx.st = stream;
    ctx.pw = &pw;
    if (on_demand) {
        ctx.db = expert_db;
        ctx.d_pack_stage = d_pack_stage;
        ctx.d_scale_stage = d_scale_stage;
    } else {
        ctx.experts = experts;
    }
    ctx.hidden = hidden;
    ctx.inter = inter;
    ctx.vocab = vocab;
    ctx.d_state = arena.d_state;
    ctx.d_attn = arena.d_attn;
    ctx.d_normed = arena.d_normed;
    ctx.d_moe = arena.d_moe;
    ctx.d_gate = arena.d_gate;
    ctx.d_up = arena.d_up;
    ctx.d_logits = arena.d_logits;
    ctx.d_routing = arena.d_routing;
    ctx.d_scratch_e = arena.d_scratch;
    ctx.d_hidrow = arena.d_hidrow;
    ctx.d_mtp = arena.d_mtp;
    ctx.d_hmtp = arena.d_hmtp;
    ctx.d_h78 = arena.d_h78;
    ctx.d_hn = arena.d_hn;
    ctx.d_logits_head = arena.d_logits_head;
    ctx.d_gate_bias = arena.d_gate_bias;
    ctx.d_argmax = arena.d_argmax;
    ctx.h_routing = arena.h_routing_staging;

    // ── Prefill (turn 0): embed the prompt and forward all 79 layers. ─────
    g_stage = "prefill-embed";
    e = cudaMemcpyAsync(arena.d_ids, prompt.data(),
                        static_cast<size_t>(P) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return fail(e);
    const int emb_blocks = static_cast<int>(
        (static_cast<size_t>(P) * hidden + kGlmLoopBlock - 1) / kGlmLoopBlock);
    GlmEmbedKernel<<<emb_blocks, kGlmLoopBlock, 0, stream>>>(
        pw.d_embed, arena.d_ids, arena.d_cur, P, hidden);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(e);

    __half *cur = arena.d_cur;
    __half *nxt = arena.d_alt;
    int block_rows = P;
    for (int l = 0; l < kGlmNumLayers; ++l) {
        g_stage = "prefill-layer";
        log::debug("[glm_gen] prefill l=" + std::to_string(l) +
                   " seq=" + std::to_string(block_rows));
        e = RunGlmLoopLayer(ctx, l, 0, block_rows, cur, nxt);
        if (e != cudaSuccess) return fail(e);
        // Dense layers 0-2 collapse the block to row 0 (see RunGlmLoopLayer)
        // as soon as their MLP weights are fully loaded — whichever of the
        // three is the FIRST to be fully loaded, not necessarily layer 0.
        // Re-checking only at l==0 left `block_rows` stuck at P whenever
        // layer 0 lacked its dense weights but layer 1 or 2 had theirs:
        // RunGlmLoopLayer would then silently collapse ITS OWN output to
        // row 0 while every later layer kept being invoked (and kept
        // reading downstream buffers) with seq=block_rows=P, consuming
        // rows 1..P-1 that were never written by the previous layer.
        if (l < 3 && block_rows != 1) {
            const GlmLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
            if (lw.d_post_attn_norm && lw.d_mlp_gate &&
                lw.d_mlp_up && lw.d_mlp_down) {
                block_rows = 1;  // this dense branch collapsed the block
            }
        }
        if (l < kGlmNumLayers - 1) std::swap(cur, nxt);
    }

    // ── Greedy decode: argmax the layer-78 logits row, embed, re-forward. ──
    for (int step = 0; step < max_new; ++step) {
        g_stage = "decode-argmax";
        e = sampler.Sample(stream, arena.d_logits_head, vocab,
                           lsxcommon::sample::LogitType::kF32Rounded16,
                           &arena.d_argmax->best, &arena.d_argmax->maxv);
        if (e != cudaSuccess) return fail(e);
        e = cudaMemcpyAsync(arena.h_argmax, arena.d_argmax,
                            sizeof(GlmArgmaxSlot), cudaMemcpyDeviceToHost,
                            stream);
        if (e != cudaSuccess) return fail(e);
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return fail(e);
        const int best = arena.h_argmax->best;
        result.tokens.push_back(best);
        if (step == 0) result.first_maxv = arena.h_argmax->maxv;
        if (best == kGlmEosIds[0] || best == kGlmEosIds[1] ||
            best == kGlmEosIds[2])
            break;

        int32_t id = best;
        e = cudaMemcpyAsync(arena.d_ids, &id, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return fail(e);
        const int emb_blocks = (hidden + kGlmLoopBlock - 1) / kGlmLoopBlock;
        GlmEmbedKernel<<<emb_blocks, kGlmLoopBlock, 0, stream>>>(
            pw.d_embed, arena.d_ids, arena.d_cur, 1, hidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(e);

        cur = arena.d_cur;
        nxt = arena.d_alt;
        g_stage = "decode-forward";
        log::debug("[glm_gen] decode step=" + std::to_string(step));
        for (int l = 0; l < kGlmNumLayers; ++l) {
            e = RunGlmLoopLayer(ctx, l, step + 1, 1, cur, nxt);
            if (e != cudaSuccess) return fail(e);
            if (l < kGlmNumLayers - 1) std::swap(cur, nxt);
        }
    }

    // ── Final logits row: the raw [vocab] logits the last layer-78 MTP forward
    //    produced (the row whose argmax would be the next hypothetical token).
    //    Published for the SKWOS regression harness, which compares it
    //    element-for-element against the CPU FP64 reference. ──────────────
    result.final_logits.resize(static_cast<size_t>(vocab));
    g_stage = "final-logits";
    e = cudaMemcpyAsync(result.final_logits.data(), arena.d_logits_head,
                        static_cast<size_t>(vocab) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return fail(e);
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return fail(e);

    // ── KV slot readback: prompt length + generated tokens. ────────────────
    int h_n_used = 0;
    e = cudaMemcpyAsync(&h_n_used, &arena.d_state->n_used, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return fail(e);
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return fail(e);
    result.kv_slots_used = h_n_used;
    result.ok = true;
    cleanup();
    return cudaSuccess;
}

cudaError_t RunGlmGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, const GlmExpertTables &experts,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result,
    lsxcommon::sample::ISampler &sampler)
{
    return RunGlmGenerationImpl(cublas_h, stream, pw, &experts, nullptr,
                                nullptr, nullptr, prompt, max_new, result,
                                sampler);
}

cudaError_t RunGlmGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, GlmExpertDb &expert_db,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result,
    lsxcommon::sample::ISampler &sampler)
{
    // Two reusable per-call staging buffers, one expert at a time: the packed
    // E2M1 payloads (w1+w2+w3) and the per-block E4M3 scales. Freed on every
    // exit path, so a 385 GiB model never exceeds the pinned weights + ~2 ×
    // one-expert staging in VRAM.
    constexpr int64_t kMatEl = kGlmExpertMatEl;
    constexpr size_t pack_bytes = static_cast<size_t>(kMatEl / 2) * 3;
    constexpr size_t scale_bytes =
        static_cast<size_t>(kMatEl / kGlmNvfp4Block) * 3;

    uint8_t *d_pack = nullptr;
    uint8_t *d_scale = nullptr;
    cudaError_t e = cudaMalloc(&d_pack, pack_bytes);
    if (e != cudaSuccess) {
        result = GlmGenerationResult{};
        result.error_msg = cudaGetErrorString(e);
        return e;
    }
    e = cudaMalloc(&d_scale, scale_bytes);
    if (e != cudaSuccess) {
        cudaFree(d_pack);
        result = GlmGenerationResult{};
        result.error_msg = cudaGetErrorString(e);
        return e;
    }

    e = RunGlmGenerationImpl(cublas_h, stream, pw, nullptr, &expert_db,
                             d_pack, d_scale, prompt, max_new, result, sampler);    cudaFree(d_scale);
    cudaFree(d_pack);
    return e;
}

// ── Faithful GLM-5.2 generation: real MLA + sigmoid MoE forward ─────────────
namespace {

constexpr float kGlmRopeTheta = 8000000.0f;   // config rope_theta
constexpr int   kGlmDenseInter = 12288;       // config intermediate_size
constexpr float kGlmAttnScale = 0.0625f;      // qk_head_dim^-0.5 = 256^-0.5
constexpr float kGlmIndexScale = 0.0883883476f; // (unused: indexer is a no-op ≤ 1024 slots)

// Interleaved RoPE (DeepSeek layout) applied to the rotary half of the q
// matrix in place: q = [heads][kGlmQkNope+kGlmQkRope], rot slice
// [kGlmQkNope, kGlmQkNope+kGlmQkRope). Pair i of each head is rotated by
// pos·θ^(-(2i)/kGlmQkNope) — matches apply_rotary_pos_emb_interleave with
// cos/sin from the head_dim=kGlmQkNope inv_freq table (first kGlmQkRope/2
// pairs).
//
// KNOWN BIT-EXACTNESS GAP: unlike the softmax path (GlmExpfGlibc) and rsqrt
// path (GlmRsqrtfGlibc, glm_device_math.h), this kernel's powf/cosf/sinf are
// raw CUDA math, not glibc-bit-exact replicas. Reproducing glibc's table-
// based sincosf/powf bit-for-bit would be a substantial reverse-engineering
// effort in its own right (glm_device_math.h's GlmExpfGlibc alone is ~60
// lines replicating one glibc table), and — critically — it cannot be
// verified here: doing so blind, without the reference CPU output + a
// token-for-token diff harness against a real GLM-5.2 checkpoint, risks
// introducing a subtly wrong "bit-exact" implementation that is worse than
// today's honestly-approximate one. Left as a documented gap for whoever
// has that reference harness available, rather than attempted speculatively.
__global__ void GlmQropeKernel(__half *__restrict__ q, int pos, float theta,
                               int heads)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * (kGlmQkRope / 2)) return;
    const int h = idx / (kGlmQkRope / 2);
    const int i = idx % (kGlmQkRope / 2);
    const float angle = static_cast<float>(pos) *
        powf(theta, -static_cast<float>(2 * i) / static_cast<float>(kGlmQkNope));
    const float c = cosf(angle);
    const float s = sinf(angle);
    __half *qr = q + static_cast<size_t>(h) * (kGlmQkNope + kGlmQkRope)
                    + kGlmQkNope + 2 * i;
    const float a = __half2float(qr[0]);
    const float b = __half2float(qr[1]);
    qr[0] = __float2half(a * c - b * s);
    qr[1] = __float2half(b * c + a * s);
}

// Same rotation for the contiguous [kGlmQkRope] shared k_rot vector.
__global__ void GlmKropeKernel(__half *__restrict__ rot, int pos, float theta)
{
    const int i = threadIdx.x;
    if (i >= kGlmQkRope / 2) return;
    const float angle = static_cast<float>(pos) *
        powf(theta, -static_cast<float>(2 * i) / static_cast<float>(kGlmQkNope));
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float a = __half2float(rot[2 * i]);
    const float b = __half2float(rot[2 * i + 1]);
    rot[2 * i]     = __float2half(a * c - b * s);
    rot[2 * i + 1] = __float2half(b * c + a * s);
}

// Expands kv_b_proj output into the per-layer K/V cache. kvb = [heads]
// [kGlmQkNope+kGlmHeadV] per head (k_nope[kGlmQkNope] | v[kGlmHeadV]);
// krot[kGlmQkRope] is the shared rotated k_rot. Writes key[h] =
// [k_nope[h] | krot] and value[h] = v[h] into cache slot `pos`.
__global__ void GlmMlaStoreKvKernel(const __half *__restrict__ kvb,
                                    const __half *__restrict__ krot,
                                    __half *__restrict__ k_cache,
                                    __half *__restrict__ v_cache,
                                    int slot, int cap, int heads)
{
    const int64_t total = static_cast<int64_t>(heads) * kGlmHeadV;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= total) return;
    const int h = static_cast<int>(i / kGlmHeadV);
    const int d = static_cast<int>(i % kGlmHeadV);
    const __half *kr = kvb + static_cast<int64_t>(h) * (kGlmQkNope + kGlmHeadV);
    const __half val = (d < kGlmQkNope) ? kr[d] : krot[d - kGlmQkNope];
    const int64_t at = (static_cast<int64_t>(slot) * heads + h) * kGlmHeadV + d;
    k_cache[at] = val;
    v_cache[at] = kr[kGlmQkNope + d];
}

// Causal MLA attention for one token over its layer's K/V cache (slots 0..T-1):
//   out[h][d] = Σ_t softmax_t((q·k)·scale) · v[t][h][d],   scale = kGlmHeadV^-0.5
// One block per head, kGlmHeadV threads. Pass 1 block-reduces each key score
// (FP32), pass 2 accumulates the weighted value projection per dimension.
// FP16 in/out, FP32 math, single FP16 rounding at the end (like the eager
// attention path).
__global__ void GlmMlaAttnKernel(const __half *__restrict__ q,
                                 const __half *__restrict__ k_cache,
                                 const __half *__restrict__ v_cache,
                                 __half *__restrict__ out,
                                 int cap, int T, int heads, float scale)
{
    const int h = blockIdx.x;
    __shared__ float s_q[kGlmHeadV];
    __shared__ float s_red[8];
    __shared__ float s_score[kGlmIndexMaxSlots];
    __shared__ float s_p[kGlmIndexMaxSlots];
    __shared__ float s_lmax;
    __shared__ float s_psum;
    if (threadIdx.x < kGlmHeadV)
        s_q[threadIdx.x] = __half2float(q[h * kGlmHeadV + threadIdx.x]);
    __syncthreads();

    float lmax = -FLT_MAX;
    for (int t = 0; t < T; ++t) {
        const __half *kr =
            k_cache + (static_cast<int64_t>(t) * heads + h) * kGlmHeadV;
        float v = s_q[threadIdx.x] * __half2float(kr[threadIdx.x]);
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) s_red[warp] = v;
        __syncthreads();
        if (warp == 0) {
            float w = (threadIdx.x < 8) ? s_red[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                w += __shfl_down_sync(0xffffffffu, w, off);
            if (lane == 0) {
                s_score[t] = w * scale;
                lmax = fmaxf(lmax, s_score[t]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_lmax = lmax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float psum = 0.0f;
        for (int t = 0; t < T; ++t) {
            // Bit-exact glibc expf shim — this whole forward pass
            // (RunGlmRealGeneration) is built to replicate the CPU/glibc
            // reference bit-for-bit under -fmad=false; a raw CUDA expf()
            // here would use a different polynomial/table and silently
            // break that parity for every MLA attention softmax.
            s_p[t] = GlmExpfGlibc(s_score[t] - s_lmax);
            psum += s_p[t];
        }
        s_psum = psum;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) {
        const float p = s_p[t] / s_psum;
        acc += p * __half2float(
                       v_cache[(static_cast<int64_t>(t) * heads + h) * kGlmHeadV +
                               threadIdx.x]);
    }
    out[h * kGlmHeadV + threadIdx.x] = __float2half(acc);
}

// GlmRealSwigluKernel: single textual source in glm_expert_kernels.cuh.

// Per-token scratch + persistent per-layer expanded K/V caches of the real
// GLM-5.2 generation pass. One contiguous device allocation + one pinned host
// staging block.
struct GlmRealCtx {
    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const GlmPinnedWeights *pw = nullptr;
    GlmExpertDb *db = nullptr;

    __half *d_hid = nullptr;      // [hidden]          working residual stream
    __half *d_norm = nullptr;     // [hidden]          input_layernorm out
    __half *d_q_resid = nullptr;  // [q_lora]          q_a_proj out
    __half *d_q_resid_n = nullptr;// [q_lora]          q_a_layernorm out
    __half *d_q = nullptr;        // [64*256]          q_b_proj out
    __half *d_kva = nullptr;      // [576]             kv_a_proj out
    __half *d_kvp = nullptr;      // [512]             kv_a_layernorm out
    __half *d_kvb = nullptr;      // [64*448]          kv_b_proj out
    __half *d_krot = nullptr;     // [64]              rotated shared k_rot
    __half *d_attn = nullptr;     // [64*256]          per-head attention out
    __half *d_off = nullptr;      // [hidden]          o_proj / dense-down out
    __half *d_mlpn = nullptr;     // [hidden]          post_attn_layernorm out
    __half *d_routed = nullptr;   // [hidden]          routed-expert output
    __half *d_shared = nullptr;   // [hidden]          shared-expert output
    __half *d_dgate = nullptr;    // [dense_inter]     dense gate_proj out
    __half *d_dup = nullptr;      // [dense_inter]     dense up_proj out
    float  *d_logits = nullptr;   // [256]             router logits
    GlmRoutedSlot *d_routing = nullptr; // [8]         router grid
    float  *d_gate_bias = nullptr;// [256]             router-bias staging
    __half *d_scratch_e = nullptr;// expert dequant/inter/gate workspace
    uint8_t *d_pack = nullptr;    // on-demand expert staging
    uint8_t *d_scale = nullptr;   // on-demand expert staging
    GlmRoutedSlot *h_routing = nullptr; // [8] pinned host routing staging

    __half *d_hn = nullptr;       // [hidden]          model.norm out
    float  *d_logits_head = nullptr; // [vocab]        head logits row
    GlmArgmaxSlot *d_argmax = nullptr; // [1]
    GlmArgmaxSlot *h_argmax = nullptr; // pinned host argmax
    __half *d_cur = nullptr;      // [cap][hidden]     token residual stream
    int32_t *d_ids = nullptr;     // [cap]             id staging
    void   *d_base = nullptr;     // single device allocation
    __half *d_kv_k_base = nullptr; // [layers][cap][64][256] expanded keys
    __half *d_kv_v_base = nullptr; // [layers][cap][64][256] expanded values
    int kv_cap = 0;
    size_t bytes = 0;

    void Free()
    {
        if (h_routing) cudaFreeHost(h_routing);
        if (h_argmax)  cudaFreeHost(h_argmax);
        if (d_base)    cudaFree(d_base);
        h_routing = nullptr; h_argmax = nullptr; d_base = nullptr;
        d_hid = d_norm = d_q_resid = d_q_resid_n = d_q = nullptr;
        d_kva = d_kvp = d_kvb = d_krot = d_attn = d_off = nullptr;
        d_mlpn = d_routed = d_shared = d_dgate = d_dup = nullptr;
        d_logits = nullptr; d_routing = nullptr; d_gate_bias = nullptr;
        d_scratch_e = nullptr; d_pack = nullptr; d_scale = nullptr;
        d_hn = nullptr; d_logits_head = nullptr; d_argmax = nullptr;
        d_cur = nullptr; d_ids = nullptr;
        d_kv_k_base = nullptr; d_kv_v_base = nullptr;
        kv_cap = 0; bytes = 0;
    }
};

// Allocates the full real-generation footprint (per-token scratch + per-layer
// expanded K/V caches + head + residual stream) as one device block plus one
// pinned host staging block. cap = P + max_new (≤ kGlmIndexMaxSlots).
cudaError_t GlmRealCtxAllocate(GlmRealCtx &c, int cap, int vocab)
{
    c.Free();
    constexpr int64_t kMatEl = kGlmExpertMatEl;
    const int64_t hidden = kGlmHidden;
    const int64_t dinter = kGlmDenseInter;
    const int64_t layers = kGlmNumLayers;
    const int64_t heads256 = static_cast<int64_t>(kGlmNumHeads) * kGlmHeadV;
    const int64_t kv_bytes = layers * cap * heads256 * 2 * sizeof(__half);

    int64_t off = 0;
    auto page = [&](int64_t bytes_, int64_t align) -> int64_t {
        off = (off + align - 1) / align * align;
        const int64_t at = off;
        off += bytes_;
        return at;
    };
    const int64_t o_hid    = page(hidden * sizeof(__half), 16);
    const int64_t o_norm   = page(hidden * sizeof(__half), 16);
    const int64_t o_qr     = page(kGlmQLoraRank * sizeof(__half), 16);
    const int64_t o_qrn    = page(kGlmQLoraRank * sizeof(__half), 16);
    const int64_t o_q      = page(heads256 * sizeof(__half), 16);
    const int64_t o_kva    = page((kGlmKVLoraRank + kGlmQkRope) * sizeof(__half), 16);
    const int64_t o_kvp    = page(kGlmKVLoraRank * sizeof(__half), 16);
    const int64_t o_kvb    = page(kGlmNumHeads * (kGlmQkNope + kGlmHeadV) * sizeof(__half), 16);
    const int64_t o_krot   = page(kGlmQkRope * sizeof(__half), 16);
    const int64_t o_attn   = page(heads256 * sizeof(__half), 16);
    const int64_t o_off    = page(hidden * sizeof(__half), 16);
    const int64_t o_mlpn   = page(hidden * sizeof(__half), 16);
    const int64_t o_routed = page(hidden * sizeof(__half), 16);
    const int64_t o_shared = page(hidden * sizeof(__half), 16);
    const int64_t o_dgate  = page(dinter * sizeof(__half), 16);
    const int64_t o_dup    = page(dinter * sizeof(__half), 16);
    const int64_t o_logits = page(kGlmNumRouted * sizeof(float), 16);
    const int64_t o_route  = page(kGlmMoETopK * sizeof(GlmRoutedSlot), 16);
    const int64_t o_bias   = page(kGlmNumRouted * sizeof(float), 16);
    const int64_t o_scratch = page((kMatEl + hidden + 2 * kGlmMoeIntermediate) * sizeof(__half), 16);
    const int64_t o_pack   = page(kMatEl / 2 * 3, 16);
    const int64_t o_scale  = page(kMatEl / kGlmNvfp4Block * 3, 16);
    const int64_t o_hn     = page(hidden * sizeof(__half), 16);
    const int64_t o_lhead  = page(int64_t(vocab) * sizeof(float), 16);
    const int64_t o_argmax = page(sizeof(GlmArgmaxSlot), 16);
    const int64_t o_cur    = page(int64_t(cap) * hidden * sizeof(__half), 16);
    const int64_t o_ids    = page(int64_t(cap) * sizeof(int32_t), 16);
    const int64_t o_kv     = page(kv_bytes, 16);
    c.bytes = static_cast<size_t>(off);

    uint8_t *base = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&base), c.bytes);
    if (e != cudaSuccess) { c.bytes = 0; return e; }
    c.d_base = base;
    c.d_hid     = reinterpret_cast<__half *>(base + o_hid);
    c.d_norm    = reinterpret_cast<__half *>(base + o_norm);
    c.d_q_resid = reinterpret_cast<__half *>(base + o_qr);
    c.d_q_resid_n = reinterpret_cast<__half *>(base + o_qrn);
    c.d_q       = reinterpret_cast<__half *>(base + o_q);
    c.d_kva     = reinterpret_cast<__half *>(base + o_kva);
    c.d_kvp     = reinterpret_cast<__half *>(base + o_kvp);
    c.d_kvb     = reinterpret_cast<__half *>(base + o_kvb);
    c.d_krot    = reinterpret_cast<__half *>(base + o_krot);
    c.d_attn    = reinterpret_cast<__half *>(base + o_attn);
    c.d_off     = reinterpret_cast<__half *>(base + o_off);
    c.d_mlpn    = reinterpret_cast<__half *>(base + o_mlpn);
    c.d_routed  = reinterpret_cast<__half *>(base + o_routed);
    c.d_shared  = reinterpret_cast<__half *>(base + o_shared);
    c.d_dgate   = reinterpret_cast<__half *>(base + o_dgate);
    c.d_dup     = reinterpret_cast<__half *>(base + o_dup);
    c.d_logits  = reinterpret_cast<float *>(base + o_logits);
    c.d_routing = reinterpret_cast<GlmRoutedSlot *>(base + o_route);
    c.d_gate_bias = reinterpret_cast<float *>(base + o_bias);
    c.d_scratch_e = reinterpret_cast<__half *>(base + o_scratch);
    c.d_pack    = base + o_pack;
    c.d_scale   = base + o_scale;
    c.d_hn      = reinterpret_cast<__half *>(base + o_hn);
    c.d_logits_head = reinterpret_cast<float *>(base + o_lhead);
    c.d_argmax  = reinterpret_cast<GlmArgmaxSlot *>(base + o_argmax);
    c.d_cur     = reinterpret_cast<__half *>(base + o_cur);
    c.d_ids     = reinterpret_cast<int32_t *>(base + o_ids);
    c.d_kv_k_base = reinterpret_cast<__half *>(base + o_kv);
    c.d_kv_v_base = c.d_kv_k_base + layers * cap * heads256;
    c.kv_cap    = cap;

    uint8_t *hbase = nullptr;
    const int64_t h_route = kGlmMoETopK * sizeof(GlmRoutedSlot);
    e = cudaMallocHost(reinterpret_cast<void **>(&hbase),
                       static_cast<size_t>(h_route + sizeof(GlmArgmaxSlot)));
    if (e != cudaSuccess) { c.Free(); return e; }
    c.h_routing = reinterpret_cast<GlmRoutedSlot *>(hbase);
    c.h_argmax  = reinterpret_cast<GlmArgmaxSlot *>(hbase + h_route);
    return cudaSuccess;
}

// One layer of the real GLM-5.2 forward for a single token at position `pos`
// (its K/V is stored at cache slot `pos`). d_hid is the in/out residual stream.
cudaError_t RunGlmRealLayer(GlmRealCtx &c, int l, int pos,
                            const PackedExpert *shared_host)
{
    const GlmPinnedWeights &pw = *c.pw;
    const GlmLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
    const int hidden = pw.hidden;
    const __half *dkv = c.d_kv_k_base +
        static_cast<int64_t>(l) * c.kv_cap * kGlmNumHeads * kGlmHeadV;
    const __half *dvv = c.d_kv_v_base +
        static_cast<int64_t>(l) * c.kv_cap * kGlmNumHeads * kGlmHeadV;
    cudaError_t e;

    // input_layernorm
    e = RunGlmRmsNorm(c.st, c.d_hid, lw.d_input_norm, c.d_norm, 1, hidden,
                      kGlmRmsEps);
    if (e != cudaSuccess) return e;

    // q_resid = q_a_proj(x); q_resid = q_a_layernorm(q_resid)
    e = RunGemmFp64(c.st, lw.d_q_a_proj, c.d_norm, c.d_q_resid,
                    kGlmQLoraRank, hidden, 1);
    if (e != cudaSuccess) return e;
    e = RunGlmRmsNorm(c.st, c.d_q_resid, lw.d_q_a_norm, c.d_q_resid_n,
                      1, kGlmQLoraRank, kGlmRmsEps);
    if (e != cudaSuccess) return e;
    // q = q_b_proj(q_resid_n)  [64, 256]
    e = RunGemmFp64(c.st, lw.d_q_b_proj, c.d_q_resid_n, c.d_q,
                    kGlmNumHeads * (kGlmQkNope + kGlmQkRope), kGlmQLoraRank, 1);
    if (e != cudaSuccess) return e;

    // compressed_kv = kv_a_proj_with_mqa(x)  [576]
    e = RunGemmFp64(c.st, lw.d_kv_a_proj, c.d_norm, c.d_kva,
                    kGlmKVLoraRank + kGlmQkRope, hidden, 1);
    if (e != cudaSuccess) return e;
    // k_pass = kv_a_layernorm(kv_pass); k_rot_raw = kv_a[512:576]
    e = RunGlmRmsNorm(c.st, c.d_kva, lw.d_kv_a_norm, c.d_kvp, 1,
                      kGlmKVLoraRank, kGlmRmsEps);
    if (e != cudaSuccess) return e;
    {
        const int blocks = (kGlmQkRope + kGlmLoopBlock - 1) / kGlmLoopBlock;
        GlmCopyKernel<<<blocks, kGlmLoopBlock, 0, c.st>>>(
            c.d_kva + kGlmKVLoraRank, c.d_krot, kGlmQkRope);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    // kv_b = kv_b_proj(k_pass)  [64, 448]
    e = RunGemmFp64(c.st, lw.d_kv_b_proj, c.d_kvp, c.d_kvb,
                    kGlmNumHeads * (kGlmQkNope + kGlmHeadV),
                    kGlmKVLoraRank, 1);
    if (e != cudaSuccess) return e;

    // Interleaved RoPE on q[..., kGlmQkNope:kGlmQkNope+kGlmQkRope] and
    // k_rot, then store K/V.
    {
        GlmQropeKernel<<<(kGlmNumHeads * (kGlmQkRope / 2) + 255) / 256, 256, 0, c.st>>>(
            c.d_q, pos, kGlmRopeTheta, kGlmNumHeads);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    GlmKropeKernel<<<1, kGlmQkRope / 2, 0, c.st>>>(c.d_krot, pos, kGlmRopeTheta);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    GlmMlaStoreKvKernel<<<kGlmNumHeads, kGlmHeadV, 0, c.st>>>(
        c.d_kvb, c.d_krot,
        c.d_kv_k_base + static_cast<int64_t>(l) * c.kv_cap * kGlmNumHeads * kGlmHeadV,
        c.d_kv_v_base + static_cast<int64_t>(l) * c.kv_cap * kGlmNumHeads * kGlmHeadV,
        pos, c.kv_cap, kGlmNumHeads);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // Attention over slots 0..pos (causal).
    GlmMlaAttnKernel<<<kGlmNumHeads, kGlmHeadV, 0, c.st>>>(
        c.d_q, dkv, dvv, c.d_attn, c.kv_cap, pos + 1, kGlmNumHeads,
        kGlmAttnScale);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    // o_proj(attn) -> d_off; residual: hid += off
    e = RunGemmFp64(c.st, lw.d_o_proj, c.d_attn, c.d_off, hidden,
                    kGlmNumHeads * kGlmHeadV, 1);
    if (e != cudaSuccess) return e;
    GlmAddResidualKernel<<<(hidden + 255) / 256, 256, 0, c.st>>>(c.d_hid, c.d_off, hidden);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // MLP.
    if (l < 3) {
        // Dense: post_attn_norm -> gate/up -> silu swiglu -> down -> residual.
        e = RunGlmRmsNorm(c.st, c.d_hid, lw.d_post_attn_norm, c.d_mlpn, 1,
                          hidden, kGlmRmsEps);
        if (e != cudaSuccess) return e;
        e = RunGemmFp64(c.st, lw.d_mlp_gate, c.d_mlpn, c.d_dgate,
                        kGlmDenseInter, hidden, 1);
        if (e != cudaSuccess) return e;
        e = RunGemmFp64(c.st, lw.d_mlp_up, c.d_mlpn, c.d_dup,
                        kGlmDenseInter, hidden, 1);
        if (e != cudaSuccess) return e;
        GlmRealSwigluKernel<<<(kGlmDenseInter + 255) / 256, 256, 0, c.st>>>(
            c.d_dgate, c.d_dup, kGlmDenseInter);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = RunGemmFp64(c.st, lw.d_mlp_down, c.d_dgate, c.d_off, hidden,
                        kGlmDenseInter, 1);
        if (e != cudaSuccess) return e;
        GlmAddResidualKernel<<<(hidden + 255) / 256, 256, 0, c.st>>>(c.d_hid, c.d_off, hidden);
        return cudaGetLastError();
    }

    // MoE: real sigmoid router + routed experts + shared expert.
    if (!lw.d_post_attn_norm || !lw.d_router_gate)
        return cudaErrorInvalidValue;
    const float *d_bias = nullptr;
    if (lw.h_gate_bias && lw.gate_bias_n == kGlmNumRouted) {
        e = cudaMemcpyAsync(c.d_gate_bias, lw.h_gate_bias,
                            static_cast<size_t>(kGlmNumRouted) * sizeof(float),
                            cudaMemcpyHostToDevice, c.st);
        if (e != cudaSuccess) return e;
        d_bias = c.d_gate_bias;
    }
    e = RunGlmRealRouter(c.ch, c.st, lw, 1, c.d_hid, d_bias,
                         c.d_mlpn, c.d_logits, c.d_routing);
    if (e != cudaSuccess) return e;
    e = RunGlmExpertsStackOnDemand(c.ch, c.st, c.d_routing, 1, c.d_mlpn,
                                   *c.db, l, kGlmNumRouted, c.d_routed,
                                   c.d_scratch_e, c.d_pack, c.d_scale,
                                   c.h_routing);
    if (e != cudaSuccess) return e;
    // Shared expert (w1=gate, w2=down, w3=up, same NVFP4 layout as a routed
    // expert): streamed from the host-side shared_host into the reusable
    // staging buffers and evaluated with the identical RunGlmExpertSlot chain,
    // so the full ~1.4 GiB shared set costs no VRAM. Inactive/foreign payloads
    // (missing w1/w2/w3 or mismatched sizes) contribute nothing.
    constexpr int64_t kPackPerMat = kGlmExpertMatEl / 2;
    constexpr int64_t kScalePerMat = kGlmExpertMatEl / kGlmNvfp4Block;
    bool shared_ok = false;
    if (shared_host && !shared_host->w1_packed.empty() &&
        !shared_host->w2_packed.empty() && !shared_host->w3_packed.empty() &&
        !shared_host->w1_scales.empty() && !shared_host->w2_scales.empty() &&
        !shared_host->w3_scales.empty() &&
        shared_host->w1_packed.size() == static_cast<size_t>(kPackPerMat) &&
        shared_host->w2_packed.size() == static_cast<size_t>(kPackPerMat) &&
        shared_host->w3_packed.size() == static_cast<size_t>(kPackPerMat) &&
        shared_host->w1_scales.size() == static_cast<size_t>(kScalePerMat) &&
        shared_host->w2_scales.size() == static_cast<size_t>(kScalePerMat) &&
        shared_host->w3_scales.size() == static_cast<size_t>(kScalePerMat)) {
        uint8_t *p1 = c.d_pack;
        uint8_t *p2 = p1 + kPackPerMat;
        uint8_t *p3 = p2 + kPackPerMat;
        uint8_t *s1 = c.d_scale;
        uint8_t *s2 = s1 + kScalePerMat;
        uint8_t *s3 = s2 + kScalePerMat;
        GlmExpertDeviceBlock seb;
        seb.d_w1_packed = p1;  seb.d_w1_scales = s1;
        seb.w1_global_scale = shared_host->w1_global_scale;
        seb.w1_packed_bytes = kPackPerMat;
        seb.w1_scale_bytes = kScalePerMat;
        seb.d_w2_packed = p2;  seb.d_w2_scales = s2;
        seb.w2_global_scale = shared_host->w2_global_scale;
        seb.w2_packed_bytes = kPackPerMat;
        seb.w2_scale_bytes = kScalePerMat;
        seb.d_w3_packed = p3;  seb.d_w3_scales = s3;
        seb.w3_global_scale = shared_host->w3_global_scale;
        seb.w3_packed_bytes = kPackPerMat;
        seb.w3_scale_bytes = kScalePerMat;
        e = cudaMemcpyAsync(p1, shared_host->w1_packed.data(),
                            static_cast<size_t>(kPackPerMat),
                            cudaMemcpyHostToDevice, c.st);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(p2, shared_host->w2_packed.data(),
                                static_cast<size_t>(kPackPerMat),
                                cudaMemcpyHostToDevice, c.st);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(p3, shared_host->w3_packed.data(),
                                static_cast<size_t>(kPackPerMat),
                                cudaMemcpyHostToDevice, c.st);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(s1, shared_host->w1_scales.data(),
                                static_cast<size_t>(kScalePerMat),
                                cudaMemcpyHostToDevice, c.st);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(s2, shared_host->w2_scales.data(),
                                static_cast<size_t>(kScalePerMat),
                                cudaMemcpyHostToDevice, c.st);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(s3, shared_host->w3_scales.data(),
                                static_cast<size_t>(kScalePerMat),
                                cudaMemcpyHostToDevice, c.st);
        if (e != cudaSuccess) return e;
        e = cudaMemsetAsync(c.d_shared, 0,
                            static_cast<size_t>(hidden) * sizeof(__half), c.st);
        if (e != cudaSuccess) return e;
        e = RunGlmExpertSlot(c.ch, c.st, seb, c.d_mlpn, c.d_shared,
                             c.d_scratch_e, 1.0f);
        if (e != cudaSuccess) return e;
        shared_ok = true;
    }
    if (!shared_ok) {
        cudaMemsetAsync(c.d_shared, 0,
                        static_cast<size_t>(hidden) * sizeof(__half), c.st);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    // mlp_out = routed + shared; residual: hid += mlp_out.
    GlmResidualKernel<<<(hidden + 255) / 256, 256, 0, c.st>>>(
        c.d_off, c.d_routed, c.d_shared, hidden);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    GlmAddResidualKernel<<<(hidden + 255) / 256, 256, 0, c.st>>>(c.d_hid, c.d_off, hidden);
    return cudaGetLastError();
}

} // anonymous namespace

cudaError_t RunGlmRealGeneration(cublasHandle_t cublas_h, cudaStream_t stream,
                                 const GlmPinnedWeights &pw,
                                 GlmExpertDb &expert_db,
                                 const std::vector<int32_t> &prompt,
                                 int max_new, GlmGenerationResult &result,
                                 lsxcommon::sample::ISampler &sampler)
{
    result = GlmGenerationResult{};
    const char *g_stage = "validate";
    char g_stage_buf[64];
    const cudaError_t g_pending = cudaPeekAtLastError();
    if (g_pending != cudaSuccess) {
        log::warn("[glm_real] stale cuda error before entry: " +
                  std::string(cudaGetErrorString(g_pending)));
        cudaGetLastError();
    }
    const int P = static_cast<int>(prompt.size());
    const int vocab = pw.vocab_size;
    if (!cublas_h || !stream || P <= 0 || max_new <= 0 || vocab <= 0 ||
        !pw.d_embed || !pw.d_head ||
        pw.layers.size() != static_cast<size_t>(kGlmNumLayers))
        return cudaErrorInvalidValue;
    if (expert_db.num_layers != kGlmNumLayers ||
        expert_db.experts_per_layer != kGlmNumRouted)
        return cudaErrorInvalidValue;
    const int cap = P + max_new;
    if (cap > kGlmIndexMaxSlots) {
        log::error("[glm_real] prompt+" + std::to_string(max_new) +
                   " tokens exceed the decoder KV capacity (" +
                   std::to_string(kGlmIndexMaxSlots) + " slots)");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < P; ++i) {
        const int32_t id = prompt[static_cast<size_t>(i)];
        if (id < 0 || id >= vocab) return cudaErrorInvalidValue;
    }

    GlmRealCtx c;
    c.ch = cublas_h;
    c.st = stream;
    c.pw = &pw;
    c.db = &expert_db;
    // Staging buffers for the on-demand expert path.
    cudaError_t e = GlmRealCtxAllocate(c, cap, vocab);
    if (e != cudaSuccess) {
        result.ok = false;
        result.error_msg = cudaGetErrorString(e);
        return e;
    }

    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    auto fail = [&](cudaError_t err) -> cudaError_t {
        result.ok = false;
        result.error_msg = cudaGetErrorString(err);
        log::error("[glm_real] FAIL (stage=" + std::string(g_stage) + "): " +
                   std::string(cudaGetErrorString(err)));
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        c.Free();
        return err;
    };

    // Wall-clock timing (stream-delta, GPU time only): t_start at entry,
    // t_ttft at the first generated token, t_stop after the final head copy.
    e = cudaEventCreate(&t_start);
    if (e == cudaSuccess) e = cudaEventCreate(&t_ttft);
    if (e == cudaSuccess) e = cudaEventCreate(&t_stop);
    if (e != cudaSuccess) return fail(e);
    cudaEventRecord(t_start, stream);

    const int hidden = pw.hidden;
    const auto shared_for = [&pw](int l) -> const PackedExpert * {
        return (l >= 0 && static_cast<size_t>(l) < pw.shared_host.size())
                   ? &pw.shared_host[static_cast<size_t>(l)]
                   : nullptr;
    };

    // Embed the prompt (all tokens) into the residual stream.
    g_stage = "embed";
    e = cudaMemcpyAsync(c.d_ids, prompt.data(),
                        static_cast<size_t>(P) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return fail(e);
    {
        const int blocks = static_cast<int>(
            (static_cast<size_t>(P) * hidden + kGlmLoopBlock - 1) /
            kGlmLoopBlock);
        GlmEmbedKernel<<<blocks, kGlmLoopBlock, 0, stream>>>(
            pw.d_embed, c.d_ids, c.d_cur, P, hidden);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(e);
    }

    // Prefill: forward each prompt token through all 78 layers (causal), storing
    // its K/V at slot = position.
    g_stage = "prefill";
    for (int pos = 0; pos < P; ++pos) {
        e = GlmCopyAsync(stream, c.d_cur + static_cast<size_t>(pos) * hidden,
                         c.d_hid, hidden);
        if (e != cudaSuccess) return fail(e);
        for (int l = 0; l < kGlmNumLayers; ++l) {
            if (l == 78) break;  // layer 78 is the MTP block, not in the main model
            snprintf(g_stage_buf, sizeof(g_stage_buf), "prefill-p%d-l%d", pos, l);
            g_stage = g_stage_buf;
            e = RunGlmRealLayer(c, l, pos, shared_for(l));
            if (e != cudaSuccess) return fail(e);
        }
        e = GlmCopyAsync(stream, c.d_hid,
                         c.d_cur + static_cast<size_t>(pos) * hidden, hidden);
        if (e != cudaSuccess) return fail(e);
    }

    // Decode: head on the last produced hidden, then forward the winner.
    int last_idx = P - 1;
    for (int step = 0; step < max_new; ++step) {
        g_stage = "head";
        e = GlmCopyAsync(stream, c.d_cur + static_cast<size_t>(last_idx) * hidden,
                         c.d_hid, hidden);
        if (e != cudaSuccess) return fail(e);
        e = RunGlmRmsNorm(stream, c.d_hid, pw.d_final_norm, c.d_hn, 1, hidden,
                          kGlmRmsEps);
        if (e != cudaSuccess) return fail(e);
        {
            const int blocks = (vocab + kGlmLoopBlock - 1) / kGlmLoopBlock;
            GlmHeadLogitsKernel<<<blocks, kGlmLoopBlock, 0, stream>>>(
                pw.d_head, c.d_hn, c.d_logits_head, vocab, hidden);
            e = cudaGetLastError();
            if (e != cudaSuccess) return fail(e);
        }
        e = sampler.Sample(stream, c.d_logits_head, vocab,
                           lsxcommon::sample::LogitType::kF32Rounded16,
                           &c.d_argmax->best, &c.d_argmax->maxv);
        if (e != cudaSuccess) return fail(e);
        e = cudaMemcpyAsync(c.h_argmax, c.d_argmax, sizeof(GlmArgmaxSlot),
                            cudaMemcpyDeviceToHost, stream);        if (e != cudaSuccess) return fail(e);
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return fail(e);
        const int best = c.h_argmax->best;
        result.tokens.push_back(best);
        if (step == 0) result.first_maxv = c.h_argmax->maxv;
        if (step == 0) cudaEventRecord(t_ttft, stream);
        if (best == kGlmEosIds[0] || best == kGlmEosIds[1] ||
            best == kGlmEosIds[2]) {
            result.eos_terminated = true;
            break;
        }

        g_stage = "decode";
        const int pos = last_idx + 1;
        e = cudaMemcpyAsync(c.d_ids, &best, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return fail(e);
        {
            const int blocks = (hidden + kGlmLoopBlock - 1) / kGlmLoopBlock;
            GlmEmbedKernel<<<blocks, kGlmLoopBlock, 0, stream>>>(
                pw.d_embed, c.d_ids, c.d_cur + static_cast<size_t>(pos) * hidden,
                1, hidden);
            e = cudaGetLastError();
            if (e != cudaSuccess) return fail(e);
        }
        e = GlmCopyAsync(stream, c.d_cur + static_cast<size_t>(pos) * hidden,
                         c.d_hid, hidden);
        if (e != cudaSuccess) return fail(e);
        for (int l = 0; l < kGlmNumLayers; ++l) {
            if (l == 78) break;
            e = RunGlmRealLayer(c, l, pos, shared_for(l));
            if (e != cudaSuccess) return fail(e);
        }
        e = GlmCopyAsync(stream, c.d_hid,
                         c.d_cur + static_cast<size_t>(pos) * hidden, hidden);
        if (e != cudaSuccess) return fail(e);
        last_idx = pos;
    }

    // Publish the raw [vocab] logits row of the last head forward.
    g_stage = "final-logits";
    result.final_logits.resize(static_cast<size_t>(vocab));
    e = cudaMemcpyAsync(result.final_logits.data(), c.d_logits_head,
                        static_cast<size_t>(vocab) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return fail(e);
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return fail(e);
    cudaEventRecord(t_stop, stream);
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return fail(e);
    float ms_total = 0.0f, ms_ttft = 0.0f;
    e = cudaEventElapsedTime(&ms_total, t_start, t_stop);
    if (e != cudaSuccess) return fail(e);
    e = cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    if (e != cudaSuccess) return fail(e);
    result.total_ms = ms_total;
    result.ttft_ms = ms_ttft;
    result.resident_bytes = c.bytes + pw.resident_bytes;
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    result.kv_slots_used = last_idx + 1;
    result.ok = true;
    c.Free();
    return cudaSuccess;
}

} // namespace glm
} // namespace lsxcommon
