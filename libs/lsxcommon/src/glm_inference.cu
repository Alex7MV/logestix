#include "lsxcommon/glm_inference.h"
#include "lsxcommon/log.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "glm_device_math.h"

namespace lsxcommon {
namespace glm {

namespace {

using device::GlmExpfGlibc;
using device::GlmRsqrtfGlibc;

constexpr float kGlmSwigluLimit = 10.0f;
constexpr int kGlmRmsBlock = 256;

// RMSNorm over a [rows, dim] FP16 matrix, one block per row. Bit-exact mirror
// of the host RmsNormFP32 reference: a SINGLE sequential fp32 scan over
// 0..dim-1 accumulates the square sum, and the output scale is
// GlmRsqrtfGlibc(acc/dim + eps) (= glibc rsqrtf = 1.0f/sqrtf) — no strided
// partials / butterfly reduction. The
// accumulation must stay a plain (non-FMA-contracted) mul+add sequence, which
// the target build enforces via -fmad=false (the host baseline, compiled at
// -O0 without FMA ISA, cannot contract either).
__global__ void GlmRmsNormKernel(const __half *__restrict__ x,
                                 const __half *__restrict__ w,
                                 __half *__restrict__ y,
                                 int dim, float eps)
{
    constexpr int kBlock = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_rms;
    if (tid == 0) {
        const __half *xr = x + row * dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float v = __half2float(xr[i]);
            acc += v * v;
        }
        s_rms = GlmRsqrtfGlibc(acc / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float rinv = s_rms;
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(x[row * dim + i]);
        y[row * dim + i] = __float2half(v * rinv * __half2float(w[i]));
    }
}

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
// layout the code previously produced. The accumulation must stay a plain
// (non-FMA-contracted) double mul+add sequence: nvcc's default -fmad=true
// would emit DFMA, which the -O0 host baseline cannot.
__global__ void GlmGemmFp64Kernel(const __half *__restrict__ w,
                                  const __half *__restrict__ x,
                                  __half *__restrict__ out,
                                  int m, int k, int n)
{
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (idx >= total) return;
    const int t = static_cast<int>(idx / m);
    const int i = static_cast<int>(idx - static_cast<int64_t>(t) * m);
    const __half *wr = w + static_cast<int64_t>(i) * k;
    const __half *xr = x + static_cast<int64_t>(t) * k;
    double acc = 0.0;
    for (int q = 0; q < k; ++q)
        acc += static_cast<double>(__half2float(wr[q])) *
               static_cast<double>(__half2float(xr[q]));
    out[idx] = __float2half(static_cast<float>(acc));
}

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
    GlmGenerationResult &result)
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
        if (l == 0) {
            const GlmLayerWeights &lw = pw.layers[0];
            block_rows = (lw.d_post_attn_norm && lw.d_mlp_gate &&
                          lw.d_mlp_up && lw.d_mlp_down)
                             ? 1
                             : P;  // a loaded dense branch collapses the block
        }
        if (l < kGlmNumLayers - 1) std::swap(cur, nxt);
    }

    // ── Greedy decode: argmax the layer-78 logits row, embed, re-forward. ──
    for (int step = 0; step < max_new; ++step) {
        g_stage = "decode-argmax";
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
    GlmGenerationResult &result)
{
    return RunGlmGenerationImpl(cublas_h, stream, pw, &experts, nullptr,
                                nullptr, nullptr, prompt, max_new, result);
}

cudaError_t RunGlmGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, GlmExpertDb &expert_db,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result)
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
                             d_pack, d_scale, prompt, max_new, result);    cudaFree(d_scale);
    cudaFree(d_pack);
    return e;
}

} // namespace glm
} // namespace lsxcommon
