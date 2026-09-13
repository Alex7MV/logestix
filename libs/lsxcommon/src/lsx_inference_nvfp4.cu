// Model-neutral NVFP4 inference layer (see lsx_inference_nvfp4.h).
// Extracted from qwen_inference.cu (model-independence split, 2026-09):
// the GEMM kernels, the seq==1 streaming GEMV and the kernel-routing
// toggles serve several model families and live under lsxcommon::lsx_nvfp4.
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <atomic>
#include <cstdlib>
#include <memory>

#include "nvfp4_codec.h"
#include "nvfp4_gemv_math.cuh"
#include "lsxcommon/lsx_tc_gemm.h"

namespace lsxcommon {
namespace lsx_nvfp4 {

static std::atomic<bool> g_lsx_tc_fast_math{false};
static std::atomic<bool> g_lsx_seq1_gemv{false};

void SetLsxTcFastMath(bool on)
{
    g_lsx_tc_fast_math.store(on, std::memory_order_relaxed);
}

bool LsxTcFastMath()
{
    return g_lsx_tc_fast_math.load(std::memory_order_relaxed);
}

void SetLsxSeq1Gemv(bool on)
{
    g_lsx_seq1_gemv.store(on, std::memory_order_relaxed);
}

bool LsxSeq1Gemv()
{
    return g_lsx_seq1_gemv.load(std::memory_order_relaxed);
}

static std::atomic<int64_t> g_lsx_gemv_max_bytes{int64_t{32} << 20};
static std::atomic<bool> g_lsx_gemv_large{false};
static std::atomic<bool> g_lsx_dual_stream{true};

void SetLsxGemmMaxMib(int mib)
{
    if (mib > 0)
        g_lsx_gemv_max_bytes.store(int64_t{mib} << 20,
                                   std::memory_order_relaxed);
}

int LsxGemmMaxMib()
{
    return static_cast<int>(g_lsx_gemv_max_bytes.load(
               std::memory_order_relaxed) >> 20);
}

void SetLsxGemmLarge(bool on)
{
    g_lsx_gemv_large.store(on, std::memory_order_relaxed);
}

bool LsxGemmLarge()
{
    return g_lsx_gemv_large.load(std::memory_order_relaxed);
}

int LsxGemmLargeEnvState()
{
    static const int s = [] {
        const char *v = std::getenv("LSX_GEMV_LARGE");
        return v ? (v[0] == '0' ? -1 : 1) : 0;
    }();
    return s;
}

bool LsxGemmLargeActive()
{
    const int env = LsxGemmLargeEnvState();
    return env == 1 ||
           (env == 0 && g_lsx_gemv_large.load(std::memory_order_relaxed));
}

void SetLsxDualStreamStaging(bool on)
{
    g_lsx_dual_stream.store(on, std::memory_order_relaxed);
}

bool LsxDualStreamStaging()
{
    return g_lsx_dual_stream.load(std::memory_order_relaxed);
}

namespace {

// Shared NVFP4 decode primitives (nvfp4_codec.h).
using gpu::nvfp4::E2M1Value;
using gpu::nvfp4::E4M3ScaleToFloat;

__global__ void lsx_nvfp4_gemm_kernel(
    const __half *__restrict__ x, const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float global_scale, __half *__restrict__ y, int seq, int K, int N, bool acc_in)
{
    const int64_t total = (int64_t)seq * N;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int n = (int)(idx % N);
    const int t = (int)(idx / N);

    const __half *xrow = x + (int64_t)t * K;

    float acc = acc_in ? __half2float(y[idx]) : 0.0f;
    for (int k = 0; k < K; ++k) {
        // NVFP4 layout: element e lives at byte e>>1, low nibble if even,
        // high nibble if odd; scale is per-16 elements (e>>4).
        const int64_t e = (int64_t)n * K + k;
        const uint8_t byte = packed[e >> 1];
        const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
        const float scl = E4M3ScaleToFloat(scales[e >> 4]) * global_scale;
        acc += __half2float(xrow[k]) * E2M1Value(nib) * scl;
    }
    y[idx] = __float2half(acc);
}

// Tile NVFP4 dense GEMM: block computes a [BM x BN*VPT] output tile for one
// token tile. X rows and the NVFP4-dequantized weight rows are staged in
// shared memory once per K-block; E8M0 scales are decoded once per 16
// elements. Threads == BM * BN; each thread owns VPT output columns
// (n = n0 + v*BN + bn), i.e. VPT independent accumulator chains (audit F-01:
// the single-chain variant is latency-bound at ~18 GB/s effective BW with
// zero register pressure). Bit-exact by construction: every output element
// keeps the identical BK-chunked sequential accumulation order regardless of
// VPT.
template <int BM, int BN, int BK, int VPT>
__global__ void lsx_nvfp4_gemm_tile_kernel(
    const __half *__restrict__ x, const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float global_scale, __half *__restrict__ y, int seq, int K, int N, bool acc_in)
{
    static_assert(BM * BN == 256, "one thread per output element row/col pair");

    // Audit F-01 (measured): reading kE2M1Table straight from __constant__
    // serializes 16-way inside the staging loop because adjacent threads
    // decode different nibbles, and constant-cache broadcast only covers
    // warp-uniform addresses. Mirror the table into shared memory once per
    // block: identical float values, so results stay bit-exact.
    __shared__ float sE2M1[16];
    __shared__ __half sX[BM][BK];
    __shared__ __half sW[BN * VPT][BK];
    if (threadIdx.x < 16) sE2M1[threadIdx.x] = E2M1Value(threadIdx.x);
    __syncthreads();
    const int m0 = blockIdx.x * BM;      // token tile
    const int n0 = blockIdx.y * BN * VPT; // output-column tile
    const int bi = threadIdx.x;          // 0 .. BM*BN-1
    const int bm = bi / BN;              // token index within tile
    const int bn = bi % BN;              // base output-column index within tile

    const int t = m0 + bm;

    float acc[VPT];
#pragma unroll
    for (int v = 0; v < VPT; ++v) {
        const int n = n0 + v * BN + bn;
        acc[v] = (acc_in && t < seq && n < N)
                     ? __half2float(y[(int64_t)t * N + n])
                     : 0.0f;
    }

    // Audit F-01 (measured): the chunk loop is latency-serialized — each
    // iteration's global loads only start after the previous iteration's FMAs
    // and barriers complete (~3.7 us/chunk x K/BK chunks ~= observed kernel
    // time). Prefetch the next chunk's RAW BYTES into registers while the
    // current chunk computes; values and accumulation order are unchanged, so
    // results stay bit-exact.
    constexpr int TPB = BM * BN;
    constexpr int W_PER_T = (BN * VPT * BK) / TPB;
    constexpr int X_PER_T = (BM * BK) / TPB;

    uint8_t rW_byte[W_PER_T], rW_scale[W_PER_T];
    uint16_t rX_bits[X_PER_T];

    auto load_chunk = [&](int kb) {
#pragma unroll
        for (int j = 0; j < W_PER_T; ++j) {
            const int i = bi + j * TPB;
            const int w_bn = i / BK, kk = i % BK;
            const int n = n0 + w_bn;
            rW_byte[j] = 0;
            rW_scale[j] = 0;
            if (n < N && kb + kk < K) {
                const int64_t e = (int64_t)n * K + (kb + kk);
                rW_byte[j] = packed[e >> 1];
                rW_scale[j] = scales[e >> 4];
            }
        }
#pragma unroll
        for (int j = 0; j < X_PER_T; ++j) {
            const int i = bi + j * TPB;
            const int x_bm = i / BK, kk = i % BK;
            const int tt = m0 + x_bm;
            const __half hv = (tt < seq && kb + kk < K)
                                  ? x[(int64_t)tt * K + (kb + kk)]
                                  : __float2half(0.0f);
            rX_bits[j] = __half_as_ushort(hv);
        }
    };
    auto store_chunk = [&](int kb) {
#pragma unroll
        for (int j = 0; j < W_PER_T; ++j) {
            const int i = bi + j * TPB;
            const int w_bn = i / BK, kk = i % BK;
            const int n = n0 + w_bn;
            float wv = 0.0f;
            if (n < N && kb + kk < K) {
                const uint8_t byte = rW_byte[j];
                const uint8_t nib = ((kb + kk) & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scl =
                    E4M3ScaleToFloat(rW_scale[j]) * global_scale;
                wv = sE2M1[nib] * scl;
            }
            sW[w_bn][kk] = __float2half(wv);
        }
#pragma unroll
        for (int j = 0; j < X_PER_T; ++j) {
            const int i = bi + j * TPB;
            sX[i / BK][i % BK] = __ushort_as_half(rX_bits[j]);
        }
    };

    load_chunk(0);
    store_chunk(0);
    __syncthreads();

    for (int kb = 0; kb < K; kb += BK) {
        const bool has_next = (kb + BK < K);
        if (has_next) load_chunk(kb + BK);   // global latency overlaps compute

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float xv = __half2float(sX[bm][kk]);
#pragma unroll
            for (int v = 0; v < VPT; ++v)
                acc[v] += xv * __half2float(sW[v * BN + bn][kk]);
        }

        __syncthreads();                     // all threads done reading smem
        if (has_next) store_chunk(kb + BK);  // refill the same buffers
        __syncthreads();
    }

#pragma unroll
    for (int v = 0; v < VPT; ++v) {
        const int n = n0 + v * BN + bn;
        if (t < seq && n < N)
            y[(int64_t)t * N + n] = __float2half(acc[v]);
    }
}

// ── seq==1 weight-streaming NVFP4 GEMV (round-18 decode lever) ──────────────
// At seq==1 every NVFP4 GEMM degenerates into a batch of dot products over
// rows of W. Both the 16x16x32 tile kernel and the TC-MMA decode kernel
// stage weight slabs through shared memory; with a single activation row
// that staging throttles effective bandwidth to ~30 GB/s against a
// multi-TB/s DRAM ceiling. This kernel streams packed E2M1 nibbles straight
// from global memory with coalesced uint4 loads (32 elements per lane load)
// -- no shared-memory slab, no MMA, no cp.async pipeline.
//
// K parallelism: one warp owns a [1024/KP .. ]-element K-slice of one output
// row; per-row partials meet in shared memory and are summed in slice order
// by lane 0 (fixed partitioning + fixed order => run-to-run bit-stable).
// Warp reduction uses __shfl_down_sync only.
//
// Host contract: seq==1 fast-math dispatch only; requires K % 32 == 0
// (16 B alignment of every packed row) and K % 16 == 0 so scale blocks never
// straddle rows. Numeric class: FP32 accumulate, k-order free -- tolerance
// vs the scalar path exactly like the TC fast-math routes (see
// docs/NVIDIA_DOCS_NVFP4_MEMORY.md §1 decode semantics).
template <int KP>
__global__ void lsx_nvfp4_gemv_kernel(
    const __half *__restrict__ x,
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ scales,
    float global_scale, float global_scale_up, int n_half,
    __half *__restrict__ y,
    int K, int N, bool acc_in)
{
    constexpr int WARPS = 8;
    constexpr int ROWS = WARPS / KP;
    static_assert(ROWS >= 1 && KP <= WARPS);
    __shared__ float sp[WARPS];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // No early return below: the __syncthreads() in the partial-sum merge
    // must be reached uniformly by all threads of the block.
    float acc = 0.0f;
    const int row = blockIdx.x * ROWS + (warp / KP);
    // Per-row global scale: gate/up pair packs scale rows [0, n_half) with
    // global_scale and rows [n_half, N) with global_scale_up; plain launches
    // pass n_half == N so every row uses global_scale (unchanged behavior).
    const float row_scale =
        (row < n_half) ? global_scale : global_scale_up;
    if (row < N) {
        const int k_half = K >> 1;
        const uint8_t *wrow =
            packed + static_cast<int64_t>(row) * k_half;
        const uint8_t *srow =
            scales + static_cast<int64_t>(row) * (K >> 4);
        const int kb = (K / KP) * (warp % KP);       // slice base element
        const uint4 *wv = reinterpret_cast<const uint4 *>(wrow + (kb >> 1));
        const int chunks = (K / KP) >> 5;            // 32-element units
        // Two independent accumulator chains (ILP x2), merged in fixed
        // order -- run-to-run deterministic. NOTE: keep this the plain
        // 2-chain loop. An ILP4 strata variant (4 accs, 64-chunk windows)
        // measured ~7% WORSE end-to-end on MoE models (qwen3.5/3.6, gemma:
        // 134.8 -> 125 t/s, bisected 2026-09-13): expert rows are short
        // (chunks ~ 16 at K=512), where the extra loop/break machinery
        // costs more than the ILP buys. Large deep-K dense shapes use the
        // dedicated warp-per-row kernel instead (K >= 4096 gate).
        float acc0 = 0.0f, acc1 = 0.0f;
        const bool x_vec =
            (reinterpret_cast<uintptr_t>(x) & 3u) == 0;
        for (int c = lane; c < chunks; c += 32) {
            const uint4 pw = wv[c];
            const uint8_t *pb = reinterpret_cast<const uint8_t *>(&pw);
            // One 32-element chunk spans two per-16 scale blocks:
            // elements 0..15 of the chunk -> srow[sb], 16..31 -> sb+1.
            // Byte j of the packed word holds elements 2j (low nibble)
            // and 2j+1 (high nibble).
            const int sb = (kb >> 4) + 2 * c;
            const float sc0 = E4M3ScaleToFloat(srow[sb]);
            const float sc1 = E4M3ScaleToFloat(srow[sb + 1]);
            const int xb = kb + c * 32;
            const __half2 *xh = x_vec
                ? reinterpret_cast<const __half2 *>(x + xb) : nullptr;
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const bool first_block = b < 8;
                const float scl = first_block ? sc0 : sc1;
                float xv0, xv1;
                if (x_vec) {
                    const __half2 pair = xh[b];
                    xv0 = __half2float(pair.x);
                    xv1 = __half2float(pair.y);
                } else {
                    xv0 = __half2float(x[xb + 2 * b]);
                    xv1 = __half2float(x[xb + 2 * b + 1]);
                }
                acc0 += xv0 * E2M1Value(pb[b] & 0xF) * scl;
                acc1 += xv1 * E2M1Value(pb[b] >> 4) * scl;
            }
        }
        acc = acc0 + acc1;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) sp[warp] = acc;
    __syncthreads();
    // Lane 0 of slice 0 sums its row's slices in fixed order.
    if (lane == 0 && (warp % KP) == 0) {
        const int wbase = (warp / KP) * KP;
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < KP; ++j) t += sp[wbase + j];
        const int r = blockIdx.x * ROWS + (warp / KP);
        if (r < N) {
            const float v = t * row_scale;
            y[r] = acc_in ? __float2half(__half2float(y[r]) + v)
                          : __float2half(v);
        }
    }
}

// seq==1 warp-per-row GEMV (27B decode round): one warp owns a whole
// output row; 8 warps per block. No shared-memory partial merge and no
// __syncthreads -- the per-row reduction is warp-local (__shfl_down_sync),
// which keeps the whole row's 16B loads in flight per warp. Isolated probe
// (bwprobe5, RTX 5090): ~894 GB/s on the 27B gate/up shape vs ~428 GB/s
// for the sliced-kp kernel. Numeric class: FP32 accumulate, fixed merge
// order (two half-K chains, then pairs) -- run-to-run deterministic.
// Contract: K % 64 == 0 (the lane stratum walks 64-element windows).
__global__ void __launch_bounds__(256)
lsx_nvfp4_gemv_warprow_kernel(
    const __half *__restrict__ x,
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ scales,
    float global_scale, float global_scale_up, int n_half,
    __half *__restrict__ y,
    int K, int N, bool acc_in)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const float row_scale =
        (row < n_half) ? global_scale : global_scale_up;
    const int k_half = K >> 1;
    const uint8_t *wrow = packed + static_cast<int64_t>(row) * k_half;
    const uint8_t *srow = scales + static_cast<int64_t>(row) * (K >> 4);
    const uint4 *wv = reinterpret_cast<const uint4 *>(wrow);
    const int chunks = K >> 5;               // 32-element 16B units per row
    const int half_chunks = chunks >> 1;     // per-lane stratum (chunks/2)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const bool x_vec = (reinterpret_cast<uintptr_t>(x) & 3u) == 0;
    // Lane stratum: lane covers chunk strata [lane, lane+32) of the first
    // half and the mirrored strata of the second half. Fixed merge order.
    const int lim0 = half_chunks;
    const int lim1 = chunks;
    for (int c = lane; c < lim0; c += 32) {
        const uint4 pw = wv[c];
        const uint8_t *pb = reinterpret_cast<const uint8_t *>(&pw);
        const int sb = 2 * c;
        const float sc0 = E4M3ScaleToFloat(srow[sb]);
        const float sc1 = E4M3ScaleToFloat(srow[sb + 1]);
        const int xb = c * 32;
        const __half2 *xh = x_vec
            ? reinterpret_cast<const __half2 *>(x + xb) : nullptr;
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float scl = (b < 8) ? sc0 : sc1;
            float xv0, xv1;
            if (x_vec) {
                const __half2 pair = xh[b];
                xv0 = __half2float(pair.x);
                xv1 = __half2float(pair.y);
            } else {
                xv0 = __half2float(x[xb + 2 * b]);
                xv1 = __half2float(x[xb + 2 * b + 1]);
            }
            acc0 += xv0 * E2M1Value(pb[b] & 0xF) * scl;
            acc1 += xv1 * E2M1Value(pb[b] >> 4) * scl;
        }
    }
    for (int c = lim0 + lane; c < lim1; c += 32) {
        const uint4 pw = wv[c];
        const uint8_t *pb = reinterpret_cast<const uint8_t *>(&pw);
        const int sb = 2 * c;
        const float sc0 = E4M3ScaleToFloat(srow[sb]);
        const float sc1 = E4M3ScaleToFloat(srow[sb + 1]);
        const int xb = c * 32;
        const __half2 *xh = x_vec
            ? reinterpret_cast<const __half2 *>(x + xb) : nullptr;
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float scl = (b < 8) ? sc0 : sc1;
            float xv0, xv1;
            if (x_vec) {
                const __half2 pair = xh[b];
                xv0 = __half2float(pair.x);
                xv1 = __half2float(pair.y);
            } else {
                xv0 = __half2float(x[xb + 2 * b]);
                xv1 = __half2float(x[xb + 2 * b + 1]);
            }
            acc2 += xv0 * E2M1Value(pb[b] & 0xF) * scl;
            acc3 += xv1 * E2M1Value(pb[b] >> 4) * scl;
        }
    }
    float acc = (acc0 + acc2) + (acc1 + acc3);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        const float v = acc * row_scale;
        y[row] = acc_in ? __float2half(__half2float(y[row]) + v)
                        : __float2half(v);
    }
}

cudaError_t LaunchLsxNvfp4GemmSeq1GemmV(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int K, int N, bool acc_in)
{
    if (!x || !packed || !scales || !y || K <= 0 || N <= 0) return cudaErrorInvalidValue;
    if ((K & 31) != 0) return cudaErrorInvalidValue;
    // 27B decode profile (2026-09-13, RTX 5090): the big dense shapes
    // (gate/up K=5120 N=34816, down K=17408 N=5120, head K=5120 N=248320)
    // are DRAM-bound; the sliced-kp kernel reaches only ~428 GB/s there,
    // while an isolated warp-per-row probe with identical decode math
    // sustains ~894 GB/s. Route large-N shapes to the warp-per-row kernel:
    // one warp per output row (8 rows/block), no shared-memory partial
    // merge, no __syncthreads. Requires K % 64 == 0 (chunks/2 per lane is
    // integral) and N >= 2048 (block count covers the SM array); smaller
    // shapes keep the sliced kernel (its kp heuristic is better tuned for
    // short rows).
    if ((K & 63) == 0 && N >= 2048 &&
        g_lsx_tc_fast_math.load(std::memory_order_relaxed)) {
        constexpr int WARPS = 8;   // rows per block (probe: 8 > 4 or 16)
        const int grid = (N + WARPS - 1) / WARPS;
        lsx_nvfp4_gemv_warprow_kernel<<<grid, WARPS * 32, 0, stream>>>(
            x, packed, scales, global_scale, global_scale, N, y, K, N,
            acc_in);
        return cudaGetLastError();
    }
    // Slice width floor 1024 elements keeps lanes busy (>=32 chunks/slice).
    // Raise KP while enough warps remain resident to cover the SM array.
    // 27B decode note (2026-09-13 profile): the big dense shapes
    // (gate/up K=5120 N=34816, down K=17408 N=5120, head K=5120 N=248320)
    // are DRAM-bound and saturate below 24% of HBM peak at kp=1. Deeper
    // per-lane ILP (2 chains x 2 unrolled chunk strata) lifts the streaming
    // rate; kp=4 is safe for any (K,N) passing the floor check because the
    // chain merge stays a fixed in-slice order.
    int kp = 1;
    for (int cand : {8, 4, 2}) {
        if ((K / cand) >= 1024 &&
            (static_cast<int64_t>(N) * cand) >= 1024) { kp = cand; break; }
    }
    constexpr int WARPS = 8;
    auto launch = [&](auto kp_tag) -> cudaError_t {
        constexpr int KPc = decltype(kp_tag)::value;
        constexpr int ROWS = WARPS / KPc;
        const dim3 grid((N + ROWS - 1) / ROWS);
        lsx_nvfp4_gemv_kernel<KPc><<<grid, WARPS * 32, 0, stream>>>(
            x, packed, scales, global_scale, global_scale, N, y, K, N, acc_in);
        return cudaGetLastError();
    };
    switch (kp) {
    case 8:  return launch(std::integral_constant<int, 8>{});
    case 4:  return launch(std::integral_constant<int, 4>{});
    case 2:  return launch(std::integral_constant<int, 2>{});
    default: return launch(std::integral_constant<int, 1>{});
    }
}

// Fused gate/up streaming GEMV: W is the contiguous [2*n_half, K] NVFP4
// pack ([n_half, K] gate rows followed by [n_half, K] up rows; the E4M3
// scale rows follow the same split). Rows < n_half scale by gs_gate,
// rows >= n_half by gs_up. KP is picked from n_half (not 2*n_half) so the
// per-row k-slicing matches the plain single GEMV bit-for-bit.
cudaError_t LaunchLsxNvfp4GemmSeq1GateUpPair(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales,
    float gs_gate, float gs_up, __half *y, int K, int n_half, bool acc_in)
{
    if (!x || !packed || !scales || !y || K <= 0 || n_half <= 0)
        return cudaErrorInvalidValue;
    if ((K & 31) != 0) return cudaErrorInvalidValue;
    int kp = 1;
    for (int cand : {8, 4, 2}) {
        if ((K / cand) >= 1024 &&
            (static_cast<int64_t>(n_half) * cand) >= 1024) { kp = cand; break; }
    }
    constexpr int WARPS = 8;
    auto launch = [&](auto kp_tag) -> cudaError_t {
        constexpr int KPc = decltype(kp_tag)::value;
        constexpr int ROWS = WARPS / KPc;
        const int n_total = 2 * n_half;
        const dim3 grid((n_total + ROWS - 1) / ROWS);
        lsx_nvfp4_gemv_kernel<KPc><<<grid, WARPS * 32, 0, stream>>>(
            x, packed, scales, gs_gate, gs_up, n_half, y, K, n_total, acc_in);
        return cudaGetLastError();
    };
    switch (kp) {
    case 8:  return launch(std::integral_constant<int, 8>{});
    case 4:  return launch(std::integral_constant<int, 4>{});
    case 2:  return launch(std::integral_constant<int, 2>{});
    default: return launch(std::integral_constant<int, 1>{});
    }
}
} // namespace

cudaError_t RunLsxNvfp4Gemm(    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N, bool acc /* = false */,
    const float *seg_wgs_dev, const int *seg_bounds_dev)
{
    if (!x || !packed || !scales || !y || seq <= 0 || K <= 0 || N <= 0)
        return cudaErrorInvalidValue;
    // Round-18 decode lever: opt-in via SetQwenSeq1Gemv(true). At seq==1
    // the weight-streaming GEMV (CUDA-core direct global reads) beats both
    // the TC tile kernel and the TC-MMA decode kernel on small weight
    // tensors -- MoE expert shapes (~0.6 MB) see a 3.6-5x kernel-time cut.
    // Eligibility is size-bounded; the bound is a perf-policy knob
    // (SetLsxGemmMaxMib, default 32 MiB; the LSX_GEMV_MAX_MIB env read moved
    // to perf::ResolvePerfKnobs): flash KDA q/k/v/o (18.9 MB) and
    // dense-MLP gate/up (28.1 MB) measure 2.5-3x faster on the streaming
    // GEMV than on TC M=64 tiles at seq==1 (profiled 2026-09, RTX 5090;
    // 32 MiB default picked by A/B: 15.6 -> 17.1 t/s, 64 MiB regresses to
    // 16.4 t/s on the LM head).
    // LSX_GEMV_HEAD=1 additionally opts the LM-head class
    // (K >= 1024 && N >= 16384) into the streaming GEMV regardless of
    // tensor size: micro data puts it at parity-to-slight win vs TC on
    // DRAM-bound shapes, and the flat-row grid saturates the SM array
    // where the MMA path underfills at M == 1.
    static const bool kGemvHead =
        std::getenv("LSX_GEMV_HEAD") != nullptr;
    // 27B-class dense decode (2026-09-13 profile): shapes with deep K and
    // wide N (MLP gate/up K=5120 N=34816, down K=17408 N=5120, head
    // K=5120 N=248320) are DRAM-bound; the warp-per-row streaming GEMV
    // sustains ~2x the effective bandwidth of the TC M=64 tile path at
    // seq==1 (probe: 894 vs ~428 GB/s).
    // K floor 4096 is load-bearing: shallow-K shapes lose on warp-per-row
    // (qwen3.5 lm_head K=2048 N=151936 / q_proj K=2048 N=4096 measured
    // 139 -> 130 tok/s with the route on; A/B bisected 2026-09-13) --
    // short rows cannot keep the per-warp chunk strata busy.
    // Controlled by the perf policy (gemv_large_default; LSX_GEMV_LARGE env
    // kept as a tri-state override: 1=force on, 0=force off, unset=policy):
    // routes ONLY large-N plain shapes; small-N / MoE shapes and the
    // segmented fused qkv|z|a|b path are untouched.
    const bool gemv_large =
        LsxGemmLargeActive() &&
        seq == 1 && !seg_wgs_dev && !seg_bounds_dev &&
        K >= 4096 && N >= 4096;
    const bool gemv_small =
        g_lsx_tc_fast_math.load(std::memory_order_relaxed) &&
        LsxSeq1Gemv() &&
        (int64_t)(K / 2) * N + (int64_t)((K + 15) / 16) * N <=
            g_lsx_gemv_max_bytes.load(std::memory_order_relaxed);
    const bool gemv_head =
        kGemvHead && K >= 1024 && N >= 16384;
    if (((gemv_small && seq == 1) || (gemv_head && seq == 1) ||
         gemv_large) &&
        K >= 256 && !seg_wgs_dev && !seg_bounds_dev) {
        const cudaError_t gv = LaunchLsxNvfp4GemmSeq1GemmV(
            stream, x, packed, scales, global_scale, y, K, N, acc);
        // InvalidValue = shape contract mismatch -> keep falling back.
        if (gv != cudaErrorNotSupported && gv != cudaErrorInvalidValue)
            return gv;
    }
    // Blackwell tensor-core fast path (audit Phase 3 / F-01): W4A4 mxf4nvf4
    // MMA. Tolerance-class numerics -- only reachable when the runtime
    // fast-math toggle is set. Falls back to the bit-exact scalar path on
    // unsupported devices/shapes (NotSupported / InvalidValue).
    if (g_lsx_tc_fast_math.load(std::memory_order_relaxed) &&
        (K & 15) == 0 && N >= 8 && K >= 64) {
        const cudaError_t tc = lsx_tc::RunLsxTcNvfp4Gemm(
            stream, x, packed, scales, global_scale, y, seq, K, N, acc,
            seg_wgs_dev, seg_bounds_dev);
        if (tc != cudaErrorNotSupported && tc != cudaErrorInvalidValue)
            return tc;
    }
    if (N < 8 || K < 16) {
        const int threads = 256;
        const int64_t total = (int64_t)seq * N;
        const int blocks = (int)((total + threads - 1) / threads);
        lsx_nvfp4_gemm_kernel<<<blocks, threads, 0, stream>>>(
            x, packed, scales, global_scale, y, seq, K, N, acc);
        return cudaGetLastError();
    }
    // Audit F-01: VPT=2 gives each thread two independent accumulator chains
    // (ILP x2) while preserving the per-element BK-chunked accumulation order
    // — bit-exact vs the VPT=1 reference pinned by test_qwen_nvfp4_tile_gemm.
    constexpr int BM = 16, BN = 16, BK = 32, VPT = 2;
    const int gx = (int)((seq + BM - 1) / BM);
    const int gy = (int)((N + BN * VPT - 1) / (BN * VPT));
    lsx_nvfp4_gemm_tile_kernel<BM, BN, BK, VPT><<<dim3(gx, gy), BM * BN, 0, stream>>>(
        x, packed, scales, global_scale, y, seq, K, N, acc);
    return cudaGetLastError();
}

cudaError_t LaunchNvfp4Gemm(const LsxNvfp4Weight &w, const __half *x, int seq,
                            __half *y, cudaStream_t stream)
{
    if (!w.d_packed || w.rows <= 0 || w.cols <= 0) return cudaErrorInvalidValue;
    return RunLsxNvfp4Gemm(stream, x, w.d_packed, w.d_scales,
                            w.global_scale, y, seq, w.cols, w.rows, /*acc=*/false);
}

// seq==1 decode-path weight-streaming GEMV entry (external linkage so tests
// can call it directly). Requires K % 32 == 0 (16 B row alignment); the
// k-partition heuristic lives in the launch helper.
cudaError_t RunLsxNvfp4GemmSeq1GemmV(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int K, int N, bool acc)
{
    return LaunchLsxNvfp4GemmSeq1GemmV(stream, x, packed, scales,
                                        global_scale, y, K, N, acc);
}

cudaError_t RunLsxNvfp4GemmGateUpPair(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float gs_gate,
    float gs_up, __half *y, int K, int n_half, bool acc)
{
    if (!x || !packed || !scales || !y || K <= 0 || n_half <= 0)
        return cudaErrorInvalidValue;
    // Same size-bounded eligibility as the plain streaming GEMV, doubled
    // for the two halves this launch covers.
    const int64_t pair_bytes =
        2 * ((int64_t)(K / 2) * n_half + (int64_t)((K + 15) / 16) * n_half);
    const bool gemv =
        g_lsx_tc_fast_math.load(std::memory_order_relaxed) && LsxSeq1Gemv() &&
        pair_bytes <= g_lsx_gemv_max_bytes.load(std::memory_order_relaxed);
    if (gemv && K >= 256) {
        const cudaError_t gv = LaunchLsxNvfp4GemmSeq1GateUpPair(
            stream, x, packed, scales, gs_gate, gs_up, y, K, n_half, acc);
        // InvalidValue = shape contract mismatch -> keep falling back.
        if (gv != cudaErrorNotSupported && gv != cudaErrorInvalidValue)
            return gv;
    }
    // Bit-exact fallback: two independent plain GEMVs over the sub-ranges.
    const int64_t pack_half = (int64_t)K * n_half / 2;
    const int64_t scale_half = (int64_t)K * n_half / 16;
    cudaError_t e = RunLsxNvfp4Gemm(stream, x, packed, scales, gs_gate, y, 1,
                                    K, n_half, acc);
    if (e != cudaSuccess) return e;
    return RunLsxNvfp4Gemm(stream, x, packed + pack_half, scales + scale_half,
                           gs_up, y + n_half, 1, K, n_half, acc);
}

} // namespace lsx_nvfp4
} // namespace lsxcommon
