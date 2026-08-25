// qwen_tc_gemm.cu - Blackwell tensor-core NVFP4 W4A4 fast path (audit Phase 3,
// F-01). See qwen_tc_gemm.h for the compilation-target constraint.
//
// Contents:
//  * ActQuantKernel + helpers: FP16 -> NVFP4 activation quantization
//    (per-16 pow2-E4M3 scale + E2M1 nibble pairs, repo-codec-compatible).
//  * TcGemmKernel: 64x64 tile GEMM on warp-level mxf4nvf4 block-scaled MMA.
//    Raw packed K64-slabs are staged in shared memory; fragments are built
//    from smem; the hardware MMA performs all decoding. Weights are consumed
//    in the EXISTING repo layout ([N,K] nibbles + [N,K/16] pow2-E4M3 scales).
#include "lsxcommon/qwen_tc_gemm.h"

#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "nvfp4_codec.h"
#include "nvfp4_scale.h"
#include "lsx_kernels_quant.cuh"

namespace lsxcommon {
namespace qwen {
namespace tc {

namespace {

// Block-scale and E2M1 encode primitives are the shared canonical codecs:
// lsxcommon::nvfp4::FloatToE4m3P2 (nvfp4_scale.h, ceil pow2-E4M3) and
// lsxcommon::kernels::QuantizeE2m1TieUp (lsx_kernels_quant.cuh).

__global__ void ActQuantKernel(const __half *__restrict__ x, int64_t n,
                               uint8_t *__restrict__ packed,
                               uint8_t *__restrict__ scales)
{
    const int64_t blk = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t base = blk * 16;
    if (base >= n) return;
    const int len =
        static_cast<int>(min(static_cast<long long>(16),
                             static_cast<long long>(n - base)));

    float absmax = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float v = fabsf(__half2float(x[base + i]));
        absmax = fmaxf(absmax, v);
    }

    // scale = 2^ceil(log2(absmax/6)); quantized magnitude in [~0.75, 6].
    // Canonical clamp e4 in [1,14] (nvfp4_scale.h): saturates at 6*2^7=768,
    // i.e. fp16 activations above ~768 clip one octave earlier than the
    // historical frexp encode — required canonical parity, do not "fix".
    const uint8_t sb =
        lsxcommon::nvfp4::FloatToE4m3P2(absmax * (1.0f / 6.0f));
    scales[blk] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;

    for (int i = 0; i < len; i += 2) {
        const uint8_t lo =
            kernels::QuantizeE2m1TieUp(__half2float(x[base + i]) * inv);
        const uint8_t hi =
            (i + 1 < len)
                ? kernels::QuantizeE2m1TieUp(__half2float(x[base + i + 1]) *
                                             inv)
                : 0;
        packed[(base + i) / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

constexpr int kTileM = 64;
constexpr int kTileN = 64;

// ---- cp.async primitives (sm_80+, supported on sm_120a) --------------------
__device__ __forceinline__ void cp_async16(void *smem_dst, const void *gmem)
{
    const unsigned s =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

// 16-byte async copy with zero-fill predicate (src_size=0 -> dst zeroed).
__device__ __forceinline__ void cp_async16(void *smem_dst,
                                           const void *gmem_src, bool valid)
{
    const unsigned saddr = static_cast<unsigned>(
        __cvta_generic_to_shared(smem_dst));
    const int src_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
        saddr),
                 "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_wait3()
{
    asm volatile("cp.async.wait_group 3;\n");
}

__device__ __forceinline__ void cp_async_wait2()
{
    asm volatile("cp.async.wait_group 2;\n");
}

__device__ __forceinline__ void cp_async_wait1()
{
    asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ void cp_async_wait0()
{
    asm volatile("cp.async.wait_group 0;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Decode-specialized GEMM (seq == 1, K % 64 == 0, 16B-aligned buffers).
//
// At seq==1 the whole op is a weight-streaming GEMV batch: the quantized
// activation row (2 KB packed + scales) is static across the K loop and is
// loaded into shared memory exactly once; ONLY the weight slabs stream, one
// 64-column x 64-k chunk at a time, double-buffered through cp.async so the
// next chunk's global loads overlap the current chunk's MMAs. No syncthreads
// pair per chunk beyond the consumer fence -- the pipeline keeps ~one chunk
// of load latency in flight, which is what separates ~100 GB/s naive staging
// from the >300 GB/s needed for the 70-80 tok/s target (audit F-01).
constexpr int kDecCols = 128;             // n-columns per block (8 warps x 4 atoms x 8)
constexpr int kDecBufs = 4;               // cp.async pipeline depth (chunks in flight + 1)
constexpr int kDecChunkBytes = 32;        // packed bytes per column per k64

// ── B3: TMA weight-map cache ────────────────────────────────────────────────
// cuTensorMapEncodeTiled costs ~1-2us host-side and decode GEMMs repeat with
// identical (ptr,K,N) across tokens -> encode once, reuse from a bounded map.
namespace {

struct WeightMapKey {
    const void *ptr;
    int k_half;
    int n;
    bool operator==(const WeightMapKey &o) const {
        return ptr == o.ptr && k_half == o.k_half && n == o.n;
    }
};

constexpr int kMaxCachedMaps = 512;
std::mutex g_tma_mu;
std::vector<std::pair<WeightMapKey, CUtensorMap>> g_tma_cache;

cudaError_t GetWeightTmaMap(const uint8_t *w_packed, int N, int k_half,
                            CUtensorMap *out)
{
    const WeightMapKey key{w_packed, k_half, N};
    {
        std::lock_guard<std::mutex> lk(g_tma_mu);
        for (auto &kv : g_tma_cache)
            if (kv.first == key) { *out = kv.second; return cudaSuccess; }
    }
    if ((reinterpret_cast<uintptr_t>(w_packed) & 15) != 0 || (k_half & 15) != 0)
        return cudaErrorInvalidValue;   // TMA global-address / stride alignment
    CUtensorMap tmap{};
    const cuuint64_t gdim[2] = {static_cast<cuuint64_t>(k_half),
                                static_cast<cuuint64_t>(N)};
    const cuuint64_t gstride[1] = {static_cast<cuuint64_t>(k_half)};
    const cuuint32_t bdim[2] = {kDecChunkBytes, kDecCols};
    const cuuint32_t estride[2] = {1, 1};
    const CUresult cr = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<uint8_t *>(w_packed), gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) return cudaErrorNotSupported;
    std::lock_guard<std::mutex> lk(g_tma_mu);
    if (g_tma_cache.size() < kMaxCachedMaps)
        g_tma_cache.emplace_back(key, tmap);
    *out = tmap;
    return cudaSuccess;
}

} // namespace

__global__ void TcGemmDecodeKernel(const uint8_t *__restrict__ ap,
                                   const uint8_t *__restrict__ as,
                                   const uint8_t *__restrict__ bp,
                                   const uint8_t *__restrict__ bs,
                                   float wgs, __half *__restrict__ y,
                                   int K, int N, bool acc)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wn = warp >> 1;             // 0..1
    const int g = lane >> 2;
    const int tig = lane & 3;

    const int col_base = blockIdx.x * kDecCols;
    const int k_half = K / 2;
    const int k_chunks = K >> 4;

    // Dynamic activation side: packed row (K/2 bytes) + per-16 scale bytes,
    // sized from K at launch. Removes the old K<=4096 static cap that forced
    // mlp_down (K=12288) onto the tile-kernel fallback every decode token.
    extern __shared__ __align__(16) uint8_t sXp[];   // [k_half] then [k_chunks]
    uint8_t *const sXsc = sXp + ((k_half + 15) & ~15);
    // Double-buffered weight slab + scales. B3 root cause (empirical):
    // bulk-tensor dst requires 128-BYTE alignment (16 is not enough).
    // Scale slab declared BEFORE the weight ring: empirically the bulk-
    // tensor staging can transiently alias bytes adjacent to the ring
    // destination; keep scales outside its reach.
    __shared__ uint8_t sWs[kDecBufs][kDecCols][4];
    __shared__ __align__(128) uint8_t sW[kDecBufs][kDecCols][kDecChunkBytes];


    // One-time activation load (row 0 is the only row at seq == 1).
    for (int i = tid; i < k_half; i += 256)
        sXp[i] = ap[i];
    for (int c = tid; c < k_chunks; c += 256)
        sXsc[c] = as[c];
    __syncthreads();

    // SFA per lane: carrier lanes tig0/tig1 hold row 0 chunk bytes; for
    // k64 atoms the four bytes are chunks kb/16 .. kb/16+3.
    auto sfa_for = [&](int kb) {
        uint32_t v = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int ch = kb / 16 + c;
            v |= uint32_t(ch < k_chunks ? sXsc[ch] : uint8_t{0}) << (c * 8);
        }
        return v;
    };

    // ---- cp.async pipeline: stage_chunk() copies one k64 weight chunk
    // into ring buffer `buf`; up to kDecBufs chunks stay in flight.
    // ---- B3 TMA pipeline: one bulk-tensor copy stages a whole [128 cols x
    // 32 bytes] k64 chunk (OOB rows zero-filled by hardware); scales remain
    // synchronous plain stores. Thread 0 arms the buffer mbarrier with
    // expect_tx and issues the copy (.shared::cta dst, 128B aligned).
    auto stage_chunk = [&](int chunk, int buf) {
        const int kb = chunk * 64;
        const uint4 *b4 = reinterpret_cast<const uint4 *>(
            bp + static_cast<int64_t>(col_base) * k_half + kb / 2);
        uint4 *d4 = reinterpret_cast<uint4 *>(&sW[buf][0][0]);
        for (int idx = tid; idx < kDecCols * 2; idx += 256) {
            const int c = idx >> 1;
            d4[idx] = (col_base + c < N)
                          ? b4[static_cast<int64_t>(c) * (k_half / 16) +
                               (idx & 1)]
                          : make_uint4(0, 0, 0, 0);
        }
        for (int idx = tid; idx < kDecCols; idx += 256) {
            const int c = idx;
            for (int t = 0; t < 4; ++t) {
                const int chg = kb / 16 + t;
                sWs[buf][c][t] =
                    (col_base + c < N && chg < k_chunks)
                        ? bs[static_cast<int64_t>(col_base + c) * k_chunks +
                             chg]
                        : uint8_t{0};
            }
        }
        cp_async_commit();
    };

    const int n_k64 = K >> 6;
    stage_chunk(0, 0);
    if (n_k64 > 1) stage_chunk(1, 1);
    if (n_k64 > 2) stage_chunk(2, 2);

    int a_row[32], a_koff[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        a_row[i] = 0;   // seq == 1: only row 0 carries activations
        a_koff[i] = tig * 8 + (i & 7) + ((i >= 16) ? 32 : 0);
    }

    float d[4][4] = {};

    for (int i = 0; i < n_k64; ++i) {
        const int buf = i & (kDecBufs - 1);
        const int nxt = i + kDecBufs - 1;
        if (nxt < n_k64) stage_chunk(nxt, nxt & (kDecBufs - 1));
        // Chunks i..min(i+3,n-1) may be outstanding; chunk i must be done.
        cp_async_wait<kDecBufs - 2>();
        __syncthreads();
        const int kb = i * 64;
        //   ar[0] = row g   k32-lo, ar[1] = row g+8 k32-lo,
        //   ar[2] = row g   k32-hi, ar[3] = row g+8 k32-hi;
        // SFA carrier lanes: tig0 -> row g chunk bytes, tig1 -> row g+8.
        // At seq==1 only row 0 exists: rows g!=0 contribute zero fragments
        // and zero scales (the historical "layout identity" u32-load form
        // fed K-halves into row-g+8 slots -- garbage outputs; see
        // "[tc][gemm][acc]" host-model test).
        uint32_t ar[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i2 = 0; i2 < 32; ++i2) {
            const int arow = (i2 < 8 || (i2 >= 16 && i2 < 24)) ? g : g + 8;
            if (arow != 0) continue;          // seq == 1: only row 0 exists
            const int koff = tig * 8 + (i2 & 7) + ((i2 >= 16) ? 32 : 0);
            const uint8_t byte = sXp[(kb + koff) >> 1];   // absolute k
            const uint8_t code =
                (koff & 1) ? static_cast<uint8_t>(byte >> 4)
                           : static_cast<uint8_t>(byte & 0x0F);
            ar[i2 / 8] |= uint32_t(code)
                          << (((i2 % 8) / 2) * 8 + ((i2 % 8) % 2) * 4);
        }
        uint32_t sfa;
        {
            const int sr = (tig == 1) ? (g + 8) : g;
            sfa = (sr == 0) ? sfa_for(kb) : 0u;
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n_local = wn * 32 + j * 8 + g;
            const uint32_t b0 =
                *reinterpret_cast<const uint32_t *>(&sW[buf][n_local][tig * 4]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t *>(
                &sW[buf][n_local][16 + tig * 4]);
            uint32_t br[2] = {b0, b1};
            // NOTE: index the 3D scale slab directly -- a flat
            // uint32_t(sWs[sfb_off]) form miscompiles under nvcc 13.x
            // (array glvalue -> scalar cast yields garbage).
            const uint32_t sfb =
                uint32_t(sWs[buf][n_local][0]) |
                (uint32_t(sWs[buf][n_local][1]) << 8) |
                (uint32_t(sWs[buf][n_local][2]) << 16) |
                (uint32_t(sWs[buf][n_local][3]) << 24);

            float &d0 = d[j][0], &d1 = d[j][1], &d2 = d[j][2], &d3 = d[j][3];
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
                "%14, {0, 0}, %15, {0, 0};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
                  "r"(br[0]), "r"(br[1]), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(sfa), "r"(sfb));
        }
        __syncthreads();   // all lanes done reading buf before next overwrite
    }

    // ---- Epilogue: D * global_scale -> y row 0 (acc=true: += into the
    // residual stream, matching the tile-kernel contract).
    // m16n8 D layout: only lanes with groupID g==0 hold ROW 0 of the
    // accumulator (lanes g>0 hold rows 1..7 -- absent at seq==1). Without
    // this guard four lanes raced each output cell with different rows'
    // values.
    if (g != 0) return;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n0 = col_base + wn * 32 + j * 8 + tig * 2;
        if (acc) {
            if (n0 < N)
                y[n0] = __float2half(__half2float(y[n0]) + d[j][0] * wgs);
            if (n0 + 1 < N)
                y[n0 + 1] =
                    __float2half(__half2float(y[n0 + 1]) + d[j][1] * wgs);
        } else {
            if (n0 < N) y[n0] = __float2half(d[j][0] * wgs);
            if (n0 + 1 < N) y[n0 + 1] = __float2half(d[j][1] * wgs);
        }
    }
}

// Tile GEMM: block computes a [64 x 64] output tile; 256 threads = 8 warps
// arranged 4x2; each warp covers 16 rows x 32 cols via four m16n8k64 mxf4nvf4
// atoms per K64 chunk. Fragment packing matches the empirically verified
// layout in test_blackwell_tc_mma.cu.
__global__ void TcGemmKernel(const uint8_t *__restrict__ ap,
                             const uint8_t *__restrict__ as,
                             const uint8_t *__restrict__ bp,
                             const uint8_t *__restrict__ bs,
                             float wgs, __half *__restrict__ y,
                             int seq, int K, int N, bool acc_in,
                             int split_k, float *__restrict__ y_f32)
{
    // split_k > 1: blockIdx.z owns a K-slice; RAW fp32 partials land in
    // y_f32[z*N+n] (disjoint, deterministic); TcSplitKFinalize sums the
    // slices and applies wgs (+acc) afterwards.
    const int k_lo = static_cast<int>(blockIdx.z) * (K / split_k);
    const int k_hi = k_lo + K / split_k;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 1;          // 0..3
    const int wn = warp & 1;           // 0..1
    const int g = lane >> 2;
    const int tig = lane & 3;

    const int row_base = blockIdx.y * kTileM;
    const int col_base = blockIdx.x * kTileN;

    // Double-buffered slabs: chunk i+1 streams via cp.async while chunk i
    // computes (removes the per-chunk staging stall; ~30% of GEMM time).
    // 4-deep slab ring: deeper pipeline hides more DRAM latency at
    // decode (few CTAs, long K-chunk chains). Numerically identical.
    constexpr int kPipe = 4;
    __shared__ __align__(16) uint8_t sA[kPipe][kTileM][32];
    __shared__ __align__(16) uint8_t sB[kPipe][kTileN][32];
    __shared__ uint8_t sAs[kPipe][kTileM][4];
    __shared__ uint8_t sBs[kPipe][kTileN][4];

    const int k_half = K / 2;
    const int k_chunks = (K + 15) >> 4;

    float d[4][4] = {};

    // Accumulate-in: seed D from existing y (fp16 residual) once; the mma
    // chains C=D across all k-chunks.
    if (acc_in && seq > 0 && wgs != 0.0f) {
        // Seed D with prev / wgs so the shared epilogue multiply
        // ((D) * wgs) yields prev + sum*wgs -- NOT (prev+sum)*wgs.
        const float inv_wgs = 1.0f / wgs;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n0 = col_base + wn * 32 + j * 8 + tig * 2;
            const int r0 = row_base + g;
            if (r0 < seq && n0 < N)
                d[j][0] = __half2float(y[static_cast<int64_t>(r0) * N + n0]) *
                          inv_wgs;
            if (r0 < seq && n0 + 1 < N)
                d[j][1] = __half2float(y[static_cast<int64_t>(r0) * N + n0 +
                                         1]) *
                          inv_wgs;
            if (r0 + 8 < seq && n0 < N)
                d[j][2] =
                    __half2float(y[static_cast<int64_t>(r0 + 8) * N + n0]);
            if (r0 + 8 < seq && n0 + 1 < N)
                d[j][3] =
                    __half2float(y[static_cast<int64_t>(r0 + 8) * N + n0 + 1]);
        }
    }

    int a_row[32], a_koff[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        a_row[i] = (i < 8 || (i >= 16 && i < 24)) ? g : g + 8;
        a_koff[i] = tig * 8 + (i & 7) + ((i >= 16) ? 32 : 0);
    }

    // Double-buffered pipeline: slabs for chunk i+1 stream via cp.async
    // while chunk i computes (removes the per-chunk staging stall).
    const int k_chunks_total = (K + 63) / 64;
    auto stage_slabs = [&](const int kb, const int buf) {
        const bool vec_ok = ((K & 63) == 0) &&
                            (((reinterpret_cast<uintptr_t>(ap) |
                               reinterpret_cast<uintptr_t>(bp)) & 15) == 0);
        if (vec_ok) {
            // Slab traffic is issued via cp.async so the NEXT chunk streams
            // while the current one computes (no blocking store->sync chain).
            const char *a_g = reinterpret_cast<const char *>(ap) +
                              static_cast<int64_t>(row_base) * k_half + kb / 2;
            uint8_t *a_s = &sA[buf][0][0];
            for (int idx = tid; idx < kTileM * 2; idx += 256) {
                const int r = idx >> 1;
                cp_async16(a_s + idx * 16,
                           a_g + static_cast<int64_t>(r) * k_half +
                               (idx & 1) * 16,
                           row_base + r < seq);
            }
            const char *b_g = reinterpret_cast<const char *>(bp) +
                              static_cast<int64_t>(col_base) * k_half + kb / 2;
            uint8_t *b_s = &sB[buf][0][0];
            for (int idx = tid; idx < kTileN * 2; idx += 256) {
                const int c = idx >> 1;
                cp_async16(b_s + idx * 16,
                           b_g + static_cast<int64_t>(c) * k_half +
                               (idx & 1) * 16,
                           col_base + c < N);
            }
        } else {
            // Tail-K: clamp byte reads to the row (k_half may not be a
            // multiple of 32); out-of-range bytes contribute code 0.
            const int kb_b0 = kb / 2;
            for (int idx = tid; idx < kTileM * 32; idx += 256) {
                const int r = idx / 32, b = idx % 32;
                const bool ok = row_base + r < seq && kb_b0 + b < k_half;
                sA[buf][r][b] =
                    ok ? ap[static_cast<int64_t>(row_base + r) * k_half +
                            kb_b0 + b]
                       : uint8_t{0};
            }
            for (int idx = tid; idx < kTileN * 32; idx += 256) {
                const int c = idx / 32, b = idx % 32;
                const bool ok = col_base + c < N && kb_b0 + b < k_half;
                sB[buf][c][b] =
                    ok ? bp[static_cast<int64_t>(col_base + c) * k_half +
                            kb_b0 + b]
                       : uint8_t{0};
            }
        }
        for (int idx = tid; idx < kTileM * 4; idx += 256) {
            const int r = idx / 4, ch = idx % 4;
            const int ch_g = kb / 16 + ch;
            sAs[buf][r][ch] =
                (row_base + r < seq && ch_g < k_chunks)
                    ? as[static_cast<int64_t>(row_base + r) * k_chunks +
                         ch_g]
                    : uint8_t{0};
        }
        for (int idx = tid; idx < kTileN * 4; idx += 256) {
            const int c = idx / 4, ch = idx % 4;
            const int ch_g = kb / 16 + ch;
            sBs[buf][c][ch] =
                (col_base + c < N && ch_g < k_chunks)
                    ? bs[static_cast<int64_t>(col_base + c) * k_chunks +
                         ch_g]
                    : uint8_t{0};
        }
        cp_async_commit();
    };

    const int c_begin = k_lo / 64;
    const int c_end = (k_hi + 63) / 64;
    const int n_loc = c_end - c_begin;
    for (int j = 0; j < n_loc && j < kPipe - 1; ++j)
        stage_slabs((c_begin + j) * 64, j % kPipe);
    for (int i = 0; i < n_loc; ++i) {
        const int buf = i % kPipe;
        const int nxt = i + kPipe - 1;
        if (nxt < n_loc) stage_slabs((c_begin + nxt) * 64, nxt % kPipe);
        const int w = (nxt < n_loc) ? kPipe - 1 : (n_loc - 1 - i);
        if (w >= 3)
            cp_async_wait3();
        else if (w == 2)
            cp_async_wait2();
        else if (w == 1)
            cp_async_wait1();
        else
            cp_async_wait0();
        __syncthreads();

        // ---- Build A fragment regs from smem.
        uint32_t ar[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const uint8_t byte = sA[buf][a_row[i]][a_koff[i] >> 1];
            const uint8_t code =
                (a_koff[i] & 1) ? static_cast<uint8_t>(byte >> 4)
                                : static_cast<uint8_t>(byte & 0x0F);
            ar[i / 8] |= uint32_t(code)
                         << (((i % 8) / 2) * 8 + ((i % 8) % 2) * 4);
        }
        // SFA: carrier lanes tig 0/1 hold rows g / g+8 chunk bytes [0..3].
        uint32_t sfa;
        {
            const int sr = (tig == 1) ? (g + 8) : g;
            sfa = uint32_t(sAs[buf][sr][0]) |
                  (uint32_t(sAs[buf][sr][1]) << 8) |
                  (uint32_t(sAs[buf][sr][2]) << 16) |
                  (uint32_t(sAs[buf][sr][3]) << 24);
        }

        // ---- Four n-atoms.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n_local = wn * 32 + j * 8 + g;
            uint32_t br[2] = {0, 0};
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int kk = tig * 8 + (i & 7) + ((i >= 8) ? 32 : 0);
                const uint8_t byte = sB[buf][n_local][kk >> 1];
                const uint8_t code =
                    (kk & 1) ? static_cast<uint8_t>(byte >> 4)
                             : static_cast<uint8_t>(byte & 0x0F);
                br[i / 8] |= uint32_t(code)
                             << (((i % 8) / 2) * 8 + ((i % 8) % 2) * 4);
            }
            const uint32_t sfb =
                uint32_t(sBs[buf][n_local][0]) |
                (uint32_t(sBs[buf][n_local][1]) << 8) |
                (uint32_t(sBs[buf][n_local][2]) << 16) |
                (uint32_t(sBs[buf][n_local][3]) << 24);

            float &d0 = d[j][0], &d1 = d[j][1], &d2 = d[j][2], &d3 = d[j][3];
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
                "%14, {0, 0}, %15, {0, 0};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
                  "r"(br[0]), "r"(br[1]), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(sfa), "r"(sfb));
        }
        __syncthreads();   // all lanes done reading buf before next overwrite
    }

    // ---- Epilogue.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n0 = col_base + wn * 32 + j * 8 + tig * 2;
        const int r0 = row_base + g;
        if (split_k > 1) {
            // RAW partial: wgs applied once in TcSplitKFinalize. Partials
            // are disjoint per z -> deterministic, no atomics.
            const int64_t zbase =
                static_cast<int64_t>(blockIdx.z) * seq * N;
            if (r0 < seq && n0 < N)
                y_f32[zbase + static_cast<int64_t>(r0) * N + n0] = d[j][0];
            if (r0 < seq && n0 + 1 < N)
                y_f32[zbase + static_cast<int64_t>(r0) * N + n0 + 1] =
                    d[j][1];
            if (r0 + 8 < seq && n0 < N)
                y_f32[zbase + static_cast<int64_t>(r0 + 8) * N + n0] =
                    d[j][2];
            if (r0 + 8 < seq && n0 + 1 < N)
                y_f32[zbase + static_cast<int64_t>(r0 + 8) * N + n0 + 1] =
                    d[j][3];
            continue;
        }
        if (r0 < seq && n0 < N)
            y[static_cast<int64_t>(r0) * N + n0] =
                __float2half(d[j][0] * wgs);
        if (r0 < seq && n0 + 1 < N)
            y[static_cast<int64_t>(r0) * N + n0 + 1] =
                __float2half(d[j][1] * wgs);
        if (r0 + 8 < seq && n0 < N)
            y[static_cast<int64_t>(r0 + 8) * N + n0] =
                __float2half(d[j][2] * wgs);
        if (r0 + 8 < seq && n0 + 1 < N)
            y[static_cast<int64_t>(r0 + 8) * N + n0 + 1] =
                __float2half(d[j][3] * wgs);
    }
}

// Split-K finalize: y[i] = (acc ? prev : 0) + wgs * sum_z partial[z,i].
__global__ void TcSplitKFinalize(const float *__restrict__ partials,
                                 __half *__restrict__ y, float wgs, bool acc,
                                 int seq, int N, int split_k)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= static_cast<int64_t>(seq) * N) return;
    float sum = 0.f;
    for (int z = 0; z < split_k; ++z)
        sum += partials[static_cast<int64_t>(z) * seq * N + i];
    const float prev = acc ? __half2float(y[i]) : 0.f;
    y[i] = __float2half(prev + sum * wgs);
}

} // namespace

cudaError_t QuantizeActsNvfp4(cudaStream_t stream, const __half *x, int seq,
                              int K, uint8_t *d_packed, uint8_t *d_scales)
{
    if (!x || !d_packed || !d_scales || seq <= 0 || K <= 0 || (K & 15))
        return cudaErrorInvalidValue;
    const int64_t n = static_cast<int64_t>(seq) * K;
    const int64_t blocks16 = (n + 15) / 16;
    const int threads = 256;
    const int64_t grid = (blocks16 + threads - 1) / threads;
    if (grid > 0x7FFFFFFFLL) return cudaErrorInvalidValue;
    ActQuantKernel<<<static_cast<int>(grid), threads, 0, stream>>>(
        x, n, d_packed, d_scales);
    return cudaGetLastError();
}

cudaError_t RunQwenTcNvfp4Gemm(cudaStream_t stream, const __half *x_fp16,
                               const uint8_t *w_packed,
                               const uint8_t *w_scales, float w_global_scale,
                               __half *y, int seq, int K, int N, bool acc)
{
    if (!x_fp16 || !w_packed || !w_scales || !y || seq <= 0 || K <= 0 ||
        N <= 0 || (K & 15))
        return cudaErrorInvalidValue;

    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    cudaDeviceProp prop{};
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) return e;
    if (prop.major != 12) return cudaErrorNotSupported;

    // Stream-ordered scratch for quantized activations.
    const size_t packed_bytes = static_cast<size_t>(seq) * (K / 2);
    const size_t scale_bytes = static_cast<size_t>(seq) * ((K + 15) / 16);
    void *ap_scratch = nullptr, *as_scratch = nullptr;
    e = cudaMallocAsync(&ap_scratch, packed_bytes ? packed_bytes : 1, stream);
    if (e != cudaSuccess) return e;
    e = cudaMallocAsync(&as_scratch, scale_bytes ? scale_bytes : 1, stream);
    if (e != cudaSuccess) {
        cudaFreeAsync(ap_scratch, stream);
        return e;
    }

    e = QuantizeActsNvfp4(stream, x_fp16, seq, K,
                          static_cast<uint8_t *>(ap_scratch),
                          static_cast<uint8_t *>(as_scratch));
    bool handled = false;
    // Decode-specialized kernel: faster in isolation (micro-bench 84-172us)
    // but net-slower end-to-end than the v2 tile kernel on this workload
    // (12.9 vs 14.8 tok/s A/B). Kept for future multi-stage pipeline work.
    if (seq == 1 && (K & 63) == 0 &&
        (((reinterpret_cast<uintptr_t>(w_packed) |
           reinterpret_cast<uintptr_t>(ap_scratch)) & 15) == 0)) {
        // ROUTING (measured @5090, qwen3.5-9B e2e): the tile kernel beats
        // the decode-specialized kernel end-to-end (32.9 vs 31.1 tok/s)
        // despite losing isolated micro-benchmarks -- same conclusion as
        // the Phase-3 A/B. Default = tile; LSX_DEC_ENABLE=1 opts into the
        // decode kernel for experiments.
        static const bool kDecEnable =
            std::getenv("LSX_DEC_ENABLE") != nullptr;
        if (kDecEnable) {
            const int k_half = K / 2;
            const int k_chunks = K >> 4;
            const size_t dec_smem = static_cast<size_t>(
                ((k_half + 15) & ~15) + ((k_chunks + 15) & ~15));
            dim3 dgrid((N + kDecCols - 1) / kDecCols);
            TcGemmDecodeKernel<<<dgrid, 256, dec_smem, stream>>>(
                static_cast<uint8_t *>(ap_scratch),
                static_cast<uint8_t *>(as_scratch), w_packed, w_scales,
                w_global_scale, y, K, N, acc);
            e = cudaGetLastError();
            handled = e == cudaSuccess;
        }
    }
    if (!handled && e == cudaSuccess) {
        dim3 grid((N + kTileN - 1) / kTileN, (seq + kTileM - 1) / kTileM);
        // Split-K: when seq==1 leaves the SM array underfilled (few x-
        // blocks), spread K over blockIdx.z slices. Requires K to divide
        // into 64-wide chunks per slice; RAW fp32 partials are summed by
        // TcSplitKFinalize (deterministic slice order).
        int split_k = 1;
        float *partials = nullptr;
        if (seq == 1 && !acc && K >= 2048 && grid.x < 192 && (K % 512) == 0) {
            for (int cand : {8, 4, 2}) {
                if ((K / 64) % cand == 0 && grid.x * cand >= 96) {
                    split_k = cand;
                    break;
                }
            }
        }
        if (split_k > 1) {
            const size_t pbytes =
                static_cast<size_t>(split_k) * N * sizeof(float);
            e = cudaMallocAsync(&partials, pbytes, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemsetAsync(partials, 0, pbytes, stream);
            if (e != cudaSuccess) {
                cudaFreeAsync(partials, stream);
                return e;
            }
            dim3 grid3(grid.x, grid.y, split_k);
            TcGemmKernel<<<grid3, 256, 0, stream>>>(
                static_cast<uint8_t *>(ap_scratch),
                static_cast<uint8_t *>(as_scratch), w_packed, w_scales,
                w_global_scale, y, seq, K, N, acc, split_k, partials);
            e = cudaGetLastError();
            if (e == cudaSuccess) {
                const int threads = 256;
                const int64_t total = static_cast<int64_t>(seq) * N;
                TcSplitKFinalize<<<static_cast<int>((total + threads - 1) /
                                                    threads),
                                   threads, 0, stream>>>(partials, y,
                                                         w_global_scale,
                                                         acc, seq, N,
                                                         split_k);
                e = cudaGetLastError();
            }
            cudaFreeAsync(partials, stream);
        } else {
            TcGemmKernel<<<grid, 256, 0, stream>>>(
                static_cast<uint8_t *>(ap_scratch),
                static_cast<uint8_t *>(as_scratch), w_packed, w_scales,
                w_global_scale, y, seq, K, N, acc, 1, nullptr);
            e = cudaGetLastError();
        }
    }
    cudaFreeAsync(ap_scratch, stream);
    cudaFreeAsync(as_scratch, stream);
    return e;
}

// ── B3 capability probe: single-warp cp.async.bulk.tensor.2d round trip ────
namespace {

constexpr int kTmaRows = 64;
constexpr int kTmaRowBytes = 64;
constexpr int kTmaBoxRows = 8;

__global__ void TmaProbeKernel(const __grid_constant__ CUtensorMap tmap,
                               const uint8_t *__restrict__ expect,
                               int *__restrict__ ok)
{
    __shared__ __align__(128) uint8_t dst[kTmaBoxRows * kTmaRowBytes];
    __shared__ __align__(8) unsigned long long mbar;
    const unsigned mb =
        static_cast<unsigned>(__cvta_generic_to_shared(&mbar));
    const unsigned dd =
        static_cast<unsigned>(__cvta_generic_to_shared(dst));
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                     "r"(mb), "r"(kTmaBoxRows * kTmaRowBytes));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier"
            "::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dd),
            "l"(reinterpret_cast<unsigned long long>(&tmap)), "r"(0),
            "r"(0), "r"(mb)
            : "memory");
    }
    asm volatile(
        "{\n\t.reg .pred p;\n\tWAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
        "@!p bra WAIT;\n\t}" ::"r"(mb));
    __syncthreads();
    bool good = true;
    for (int i = threadIdx.x; i < kTmaBoxRows * kTmaRowBytes; i += blockDim.x)
        good &= dst[i] == expect[i];
    if (!good) *ok = 0;
}

} // namespace

cudaError_t ProbeTmaBulkCopy(cudaStream_t stream)
{
    CUtensorMap tmap{};
    std::vector<uint8_t> host(kTmaRows * kTmaRowBytes);
    for (int r = 0; r < kTmaRows; ++r)
        for (int c = 0; c < kTmaRowBytes; ++c)
            host[static_cast<size_t>(r) * kTmaRowBytes + c] =
                static_cast<uint8_t>((r * 31 + c * 7) & 0xFF);
    uint8_t *d_src = nullptr;
    int *d_ok = nullptr;
    cudaError_t e = cudaMalloc(&d_src, host.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d_src, host.data(), host.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d_src); return e; }
    e = cudaMalloc(&d_ok, sizeof(int));
    if (e != cudaSuccess) { cudaFree(d_src); return e; }
    e = cudaMemsetAsync(d_ok, 1, sizeof(int), stream);
    if (e != cudaSuccess) { cudaFree(d_src); cudaFree(d_ok); return e; }

    const cuuint64_t gdim[2] = {kTmaRowBytes, kTmaRows};
    const cuuint64_t gstride[1] = {kTmaRowBytes};
    const cuuint32_t bdim[2] = {kTmaRowBytes, kTmaBoxRows};
    const cuuint32_t estride[2] = {1, 1};
    const CUresult cr = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_src, gdim, gstride, bdim,
        estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) {
        cudaFree(d_src); cudaFree(d_ok);
        return cudaErrorNotSupported;
    }
    TmaProbeKernel<<<1, 128, 0, stream>>>(tmap, d_src, d_ok);
    e = cudaGetLastError();
    if (e == cudaSuccess) {
        int h_ok = 0;
        e = cudaMemcpyAsync(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost,
                            stream);
        if (e == cudaSuccess)
            e = cudaStreamSynchronize(stream) == cudaSuccess
                    ? (h_ok ? cudaSuccess : cudaErrorUnknown)
                    : cudaErrorInvalidValue;
    }
    cudaFree(d_src);
    cudaFree(d_ok);
    return e;
}

} // namespace tc
} // namespace qwen
} // namespace lsxcommon