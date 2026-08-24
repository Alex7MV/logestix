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

#include <cmath>
#include <cstdint>

#include "nvfp4_codec.h"

namespace lsxcommon {
namespace qwen {
namespace tc {

namespace {

// Power-of-two E4M3 encode matching schema.cpp float_to_e4m3_p2 and the
// device decoder in nvfp4_codec.h: byte = (e4+7)<<3 with value 2^(e4-7).
// Ceil semantics: smallest power of two >= s.
__device__ __forceinline__ uint8_t Pow2E4m3Encode(float s)
{
    if (!(s > 0.0f)) return 0; // zero / NaN guard: zero scale zeroes the block
    int exp;
    (void)frexpf(s, &exp); // s = m * 2^exp, m in [0.5, 1)
    const int e4 = exp + 7;
    if (e4 < 1) return static_cast<uint8_t>(1 << 3);      // smallest positive
    if (e4 > 15) return static_cast<uint8_t>(15 << 3);    // saturate at 2^8
    return static_cast<uint8_t>(e4 << 3);
}

__device__ __forceinline__ uint8_t QuantizeE2m1(float v)
{
    // Nearest-level E2M1 with sign bit 3; levels match kE2M1Table.
    const bool neg = v < 0.0f;
    const float a = fabsf(v);
    uint8_t mag;
    if (a < 0.25f) mag = 0;
    else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2;
    else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4;
    else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6;
    else mag = 7;
    return neg ? static_cast<uint8_t>(mag | 8) : mag;
}

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
    const uint8_t sb = Pow2E4m3Encode(absmax * (1.0f / 6.0f));
    scales[blk] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;

    for (int i = 0; i < len; i += 2) {
        const uint8_t lo = QuantizeE2m1(__half2float(x[base + i]) * inv);
        const uint8_t hi = (i + 1 < len)
                               ? QuantizeE2m1(__half2float(x[base + i + 1]) *
                                              inv)
                               : 0;
        packed[(base + i) / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

constexpr int kTileM = 64;
constexpr int kTileN = 64;

// Tile GEMM: block computes a [64 x 64] output tile; 256 threads = 8 warps
// arranged 4x2; each warp covers 16 rows x 32 cols via four m16n8k64 mxf4nvf4
// atoms per K64 chunk. Fragment packing matches the empirically verified
// layout in test_blackwell_tc_mma.cu.
__global__ void TcGemmKernel(const uint8_t *__restrict__ ap,
                             const uint8_t *__restrict__ as,
                             const uint8_t *__restrict__ bp,
                             const uint8_t *__restrict__ bs,
                             float wgs, __half *__restrict__ y,
                             int seq, int K, int N)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 1;          // 0..3
    const int wn = warp & 1;           // 0..1
    const int g = lane >> 2;
    const int tig = lane & 3;

    const int row_base = blockIdx.y * kTileM;
    const int col_base = blockIdx.x * kTileN;

    __shared__ uint8_t sA[kTileM][32];   // [m-row][packed byte over k64]
    __shared__ uint8_t sB[kTileN][32];   // [n-col][packed byte over k64]
    __shared__ uint8_t sAs[kTileM][4];   // activation chunk scales
    __shared__ uint8_t sBs[kTileN][4];   // weight chunk scales

    const int k_half = K / 2;
    const int k_chunks = (K + 15) >> 4;

    float d[4][4] = {};

    int a_row[32], a_koff[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        a_row[i] = (i < 8 || (i >= 16 && i < 24)) ? g : g + 8;
        a_koff[i] = tig * 8 + (i & 7) + ((i >= 16) ? 32 : 0);
    }

    for (int kb = 0; kb < K; kb += 64) {
        // ---- Cooperative streaming stage (4096 + 512 bytes per slab set).
        for (int idx = tid; idx < kTileM * 32; idx += 256) {
            const int r = idx / 32, b = idx % 32;
            sA[r][b] = (row_base + r < seq)
                           ? ap[static_cast<int64_t>(row_base + r) * k_half +
                                kb / 2 + b]
                           : uint8_t{0};
        }
        for (int idx = tid; idx < kTileN * 32; idx += 256) {
            const int c = idx / 32, b = idx % 32;
            sB[c][b] = (col_base + c < N)
                           ? bp[static_cast<int64_t>(col_base + c) * k_half +
                                kb / 2 + b]
                           : uint8_t{0};
        }
        for (int idx = tid; idx < kTileM * 4; idx += 256) {
            const int r = idx / 4, ch = idx % 4;
            const int ch_g = kb / 16 + ch;
            sAs[r][ch] = (row_base + r < seq && ch_g < k_chunks)
                             ? as[static_cast<int64_t>(row_base + r) *
                                      k_chunks + ch_g]
                             : uint8_t{0};
        }
        for (int idx = tid; idx < kTileN * 4; idx += 256) {
            const int c = idx / 4, ch = idx % 4;
            const int ch_g = kb / 16 + ch;
            sBs[c][ch] = (col_base + c < N && ch_g < k_chunks)
                             ? bs[static_cast<int64_t>(col_base + c) *
                                      k_chunks + ch_g]
                             : uint8_t{0};
        }
        __syncthreads();

        // ---- Build A fragment regs from smem.
        uint32_t ar[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const uint8_t byte = sA[a_row[i]][a_koff[i] >> 1];
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
            sfa = uint32_t(sAs[sr][0]) | (uint32_t(sAs[sr][1]) << 8) |
                  (uint32_t(sAs[sr][2]) << 16) | (uint32_t(sAs[sr][3]) << 24);
        }

        // ---- Four n-atoms.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n_local = wn * 32 + j * 8 + g;
            uint32_t br[2] = {0, 0};
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int kk = tig * 8 + (i & 7) + ((i >= 8) ? 32 : 0);
                const uint8_t byte = sB[n_local][kk >> 1];
                const uint8_t code =
                    (kk & 1) ? static_cast<uint8_t>(byte >> 4)
                             : static_cast<uint8_t>(byte & 0x0F);
                br[i / 8] |= uint32_t(code)
                             << (((i % 8) / 2) * 8 + ((i % 8) % 2) * 4);
            }
            const uint32_t sfb =
                uint32_t(sBs[n_local][0]) | (uint32_t(sBs[n_local][1]) << 8) |
                (uint32_t(sBs[n_local][2]) << 16) |
                (uint32_t(sBs[n_local][3]) << 24);

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
        __syncthreads();   // slab reuse guard
    }

    // ---- Epilogue: D * global_scale -> y (std m16n8 f32 acc layout).
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n0 = col_base + wn * 32 + j * 8 + tig * 2;
        const int r0 = row_base + g;
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
                               __half *y, int seq, int K, int N)
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
    if (e == cudaSuccess) {
        dim3 grid((N + kTileN - 1) / kTileN, (seq + kTileM - 1) / kTileM);
        TcGemmKernel<<<grid, 256, 0, stream>>>(
            static_cast<uint8_t *>(ap_scratch),
            static_cast<uint8_t *>(as_scratch), w_packed, w_scales,
            w_global_scale, y, seq, K, N);
        e = cudaGetLastError();
    }
    cudaFreeAsync(ap_scratch, stream);
    cudaFreeAsync(as_scratch, stream);
    return e;
}

} // namespace tc
} // namespace qwen
} // namespace lsxcommon
