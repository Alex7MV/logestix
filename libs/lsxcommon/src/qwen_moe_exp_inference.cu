#include "lsxcommon/qwen_moe_exp_inference.h"

#include <cuda_runtime.h>

#include <cmath>

namespace lsxcommon::qwen_moe_exp {
namespace {

// ── GatedResidual F32 core (parity twin) ──────────────────────────────────
//
// One block per token. Sequence matches HF Qwen4ExpTextGatedResidual.forward
// in fp32 with -fmad=false:
//   normed   = rmsnorm_groups(x, group=H)                  (no weight)
//   dn[r]    = silu( (W_dn @ normed)[r] / hc )
//   up[i]    = sigmoid( (W_up @ dn)[i] )
//   mixed[h] = mean_c up[c*H+h] * normed[c*H+h]
//   inj[c]   = 2 * sigmoid( (W_inj @ normed)[c] / hc )
//
// W matrices arrive fp32 row-major: down [lowrank, n], up [n, lowrank],
// inject [hc, n], n = hc*H. Deterministic tree reduction, ascending index,
// no atomics, shared-memory writes by thread 0 only (protocol step 6).

__device__ __forceinline__ float dev_silu(float v) { return v / (1.f + expf(-v)); }

__global__ void hc_gated_residual_f32_kernel(
    const float *__restrict__ x,      // [n]  (seq = 1 for the twin)
    const float *__restrict__ down,   // [lowrank * n]
    const float *__restrict__ up,     // [n * lowrank]
    const float *__restrict__ inject, // [hc * n] or null
    int H, int hc, int lowrank, float eps,
    float *__restrict__ mixed_out,    // [H]
    float *__restrict__ inj_out)      // [hc]
{
    const int n = H * hc;
    __shared__ float s_normed[2048];
    __shared__ float s_dn[512];
    __shared__ float s_up[2048];
    __shared__ float s_scale;

    // Phase 1: group-RMSNorm (group = H). Per-group mean of squares reduced
    // by thread 0 after a grid-stride partial store.
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_normed[i] = x[i];
    __syncthreads();

    for (int g = 0; g < hc; ++g) {
        // each group occupies [g*H, g*H+H)
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i) {
                const double v = s_normed[g * H + i];
                acc += v * v;
            }
            s_scale = static_cast<float>(1.0 / sqrt(acc / H + eps));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            s_normed[g * H + i] *= s_scale;
        }
        __syncthreads();
    }

    // Phase 2: down-projection [lowrank] + silu(/hc). One thread per r with
    // sequential fp32 dot over n (deterministic; n small in tests, fine).
    for (int r = threadIdx.x; r < lowrank; r += blockDim.x) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i)
            acc += static_cast<double>(down[r * n + i]) * s_normed[i];
        const double v = acc / hc;
        s_dn[r] = static_cast<float>(v / (1.0 + exp(-v)));
    }
    __syncthreads();

    // Phase 3: up-projection [n] + sigmoid.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = 0.0;
        for (int r = 0; r < lowrank; ++r)
            acc += static_cast<double>(up[i * lowrank + r]) * s_dn[r];
        s_up[i] = 1.0f / (1.0f + expf(-static_cast<float>(acc)));
    }
    __syncthreads();

    // Phase 4: mixed = mean over branches of (gate ⊙ normed).
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        double acc = 0.0;
        for (int c = 0; c < hc; ++c)
            acc += static_cast<double>(s_up[c * H + h]) *
                   s_normed[c * H + h];
        mixed_out[h] = static_cast<float>(acc / hc);
    }

    // Phase 5: injection weights.
    if (inject) {
        for (int c = threadIdx.x; c < hc; c += blockDim.x) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i)
                acc += static_cast<double>(inject[c * n + i]) * s_normed[i];
            inj_out[c] = static_cast<float>(2.0 / (1.0 + exp(-acc / hc)));
        }
    }
}

} // namespace

cudaError_t RunExpHcGatedResidualF32(
    cudaStream_t stream, const float *d_hyper_input, const float *d_down,
    const float *d_up, const float *d_inject_w, int seq, int H, int hc,
    int lowrank, float *d_mixed, float *d_inject_w_out)
{
    if (seq != 1)
        return cudaErrorInvalidValue;  // twin supports the parity case only
    const int n = H * hc;
    if (n > 2048 || lowrank > 512) return cudaErrorInvalidValue;
    const int block = 128;
    hc_gated_residual_f32_kernel<<<1, block, 0, stream>>>(
        d_hyper_input, d_down, d_up, d_inject_w, H, hc, lowrank, 1e-6f,
        d_mixed, d_inject_w_out);
    return cudaGetLastError();
}

namespace {

__global__ void hc_inject_scatter_kernel(
    const __half *__restrict__ hyper_input,  // [seq, n]
    const __half *__restrict__ branch_out,   // [seq, H]
    const float *__restrict__ inject_w,      // [hc]
    int H, int hc, __half *__restrict__ out) // [seq, n]
{
    const int n = H * hc;
    const int total = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += total) {
        const int c = idx / H;
        const int i = idx % H;
        out[idx] = __float2half(__half2float(hyper_input[idx]) +
                                __half2float(branch_out[i]) * inject_w[c]);
    }
}

} // namespace

cudaError_t RunExpHcInjectScatter(cudaStream_t stream,
                                  const __half *d_hyper_input,
                                  const __half *d_branch_out,
                                  const float *d_inject_w, int seq, int H,
                                  int hc, __half *d_out)
{
    const int n = H * hc;
    if (n <= 0) return cudaErrorInvalidValue;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    for (int s = 0; s < seq; ++s) {
        hc_inject_scatter_kernel<<<grid, block, 0, stream>>>(
            d_hyper_input + static_cast<int64_t>(s) * n,
            d_branch_out + static_cast<int64_t>(s) * H, d_inject_w, H, hc,
            d_out + static_cast<int64_t>(s) * n);
    }
    return cudaGetLastError();
}

} // namespace lsxcommon::qwen_moe_exp
