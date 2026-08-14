#include "lsxcommon/deepseek_norm_kernels.h"

//==============================================================================
// rms_norm_float_kernel
//
// Each block processes one row.  The reduction is split into two phases with
// a __syncthreads() barrier between them so the entire normalisation finishes
// in a single kernel launch.
//
// Phase 1 — sum(x^2)
//   Each thread accumulates the squared elements for its strided chunk,
//   then a warp‑shuffle reduction collapses the warp partials.  After
//   __syncthreads the first warp reduces the 8 warp partials to a global sum.
//   The final inv_rms is broadcast through shared memory.
//
// Phase 2 — output
//   Element‑wise  out[i] = in[i] * inv_rms * gamma[i].
//==============================================================================

namespace lsxcommon {
namespace gpu {
namespace {

template<int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
__global__ void rms_norm_float_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float*       __restrict__ output,
    float                       eps,
    int                         hidden_dim)
{
    static constexpr int NUM_WARPS = BLOCK_DIM / 32;

    __shared__ float s_warp_partial[NUM_WARPS];

    const int row         = blockIdx.x;
    const int tid         = threadIdx.x;
    const int lane        = tid & 31;
    const int warp_id     = tid >> 5;
    const unsigned mask   = __activemask();

    const float* row_input  = input  + row * static_cast<long long>(hidden_dim);
    float*       row_output = output + row * static_cast<long long>(hidden_dim);

    // ---- Phase 1 : sum(x^2) ----
    float partial = 0.0f;

    constexpr int VEC_UNIT = 4;  // float4
    const int vec_count = hidden_dim / VEC_UNIT;

    const float4* in4 = reinterpret_cast<const float4*>(row_input);

    #pragma unroll 4
    for (int i = tid; i < vec_count; i += BLOCK_DIM) {
        float4 v = in4[i];
        partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Scalar tail for any remainder beyond the float4-aligned prefix
    const int tail_start = vec_count * VEC_UNIT;
    for (int i = tail_start + tid; i < hidden_dim; i += BLOCK_DIM) {
        float x = row_input[i];
        partial += x * x;
    }

    // Intra‑warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(mask, partial, off);
    }
    if (lane == 0) {
        s_warp_partial[warp_id] = partial;
    }
    __syncthreads();

    // Inter‑warp reduction (first warp gathers all warp partials)
    if (warp_id == 0) {
        float total = (tid < NUM_WARPS) ? s_warp_partial[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            total += __shfl_down_sync(mask, total, off);
        }
        if (tid == 0) {
            float mean_sq  = total / static_cast<float>(hidden_dim);
            float inv_rms  = rsqrtf(mean_sq + eps);
            s_warp_partial[0] = inv_rms;
        }
    }
    __syncthreads();

    // ---- Phase 2 : out = in * inv_rms * gamma ----
    const float inv_rms = s_warp_partial[0];
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    float4* out4 = reinterpret_cast<float4*>(row_output);

    #pragma unroll 4
    for (int i = tid; i < vec_count; i += BLOCK_DIM) {
        float4 x = in4[i];
        float4 g = g4[i];
        out4[i] = make_float4(
            x.x * inv_rms * g.x,
            x.y * inv_rms * g.y,
            x.z * inv_rms * g.z,
            x.w * inv_rms * g.w);
    }

    for (int i = tail_start + tid; i < hidden_dim; i += BLOCK_DIM) {
        row_output[i] = row_input[i] * inv_rms * gamma[i];
    }
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Public launch wrapper
//------------------------------------------------------------------------------
cudaError_t rmsnorm_float(
    cudaStream_t      stream,
    int               batch_size,
    int               hidden_dim,
    const float*      d_input,
    const float*      d_gamma,
    float             epsilon,
    float*            d_output)
{
    if (!d_input || !d_gamma || !d_output)
        return cudaErrorInvalidValue;
    if (batch_size <= 0 || hidden_dim <= 0)
        return cudaErrorInvalidValue;

    static constexpr int BLOCK_DIM = 256;

    dim3 block(BLOCK_DIM);
    dim3 grid(static_cast<int>(batch_size));

    rms_norm_float_kernel<BLOCK_DIM><<<grid, block, 0, stream>>>(
        d_input, d_gamma, d_output, epsilon, hidden_dim);

    return cudaGetLastError();
}

} // namespace gpu
} // namespace lsxcommon
