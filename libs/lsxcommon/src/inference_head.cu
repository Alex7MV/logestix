#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace lsxcommon {
namespace gpu {

// Argmax: find index of max element in float array
__global__ void argmax_kernel(const float *logits, int32_t *out, int vocab_size)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_max = -1e10f;
    int local_idx = 0;
    for (int i = tid; i < vocab_size; i += nthreads) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }

    extern __shared__ float s_vals[];
    int *s_idxs = reinterpret_cast<int *>(s_vals + nthreads);
    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
        *out = s_idxs[0];
}

// Compute LM head: logits = head_weight @ hidden (FP16 GEMM) + argmax
// head_weight: [vocab_size, hidden_dim] column-major (lda = vocab_size)
// hidden: [hidden_dim, 1]
// logits: [vocab_size, 1] (float32 output)
cudaError_t run_lm_head(
    cublasHandle_t cublas_h,
    cudaStream_t stream,
    const __half *d_head_weight,
    const __half *d_hidden,
    int vocab_size,
    int hidden_dim,
    float *d_logits,
    int32_t *d_next_token)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t stat = cublasGemmEx(
        cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
        vocab_size, 1, hidden_dim, &alpha,
        d_head_weight, CUDA_R_16F, vocab_size,
        d_hidden, CUDA_R_16F, hidden_dim, &beta,
        d_logits, CUDA_R_32F, vocab_size,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (stat != CUBLAS_STATUS_SUCCESS)
        return cudaErrorUnknown;

    int block = 256;
    size_t shared_sz = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    argmax_kernel<<<1, block, shared_sz, stream>>>(d_logits, d_next_token, vocab_size);
    return cudaGetLastError();
}

} // namespace gpu
} // namespace lsxcommon
