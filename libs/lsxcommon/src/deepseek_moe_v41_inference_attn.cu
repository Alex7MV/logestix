// deepseek_moe_v41_inference_attn.cu — DeepSeek-V4.1-Flash MLA attention for a
// pure-SWA (compress_ratio == 0) decode step. Composes the neutral dense
// kernels (GEMM/RMSNorm/RoPE/FP8-quant/sparse-attn); scratch is allocated per
// call (correctness first; the model layer loop reuses buffers later).
#include "deepseek_moe_v41_inference.h"

#include <cmath>
#include <vector>

#include <cuda_fp16.h>

#include "deepseek_moe_v41_dequant.h"

namespace lsxcommon {
namespace deepseek_v41 {

namespace {

__global__ void window_idxs_kernel(int *__restrict__ idxs, int window,
                                   int start_pos)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= window) return;
    int v;
    if (start_pos == 0) {
        v = t;
    } else {
        const int oldest = start_pos % window + 1;
        v = (t < window - oldest) ? (oldest + t) : (t - (window - oldest));
    }
    if (v > start_pos) v = -1;
    idxs[t] = v;
}

struct Scratch {
    uint16_t *qr = nullptr, *q = nullptr, *kv = nullptr, *o = nullptr,
             *olora = nullptr;
    int32_t *idxs = nullptr;
    void *base = nullptr;

    cudaError_t Alloc(int q_lora, int heads, int head_dim, int olora_total,
                      int window)
    {
        size_t bytes = 0;
        auto bump = [&](size_t n) { bytes += (n + 255) / 256 * 256; };
        bump(static_cast<size_t>(q_lora) * 2);
        bump(static_cast<size_t>(heads) * head_dim * 2);
        bump(static_cast<size_t>(head_dim) * 2);
        bump(static_cast<size_t>(heads) * head_dim * 2);
        bump(static_cast<size_t>(olora_total) * 2);
        bump(static_cast<size_t>(window) * sizeof(int32_t));
        if (cudaMalloc(&base, bytes) != cudaSuccess) return cudaGetLastError();
        uint8_t *p = static_cast<uint8_t *>(base);
        auto take = [&](size_t n) {
            uint8_t *r = p;
            p += (n + 255) / 256 * 256;
            return r;
        };
        qr = reinterpret_cast<uint16_t *>(take(q_lora * 2));
        q = reinterpret_cast<uint16_t *>(take(static_cast<size_t>(heads) * head_dim * 2));
        kv = reinterpret_cast<uint16_t *>(take(head_dim * 2));
        o = reinterpret_cast<uint16_t *>(take(static_cast<size_t>(heads) * head_dim * 2));
        olora = reinterpret_cast<uint16_t *>(take(olora_total * 2));
        idxs = reinterpret_cast<int32_t *>(take(window * sizeof(int32_t)));
        return cudaSuccess;
    }
    void Free()
    {
        if (base) cudaFree(base);
        base = nullptr;
    }
};

} // namespace

cudaError_t AttentionForwardDecode(
    cublasHandle_t handle, cudaStream_t stream, const uint16_t *x, int hidden,
    const uint16_t *wq_a, int q_lora, const uint16_t *q_norm,
    const uint16_t *wq_b, int heads, int head_dim, int rope_dim,
    const uint16_t *wkv, const uint16_t *kv_norm, const float *attn_sink,
    const uint16_t *wo_a, int o_groups, int o_lora_rank, const uint16_t *wo_b,
    const float *rope_cos, const float *rope_sin, uint16_t *kv_cache,
    int window, int start_pos, uint16_t *out)
{
    const int q_dim = heads * head_dim;
    Scratch sc;
    if (sc.Alloc(q_lora, heads, head_dim, o_groups * o_lora_rank, window) !=
        cudaSuccess)
        return cudaErrorInvalidValue;

    cudaError_t e = cudaSuccess;
    do {
        // qr = q_norm(wq_a @ x)
        if ((e = LaunchGemmFp16(handle, stream, wq_a, x, sc.qr, q_lora, 1,
                                hidden)) != cudaSuccess)
            break;
        if ((e = LaunchRmsNormHalf(sc.qr, q_norm, sc.qr, q_lora, 1e-20f,
                                   stream)) != cudaSuccess)
            break;
        // q = wq_b @ qr
        if ((e = LaunchGemmFp16(handle, stream, wq_b, sc.qr, sc.q, q_dim, 1,
                                q_lora)) != cudaSuccess)
            break;
        if ((e = LaunchRopeHalf(sc.q, heads, head_dim, rope_dim, rope_cos,
                                rope_sin, false, stream)) != cudaSuccess)
            break;
        // kv = kv_norm(wkv @ x) -> rope -> fp8 quant
        if ((e = LaunchGemmFp16(handle, stream, wkv, x, sc.kv, head_dim, 1,
                                hidden)) != cudaSuccess)
            break;
        if ((e = LaunchRmsNormHalf(sc.kv, kv_norm, sc.kv, head_dim, 1e-20f,
                                   stream)) != cudaSuccess)
            break;
        if ((e = LaunchRopeHalf(sc.kv, 1, head_dim, rope_dim, rope_cos,
                                rope_sin, false, stream)) != cudaSuccess)
            break;
        if ((e = LaunchFp8ActQuantInplace(sc.kv, 1, head_dim, 32, stream)) !=
            cudaSuccess)
            break;

        // Ring buffer: write the new KV, attend over the whole window.
        const int slot = (start_pos % window) * head_dim;
        if ((e = cudaMemcpyAsync(kv_cache + slot, sc.kv,
                                 static_cast<size_t>(head_dim) * 2,
                                 cudaMemcpyDeviceToDevice, stream)) !=
            cudaSuccess)
            break;
        window_idxs_kernel<<<(window + 63) / 64, 64, 0, stream>>>(
            sc.idxs, window, start_pos);
        if ((e = cudaGetLastError()) != cudaSuccess) break;

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        if ((e = LaunchSparseAttnHalf(sc.q, 1, heads, head_dim, kv_cache,
                                      window, attn_sink, sc.idxs, window,
                                      sc.o, scale, stream)) != cudaSuccess)
            break;
        if ((e = LaunchRopeHalf(sc.o, heads, head_dim, rope_dim, rope_cos,
                                rope_sin, true, stream)) != cudaSuccess)
            break;

        // Grouped wo_a: per-group [o_lora_rank, heads/o_groups*head_dim] gemm.
        const int per_group = head_dim * (heads / o_groups);
        for (int g = 0; g < o_groups; ++g) {
            const uint16_t *w_g = wo_a +
                static_cast<size_t>(g) * o_lora_rank * per_group;
            const uint16_t *x_g = sc.o + static_cast<size_t>(g) * per_group;
            uint16_t *y_g = sc.olora +
                static_cast<size_t>(g) * o_lora_rank;
            if ((e = LaunchGemmFp16(handle, stream, w_g, x_g, y_g, o_lora_rank,
                                    1, per_group)) != cudaSuccess)
                break;
        }
        if (e != cudaSuccess) break;

        // out = wo_b @ olora
        if ((e = LaunchGemmFp16(handle, stream, wo_b, sc.olora, out, hidden, 1,
                                o_groups * o_lora_rank)) != cudaSuccess)
            break;
    } while (false);

    sc.Free();
    return e;
}

} // namespace deepseek_v41
} // namespace lsxcommon
