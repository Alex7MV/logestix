// GLM-5.3-Flash CUDA kernels: KDA (conv1d + per-channel-decay gated delta),
// router top-8 (sigmoid noaux_tc), and the single-stream generation driver.
// Per-channel decay mirrors the FLA kimi-delta semantics: decay applies to
// the KEY dim (l) of the state column, unlike qwen GDN's scalar head decay.
#include "lsxcommon/glm_moe_flash_inference.h"

#include "glm_moe_flash_device_math.h"

#include <cuda_runtime.h>

namespace lsxcommon {
namespace glm_moe_flash {
namespace {

FLASH_HD float sigmoid_dev(float x)
{
    return 1.0f / (1.0f + flash_expf_glibc(-x));
}

} // namespace

// ── Conv1d test twin ─────────────────────────────────────────────────────
__global__ void flash_conv1d_f32_kernel(const float *__restrict__ x,
                                        const float *__restrict__ w,
                                        float *__restrict__ y, int channels,
                                        int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k)
        acc += w[c * kernel + k] * x[k * channels + c];
    y[c] = acc;
}

cudaError_t RunFlashConv1dF32(cudaStream_t stream, const float *x,
                              const float *w, float *y, int channels,
                              int kernel)
{
    const int threads = 128;
    const int blocks = (channels + threads - 1) / threads;
    flash_conv1d_f32_kernel<<<blocks, threads, 0, stream>>>(x, w, y, channels,
                                                            kernel);
    return cudaGetLastError();
}

// ── Gated delta test twin (per-channel decay) ─────────────────────────────
// Block per (t, v_head, d-column plane); thread per value-dim d. Registers
// hold the column S[l, d] across the time loop (l in 0..dim).
__global__ void flash_gated_delta_f32_kernel(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ beta,
    const float *__restrict__ gdecay, float *__restrict__ state_out,
    float *__restrict__ out, int seq, int heads, int dim)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= dim) return;

    float col[256];
    for (int l = 0; l < dim; ++l) col[l] = 0.0f;

    for (int tt = 0; tt <= t; ++tt) {
        // Per-channel decay along the key dim l.
        for (int l = 0; l < dim; ++l)
            col[l] *= gdecay[(tt * heads + h) * dim + l];
        float kv = 0.0f;
        for (int l = 0; l < dim; ++l) kv += col[l] * k[(tt * heads + h) * dim + l];
        const float delta = beta[tt * heads + h] * (v[(tt * heads + h) * dim + d] - kv);
        for (int l = 0; l < dim; ++l)
            col[l] += k[(tt * heads + h) * dim + l] * delta;
        float acc = 0.0f;
        for (int l = 0; l < dim; ++l) acc += col[l] * q[(tt * heads + h) * dim + l];
        out[(tt * heads + h) * dim + d] = acc;
    }

    // Write the final row's state: [H, D(l), D(d)].
    if (t == seq - 1) {
        for (int l = 0; l < dim; ++l)
            state_out[(h * dim + l) * dim + d] = col[l];
    }
}

cudaError_t RunFlashGatedDeltaF32(cudaStream_t stream, const float *q,
                                  const float *k, const float *v,
                                  const float *beta, const float *gdecay,
                                  int seq, int heads, int dim,
                                  float *state_out, float *out)
{
    if (dim > 256) return cudaErrorInvalidValue;
    const dim3 grid(seq, heads);
    int threads = 1;
    while (threads < dim) threads <<= 1;
    flash_gated_delta_f32_kernel<<<grid, threads, 0, stream>>>(
        q, k, v, beta, gdecay, state_out, out, seq, heads, dim);
    return cudaGetLastError();
}

// ── Router top-8 ─────────────────────────────────────────────────────────
// One block per token. Score = sigmoid(dot(gate_e, x)) + bias_e computed in
// FP32 over BF16-decoded values; top-8 via linear-scan selection (tie-break
// lower expert index); renorm + scale 2.5. RMW-free per-slot writes: each
// selection round writes its slot once.
__global__ void flash_router_top8_kernel(
    const uint8_t *__restrict__ gate_bf16, const float *__restrict__ bias,
    const uint8_t *__restrict__ x_bf16, qwen_moe::QwenMoeRoutedSlot *__restrict__ slots,
    int hidden, int experts)
{
    const int t = blockIdx.x;
    extern __shared__ float s_scores[];  // [experts]
    // Score pass: one thread per expert.
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        float dot = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            const uint32_t gb =
                (static_cast<uint32_t>(gate_bf16[e * hidden + h * 2])
                 | (static_cast<uint32_t>(gate_bf16[e * hidden + h * 2 + 1])
                    << 8))
                << 16;
            const uint32_t xb =
                (static_cast<uint32_t>(x_bf16[t * hidden + h * 2])
                 | (static_cast<uint32_t>(x_bf16[t * hidden + h * 2 + 1])
                    << 8))
                << 16;
            float g, xv;
            memcpy(&g, &gb, 4);
            memcpy(&xv, &xb, 4);
            dot += g * xv;
        }
        const float s = sigmoid_dev(dot) + bias[e];
        s_scores[e] = s;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    // Thread-0 selection over the staged scores.
    constexpr int kTopK = 8;
    for (int kk = 0; kk < kTopK; ++kk) {
        int best = -1;
        float best_s = -1e30f;
        for (int e = 0; e < experts; ++e) {
            bool taken = false;
            for (int j = 0; j < kk; ++j)
                if (slots[t * kTopK + j].eid == e) taken = true;
            if (taken) continue;
            if (s_scores[e] > best_s) {
                best_s = s_scores[e];
                best = e;
            }
        }
        slots[t * kTopK + kk].eid = best;
        slots[t * kTopK + kk].coeff = best_s;
    }
    float sum = 0.0f;
    for (int kk = 0; kk < kTopK; ++kk) sum += slots[t * kTopK + kk].coeff;
    for (int kk = 0; kk < kTopK; ++kk)
        slots[t * kTopK + kk].coeff =
            slots[t * kTopK + kk].coeff / sum * 2.5f;
}

cudaError_t RunFlashRouterTop8(cudaStream_t stream, const uint8_t *d_gate,
                               const float *d_bias, const uint8_t *d_x,
                               int tok, int hidden, int experts,
                               qwen_moe::QwenMoeRoutedSlot *d_slots)
{
    const int threads = 256;
    const size_t scratch = static_cast<size_t>(experts) * sizeof(float);
    flash_router_top8_kernel<<<tok, threads, scratch, stream>>>(
        d_gate, d_bias, d_x, d_slots, hidden, experts);
    return cudaGetLastError();
}

// ── Generation driver (Task 6 stub: host-side orchestration) ─────────────

cudaError_t RunGlmFlashGeneration(
    cudaStream_t stream, const FlashPinnedWeights &pw,
    qwen_moe::QwenMoeExpertStore &store, gpu::ExpertDeviceCache &cache,
    const std::vector<int32_t> &prompt, int max_new,
    const std::vector<int> &eos_ids, qwen::QwenGenerationResult &result,
    sample::ISampler &sampler)
{
    (void) stream;
    (void) pw;
    (void) store;
    (void) cache;
    (void) prompt;
    (void) max_new;
    (void) eos_ids;
    (void) result;
    (void) sampler;
    return cudaErrorNotYetImplemented;
}

} // namespace glm_moe_flash
} // namespace lsxcommon
