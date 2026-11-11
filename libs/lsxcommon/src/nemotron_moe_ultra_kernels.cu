// Nemotron-3-Ultra-550B device kernels.
//
// Shared-memory sandbox: reductions use a shared array + barriers; no atomics
// feed logits. Math pinned by nemotron_moe_ultra_device_math.h (host reference).
#include "nemotron_moe_ultra_kernels.h"

#include "nemotron_moe_ultra_constants.h"

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_ultra {

namespace {

// Compile-time MAXIMUM number of routing groups the kernel supports. This is NOT
// `frozen::kRouterGroups`: the group count is a runtime argument (the index
// carries it, and the fixtures exercise 1, 4 and 8), so sizing the shared arrays
// from the passport would silently overflow shared memory the moment a caller
// passes more groups than this checkpoint happens to use. That is exactly what
// happened when the passport moved from 8 to 1: the grouped cases started
// scribbling past gscore[]/gkeep[].
inline constexpr int kMaxRouterGroups = 64;
static_assert(frozen::kRouterGroups <= kMaxRouterGroups,
              "the passport's n_group must fit the kernel's shared arrays");
static_assert(frozen::kNumRouted <= 1024, "router shared arrays are sized to 1024");

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__global__ void conv_step_kernel(const float *__restrict__ xBC,
                                 const float *__restrict__ w,
                                 const float *__restrict__ bias,
                                 const float *__restrict__ state_in,
                                 float *__restrict__ state_out,
                                 float *__restrict__ out, int conv_dim, int K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int S = K - 1;
    float acc = bias ? bias[c] : 0.0f;
    for (int k = 0; k < S; ++k) acc += w[c * K + k] * state_in[c * S + k];
    acc += w[c * K + (K - 1)] * xBC[c];
    for (int k = 0; k < S - 1; ++k) state_out[c * S + k] = state_in[c * S + k + 1];
    state_out[c * S + (S - 1)] = xBC[c];
    out[c] = silu_f(acc);
}

// One block per (head, head_dim) pair: head_dim threads walk the state vector.
// state_size is 128 in production, so this stays a serial inner loop per thread
// (each thread owns one (head, d) row of the state, so the state update needs no
// cross-thread reduction at all).
__global__ void ssm_step_kernel(const float *__restrict__ state_in,
                                const float *__restrict__ xs,
                                const float *__restrict__ B,
                                const float *__restrict__ C,
                                const float *__restrict__ D,
                                const float *__restrict__ dt,
                                const float *__restrict__ dt_bias,
                                const float *__restrict__ A_log, int heads,
                                int head_dim, int groups, int state_size,
                                int clamp_dt, float *__restrict__ state_out,
                                float *__restrict__ out)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= heads || d >= head_dim) return;

    float dth = dt[h] + dt_bias[h];
    dth = dth > 20.0f ? dth : log1pf(expf(dth));
    // Reference split: the chunk scan (prefill) clamps dt to time_step_min, the
    // single-token selective update (decode) applies softplus only.
    if (clamp_dt && dth < frozen::kDtMin) dth = frozen::kDtMin;
    const float A = -expf(A_log[h]);
    const float dA = expf(dth * A);
    const int hpg = heads / groups;
    const int g = h / hpg;
    const float xsd = xs[h * head_dim + d];
    const float *Bg = B + g * state_size;
    const float *Cg = C + g * state_size;
    const size_t base = (static_cast<size_t>(h) * head_dim + d) * state_size;
    float acc = 0.0f;
    for (int s = 0; s < state_size; ++s) {
        const float st = state_in[base + s] * dA + dth * Bg[s] * xsd;
        state_out[base + s] = st;
        acc += st * Cg[s];
    }
    out[h * head_dim + d] = acc + xsd * D[h];
}

// Gated RMSNorm with a tree reduction so group_size can exceed the thread count
// (production group is 2048). Shared: one float per thread for the squares.
__global__ void gated_rmsnorm_kernel(const float *__restrict__ x,
                                     const float *__restrict__ gate,
                                     const float *__restrict__ weight, int n,
                                     int group_size, float eps,
                                     float *__restrict__ out)
{
    __shared__ float sq[1024];
    const int g = blockIdx.x;
    const int base = g * group_size;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    float ss = 0.0f;
    for (int i = tid; i < group_size; i += nthr) {
        const float h = x[base + i] * silu_f(gate[base + i]);
        ss += h * h;
    }
    sq[tid] = ss;
    __syncthreads();
    for (int sft = nthr / 2; sft > 0; sft >>= 1) {
        if (tid < sft) sq[tid] += sq[tid + sft];
        __syncthreads();
    }
    const float inv = rsqrtf(sq[0] / static_cast<float>(group_size) + eps);
    for (int i = tid; i < group_size; i += nthr) {
        const float h = x[base + i] * silu_f(gate[base + i]);
        out[base + i] = weight[base + i] * h * inv;
    }
}

// Grouped top-k router. One block, one thread per expert (E <= 1024).
//
// Stage 1: scores/choice per expert.
// Stage 2: the group score is the sum of the top TWO choices in each group
//          (NemotronHTopkRouter). One thread per group scans its own group, so
//          the scan is deterministic and identical to the host reference's
//          strict-`>` comparison order.
// Stage 3: thread 0 keeps the best `topk_group` groups (lowest index on a tie).
// Stage 4: parallel top-k reduction over survivors, lowest index on a tie.
__global__ void router_kernel(const float *__restrict__ logits,
                              const float *__restrict__ bias, int E, int k,
                              int n_group, int topk_group, float scale,
                              float *__restrict__ weights,
                              int *__restrict__ indices)
{
    __shared__ float s[frozen::kNumRouted];
    __shared__ float ch[frozen::kNumRouted];
    __shared__ int sel[frozen::kNumRouted];
    __shared__ float gscore[kMaxRouterGroups];
    __shared__ int gkeep[kMaxRouterGroups];
    __shared__ float rv[1024];
    __shared__ int ri[1024];

    const int e = threadIdx.x;
    const int per_group = E / n_group;

    if (e < E) {
        const float sg = 1.0f / (1.0f + expf(-logits[e]));
        s[e] = sg;
        ch[e] = (bias ? sg + bias[e] : sg);
        sel[e] = 0;
    }
    __syncthreads();

    // Stage 2: top-2 sum inside each group.
    if (e < n_group) {
        const int base = e * per_group;
        float best1 = -INFINITY, best2 = -INFINITY;
        for (int i = 0; i < per_group; ++i) {
            const float v = ch[base + i];
            if (v > best1) {
                best2 = best1;
                best1 = v;
            } else if (v > best2) {
                best2 = v;
            }
        }
        gscore[e] = best1 + (per_group > 1 ? best2 : 0.0f);
        gkeep[e] = 0;
    }
    __syncthreads();

    // Stage 3: keep the best topk_group groups (lowest index wins a tie).
    if (e == 0) {
        for (int pick = 0; pick < topk_group; ++pick) {
            int best = -1;
            for (int g = 0; g < n_group; ++g) {
                if (gkeep[g]) continue;
                if (best < 0 || gscore[g] > gscore[best]) best = g;
            }
            if (best >= 0) gkeep[best] = 1;
        }
        // Mask the survivors' losers out of the choice scores.
        for (int g = 0; g < n_group; ++g) {
            if (gkeep[g]) continue;
            for (int i = 0; i < per_group; ++i) ch[g * per_group + i] = -INFINITY;
        }
    }
    __syncthreads();

    // Stage 4: top-k over the survivors.
    for (int t = 0; t < k; ++t) {
        rv[e] = (e < E && !sel[e]) ? ch[e] : -INFINITY;
        ri[e] = e;
        __syncthreads();
        for (int sft = blockDim.x / 2; sft > 0; sft >>= 1) {
            if (e < sft) {
                if (rv[e + sft] > rv[e] ||
                    (rv[e + sft] == rv[e] && ri[e + sft] < ri[e])) {
                    rv[e] = rv[e + sft];
                    ri[e] = ri[e + sft];
                }
            }
            __syncthreads();
        }
        if (e == 0) {
            indices[t] = ri[0];
            sel[ri[0]] = 1;
        }
        __syncthreads();
    }

    if (e == 0) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            weights[t] = s[indices[t]];
            sum += weights[t];
        }
        const float denom = sum + 1e-20f;
        for (int t = 0; t < k; ++t) weights[t] = weights[t] / denom * scale;
    }
}

__global__ void attn_decode_kernel(const float *__restrict__ q,
                                   const float *__restrict__ k_cache,
                                   const float *__restrict__ v_cache, int heads,
                                   int kv_heads, int head_dim, int seq_len,
                                   int cache_stride, float scale,
                                   float *__restrict__ out)
{
    __shared__ float red[512];
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const int rep = heads / kv_heads;
    const int hk = h / rep;
    const float qd = q[h * head_dim + d];

    float m = -INFINITY;
    float l = 0.0f;
    float o = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        const size_t kv_base = (static_cast<size_t>(hk) * cache_stride + t) * head_dim;
        red[d] = qd * k_cache[kv_base + d];
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (d < st) red[d] += red[d + st];
            __syncthreads();
        }
        const float score = red[0] * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float p = expf(score - m_new);
        l = l * alpha + p;
        o = o * alpha + p * v_cache[kv_base + d];
        m = m_new;
        __syncthreads();
    }
    out[h * head_dim + d] = o / l;
}

__global__ void relu2_half_kernel(__half *__restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    const float r = v > 0.0f ? v : 0.0f;
    x[i] = __float2half(r * r);
}

__global__ void moe_weighted_sum_kernel(const __half *__restrict__ y,
                                        const float *__restrict__ w, int count,
                                        int n, float *__restrict__ acc, int add)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float s = add ? acc[j] : 0.0f;
    for (int e = 0; e < count; ++e)
        s += w[e] * __half2float(y[static_cast<size_t>(e) * n + j]);
    acc[j] = s;
}

__global__ void add_half_to_f32_kernel(const __half *__restrict__ x, int n,
                                       float *__restrict__ acc)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    acc[j] += __half2float(x[j]);
}

} // namespace

void RunUltraConvStep(const float *xBC, const float *w, const float *bias,
                      const float *state_in, float *state_out, float *out,
                      int conv_dim, int K, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (conv_dim + threads - 1) / threads;
    conv_step_kernel<<<blocks, threads, 0, stream>>>(xBC, w, bias, state_in,
                                                     state_out, out, conv_dim, K);
}

void RunUltraSelectiveStateStep(const float *state_in, const float *xs,
                                const float *B, const float *C, const float *D,
                                const float *dt, const float *dt_bias,
                                const float *A_log, int heads, int head_dim,
                                int groups, int state_size, int clamp_dt,
                                float *state_out, float *out, cudaStream_t stream)
{
    ssm_step_kernel<<<heads, head_dim, 0, stream>>>(
        state_in, xs, B, C, D, dt, dt_bias, A_log, heads, head_dim, groups,
        state_size, clamp_dt, state_out, out);
}

void RunUltraGatedRmsNorm(const float *x, const float *gate, const float *weight,
                          int n, int group_size, float eps, float *out,
                          cudaStream_t stream)
{
    const int groups = n / group_size;
    // Cap the block at 1024 threads (CC 12.x) and stride the rest: the
    // production group is 2048, so one-thread-per-element cannot launch.
    int threads = group_size < 1024 ? (group_size < 32 ? 32 : group_size) : 1024;
    // Power-of-two block for the tree reduction.
    int t = 1;
    while (t * 2 <= threads) t *= 2;
    threads = t;
    gated_rmsnorm_kernel<<<groups, threads, 0, stream>>>(x, gate, weight, n,
                                                         group_size, eps, out);
}

void RunUltraAttnDecode(const float *q, const float *k_cache,
                        const float *v_cache, int heads, int kv_heads,
                        int head_dim, int seq_len, int cache_stride, float scale,
                        float *out, cudaStream_t stream)
{
    // One block per query head; head_dim threads (head_dim <= 512).
    attn_decode_kernel<<<heads, head_dim, 0, stream>>>(
        q, k_cache, v_cache, heads, kv_heads, head_dim, seq_len, cache_stride,
        scale, out);
}

void RunUltraRouter(const float *logits, const float *e_score_bias, int E, int k,
                    int n_group, int topk_group, float scale, float *weights,
                    int *indices, cudaStream_t stream)
{
    int threads = 1;
    while (threads < E) threads <<= 1;  // power-of-two reduction width
    router_kernel<<<1, threads, 0, stream>>>(logits, e_score_bias, E, k, n_group,
                                             topk_group, scale, weights, indices);
}

void RunUltraRelu2Half(__half *x, int n, cudaStream_t stream)
{
    const int threads = 256;
    relu2_half_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, n);
}

void RunUltraMoeWeightedSum(const __half *y, const float *weights, int count, int n,
                            float *acc, bool add, cudaStream_t stream)
{
    const int threads = 256;
    moe_weighted_sum_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        y, weights, count, n, acc, add ? 1 : 0);
}

void RunUltraAddHalfToF32(const __half *x, int n, float *acc, cudaStream_t stream)
{
    const int threads = 256;
    add_half_to_f32_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x,
                                                                               n, acc);
}

} // namespace lsxcommon::nemotron_ultra
