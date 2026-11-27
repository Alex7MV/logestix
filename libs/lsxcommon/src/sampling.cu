#include "lsxcommon/sampling.h"

#include "lsxcommon/core.h"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace lsxcommon {
namespace sample {

namespace {

// Partial buffers for the two-phase greedy argmax.
// Phase 1: up to kMaxPartials blocks each produce a (value, index) partial;
// Phase 2: one block combines them in ascending block order. The tie-break
// ("smaller index wins") composes correctly with strict-> partial maxima and
// keeps results bit-identical to the old single-block kernel.
//
// The scratch is per-thread (lazily cudaMalloc'd once), NOT a process-global
// __device__ array: two concurrent Sample() calls on independent streams would
// otherwise overwrite each other's partials and return the wrong argmax
// (regression test: "GreedySampler is safe across concurrent streams").
constexpr int kMaxPartials = 512;

struct GreedyScratch {
    float *val = nullptr;
    int32_t *idx = nullptr;
    ~GreedyScratch() {
        cudaFreeIgnoreShutdown(val);
        cudaFreeIgnoreShutdown(idx);
    }
};

GreedyScratch &ScratchForThisThread()
{
    static thread_local GreedyScratch scratch;
    return scratch;
}

// Phase-1 kernel: one partial (value, index) per block into the caller's
// per-thread scratch (indexed by blockIdx.x — no atomics, fully deterministic).
template <typename T>
__global__ void kGreedyPartial(const T* __restrict__ logits, int vocab,
                               int items_per_thread, float* __restrict__ pval,
                               int32_t* __restrict__ pidx) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    const int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x
                          * items_per_thread;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items_per_thread);

    float best = -1e30f;
    int32_t best_i = -1;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float v = static_cast<float>(logits[i]);
        if (v > best) { best = v; best_i = static_cast<int32_t>(i); }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pval[blockIdx.x] = s_best[0];
        pidx[blockIdx.x] = s_idx[0];
    }
}

__global__ void kGreedyFinalize(int num_partials, const float* __restrict__ pval,
                                const int32_t* __restrict__ pidx,
                                int32_t* out_tok, float* out_maxv) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    float best = -1e30f;
    int32_t best_i = -1;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        const float v = pval[i];
        const int32_t vi = pidx[i];
        if (v > best || (v == best && vi < best_i)) { best = v; best_i = vi; }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_tok = s_idx[0];
        *out_maxv = s_best[0];
    }
}

// GLM's fused argmax semantics: each FP32 logit is rounded to FP16 (RNE) before
// comparing — a bit-exact mirror of the host baseline's FP16-stored lm_head row.
// maxv stores the rounded value, exactly as GlmArgmaxKernel does.
__global__ void kGreedyPartialRounded16(const float* __restrict__ logits,
                                        int vocab, int items_per_thread,
                                        float* __restrict__ pval,
                                        int32_t* __restrict__ pidx) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    const int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x
                          * items_per_thread;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items_per_thread);

    float best = -1e30f;
    int32_t best_i = -1;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float v = __half2float(__float2half(logits[i]));
        if (v > best) { best = v; best_i = static_cast<int32_t>(i); }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pval[blockIdx.x] = s_best[0];
        pidx[blockIdx.x] = s_idx[0];
    }
}

}  // namespace

cudaError_t GreedySampler::Sample(cudaStream_t stream, const void* d_logits,
                                  int vocab, LogitType dtype,
                                  int32_t* d_out_token, float* d_out_maxv) {
    if (!d_logits || !d_out_token || !d_out_maxv || vocab <= 0)
        return cudaErrorInvalidValue;
    constexpr int kThreads = 256;
    // Two-phase multi-block argmax: phase 1 spreads the vocab scan across
    // ~vocab/(256*32) blocks (vs one block on one SM before), phase 2 combines
    // the partials. Same tie-break semantics as the single-block kernel.
    constexpr int kItemsPerThread = 32;
    const int blocks = static_cast<int>(
        (static_cast<int64_t>(vocab) + kThreads * kItemsPerThread - 1)
        / (kThreads * kItemsPerThread));
    if (blocks > kMaxPartials) return cudaErrorInvalidValue;

    GreedyScratch &scr = ScratchForThisThread();
    if (scr.val == nullptr &&
        cudaMalloc(&scr.val, sizeof(float) * kMaxPartials) != cudaSuccess) {
        scr.val = nullptr;
        return cudaGetLastError();
    }
    if (scr.idx == nullptr &&
        cudaMalloc(&scr.idx, sizeof(int32_t) * kMaxPartials) != cudaSuccess) {
        scr.idx = nullptr;
        return cudaGetLastError();
    }

    if (dtype == LogitType::kF32) {
        kGreedyPartial<float><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, kItemsPerThread,
            scr.val, scr.idx);
    } else if (dtype == LogitType::kF16) {
        kGreedyPartial<__half><<<blocks, kThreads, 0, stream>>>(
            static_cast<const __half*>(d_logits), vocab, kItemsPerThread,
            scr.val, scr.idx);
    } else if (dtype == LogitType::kF32Rounded16) {
        // GLM's fused argmax semantics: each FP32 logit is rounded to FP16
        // (RNE) before comparing — a bit-exact mirror of the host baseline's
        // FP16-stored lm_head row. maxv stores the rounded value, exactly as
        // GlmArgmaxKernel does.
        kGreedyPartialRounded16<<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, kItemsPerThread,
            scr.val, scr.idx);
    } else {
        return cudaErrorInvalidValue;
    }
    kGreedyFinalize<<<1, kThreads, 0, stream>>>(blocks, scr.val, scr.idx,
                                                d_out_token, d_out_maxv);
    return cudaGetLastError();
}

// ── SamplingSampler ─────────────────────────────────────────────────────────
// Deterministic temperature/top-k/top-p sampler. All state is per-thread; the
// RNG is counter-based (splitmix64 over seed+counter), so a request with a
// fixed seed replays its token stream exactly and no device RNG state exists.
namespace {

constexpr int kSampleBins = 4096;

struct SampleScratch {
    float *psum = nullptr;      // [kMaxPartials] block partial sums
    float *pmax = nullptr;      // [kMaxPartials] argmax partial values
    int32_t *pidx = nullptr;    // [kMaxPartials]
    uint32_t *hist = nullptr;   // [256] radix histogram
    uint32_t *state = nullptr;  // [2] prefix / rank
    int32_t *counts = nullptr;  // [kSampleBins] top-p counts
    ~SampleScratch() {
        cudaFreeIgnoreShutdown(psum);
        cudaFreeIgnoreShutdown(pmax);
        cudaFreeIgnoreShutdown(pidx);
        cudaFreeIgnoreShutdown(hist);
        cudaFreeIgnoreShutdown(state);
        cudaFreeIgnoreShutdown(counts);
    }
};

SampleScratch &SampleScratchForThisThread()
{
    static thread_local SampleScratch s;
    if (s.psum == nullptr) {
        cudaMalloc(&s.psum, sizeof(float) * kMaxPartials);
        cudaMalloc(&s.pmax, sizeof(float) * kMaxPartials);
        cudaMalloc(&s.pidx, sizeof(int32_t) * kMaxPartials);
        cudaMalloc(&s.hist, sizeof(uint32_t) * 256);
        cudaMalloc(&s.state, sizeof(uint32_t) * 2);
        cudaMalloc(&s.counts, sizeof(int32_t) * kSampleBins);
        cudaGetLastError();
    }
    return s;
}

__device__ __forceinline__ float SampleLoad(const float *p) { return *p; }
__device__ __forceinline__ float SampleLoad(const __half *p) { return __half2float(*p); }

__device__ __forceinline__ float SampleExp(float d, float inv_temp)
{
    return expf(-d * inv_temp);
}

// Per-block sum of exp((v - max) / T) over the allowed set (d <= dthr).
template <typename T>
__global__ void kSampleSumPartial(const T *__restrict__ row, int vocab,
                                  int items, float maxv, float inv_temp,
                                  float dthr, float *__restrict__ psum) {
    __shared__ float sh[256];
    const int64_t begin =
        static_cast<int64_t>(blockIdx.x) * blockDim.x * items;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items);
    float s = 0.0f;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float d = maxv - SampleLoad(row + i);
        if (d <= dthr) s += SampleExp(d, inv_temp);
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) psum[blockIdx.x] = sh[0];
}

// One thread walks the partials in fixed order (deterministic), picks a block,
// then walks that block's slice. Chosen for determinism, not throughput: the
// slice is ~vocab/blocks elements.
template <typename T>
__global__ void kSamplePick(int nparts, const float *__restrict__ psum,
                            const T *__restrict__ row, int vocab, int items,
                            float maxv, float inv_temp, float dthr,
                            unsigned long long rng, int32_t *out_tok,
                            float *out_maxv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    for (int i = 0; i < nparts; ++i) total += psum[i];
    *out_maxv = maxv;
    if (!(total > 0.0) || !isfinite(total)) {
        // Degenerate row (all -inf / underflowed): first maximum wins, the
        // same tie-break GreedySampler uses.
        for (int i = 0; i < vocab; ++i)
            if (SampleLoad(row + i) == maxv) { *out_tok = i; return; }
        *out_tok = 0;
        return;
    }
    const double u = static_cast<double>(rng >> 11) *
                     0x1.0p-53;
    const double target = u * total;
    int chosen = nparts - 1;
    double before = 0.0;
    for (int b = 0; b < nparts; ++b) {
        if (before + psum[b] >= target) { chosen = b; break; }
        before += psum[b];
    }
    const int64_t begin =
        static_cast<int64_t>(chosen) * 256 * items;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(256) * items);
    double cum = before;
    for (int64_t i = begin; i < end; ++i) {
        const float d = maxv - SampleLoad(row + i);
        if (d > dthr) continue;
        cum += SampleExp(d, inv_temp);
        if (cum >= target) { *out_tok = static_cast<int32_t>(i); return; }
    }
    // Numeric edge: take the last allowed index in the chosen block.
    for (int64_t i = end - 1; i >= begin; --i) {
        if ((maxv - SampleLoad(row + i)) <= dthr) {
            *out_tok = static_cast<int32_t>(i);
            return;
        }
    }
    *out_tok = 0;
}

// Radix histogram for the exact k-th-largest selection on d = max - v.
template <typename T>
__global__ void kRadixHist(const T *__restrict__ row, int vocab, int items,
                           float maxv, uint32_t mask, const uint32_t *state,
                           int shift, uint32_t *__restrict__ hist) {
    const uint32_t prefix = state[0];
    __shared__ uint32_t sh[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = 0;
    __syncthreads();
    const int64_t begin =
        static_cast<int64_t>(blockIdx.x) * blockDim.x * items;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items);
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint32_t d = __float_as_uint(maxv - SampleLoad(row + i));
        if ((d & mask) == prefix) atomicAdd(&sh[(d >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

// Serial 256-bin walk (one thread): monotone, order-independent of the
// histogram accumulation.
__global__ void kRadixSelect(uint32_t *__restrict__ hist,
                             uint32_t *__restrict__ state, int shift) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t prefix = state[0];
    uint32_t rank = state[1];
    uint32_t cum = 0;
    int bin = 255;
    for (int i = 0; i < 256; ++i) {
        if (cum + hist[i] >= rank) { bin = i; break; }
        cum += hist[i];
    }
    state[0] = prefix | (static_cast<uint32_t>(bin) << shift);
    state[1] = rank - cum;
}

// Top-p histogram over the allowed set: integer counts only (deterministic),
// each bin's mass approximated from its UPPER edge so the nucleus is
// conservative (retains at least top_p, overshoots by at most one bin).
template <typename T>
__global__ void kTopPHist(const T *__restrict__ row, int vocab, int items,
                          float maxv, float dthr, float dmax,
                          int32_t *__restrict__ counts) {
    const int64_t begin =
        static_cast<int64_t>(blockIdx.x) * blockDim.x * items;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items);
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float d = maxv - SampleLoad(row + i);
        if (d > dthr) continue;
        int bin = static_cast<int>(d / dmax * kSampleBins);
        if (bin >= kSampleBins) bin = kSampleBins - 1;
        if (bin < 0) bin = 0;
        atomicAdd(&counts[bin], 1);
    }
}

__global__ void kTopPSelect(const int32_t *__restrict__ counts,
                            float inv_temp, float dmax, float top_p,
                            float *__restrict__ dthr_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float mass[kSampleBins];
    double total = 0.0;
    for (int i = 0; i < kSampleBins; ++i) {
        const float upper = (i + 1) * dmax / kSampleBins;
        mass[i] = counts[i] * SampleExp(upper, inv_temp);
        total += mass[i];
    }
    if (!(total > 0.0)) return;  // keep the caller threshold
    double cum = 0.0;
    for (int i = 0; i < kSampleBins; ++i) {
        cum += mass[i];
        if (cum >= top_p * total) {
            // Include the whole boundary bin.
            *dthr_out = (i + 1) * dmax / kSampleBins;
            return;
        }
    }
}

__host__ __device__ __forceinline__ unsigned long long SplitMix64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

template <typename T>
cudaError_t SampleImpl(cudaStream_t stream, const void *d_logits, int vocab,
                       LogitType /*dtype*/, int32_t *d_out_token,
                       float *d_out_maxv, const SamplingParams &params,
                       unsigned long long rng) {
    const T *row = static_cast<const T *>(d_logits);
    constexpr int kThreads = 256;
    constexpr int kItems = 32;
    const int blocks = static_cast<int>(
        (static_cast<int64_t>(vocab) + kThreads * kItems - 1) /
        (kThreads * kItems));
    if (blocks > kMaxPartials) return cudaErrorInvalidValue;
    SampleScratch &s = SampleScratchForThisThread();
    if (!s.psum || !s.pmax || !s.pidx || !s.hist || !s.state || !s.counts)
        return cudaErrorMemoryAllocation;

    kGreedyPartial<T><<<blocks, kThreads, 0, stream>>>(
        row, vocab, kItems, s.pmax, s.pidx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    kGreedyFinalize<<<1, kThreads, 0, stream>>>(blocks, s.pmax, s.pidx,
                                                d_out_token, d_out_maxv);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    float maxv = 0.0f;
    if (cudaMemcpyAsync(&maxv, d_out_maxv, sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return cudaGetLastError();
    if (cudaStreamSynchronize(stream) != cudaSuccess) return cudaGetLastError();

    const float inv_temp =
        1.0f / (params.temperature > 0.0f ? params.temperature : 1.0f);
    float dthr = INFINITY;

    // Exact top-k: four radix passes over the float bits of d.
    if (params.top_k > 0 && params.top_k < vocab) {
        uint32_t state[2] = {0u, static_cast<uint32_t>(params.top_k)};
        if (cudaMemcpyAsync(s.state, state, sizeof(state),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return cudaGetLastError();
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            const uint32_t mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
            if (cudaMemsetAsync(s.hist, 0, sizeof(uint32_t) * 256, stream) !=
                cudaSuccess)
                return cudaGetLastError();
            kRadixHist<T><<<blocks, kThreads, 0, stream>>>(
                row, vocab, kItems, maxv, mask, s.state, shift, s.hist);
            e = cudaGetLastError();
            if (e != cudaSuccess) return e;
            kRadixSelect<<<1, 1, 0, stream>>>(s.hist, s.state, shift);
            e = cudaGetLastError();
            if (e != cudaSuccess) return e;
        }
        uint32_t key = 0;
        if (cudaMemcpyAsync(&key, s.state, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            return cudaGetLastError();
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return cudaGetLastError();
        float key_f = 0.0f;
        std::memcpy(&key_f, &key, sizeof(key_f));
        dthr = key_f;
    }

    // Nucleus: conservative 4096-bin threshold; the boundary bin is kept.
    if (params.top_p < 1.0f) {
        const float dmax = std::max(88.0f * (1.0f / inv_temp), 1e-6f);
        if (cudaMemsetAsync(s.counts, 0, sizeof(int32_t) * kSampleBins,
                            stream) != cudaSuccess)
            return cudaGetLastError();
        kTopPHist<T><<<blocks, kThreads, 0, stream>>>(
            row, vocab, kItems, maxv, dthr, dmax, s.counts);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        // s.pmax is free after the max finalize: reuse slot 0 for the
        // device-computed nucleus threshold.
        if (cudaMemcpyAsync(s.pmax, &dthr, sizeof(float),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return cudaGetLastError();
        kTopPSelect<<<1, 1, 0, stream>>>(s.counts, inv_temp, dmax,
                                         params.top_p, s.pmax);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        float dthr_p = dthr;
        if (cudaMemcpyAsync(&dthr_p, s.pmax, sizeof(float),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            return cudaGetLastError();
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return cudaGetLastError();
        if (dthr_p < dthr) dthr = dthr_p;
    }

    kSampleSumPartial<T><<<blocks, kThreads, 0, stream>>>(
        row, vocab, kItems, maxv, inv_temp, dthr, s.psum);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    kSamplePick<T><<<1, 1, 0, stream>>>(blocks, s.psum, row, vocab, kItems,
                                        maxv, inv_temp, dthr, rng, d_out_token,
                                        d_out_maxv);
    return cudaGetLastError();
}

}  // namespace

SamplingSampler::SamplingSampler(SamplingParams params)
{
    Configure(params);
}

void SamplingSampler::Configure(const SamplingParams &params)
{
    params_ = params;
    // A greedy request must go through GreedySampler (bit-exact); Sample()
    // rejects it here rather than silently coercing a temperature.
    if (params_.top_p <= 0.0f) params_.top_p = 1.0f;
    if (params_.top_p > 1.0f) params_.top_p = 1.0f;
    if (params_.top_k < 0) params_.top_k = 0;
    counter_ = 0;  // one sampler per request: replayable from the seed
}

cudaError_t SamplingSampler::Sample(cudaStream_t stream, const void *d_logits,
                                    int vocab, LogitType dtype,
                                    int32_t *d_out_token, float *d_out_maxv)
{
    if (!d_logits || !d_out_token || !d_out_maxv || vocab <= 0 ||
        params_.temperature <= 0.0f)
        return cudaErrorInvalidValue;
    const unsigned long long rng = SplitMix64(params_.seed ^ (counter_++));
    // kF32Rounded16 uses the same FP16-rounded value the GLM argmax does.
    if (dtype == LogitType::kF32 || dtype == LogitType::kF32Rounded16) {
        return SampleImpl<float>(stream, d_logits, vocab, dtype, d_out_token,
                                 d_out_maxv, params_, rng);
    }
    if (dtype == LogitType::kF16)
        return SampleImpl<__half>(stream, d_logits, vocab, dtype, d_out_token,
                                  d_out_maxv, params_, rng);
    return cudaErrorInvalidValue;
}

ISampler& SamplerFor(const SamplingParams &params)
{
    if (params.IsGreedy()) return DefaultSampler();
    static thread_local SamplingSampler sampler;
    sampler.Configure(params);
    return sampler;
}

cudaError_t SampleToHost(cudaStream_t stream, ISampler &sampler,
                         const void *d_logits, int vocab, LogitType dtype,
                         std::int32_t *out_token, float *out_maxv)
{
    static thread_local int32_t *tok = nullptr;
    static thread_local float *maxv = nullptr;
    if (!tok && cudaMalloc(&tok, sizeof(int32_t)) != cudaSuccess) {
        tok = nullptr;
        return cudaGetLastError();
    }
    if (!maxv && cudaMalloc(&maxv, sizeof(float)) != cudaSuccess) {
        maxv = nullptr;
        return cudaGetLastError();
    }
    cudaError_t e = sampler.Sample(stream, d_logits, vocab, dtype, tok, maxv);
    if (e != cudaSuccess) return e;
    if (cudaMemcpyAsync(out_token, tok, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return cudaGetLastError();
    if (cudaMemcpyAsync(out_maxv, maxv, sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return cudaGetLastError();
    if (cudaStreamSynchronize(stream) != cudaSuccess) return cudaGetLastError();
    return cudaSuccess;
}

ISampler& DefaultSampler() {
    static GreedySampler g;
    return g;
}

}  // namespace sample
}  // namespace lsxcommon