#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Isolated sampling seam (spec item 13). HOST + CUDA safe: no absl containers,
// no device-side state. Engines keep their fused argmax kernels; GreedySampler
// is the default wrapper where a TopK/TopP sampler later reads the full row.
namespace lsxcommon {
namespace sample {

// Request-level sampling policy. The default (temperature == 0) is greedy
// argmax -- the bit-exact numeric class the DET/FSI/CODE goldens pin. Any
// positive temperature selects SamplingSampler; top_p < 1 and top_k > 0 are
// only consulted when sampling is active, so a greedy request never pays for
// them.
struct SamplingParams {
    float temperature = 0.0f;  // <= 0 => greedy
    float top_p = 1.0f;        // nucleus mass, 1 => disabled
    int top_k = 0;             // 0 => disabled
    std::uint64_t seed = 0;    // deterministic sequence per (seed, request)

    bool IsGreedy() const { return !(temperature > 0.0f); }
};

enum class LogitType {
    kF32 = 0,         // float logits rows (DeepSeek d_logits)
    kF16 = 1,         // __half logits rows (Qwen d_logits)
    kF32Rounded16 = 2 // float row, compared as FP16-rounded (GLM d_logits_head)
};

class ISampler {
public:
    virtual ~ISampler() = default;

    // Selects one token id from a [vocab] logits row on `stream`.
    //   d_logits    device pointer to the row (FP32 or FP16 per @p dtype).
    //   d_out_token device int32 slot receiving the selected id.
    //   d_out_maxv  device float slot receiving the raw max logit.
    // Returns cudaSuccess or a CUDA error code; never throws.
    virtual cudaError_t Sample(cudaStream_t stream, const void* d_logits,
                               int vocab, LogitType dtype,
                               int32_t* d_out_token, float* d_out_maxv) = 0;

    // True when this sampler reproduces greedy argmax. Families with a fused
    // decode tail use it to keep their fast path and only pay for sampling
    // when the request asked for it.
    virtual bool IsGreedy() const { return true; }
};

// Greedy argmax sampler. Matches the engines' fused-argmax semantics: on ties
// the FIRST (lowest index) maximum wins. Single block, <= 256 threads.
// Thread-safe ACROSS HOST THREADS: the two-phase partial scratch is
// thread_local (lazily allocated), so two host threads (each on its own
// stream) do not interfere. The scratch is NOT keyed by CUDA stream, so a
// single host thread must not call Sample() concurrently on two streams.
class GreedySampler : public ISampler {
public:
    cudaError_t Sample(cudaStream_t stream, const void* d_logits, int vocab,
                       LogitType dtype, int32_t* d_out_token,
                       float* d_out_maxv) override;
};

// Temperature / top-k / top-p sampler with a counter-based RNG. Deterministic
// for a fixed (seed, row sequence): Configure() resets the draw counter, so
// one sampler per request reproduces its token stream exactly. top_k is an
// exact selection (ties at the k-th value are all kept); top_p resolves the
// nucleus on a 4096-bin histogram of the temperature-scaled excess logit and
// includes the boundary bin, so the retained mass is >= top_p and exceeds it
// by at most one bin.
class SamplingSampler : public ISampler {
public:
    explicit SamplingSampler(SamplingParams params = {});
    void Configure(const SamplingParams &params);

    cudaError_t Sample(cudaStream_t stream, const void *d_logits, int vocab,
                       LogitType dtype, int32_t *d_out_token,
                       float *d_out_maxv) override;
    bool IsGreedy() const override { return false; }

private:
    SamplingParams params_;
    std::uint64_t counter_ = 0;
};

// Returns the greedy singleton for a greedy request, otherwise a thread-local
// SamplingSampler configured from `params` (one call per request: it resets
// the draw counter). Host-thread safe; not safe to call Sample() concurrently
// on two streams from one thread.
ISampler& SamplerFor(const SamplingParams &params);

// Convenience for families whose fused decode tail already produced a greedy
// token: samples into per-thread device scratch and copies the {token, maxv}
// pair back. Callers use it only when `!SamplingParams::IsGreedy()` so the
// greedy fast path is untouched. Host-thread safe; one call in flight per
// thread.
cudaError_t SampleToHost(cudaStream_t stream, ISampler &sampler,
                         const void *d_logits, int vocab, LogitType dtype,
                         std::int32_t *out_token, float *out_maxv);

// Function-local static default instance (stable address for default args).
ISampler& DefaultSampler();

}  // namespace sample
}  // namespace lsxcommon