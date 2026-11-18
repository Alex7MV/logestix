#pragma once

// Muse-Glimmer-30B runtime — Model Ingestion Protocol step 5.
//
// One stream, no per-token cudaDeviceSynchronize. Every dense projection goes
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); the
// family-local kernels are the centered norms, the weightless per-head QK
// norm + Q scale, the sliding-band / NoPE causal GQA attention, the sigmoid
// attention output gate, SwiGLU (silu), the FP16 embed gather + weightless
// embed norm and the softcapped FP16-head logits/argmax.
//
// Activations are FP16 with FP32 accumulation; the KV cache is addressed
// exclusively through KvCacheGeometry (uniform 2-head x 128 per layer,
// pos-major).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/muse_dense_constants.h"
#include "lsxcommon/muse_dense_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/token_stream.h"
#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace muse_dense {

// ── Generation ──────────────────────────────────────────────────────────────
struct MuseDenseGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<std::int32_t> tokens;
    std::vector<float> final_logits;  // last decode step (softcapped)
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
    std::uint64_t resident_bytes = 0;
};

// Full generation loop: prefill(input) + up to max_gen_tokens decode steps on
// ONE stream. @p kv must be pre-allocated for prompt + generated capacity.
// The driver body lands with the kernel suite (Task 7); the signature is the
// contract the model TU compiles against.
cudaError_t RunMuseDenseGeneration(
    cudaStream_t stream, const MuseDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, MuseDenseGenerationResult &out,
    MuseDenseKvCache &kv,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace muse_dense
} // namespace lsxcommon