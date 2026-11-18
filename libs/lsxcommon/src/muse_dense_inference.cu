// Muse-Glimmer-30B CUDA runtime — Model Ingestion Protocol steps 5/6.
// The kernel suite and the generation driver body land with Task 6/7; this
// TU carries the driver entry point the model TU links against.
#include "lsxcommon/muse_dense_inference.h"

#include "lsxcommon/log.h"

namespace lsxcommon {
namespace muse_dense {

cudaError_t RunMuseDenseGeneration(
    cudaStream_t stream, const MuseDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, MuseDenseGenerationResult &out,
    MuseDenseKvCache &kv, lsxcommon::sample::ISampler &sampler,
    const lsxcommon::GenerationStream *gs)
{
    (void)stream;
    (void)pw;
    (void)input_ids;
    (void)max_gen_tokens;
    (void)eos_ids;
    (void)kv;
    (void)sampler;
    (void)gs;
    out.ok = false;
    out.error_msg = "muse-glimmer generation driver not yet implemented";
    return cudaErrorNotYetImplemented;
}

} // namespace muse_dense
} // namespace lsxcommon