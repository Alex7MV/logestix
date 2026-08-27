#include "lsxcommon/deepseek_moe_inference_csa.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace lsxcommon {
namespace gpu {

// alloc_csa_state / free_csa_state were removed: DeepSeekInferArena
// (deepseek_moe_load.cuh) now carves the per-layer CSA state from its
// single per-run device block and reproduces the exact zero / -inf init
// (deepseek_moe_load.cu). run_csa_attention lives in
// deepseek_moe_inference_attn.cu.

} // namespace gpu
} // namespace lsxcommon
