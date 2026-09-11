// deepseek_moe_repack.h — offline DeepSeek routed-expert NVFP4 repack
// (logestix-v1 -> logestix-v1-nvfp4), migration plan Task 2
// (docs/superpowers/plans/2026-09-11-deepseek-nvfp4-migration.md).
//
// The DeepSeek vault stores routed experts as E2M1 nibbles with TRUE E8M0
// per-32 exponent scales (verified: layer_0/exp_0 bytes span [119..122] =
// 2^-8..2^-5 with gs = 1.0). The NVFP4 contour (lsx_nvfp4) decodes pow2-E4M3
// per-16 scales multiplied by a per-tensor global_scale, so the repack:
//   1. splits every per-32 scale into two identical per-16 halves;
//   2. remaps each E8M0 byte through E8m0ToE4m3Pow2Pair with a per-tensor
//      shift chosen by ChooseE8m0ScaleShift (exact whenever the byte spread
//      fits the e4 window [1..14]);
//   3. stores the folded shift as the tensor's float global_scale (2^m);
//   4. carries packed nibbles, all non-expert files and the pinned/ tree
//      verbatim, swapping the index format tag to "logestix-v1-nvfp4".
// Host-only, no CUDA. Strict: any missing/corrupt expert file is an error.
#pragma once

#include <arrow/api.h>
#include <string>

namespace lsxcommon {
namespace deepseek {

// Repack @p src (model root containing logestix/ or the logestix directory
// itself) into @p dst (same accepted layouts; created as needed).
arrow::Status RepackDeepseekExpertsNvfp4(const std::string &src,
                                         const std::string &dst);

} // namespace deepseek
} // namespace lsxcommon
