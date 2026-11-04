#pragma once

// Kimi-K3.1-Pro conversion entry (lsxforge --kimi_k31_pro).
//
// The variant is a sibling of Kimi-K3: identical KimiLinear text tower plus a
// Gated-DSA Lightning Indexer on the 24 full-attention layers. The conversion
// itself is the shared Kimi-K3 converter driven with the `k31_pro` variant
// discriminator (the `minimax-m3.1-flash` precedent), which is what enforces
// the two-way gate: a `kimi_k31` checkpoint is required here and a `kimi_k3`
// checkpoint is rejected by `ConvertKimiK3Model` and vice versa.
//
// Emitted layout: `logestix-kimi-k31-pro-nvfp4` under
// <output_root>/logestix, with the five indexer tensors per DSA layer stored
// BF16 verbatim (never quantized: they decide WHICH keys are visible).

#include <arrow/api.h>

#include <string>

namespace lsxcommon::kimi_pro {

arrow::Status ConvertKimiK31ProModel(const std::string &input_dir,
                                     const std::string &output_root);

}  // namespace lsxcommon::kimi_pro
