#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace runtime {

// Item 2 — unified decode capacity-retry. Decodes a generated token stream
// into text. Grows the output buffer on the tokenizer's -2 (capacity too
// small) return; returns "" (with an spdlog warning) on persistent failure.
// Host-only: safe to call from .cu TUs.
std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids);

// Item 3 — unified SelfCheck. Forces max(2, req.max_gen_tokens), delegates to
// model.Infer, then matches the decoded 1->2-token prefix against "Paris".
// Returns the InferenceResult with selfcheck_passed filled.
gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req);

} // namespace runtime
} // namespace lsxcommon