#pragma once

#include <string>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Host-only: mmap and load a converted model's tokenizer arrows
/// (`<model_path>/logestix/pinned/tokenizer_{vocab,merges}.arrow`, with the
/// MoE alternate roots). No GPU, no weights. Returns false with `out.error_msg`
/// set on missing files or format error.
bool LoadTokenizerArrows(const std::string& model_path, ArrowTokenizerDb& out);

}  // namespace lsxcommon
