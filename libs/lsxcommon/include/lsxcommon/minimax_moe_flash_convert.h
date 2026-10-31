#pragma once

#include <arrow/api.h>

#include <string>

namespace lsxcommon::minimax {

// Converter entry point for the MiniMax-M3.1-Flash sibling. It runs the
// shared MiniMax pipeline (the emitted artifact layout is identical, so the
// parent loader/kernels consume it unchanged) with the sibling's identity
// and its variant discriminator gate: RoPE theta 1e7 + MiniMax Sparse
// Attention on EVERY layer (source: MXFP8 dense, already-NVFP4 experts).
arrow::Status ConvertMinimaxFlashModel(const std::string &input_dir,
                                       const std::string &output_root);

} // namespace lsxcommon::minimax
