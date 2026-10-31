#pragma once

#include <arrow/api.h>

#include <string>

#include "lsxcommon/hy_moe_convert.h"

namespace lsxcommon::hy_simulmt {

// Identity + policy for the Hy-SimulMT conversion: the hy_v3 text backbone,
// own model_type/format tag, and the speech encoder (`audio_tower.*`) stripped
// and counted as ballast.
hy::HyConvertSpec SimulMTConvertSpec();

// Converts a Hy-SimulMT-30B-A3B model directory into the unified
// <output_root>/logestix arrow layout.
arrow::Status ConvertHySimulMTModel(const std::string &input_dir,
                                    const std::string &output_root);

} // namespace lsxcommon::hy_simulmt
