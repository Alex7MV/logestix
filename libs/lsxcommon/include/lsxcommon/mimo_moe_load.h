#pragma once

// MiMo-V2.6-Flash-RL layout loader (Model Ingestion Protocol step 3):
// validates the `logestix-mimo-nvfp4` index and resolves the on-disk weight
// files. Host-only; device upload lives in the model class.
#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/mimo_moe_convert.h"

namespace lsxcommon {
namespace mimo {

struct MiMoModelFiles {
    std::string root;             // <root>/logestix
    std::string global_file;      // pinned/global.arrow
    std::string experts_root;     // experts/
    std::vector<std::string> layer_raw;     // pinned/layer_NNN.arrow (may be "")
    std::vector<std::string> layer_nvfp4;   // pinned/layer_NNN_nvfp4.arrow
    int num_layers = 0;
    MiMoTopology topo;
};

// Cross-checks a parsed topology against the frozen passport. Returns an error
// naming the first mismatching field on failure.
arrow::Status ValidateMiMoGeometry(const MiMoTopology &top);

// Reads <root>/logestix/model.index.json, validates the format tag and the
// frozen geometry (unless SyntheticGeometryAllowed()), and resolves the weight
// files. Errors name the offending tensor/file.
arrow::Result<MiMoModelFiles> LoadMiMoLayout(const std::string &root);

} // namespace mimo
} // namespace lsxcommon
