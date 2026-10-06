#pragma once

// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 loader contract
// (Model Ingestion Protocol stage 3).
//
// The converted tree is byte-compatible with the BF16-distribution sibling's, so
// the heavy lifting (reading the arrow files, staging to the device, the routed
// expert host store + device slot LRU, the session allocation) is the sibling's
// `LoadNemotronIndex` / `LoadNemotronWeights`, reused verbatim.
//
// What is model-specific, and therefore here:
//
//   * `ValidateNemotronNvfp4Geometry` cross-checks the loaded index against THIS
//     model's own passport (`nemotron_nvfp4::frozen`), not the sibling's.
//   * the stop-id SET is frozen and validated, so a tree that claims a stop the
//     vault does not declare fails Load instead of running past it.
//   * `LoadNemotronNvfp4Index` additionally requires the index's `model` field
//     to name THIS distribution. Both distributions emit the same format tag, so
//     without that check the sibling's tree would load happily under this model's
//     kType - and the two models have separate FSI/CODE/DET goldens.

#include <string>
#include <vector>

#include "lsxcommon/nemotron_moe_load.h"

namespace lsxcommon {
namespace nemotron_nvfp4 {

using nemotron_moe::NemotronIndex;

// Reads <model_path>/logestix/model.index.json for THIS model: the shared
// format-tag and field parsing, plus the model-name check. Returns false and
// fills `error` on a missing/unparseable index, an unknown format tag, or an
// index that belongs to a different distribution.
bool LoadNemotronNvfp4Index(const std::string &model_path, NemotronIndex &out,
                            std::string &error);

// Cross-checks `idx` against this model's frozen passport, including the stop
// set. Honors LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 for synthetic fixtures.
// Returns false and fills `error` on the first mismatch.
bool ValidateNemotronNvfp4Geometry(const NemotronIndex &idx, std::string &error);

} // namespace nemotron_nvfp4
} // namespace lsxcommon