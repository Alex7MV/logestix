#pragma once

// GigaChat4-708B-A41B ingestion converter (Model Ingestion Protocol step 1).
// Produces the `logestix-gigachat4-708b-nvfp4` layout under <output>/logestix/:
// model.index.json, pinned/global.arrow, pinned/layer_NNN.arrow,
// pinned/layer_NNN_nvfp4.arrow, experts/layer_N/NNN/exp_E.arrow and
// mtp/layer_NNN.arrow.
//
// The checkpoint is the 37B sibling's architecture at 19x the size: 64 decoder
// layers (3 dense FFN + 61 MoE of 256 experts), 16 gated MLA layers carrying
// the DSA lightning indexer, 48 GatedDeltaNet layers, one MTP layer. It is
// mixed precision in exactly the same way (FP8 E4M3 block-128 for the
// attention/expert/GDN matrices, BF16 for the attention gate, the shared
// experts, the dense MLP and the indexer's wk/weights_proj).
//
// This entry point only supplies the member's frozen expectations to the shared
// family core (gigachat_moe_37b_convert.h); the conversion itself is shared.

#include <arrow/status.h>

#include <string>

#include "lsxcommon/gigachat_moe_37b_convert.h"

namespace lsxcommon::gigachat4_708b {

// The frozen spec of this member: format tag, model name, geometry, layer map
// and attention scale, all from gigachat_moe_708b_constants.h.
gigachat4::Gc4IngestSpec Gc708BIngestSpec();

arrow::Status ConvertGigaChat4_708BModel(const std::string &input_dir,
                                        const std::string &output_root);

} // namespace lsxcommon::gigachat4_708b
