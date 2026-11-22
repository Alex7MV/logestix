#pragma once

// TML-d24-varloop-research converter (Model Ingestion Protocol stages 1-2).
//
// The checkpoint is `inkling2_loop` with `loop_config.is_varloop: true`: the
// SAME architecture and the same artifact layout as Inkling-2-Mini, converted
// by the same pipeline. This TU therefore supplies only what differs — the
// model row's identity (card name, format tag), its own frozen geometry, and
// the pass ceiling the varloop capability needs recorded:
//
//   * format `logestix-inkling-2-varloop` — the two trees cannot be loaded as
//     each other, because the loader checks the tag before anything else;
//   * `model` = the card name, which `gpu::detect_model_type` normalizes into
//     the factory key `tml-d24-varloop-research`;
//   * `varloop: true` + `loop_passes_max: 5` — the trained range's upper bound.
//     config.json carries only `num_passes` (3, the post-trained default); the
//     1..5 range is stated by the model card, so it belongs to the passport and
//     not to the checkpoint's parser.
//
// Everything else (the mixed NVFP4/MXFP8 routed experts passed through
// verbatim, the w13 row de-interleave, the merged cross-pass queries, the
// per-loop-layer MessageGates, the stripped vision tower) is the parent's
// implementation, unchanged and untested twice.
#include <arrow/api.h>

#include <string>

#include "lsxcommon/inkling_moe_loop_convert.h"

namespace lsxcommon::inkling {

// This row's conversion contract, filled from
// libs/lsxcommon/src/inkling_moe_varloop_constants.h.
InklingLoopConvertSpec VarloopConvertSpec();

// Converts the checkpoint into the logestix-inkling-2-varloop layout under
// <output_root>/logestix. Refuses a fixed-pass checkpoint (`is_varloop: false`)
// and a tree whose declared pass count exceeds the trained ceiling.
arrow::Status ConvertInklingVarloopModel(const std::string &input_dir,
                                         const std::string &output_root);

}  // namespace lsxcommon::inkling
