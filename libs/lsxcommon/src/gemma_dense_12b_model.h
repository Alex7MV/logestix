#pragma once

#include <string>

#include "gemma_dense_model.h"
#include "lsxcommon/gemma_dense_12b_constants.h"

namespace lsxcommon {

// Gemma-4-12B-it (dense, `gemma4_unified` wrapper): 48 layers / 3840 hidden /
// 16 Q / 8 sliding KV + 1 global KV / 15360 MLP, same decoder as the 31B —
// GeGLU, 5:1 sliding/global attention with per-head QK-norm and V tied to K on
// the global layers, tied embed/lm_head, softcapped logits. The "unified" part
// of the checkpoint (a patch-embedding tower and one audio projection) is
// stripped at conversion and counted; this is a text-only runtime.
//
// Frozen geometry lives in gemma_dense_12b_constants.h. The decoder body is
// shared with the 31B (GemmaDenseModel): only the passport differs, which is
// exactly what `ValidateGeometry` compares the converted tree against.
class Gemma4_12BDenseModel final : public GemmaDenseModel {
public:
    static const std::string kType;

    explicit Gemma4_12BDenseModel(ModelInitConfig cfg);
    ~Gemma4_12BDenseModel() override;

    const std::string &Type() const override { return kType; }
};

} // namespace lsxcommon
