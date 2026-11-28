#include "gemma_dense_12b_model.h"

#include <string>

#include "fast_load_internal.h"
#include "lsxcommon/gemma_tool_format.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {

// Model Ingestion Protocol step 3: the string the factory, the manifest
// (`models/google/gemma-4-12B-it.json`) and the artifact store all agree on.
const std::string Gemma4_12BDenseModel::kType = "gemma-4-12b-it";

// fast-math policy: per-call fast-math policy (LsxFastMathScope,
// model-independence invariant) — opened by the SHARED Infer in
// gemma_dense_model.cpp, which is the body this class inherits. seq1_gemv
// follows the instance config; tc_fast_math follows the tri-state config.

Gemma4_12BDenseModel::Gemma4_12BDenseModel(ModelInitConfig cfg)
    : GemmaDenseModel(std::move(cfg), gemma_dense_12b::frozen::kFrozen) {}

Gemma4_12BDenseModel::~Gemma4_12BDenseModel() = default;

} // namespace lsxcommon
