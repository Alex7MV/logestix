#include "qwen_moe_36_model.h"
#include "qwen_moe_36_constants.h"

#include <absl/strings/str_cat.h>

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {

const std::string Qwen36MoeModel::kType = "qwen3.6-35b-a3b";

Qwen36MoeModel::Qwen36MoeModel(ModelInitConfig cfg)
    : QwenMoeModelBase{std::move(cfg)}
{
}

Qwen36MoeModel::~Qwen36MoeModel() { Unload(); }

gpu::InferenceResult Qwen36MoeModel::Infer(const ModelRequest &req)
{
    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    const lsx_nvfp4::LsxFastMathScope fm_scope(cfg_.fast_math, cfg_.fast_math);
    return RunInferShared(req);
}

bool Qwen36MoeModel::ValidateGeometry(
    const qwen_moe::QwenMoeTopology &top) const
{
    using namespace qwen_moe_36::frozen;
    if (SyntheticGeometryAllowed()) return true;

    const GeoFieldCheck fields[] = {
        {"hidden_size", kHidden, top.hidden_size},
        {"num_layers", kLayers, top.num_layers},
        {"num_attention_heads", kHeads, top.num_attention_heads},
        {"num_key_value_heads", kKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"vocab_size", kVocab, top.vocab_size},
        {"num_experts", kNumExperts, top.num_experts},
        {"num_experts_per_tok", kTopK, top.num_experts_per_tok},
        {"moe_intermediate_size", kMoeIntermediate,
         top.moe_intermediate_size},
        {"shared_expert_intermediate_size", kSharedIntermediate,
         top.shared_expert_intermediate_size},
    };
    if (!CheckGeometryFields("qwen36-moe-model", fields, 10)) return false;
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string want =
            (l % kFullAttnInterval == kFullAttnInterval - 1) ? "full_attention"
                                                             : "linear_attention";
        if (top.layer_types[static_cast<size_t>(l)] != want) {
            log::error(absl::StrCat("[qwen36-moe-model] layer_types[", l,
                                    "] != ", want));
            return false;
        }
    }
    return true;
}

} // namespace lsxcommon
