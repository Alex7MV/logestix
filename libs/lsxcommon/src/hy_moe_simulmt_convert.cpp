#include "lsxcommon/hy_moe_simulmt_convert.h"

#include "hy_moe_simulmt_constants.h"

#include "lsxcommon/hy_moe_convert.h"

namespace lsxcommon::hy_simulmt {

hy::HyConvertSpec SimulMTConvertSpec()
{
    using namespace frozen;
    hy::HyConvertSpec spec;
    spec.geometry = hy::HyPassportGeometry{
        .layers = kLayers,
        .hidden = kHidden,
        .heads = kHeads,
        .kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .rotary_dim = kRotaryDim,
        .vocab = kVocab,
        .dense_intermediate = kDenseIntermediate,
        .moe_intermediate = kMoeIntermediate,
        .shared_intermediate = kSharedIntermediate,
        .num_experts = kNumExperts,
        .top_k = kTopK,
        .first_moe_layer = kFirstMoeLayer,
    };
    spec.model_type = kType;
    spec.model_name = kModelName;
    spec.allow_audio_ballast = true;
    return spec;
}

arrow::Status ConvertHySimulMTModel(const std::string &input_dir,
                                    const std::string &output_root)
{
    return hy::ConvertHyMoeModelWithSpec(input_dir, output_root,
                                         SimulMTConvertSpec());
}

} // namespace lsxcommon::hy_simulmt
