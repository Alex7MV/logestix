#include "lsxcommon/inkling_moe_varloop_convert.h"

#include "inkling_moe_varloop_constants.h"

namespace lsxcommon::inkling {

InklingLoopConvertSpec VarloopConvertSpec()
{
    using namespace lsxcommon::inkling::varloop::frozen;
    InklingLoopConvertSpec spec;
    spec.model_name = kModelCard;
    spec.format_tag = kFormatTag;
    spec.varloop = kVarLoop;
    spec.max_passes = kLoopPassesMax;
    spec.frozen.hidden = kHidden;
    spec.frozen.layers = kLayers;
    spec.frozen.vocab = kVocab;
    spec.frozen.base_vocab = kBaseVocab;
    spec.frozen.heads = kHeads;
    spec.frozen.kv_heads = kKvHeads;
    spec.frozen.head_dim = kHeadDim;
    spec.frozen.d_rel = kDRel;
    spec.frozen.rel_extent = kRelExtent;
    spec.frozen.sliding_window = kSlidingWindow;
    spec.frozen.conv_kernel = kConvKernel;
    spec.frozen.attn_res_block = kAttnResBlock;
    spec.frozen.experts = kExperts;
    spec.frozen.top_k = kTopK;
    spec.frozen.moe_intermediate = kMoeIntermediate;
    spec.frozen.shared_experts = kSharedExperts;
    spec.frozen.loop_start = kLoopStart;
    spec.frozen.loop_end = kLoopEnd;
    spec.frozen.loop_passes = kLoopPassesDefault;
    spec.frozen.loop_gate_rank = kLoopGateRank;
    spec.frozen.log_scaling_floor = kLogScaleFloor;
    spec.frozen.nvfp4_expert_layers = kNovfp4ExpertLayerCount;
    spec.frozen.full_layers.assign(kFullLayers.begin(), kFullLayers.end());
    return spec;
}

arrow::Status ConvertInklingVarloopModel(const std::string &input_dir,
                                         const std::string &output_root)
{
    return ConvertInklingLoopModelWithSpec(input_dir, output_root,
                                           VarloopConvertSpec());
}

}  // namespace lsxcommon::inkling
