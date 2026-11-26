// GigaChat4-708B-A41B converter entry point (Model Ingestion Protocol
// stage 1/2). See gigachat_moe_708b_constants.h for the frozen geometry and
// the design doc
// docs/superpowers/specs/2026-11-27-gigachat4-708b-a41b-ingest-design.md.
//
// Nothing about the conversion differs from the 37B sibling: this file exists so
// the shared core (gigachat_moe_37b_convert.cpp) never has to know which member
// it is converting. Every member-specific value arrives in the Gc4IngestSpec.
#include "lsxcommon/gigachat_moe_708b_convert.h"

#include "gigachat_moe_708b_constants.h"

namespace lsxcommon::gigachat4_708b {

gigachat4::Gc4IngestSpec Gc708BIngestSpec()
{
    using namespace frozen;
    gigachat4::Gc4IngestSpec s;
    s.format = kFormat;
    s.model_name = kModelName;
    s.frozen.hidden = kHidden;
    s.frozen.layers = kLayers;
    s.frozen.vocab = kVocab;
    s.frozen.heads = kHeads;
    s.frozen.kv_heads = kKVHeads;
    s.frozen.num_experts = kNumRouted;
    s.frozen.top_k = kTopK;
    s.frozen.moe_intermediate = kMoeIntermediate;
    s.frozen.dense_intermediate = kDenseIntermediate;
    s.frozen.first_k_dense = kFirstKDense;
    s.frozen.q_lora = kQLoraRank;
    s.frozen.kv_lora = kKvLoraRank;
    s.frozen.qk_nope = kQkNopeHeadDim;
    s.frozen.qk_rope = kQkRopeHeadDim;
    s.frozen.v_head = kVHeadDim;
    s.frozen.index_heads = kIndexHeads;
    s.frozen.index_head_dim = kIndexHeadDim;
    s.frozen.index_topk = kIndexTopk;
    s.frozen.rms_eps = static_cast<double>(kRmsEps);
    s.frozen.routed_scale = static_cast<double>(kRoutedScale);
    s.frozen.swiglu_limit = static_cast<double>(kSwigluLimit);
    s.frozen.logit_softcap = static_cast<double>(kLogitSoftcap);
    s.frozen.bos_id = kBosId;
    s.frozen.eos_id = kEosId;
    s.frozen.pad_id = kPadId;
    s.full_attention_layers.assign(kFullAttnLayers,
                                   kFullAttnLayers + kNumFullAttn);
    s.attn_scale = kAttnScale;
    return s;
}

arrow::Status ConvertGigaChat4_708BModel(const std::string &input_dir,
                                        const std::string &output_root)
{
    return gigachat4::ConvertGigaChat4ModelWithSpec(input_dir, output_root,
                                                    Gc708BIngestSpec());
}

} // namespace lsxcommon::gigachat4_708b
