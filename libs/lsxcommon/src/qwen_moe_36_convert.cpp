#include "lsxcommon/qwen_moe_36_convert.h"

#include "qwen_moe_36_constants.h"
#include "lsxcommon/qwen_moe_convert.h"

#include <absl/strings/str_format.h>

#include <string>

namespace lsxcommon::qwen_moe_36 {
namespace {

// Cross-checks parsed config against the 3.6 frozen passport. Returns false
// and fills `why` on mismatch (etalon: qwen_moe_convert.cpp
// check_frozen_geometry).
bool check_frozen_geometry_36(const qwen_moe::QwenMoeTopology &top,
                              std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.hidden_size != kHidden)
        return bad("hidden_size", top.hidden_size, kHidden);
    if (top.num_layers != kLayers)
        return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.num_attention_heads != kHeads)
        return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKVHeads)
        return bad("num_key_value_heads", top.num_key_value_heads, kKVHeads);
    if (top.head_dim != kHeadDim)
        return bad("head_dim", top.head_dim, kHeadDim);
    if (top.linear_num_key_heads != kLinearNumKeyHeads)
        return bad("linear_num_key_heads", top.linear_num_key_heads,
                   kLinearNumKeyHeads);
    if (top.linear_num_value_heads != kLinearNumValueHeads)
        return bad("linear_num_value_heads", top.linear_num_value_heads,
                   kLinearNumValueHeads);
    if (top.linear_key_head_dim != kLinearKeyHeadDim)
        return bad("linear_key_head_dim", top.linear_key_head_dim,
                   kLinearKeyHeadDim);
    if (top.linear_value_head_dim != kLinearValueHeadDim)
        return bad("linear_value_head_dim", top.linear_value_head_dim,
                   kLinearValueHeadDim);
    if (top.vocab_size != kVocab)
        return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts)
        return bad("num_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   kMoeIntermediate);
    if (top.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   top.shared_expert_intermediate_size, kSharedIntermediate);
    if (top.bos_token_id != kBosTokenId)
        return bad("bos_token_id", top.bos_token_id, kBosTokenId);
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want = (i % kFullAttnInterval == kFullAttnInterval - 1)
                                     ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want)
            return bad("layer_types[i]", i, 0);
    }
    return true;
}

} // namespace

arrow::Status ConvertQwen36MoeModel(const std::string &input_dir,
                                    const std::string &output_root)
{
    constexpr qwen_moe::QwenMoeConvertSpec kSpec = {
        "logestix-qwen36moe", "Qwen3.6-35B-A3B",
        &check_frozen_geometry_36, /*record_bos=*/true};
    return qwen_moe::ConvertQwenMoeModelCommon(input_dir, output_root, kSpec);
}

} // namespace lsxcommon::qwen_moe_36
