#include "lsxcommon/alice_moe_load.h"

#include "alice_moe_constants.h"

#include <simdjson.h>

#include <filesystem>
#include <fstream>
#include <iterator>

namespace lsxcommon::alice {
namespace {

namespace fs = std::filesystem;

bool get_i(simdjson::ondemand::object &o, const char *k, int &out)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return false;
    out = static_cast<int>(v);
    return true;
}

bool get_d(simdjson::ondemand::object &o, const char *k, double &out)
{
    simdjson::ondemand::value x;
    if (o[k].get(x) != simdjson::SUCCESS) return false;
    if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) {
        out = d.value();
        return true;
    }
    if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS) {
        out = static_cast<double>(i.value());
        return true;
    }
    return false;
}

} // namespace

arrow::Status LoadAliceTopology(const std::string &model_path,
                                AliceMoeTopology &out)
{
    out = AliceMoeTopology{};
    const fs::path idx = fs::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(idx);
    if (!f)
        return arrow::Status::IOError("cannot open ", idx.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        out.error_msg = "index JSON parse error";
        return arrow::Status::Invalid("alice index: parse error");
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("alice index: root is not an object");

    std::string_view s;
    if (root["format"].get_string().get(s) == simdjson::SUCCESS)
        out.format.assign(s.data(), s.size());
    if (out.format != "logestix-alice-nvfp4") {
        out.error_msg = "unexpected format: " + out.format;
        return arrow::Status::Invalid("alice loader: expected format "
                                      "logestix-alice-nvfp4, got ",
                                      out.format);
    }
    if (root["model"].get_string().get(s) == simdjson::SUCCESS)
        out.model.assign(s.data(), s.size());

    get_i(root, "num_layers", out.num_layers);
    get_i(root, "hidden_size", out.hidden_size);
    get_i(root, "vocab_size", out.vocab_size);
    get_i(root, "num_attention_heads", out.num_attention_heads);
    get_i(root, "num_key_value_heads", out.num_key_value_heads);
    get_i(root, "head_dim", out.head_dim);
    get_i(root, "linear_num_key_heads", out.linear_num_key_heads);
    get_i(root, "linear_num_value_heads", out.linear_num_value_heads);
    get_i(root, "linear_key_head_dim", out.linear_key_head_dim);
    get_i(root, "linear_value_head_dim", out.linear_value_head_dim);
    get_i(root, "linear_conv_kernel_dim", out.linear_conv_kernel_dim);
    get_i(root, "num_experts", out.num_experts);
    get_i(root, "num_experts_per_tok", out.num_experts_per_tok);
    get_i(root, "moe_intermediate_size", out.moe_intermediate_size);
    get_i(root, "shared_expert_intermediate_size",
          out.shared_expert_intermediate_size);
    get_i(root, "block_attn_res_block_size", out.block_attn_res_block_size);
    get_i(root, "mtp_num_hidden_layers", out.mtp_num_hidden_layers);
    get_i(root, "bos_id", out.bos_id);
    get_i(root, "eos_id", out.eos_id);
    get_i(root, "pad_id", out.pad_id);
    get_d(root, "rms_norm_eps", out.rms_norm_eps);
    get_d(root, "partial_rotary_factor", out.partial_rotary_factor);
    get_d(root, "rope_theta", out.rope_theta);

    if (auto lt = root["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                out.layer_types.emplace_back(sv);
        }
    }
    if (out.num_layers <= 0 ||
        static_cast<int>(out.layer_types.size()) != out.num_layers)
        return arrow::Status::Invalid(
            "alice index: layer_types size mismatch");
    return arrow::Status::OK();
}

bool AliceTopologyMatchesFrozen(const AliceMoeTopology &top)
{
    using namespace frozen;
    if (top.hidden_size != kHidden || top.num_layers != kLayers ||
        top.vocab_size != kVocab || top.num_attention_heads != kHeads ||
        top.num_key_value_heads != kKVHeads || top.head_dim != kHeadDim ||
        top.linear_num_key_heads != kLinearKeyHeads ||
        top.linear_num_value_heads != kLinearValueHeads ||
        top.linear_key_head_dim != kLinearKeyHeadDim ||
        top.linear_value_head_dim != kLinearValueHeadDim ||
        top.linear_conv_kernel_dim != kLinearConvKernel ||
        top.num_experts != kNumExperts ||
        top.num_experts_per_tok != kTopK ||
        top.moe_intermediate_size != kMoeIntermediate ||
        top.shared_expert_intermediate_size != kSharedIntermediate ||
        top.block_attn_res_block_size != kBlockAttnRes)
        return false;
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want =
            (i % kFullAttnInterval == kFullAttnInterval - 1)
                ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want) return false;
    }
    return true;
}

} // namespace lsxcommon::alice
