#include "lsxcommon/inkling_moe_load.h"

#include "inkling_moe_constants.h"

#include <absl/strings/str_format.h>
#include <simdjson.h>

#include "lsxcommon/synthetic_geometry.h"

#include <filesystem>
#include <fstream>
#include <iterator>

namespace lsxcommon::inkling {
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

bool get_b(simdjson::ondemand::object &o, const char *k, bool &out)
{
    bool v;
    if (o[k].get_bool().get(v) != simdjson::SUCCESS) return false;
    out = v;
    return true;
}

}  // namespace

arrow::Status LoadInklingTopology(const std::string &model_path,
                                  InklingTopology &out)
{
    out = InklingTopology{};
    const fs::path idx = fs::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(idx);
    if (!f) return arrow::Status::IOError("cannot open ", idx.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        out.error_msg = "index JSON parse error";
        return arrow::Status::Invalid("inkling index: parse error");
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("inkling index: root is not an object");

    std::string_view s;
    if (root["format"].get_string().get(s) == simdjson::SUCCESS)
        out.format.assign(s.data(), s.size());
    if (out.format != frozen::kFormatTag) {
        out.error_msg = "unexpected format: " + out.format;
        return arrow::Status::Invalid("inkling loader: expected format ",
                                      frozen::kFormatTag, ", got ",
                                      out.format);
    }
    if (root["source_model_type"].get_string().get(s) == simdjson::SUCCESS)
        out.source_model_type.assign(s.data(), s.size());

    get_i(root, "num_layers", out.num_layers);
    get_i(root, "hidden_size", out.hidden);
    get_i(root, "vocab_size", out.vocab);
    get_i(root, "unpadded_vocab_size", out.unpadded_vocab);
    get_i(root, "num_attention_heads", out.num_heads);
    get_i(root, "num_key_value_heads", out.num_kv_heads);
    get_i(root, "head_dim", out.head_dim);
    get_i(root, "d_rel", out.d_rel);
    get_i(root, "rel_extent", out.rel_extent);
    get_i(root, "sliding_window_size", out.sliding_window);
    get_i(root, "conv_kernel", out.conv_kernel);
    get_i(root, "attn_res_block", out.attn_res_block);
    get_i(root, "num_experts", out.num_experts);
    get_i(root, "num_experts_per_tok", out.top_k);
    get_i(root, "moe_intermediate_size", out.moe_intermediate);
    get_i(root, "num_shared_experts", out.shared_experts);
    get_i(root, "log_scaling_n_floor", out.log_scaling_n_floor);
    get_i(root, "bos_id", out.bos_id);
    get_i(root, "eos_id", out.eos_id);
    get_i(root, "stop_id_alt", out.stop_id_alt);
    get_i(root, "pad_id", out.pad_id);
    get_i(root, "tokenizer_kind", out.tokenizer_kind);
    get_i(root, "skipped_visual_tensors", out.skipped_visual);
    get_d(root, "attn_scale_denom", out.attn_scale_denom);
    get_d(root, "rms_norm_eps", out.rms_eps);
    get_d(root, "qk_l2_clip", out.qk_l2_clip);
    get_d(root, "rel_l2_clip", out.rel_l2_clip);
    get_d(root, "log_scaling_alpha", out.log_scaling_alpha);
    get_d(root, "mlp_gate_softcap", out.mlp_gate_softcap);
    get_d(root, "mlp_up_softcap", out.mlp_up_softcap);
    get_d(root, "final_logit_softcap", out.final_logit_softcap);
    get_d(root, "routed_scaling_factor", out.route_scale);
    get_b(root, "moe_renormalize", out.moe_renormalize);

    if (auto arr = root["full_layers"].get_array();
        arr.error() == simdjson::SUCCESS)
        for (auto e : arr.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.full_layers.push_back(static_cast<int>(v));
        }
    if (auto arr = root["sliding_layers"].get_array();
        arr.error() == simdjson::SUCCESS)
        for (auto e : arr.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.sliding_layers.push_back(static_cast<int>(v));
        }
    if (auto arr = root["expert_nvfp4"].get_array();
        arr.error() == simdjson::SUCCESS)
        for (auto e : arr.value()) {
            bool v = false;
            if (e.get_bool().get(v) == simdjson::SUCCESS)
                out.expert_nvfp4.push_back(v);
        }

    if (out.num_layers <= 0 || out.hidden <= 0 || out.num_experts <= 0 ||
        static_cast<int>(out.full_layers.size() + out.sliding_layers.size()) !=
            out.num_layers) {
        out.error_msg = "index is missing the layer partition";
        return arrow::Status::Invalid("inkling index: ", out.error_msg);
    }
    return arrow::Status::OK();
}

bool InklingTopologyMatchesFrozen(const InklingTopology &top)
{
    if (SyntheticGeometryAllowed()) return true;
    return top.num_layers == frozen::kLayers && top.hidden == frozen::kHidden &&
           top.vocab == frozen::kVocab &&
           top.unpadded_vocab == frozen::kBaseVocab &&
           top.num_heads == frozen::kHeads &&
           top.num_kv_heads == frozen::kKvHeads &&
           top.head_dim == frozen::kHeadDim && top.d_rel == frozen::kDRel &&
           top.rel_extent == frozen::kRelExtent &&
           top.sliding_window == frozen::kSlidingWindow &&
           top.conv_kernel == frozen::kConvKernel &&
           top.attn_res_block == frozen::kAttnResBlock &&
           top.num_experts == frozen::kExperts && top.top_k == frozen::kTopK &&
           top.moe_intermediate == frozen::kMoeIntermediate &&
           top.shared_experts == frozen::kSharedExperts &&
           static_cast<int>(top.full_layers.size()) == frozen::kFullLayerCount;
}

}  // namespace lsxcommon::inkling
