#include "lsxcommon/nemotron_moe_ultra_load.h"

#include "nemotron_moe_ultra_constants.h"
#include "fast_load_internal.h"

#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::nemotron_ultra {
namespace {
// Dedup Ф5: single-tensor upload is the shared lsxcommon::fastload body.
using lsxcommon::fastload::UploadOne;


constexpr const char *kFormatTag = "logestix-nemotron-ultra-nvfp4";

bool get_i(simdjson::ondemand::object &o, const char *k, int &out)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return false;
    out = static_cast<int>(v);
    return true;
}

} // namespace

bool LoadUltraIndex(const std::string &model_path, UltraIndex &out,
                    std::string &error)
{
    namespace fs = std::filesystem;
    const fs::path idx_path = fs::path(model_path) / "model.index.json";
    std::error_code ec;
    if (!fs::exists(idx_path, ec)) {
        error = "model.index.json not found: " + idx_path.string();
        return false;
    }
    std::ifstream f(idx_path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        error = "model.index.json parse error";
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        error = "model.index.json root is not an object";
        return false;
    }

    std::string_view fmt;
    if (root["format"].get_string().get(fmt) != simdjson::SUCCESS) {
        error = "index missing 'format' tag";
        return false;
    }
    out.format = std::string(fmt);
    if (out.format != kFormatTag) {
        error = absl::StrCat("unknown format tag: ", out.format, " (want ",
                             kFormatTag, ")");
        return false;
    }
    // simdjson on-demand is FORWARD-ONLY: keys must be read in the order the
    // document stores them, or a later lookup silently fails. The order below is
    // exactly the order ConvertNemotronUltraModel writes the index - keep them in
    // step, and prefer a parser with a DOM if a future field has to be optional
    // in an arbitrary position.
    std::string_view model;
    if (root["model"].get_string().get(model) == simdjson::SUCCESS)
        out.model = std::string(model);

    get_i(root, "num_layers", out.num_layers);

    if (auto lt = root["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) out.layer_types.emplace_back(sv);
        }
    }
    if (auto ai = root["attn_layer_idx"].get_array();
        ai.error() == simdjson::SUCCESS) {
        for (auto e : ai.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.attn_layer_idx.push_back(static_cast<int>(v));
        }
    }

    get_i(root, "hidden_size", out.hidden_size);
    get_i(root, "vocab_size", out.vocab_size);
    get_i(root, "mamba_num_heads", out.mamba_num_heads);
    get_i(root, "mamba_head_dim", out.mamba_head_dim);
    get_i(root, "ssm_state_size", out.ssm_state_size);
    get_i(root, "n_groups", out.n_groups);
    get_i(root, "conv_kernel", out.conv_kernel);
    get_i(root, "num_attention_heads", out.num_attention_heads);
    get_i(root, "num_key_value_heads", out.num_key_value_heads);
    get_i(root, "head_dim", out.head_dim);
    double rd;
    if (root["rope_theta"].get_double().get(rd) == simdjson::SUCCESS)
        out.rope_theta = rd;
    if (root["rms_norm_eps"].get_double().get(rd) == simdjson::SUCCESS)
        out.rms_norm_eps = rd;
    get_i(root, "num_experts", out.num_experts);
    get_i(root, "num_experts_per_tok", out.num_experts_per_tok);
    get_i(root, "moe_intermediate_size", out.moe_intermediate_size);
    get_i(root, "shared_expert_intermediate_size",
          out.shared_expert_intermediate_size);
    get_i(root, "moe_latent_size", out.moe_latent_size);
    get_i(root, "n_group", out.n_group);
    get_i(root, "topk_group", out.topk_group);
    if (root["routed_scaling_factor"].get_double().get(rd) == simdjson::SUCCESS)
        out.routed_scaling_factor = static_cast<float>(rd);
    bool b;
    if (root["norm_topk_prob"].get_bool().get(b) == simdjson::SUCCESS)
        out.norm_topk_prob = b;
    if (root["experts_no_gate"].get_bool().get(b) == simdjson::SUCCESS)
        out.experts_no_gate = b;
    get_i(root, "bos_token_id", out.bos_token_id);
    // eos_token_id is an int in some distributions and a LIST here ([2, 11]);
    // both spellings must load, and the scalar is always the first stop.
    {
        simdjson::ondemand::value eos;
        if (root["eos_token_id"].get(eos) == simdjson::SUCCESS) {
            if (auto arr = eos.get_array(); arr.error() == simdjson::SUCCESS) {
                for (auto e : arr.value()) {
                    int64_t v;
                    if (e.get_int64().get(v) == simdjson::SUCCESS)
                        out.eos_token_ids.push_back(static_cast<int>(v));
                }
            } else {
                int64_t v;
                if (eos.get_int64().get(v) == simdjson::SUCCESS)
                    out.eos_token_ids.push_back(static_cast<int>(v));
            }
        }
    }
    if (!out.eos_token_ids.empty()) out.eos_token_id = out.eos_token_ids.front();
    get_i(root, "pad_token_id", out.pad_token_id);
    get_i(root, "mtp_tensors", out.mtp_tensors);
    if (root["partial"].get_bool().get(b) == simdjson::SUCCESS) out.partial = b;
    if (auto lc = root["layers_converted"].get_array();
        lc.error() == simdjson::SUCCESS) {
        for (auto e : lc.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.layers_converted.push_back(static_cast<int>(v));
        }
    }
    if (auto vr = root["vram"].get_object(); vr.error() == simdjson::SUCCESS) {
        uint64_t u;
        auto vo = vr.value();
        if (vo["min_bytes"].get_uint64().get(u) == simdjson::SUCCESS)
            out.vram.min_bytes = u;
        if (vo["optimal_bytes"].get_uint64().get(u) == simdjson::SUCCESS)
            out.vram.optimal_bytes = u;
        if (vo["max_bytes"].get_uint64().get(u) == simdjson::SUCCESS)
            out.vram.max_bytes = u;
        int64_t c;
        if (vo["context_tokens"].get_int64().get(c) == simdjson::SUCCESS)
            out.vram.context_tokens = c;
    }
    get_i(root, "total_pinned_tensors", out.total_pinned_tensors);
    get_i(root, "total_experts", out.total_experts);

    if (out.num_layers <= 0 || out.hidden_size <= 0 || out.vocab_size <= 0 ||
        static_cast<int>(out.layer_types.size()) != out.num_layers) {
        error = "index missing required NemotronH fields";
        return false;
    }
    // Grouped routing must be coherent before the router kernel consumes it.
    if (out.n_group <= 0 || out.num_experts % out.n_group != 0) {
        error = absl::StrFormat("index n_group %d does not divide %d experts",
                                out.n_group, out.num_experts);
        return false;
    }
    if (out.topk_group <= 0 || out.topk_group > out.n_group) {
        error = absl::StrFormat("index topk_group %d out of range for n_group %d",
                                out.topk_group, out.n_group);
        return false;
    }
    if (out.moe_latent_size < 0 || out.moe_latent_size >= out.hidden_size) {
        error = absl::StrFormat("index moe_latent_size %d is not a compression of "
                                "hidden_size %d",
                                out.moe_latent_size, out.hidden_size);
        return false;
    }
    return true;
}

bool ValidateUltraGeometry(const UltraIndex &idx, std::string &error)
{
    using namespace frozen;
    auto bad = [&error](const char *field, long long got, long long want) {
        error = absl::StrFormat("%s: got %lld, want %lld", field, got, want);
        return false;
    };

    // A partial tree is refused for every caller, synthetic or not: the layers
    // it does not hold are exactly the ones Infer would read.
    if (idx.partial) {
        int missing = -1;
        for (int l = 0; l < idx.num_layers; ++l) {
            bool have = false;
            for (int c : idx.layers_converted)
                if (c == l) have = true;
            if (!have) {
                missing = l;
                break;
            }
        }
        if (missing >= 0) {
            error = absl::StrFormat(
                "tree is partial: layer %d was not converted (%zu of %d layers "
                "present; rerun lsxforge --nemotron_ultra on that layer window)",
                missing, idx.layers_converted.size(), idx.num_layers);
            return false;
        }
    }

    if (SyntheticGeometryAllowed()) return true;

    if (idx.num_layers != kLayers) return bad("num_layers", idx.num_layers, kLayers);
    if (idx.hidden_size != kHidden) return bad("hidden_size", idx.hidden_size, kHidden);
    if (idx.vocab_size != kVocab) return bad("vocab_size", idx.vocab_size, kVocab);
    if (idx.mamba_num_heads != kMambaHeads)
        return bad("mamba_num_heads", idx.mamba_num_heads, kMambaHeads);
    if (idx.mamba_head_dim != kMambaHeadDim)
        return bad("mamba_head_dim", idx.mamba_head_dim, kMambaHeadDim);
    if (idx.ssm_state_size != kSsmStateSize)
        return bad("ssm_state_size", idx.ssm_state_size, kSsmStateSize);
    if (idx.n_groups != kSsmGroups) return bad("n_groups", idx.n_groups, kSsmGroups);
    if (idx.conv_kernel != kMambaConvKernel)
        return bad("conv_kernel", idx.conv_kernel, kMambaConvKernel);
    if (idx.num_attention_heads != kHeads)
        return bad("num_attention_heads", idx.num_attention_heads, kHeads);
    if (idx.num_key_value_heads != kKVHeads)
        return bad("num_key_value_heads", idx.num_key_value_heads, kKVHeads);
    if (idx.head_dim != kHeadDim) return bad("head_dim", idx.head_dim, kHeadDim);
    if (idx.num_experts != kNumRouted)
        return bad("num_experts", idx.num_experts, kNumRouted);
    if (idx.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", idx.num_experts_per_tok, kTopK);
    if (idx.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", idx.moe_intermediate_size, kMoeIntermediate);
    if (idx.moe_latent_size != kMoeLatent)
        return bad("moe_latent_size", idx.moe_latent_size, kMoeLatent);
    if (idx.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   idx.shared_expert_intermediate_size, kSharedIntermediate);
    if (idx.n_group != kRouterGroups) return bad("n_group", idx.n_group, kRouterGroups);
    if (idx.topk_group != kRouterTopkGroup)
        return bad("topk_group", idx.topk_group, kRouterTopkGroup);
    if (!idx.experts_no_gate) return bad("experts_no_gate", 0, 1);

    int mamba = 0, moe = 0, attn = 0;
    for (const auto &t : idx.layer_types) {
        if (t == "mamba") ++mamba;
        else if (t == "moe") ++moe;
        else if (t == "attention") ++attn;
        else return bad("layer_types", 0, 0);
    }
    if (mamba != kMambaLayers || moe != kMoeLayers || attn != kAttnLayers)
        return bad("layer mix", mamba * 10000 + moe * 100 + attn,
                   kMambaLayers * 10000 + kMoeLayers * 100 + kAttnLayers);

    if (static_cast<int>(idx.attn_layer_idx.size()) != kAttnLayers)
        return bad("attn_layer_idx size",
                   static_cast<long long>(idx.attn_layer_idx.size()), kAttnLayers);
    for (int i = 0; i < kAttnLayers; ++i)
        if (idx.attn_layer_idx[i] != kAttnLayerIdx[i])
            return bad("attn_layer_idx", idx.attn_layer_idx[i], kAttnLayerIdx[i]);
    return true;
}

// ── Device weight graph ─────────────────────────────────────────────────────

void UltraLayerWeights::Free()
{
    cudaFree(d_norm); d_norm = nullptr;
    in_proj.Free(); out_proj.Free();
    cudaFree(d_conv_w); d_conv_w = nullptr;
    cudaFree(d_conv_b); d_conv_b = nullptr;
    cudaFree(d_A_log); d_A_log = nullptr;
    cudaFree(d_D); d_D = nullptr;
    cudaFree(d_dt_bias); d_dt_bias = nullptr;
    cudaFree(d_mamba_norm); d_mamba_norm = nullptr;
    q.Free(); k.Free(); v.Free(); o.Free();
    cudaFree(d_router); d_router = nullptr;
    cudaFree(d_router_bias); d_router_bias = nullptr;
    latent_up.Free(); latent_down.Free();
    shared_up.Free(); shared_down.Free();
}

void UltraWeights::Free()
{
    cudaFree(d_embed); d_embed = nullptr;
    cudaFree(d_head); d_head = nullptr;
    cudaFree(d_final_norm); d_final_norm = nullptr;
    for (auto &l : layers) l.Free();
    layers.clear();
    resident_bytes = 0;
}

namespace {

using fastload::CvKind;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::PinnedStagingRing;
using fastload::UploadJob;
using fastload::UploadNvfp4Weight;

} // namespace

bool LoadUltraWeights(const std::string &model_path, const UltraIndex &idx,
                      UltraWeights &w, cudaStream_t stream, std::string &error)
{
    namespace fs = std::filesystem;
    const fs::path base = fs::path(model_path);
    PinnedStagingRing ring;
    if (!ring.ok()) {
        error = "pinned staging ring allocation failed";
        return false;
    }
    fastload::StreamDrain drain{stream};
    std::uint64_t acc = 0;

    auto global = LoadTensorBatchViews(base / "pinned" / "global.arrow");
    if (!global.ok()) {
        error = "cannot open pinned/global.arrow: " + global.status().ToString();
        return false;
    }
    if (!UploadOne(FindTensorSuffix(*global, "embeddings.weight"),
                   CvKind::kBf16ToHalf, reinterpret_cast<void **>(&w.d_embed),
                   "embeddings.weight", stream, ring, acc, error))
        return false;
    if (!UploadOne(FindTensorSuffix(*global, "lm_head.weight"),
                   CvKind::kBf16ToHalf, reinterpret_cast<void **>(&w.d_head),
                   "lm_head.weight", stream, ring, acc, error))
        return false;
    if (!UploadOne(FindTensorSuffix(*global, "norm_f.weight"), CvKind::kBf16ToHalf,
                   reinterpret_cast<void **>(&w.d_final_norm), "norm_f.weight",
                   stream, ring, acc, error))
        return false;

    w.layers.resize(static_cast<std::size_t>(idx.num_layers));

    for (int l = 0; l < idx.num_layers; ++l) {
        UltraLayerWeights &lw = w.layers[static_cast<std::size_t>(l)];
        const auto nv_path =
            base / "pinned" / absl::StrCat("layer_", l, "_nvfp4.arrow");
        const auto raw_path = base / "pinned" / absl::StrCat("layer_", l, ".arrow");
        auto nv = LoadDenseNvfp4BatchViews(nv_path);
        auto raw = LoadTensorBatchViews(raw_path);
        if (!nv.ok() || !raw.ok()) {
            error = absl::StrCat("cannot open layer ", l, " arrow files (",
                                 nv_path.filename().string(), " / ",
                                 raw_path.filename().string(), ")");
            w.Free();
            return false;
        }
        const std::string lp = absl::StrCat("layers.", l, ".");
        auto raw_suffix = [&](const char *s) {
            return FindTensorSuffix(*raw, lp + s);
        };

        if (!UploadOne(raw_suffix("norm.weight"), CvKind::kBf16ToHalf,
                       reinterpret_cast<void **>(&lw.d_norm), "layer norm", stream,
                       ring, acc, error)) {
            w.Free();
            return false;
        }

        const std::string &lt = idx.layer_types[static_cast<std::size_t>(l)];
        if (lt == "mamba") {
            if (UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mixer.in_proj.weight"),
                                  lw.in_proj, stream, ring, acc, &error) !=
                    cudaSuccess ||
                UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mixer.out_proj.weight"),
                                  lw.out_proj, stream, ring, acc, &error) !=
                    cudaSuccess) {
                w.Free();
                return false;
            }
            if (!UploadOne(raw_suffix("mixer.conv1d.weight"), CvKind::kBf16ToF32,
                           reinterpret_cast<void **>(&lw.d_conv_w), "conv1d.w",
                           stream, ring, acc, error) ||
                !UploadOne(raw_suffix("mixer.conv1d.bias"), CvKind::kBf16ToF32,
                           reinterpret_cast<void **>(&lw.d_conv_b), "conv1d.b",
                           stream, ring, acc, error) ||
                !UploadOne(raw_suffix("mixer.A_log"), CvKind::kBf16ToF32,
                           reinterpret_cast<void **>(&lw.d_A_log), "A_log", stream,
                           ring, acc, error) ||
                !UploadOne(raw_suffix("mixer.D"), CvKind::kBf16ToF32,
                           reinterpret_cast<void **>(&lw.d_D), "D", stream, ring, acc,
                           error) ||
                !UploadOne(raw_suffix("mixer.dt_bias"), CvKind::kBf16ToF32,
                           reinterpret_cast<void **>(&lw.d_dt_bias), "dt_bias",
                           stream, ring, acc, error) ||
                !UploadOne(raw_suffix("mixer.norm.weight"), CvKind::kBf16ToF32,
                           reinterpret_cast<void **>(&lw.d_mamba_norm), "mamba norm",
                           stream, ring, acc, error)) {
                w.Free();
                return false;
            }
        } else if (lt == "attention") {
            // NoPE: q/k are used as-is, so there is no rotary table to load.
            if (UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mixer.q_proj.weight"), lw.q,
                                  stream, ring, acc, &error) != cudaSuccess ||
                UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mixer.k_proj.weight"), lw.k,
                                  stream, ring, acc, &error) != cudaSuccess ||
                UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mixer.v_proj.weight"), lw.v,
                                  stream, ring, acc, &error) != cudaSuccess ||
                UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mixer.o_proj.weight"), lw.o,
                                  stream, ring, acc, &error) != cudaSuccess) {
                w.Free();
                return false;
            }
        } else {  // moe
            const TensorView *gate = raw_suffix("mixer.gate.weight");
            const CvKind gate_kind =
                (gate && gate->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
            if (!UploadOne(gate, gate_kind, reinterpret_cast<void **>(&lw.d_router),
                           "router", stream, ring, acc, error) ||
                !UploadOne(raw_suffix("mixer.gate.e_score_correction_bias"),
                           CvKind::kF32Raw,
                           reinterpret_cast<void **>(&lw.d_router_bias), "router bias",
                           stream, ring, acc, error)) {
                w.Free();
                return false;
            }
            // The latent bridge is mandatory exactly when the index declares a
            // latent width: without it the routed path would be computed at the
            // wrong width against weights that expect the other one.
            if (idx.moe_latent_size > 0) {
                const DenseNvfp4View *up =
                    FindNvfp4Suffix(*nv, "mixer.fc1_latent_proj.weight");
                const DenseNvfp4View *down =
                    FindNvfp4Suffix(*nv, "mixer.fc2_latent_proj.weight");
                if (!up) {
                    error = absl::StrCat(
                        "layer ", l, ": index declares moe_latent_size=",
                        idx.moe_latent_size,
                        " but mixer.fc1_latent_proj.weight is not in ",
                        nv_path.filename().string());
                    w.Free();
                    return false;
                }
                if (!down) {
                    error = absl::StrCat(
                        "layer ", l, ": index declares moe_latent_size=",
                        idx.moe_latent_size,
                        " but mixer.fc2_latent_proj.weight is not in ",
                        nv_path.filename().string());
                    w.Free();
                    return false;
                }
                if (UploadNvfp4Weight(up, lw.latent_up, stream, ring, acc, &error) !=
                        cudaSuccess ||
                    UploadNvfp4Weight(down, lw.latent_down, stream, ring, acc,
                                      &error) != cudaSuccess) {
                    w.Free();
                    return false;
                }
            }
            if (UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "mixer.shared_experts.up_proj.weight"),
                    lw.shared_up, stream, ring, acc, &error) != cudaSuccess ||
                UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "mixer.shared_experts.down_proj.weight"),
                    lw.shared_down, stream, ring, acc, &error) != cudaSuccess) {
                w.Free();
                return false;
            }
        }
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        error = "weight upload stream sync failed";
        w.Free();
        return false;
    }
    w.resident_bytes = acc;
    return true;
}

} // namespace lsxcommon::nemotron_ultra
