#include "lsxcommon/kimi_moe_k3_load.h"
#include "fast_load_internal.h"
#include "lsxcommon/config_json.h"

#include "kimi_moe_k3_constants.h"
#include "kimi_moe_pro_constants.h"

#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>
#include <vector>

namespace lsxcommon::kimi_k3 {
namespace {
// Dedup Ф5: BF16→FP16 host conversion is the shared lsxcommon::fastload body.
using lsxcommon::fastload::Bf16ToHalf;

// Dedup Ф5: numeric config field reader is the shared lsxcommon::config body.
using lsxcommon::config::GetDouble;
inline bool get_d(simdjson::ondemand::object &o, const char *k, double &out)
{
    return GetDouble(o, k, out);
}


namespace fs = std::filesystem;

bool get_i(simdjson::ondemand::object &o, const char *k, int &out)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return false;
    out = static_cast<int>(v);
    return true;
}

bool get_b(simdjson::ondemand::object &o, const char *k, bool &out)
{
    bool v;
    if (o[k].get_bool().get(v) != simdjson::SUCCESS) return false;
    out = v;
    return true;
}

__half *UploadHalf(const std::vector<uint8_t> &data, cudaStream_t s)
{
    if (data.empty() || data.size() % 2 != 0) return nullptr;
    const auto h = Bf16ToHalf(data);
    __half *d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(__half)) != cudaSuccess) return nullptr;
    cudaMemcpyAsync(d, h.data(), h.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, s);
    return d;
}

// F32 source tensors (KDA conv / A_log / dt_bias / o_norm / router bias).
float *UploadF32(const std::vector<uint8_t> &data, cudaStream_t s)
{
    if (data.empty() || data.size() % 4 != 0) return nullptr;
    float *d = nullptr;
    if (cudaMalloc(&d, data.size()) != cudaSuccess) return nullptr;
    cudaMemcpyAsync(d, data.data(), data.size(), cudaMemcpyHostToDevice, s);
    return d;
}

bool UploadNvfp4(const DenseNvfp4Record &r, lsx_nvfp4::LsxNvfp4Weight &w,
                 cudaStream_t s)
{
    w = lsx_nvfp4::LsxNvfp4Weight{};
    if (r.shape.size() != 2 || r.packed.empty() || r.scales.empty())
        return false;
    w.rows = static_cast<int>(r.shape[0]);
    w.cols = static_cast<int>(r.shape[1]);
    w.global_scale = r.global_scale;
    void *p = nullptr, *sc = nullptr;
    if (cudaMalloc(&p, r.packed.size()) != cudaSuccess) return false;
    if (cudaMalloc(&sc, r.scales.size()) != cudaSuccess) {
        cudaFree(p);
        return false;
    }
    cudaMemcpyAsync(p, r.packed.data(), r.packed.size(),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(sc, r.scales.data(), r.scales.size(),
                    cudaMemcpyHostToDevice, s);
    w.d_packed = static_cast<const uint8_t *>(p);
    w.d_scales = static_cast<const uint8_t *>(sc);
    w.packed_bytes = static_cast<int64_t>(r.packed.size());
    w.scale_bytes = static_cast<int64_t>(r.scales.size());
    return true;
}

const TensorRecord *FindRaw(const std::vector<TensorRecord> &v,
                            std::string_view name)
{
    for (const auto &r : v)
        if (r.name == name) return &r;
    return nullptr;
}

const DenseNvfp4Record *FindNvfp4(const std::vector<DenseNvfp4Record> &v,
                                  std::string_view name)
{
    for (const auto &r : v)
        if (r.name == name) return &r;
    return nullptr;
}

const DenseNvfp4View *FindView(const DenseNvfp4BatchViews &v,
                               std::string_view name)
{
    for (const auto &r : v.records)
        if (r.name == name) return &r;
    return nullptr;
}

int64_t StagingBytesFor(const KimiK3StreamedLayer &s)
{
    int64_t n = 0;
    for (const KimiK3HostNvfp4 *w : {&s.latent_down, &s.latent_up, &s.shared_gate,
                                     &s.shared_up, &s.shared_down})
        n += w->packed_bytes + w->scale_bytes;
    return n;
}

}  // namespace

arrow::Status LoadKimiK3Topology(const std::string &model_path,
                                 KimiK3Topology &out)
{
    out = KimiK3Topology{};
    const fs::path idx = fs::path(model_path) / "model.index.json";
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
        return arrow::Status::Invalid("kimi-k3 index: parse error");
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("kimi-k3 index: root is not an object");

    std::string_view s;
    if (root["format"].get_string().get(s) == simdjson::SUCCESS)
        out.format.assign(s.data(), s.size());
    // Two trees, one loader: the K3 tag is the plain KimiLinear tower, the Pro
    // tag adds the indexer plane. Anything else is not this family.
    const bool is_k3 = out.format == frozen::kFormatTag;
    const bool is_pro = out.format == kimi_pro::frozen::kFormatTag;
    if (!is_k3 && !is_pro) {
        out.error_msg = "unexpected format: " + out.format;
        return arrow::Status::Invalid("kimi-k3 loader: expected format ",
                                      frozen::kFormatTag, " or ",
                                      kimi_pro::frozen::kFormatTag, ", got ",
                                      out.format);
    }
    if (root["model"].get_string().get(s) == simdjson::SUCCESS)
        out.model.assign(s.data(), s.size());

    get_i(root, "num_layers", out.num_layers);
    get_i(root, "num_dense_layers", out.dense_layers);
    get_i(root, "hidden_size", out.hidden);
    get_i(root, "intermediate_size", out.dense_intermediate);
    get_i(root, "vocab_size", out.vocab);
    get_i(root, "attn_res_block_size", out.attn_res_block);
    get_i(root, "mla_heads", out.mla_heads);
    get_i(root, "mla_kv_heads", out.kv_heads);
    get_i(root, "q_lora_rank", out.q_lora);
    get_i(root, "kv_lora_rank", out.kv_lora);
    get_i(root, "qk_nope_head_dim", out.qk_nope);
    get_i(root, "qk_rope_head_dim", out.qk_rope);
    get_i(root, "v_head_dim", out.v_head);
    get_i(root, "kda_heads", out.kda_heads);
    get_i(root, "kda_head_dim", out.kda_head_dim);
    get_i(root, "kda_conv_kernel", out.conv_kernel);
    get_i(root, "num_experts", out.num_experts);
    get_i(root, "num_experts_per_tok", out.top_k);
    get_i(root, "moe_intermediate_size", out.moe_intermediate);
    get_i(root, "routed_expert_hidden_size", out.latent);
    get_i(root, "num_shared_experts", out.shared_experts);
    get_i(root, "shared_intermediate_size", out.shared_intermediate);
    get_i(root, "bos_id", out.bos_id);
    get_i(root, "eos_id", out.eos_id);
    get_i(root, "pad_id", out.pad_id);
    get_i(root, "tokenizer_kind", out.tokenizer_kind);
    get_i(root, "skipped_visual_tensors", out.skipped_visual);
    get_i(root, "skipped_projector_tensors", out.skipped_projector);
    get_d(root, "rms_norm_eps", out.rms_eps);
    get_d(root, "activation_situ_beta", out.situ_beta);
    get_d(root, "activation_situ_linear_beta", out.situ_linear_beta);
    get_d(root, "kda_gate_lower_bound", out.kda_gate_lower_bound);
    get_d(root, "routed_scaling_factor", out.route_scale);
    get_b(root, "latent_moe_use_norm", out.latent_norm);
    get_b(root, "moe_renormalize", out.moe_renormalize);
    get_b(root, "mla_use_nope", out.mla_nope);
    get_b(root, "mla_use_output_gate", out.mla_gate);
    get_b(root, "kda_full_rank_gate", out.kda_full_rank_gate);

    if (auto arr = root["mla_layers"].get_array(); arr.error() == simdjson::SUCCESS)
        for (auto e : arr.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.mla_layers.push_back(static_cast<int>(v));
        }
    if (auto arr = root["kda_layers"].get_array(); arr.error() == simdjson::SUCCESS)
        for (auto e : arr.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.kda_layers.push_back(static_cast<int>(v));
        }

    // ── Gated DSA block (Kimi-K3.1-Pro) ──
    // The Pro tag REQUIRES the flag and the flag must be true; the K3 tag may
    // omit it (trees converted before the variant existed) but may never carry
    // a true one.
    bool indexer_flag = false;
    const bool has_indexer_flag = get_b(root, "indexer", indexer_flag);
    if (is_pro ? (!has_indexer_flag || !indexer_flag)
               : (has_indexer_flag && indexer_flag)) {
        out.error_msg = "index: the indexer flag disagrees with the format "
                        "(format says " +
                        std::string(is_pro ? "kimi-k3.1-pro" : "kimi-k3") + ")";
        return arrow::Status::Invalid("kimi-k3 index: ", out.error_msg);
    }
    out.dsa = is_pro;
    if (out.dsa) {
        get_i(root, "index_head_dim", out.index_head_dim);
        get_i(root, "index_num_heads", out.index_heads);
        get_i(root, "index_topk", out.index_topk);
        get_i(root, "index_rope_dim", out.index_rope_dim);
        get_d(root, "index_norm_eps", out.index_norm_eps);
        if (auto arr = root["dsa_layers"].get_array();
            arr.error() == simdjson::SUCCESS)
            for (auto e : arr.value()) {
                int64_t v = 0;
                if (e.get_int64().get(v) == simdjson::SUCCESS)
                    out.dsa_layers.push_back(static_cast<int>(v));
            }
        if (out.index_heads <= 0 || out.index_head_dim <= 0 ||
            out.index_topk <= 0 || out.index_rope_dim < 0 ||
            !(out.index_norm_eps > 0.0)) {
            out.error_msg = "index: the DSA block is incomplete";
            return arrow::Status::Invalid("kimi-k3 index: ", out.error_msg);
        }
        // The indexer runs on every full-attention layer or not at all: a
        // partial table means the tree was written by something that did not
        // understand the variant, and a per-layer selection would silently
        // differ from the source.
        if (out.dsa_layers.size() != out.mla_layers.size()) {
            out.error_msg =
                "index: dsa_layers is not the mla table (count mismatch)";
            return arrow::Status::Invalid("kimi-k3 index: ", out.error_msg);
        }
        for (int l : out.dsa_layers)
            if (!out.IsMla(l)) {
                out.error_msg = "index: dsa_layers contains a non-MLA layer " +
                                std::to_string(l);
                return arrow::Status::Invalid("kimi-k3 index: ",
                                              out.error_msg);
            }
    } else if (!out.dsa_layers.empty() || out.index_heads != 0 ||
               out.index_head_dim != 0 || out.index_topk != 0 ||
               out.index_rope_dim != 0 || out.index_norm_eps != 0.0) {
        out.error_msg = "index: a Kimi-K3 tree carries DSA fields";
        return arrow::Status::Invalid("kimi-k3 index: ", out.error_msg);
    }

    if (out.num_layers <= 0 || out.hidden <= 0 ||
        static_cast<int>(out.mla_layers.size() + out.kda_layers.size()) !=
            out.num_layers) {
        out.error_msg = "index is missing the layer partition";
        return arrow::Status::Invalid("kimi-k3 index: ", out.error_msg);
    }
    return arrow::Status::OK();
}

bool KimiK3TopologyMatchesFrozen(const KimiK3Topology &top)
{
    return KimiK3TopologyMatchesFrozen(top, /*expect_dsa=*/false,
                                       /*expect_topk=*/0);
}

bool KimiK3TopologyMatchesFrozen(const KimiK3Topology &top, bool expect_dsa,
                                 int expect_topk)
{
    using namespace frozen;
    if (top.num_layers != kLayers || top.hidden != kHidden ||
        top.vocab != kVocab || top.dense_layers != kDenseLayers ||
        top.attn_res_block != kAttnResBlock || top.mla_heads != kMlaHeads ||
        top.kv_heads != kMlaKVHeads || top.q_lora != kQLoraRank ||
        top.kv_lora != kKvLoraRank || top.qk_nope != kQkNopeHeadDim ||
        top.qk_rope != kQkRopeHeadDim || top.v_head != kVHeadDim ||
        top.kda_heads != kKdaHeads || top.kda_head_dim != kKdaHeadDim ||
        top.conv_kernel != kKdaConvKernel || top.num_experts != kExperts ||
        top.top_k != kTopK || top.moe_intermediate != kMoeIntermediate ||
        top.latent != kLatentDim || top.shared_experts != kSharedExperts ||
        top.shared_intermediate != kSharedIntermediate)
        return false;
    if (static_cast<int>(top.mla_layers.size()) != kMlaLayerCount)
        return false;
    for (int l : kMlaLayers) {
        bool found = false;
        for (int x : top.mla_layers)
            if (x == l) found = true;
        if (!found) return false;
    }
    // ── variant half ──
    if (top.dsa != expect_dsa) return false;
    if (expect_dsa) {
        namespace prof = kimi_pro::frozen;
        if (top.index_topk != expect_topk) return false;
        if (top.index_rope_dim != prof::kIndexRopeDim) return false;
        // The key LayerNorm's eps is part of the indexer's arithmetic, not a
        // free knob: a tree that writes a different one was built by a
        // different recipe.
        if (top.index_norm_eps != prof::kIndexNormEps) return false;
        // Every full-attention layer is indexed, and nothing else is. The
        // per-tensor indexer geometry (heads / head dim) is the caller's
        // passport check: this function only owns the structural table.
        if (static_cast<int>(top.dsa_layers.size()) != prof::kDsaLayerCount)
            return false;
        for (int l : prof::kDsaLayers) {
            bool found = false;
            for (int x : top.dsa_layers)
                if (x == l) found = true;
            if (!found) return false;
        }
    }
    return true;
}

bool LoadKimiK3PinnedWeights(const std::string &root, KimiK3PinnedWeights &pw,
                             cudaStream_t stream)
{
    const fs::path base = fs::path(root);
    fs::path pinned = base / "pinned";
    if (!fs::exists(pinned)) pinned = fs::path(root) / "pinned";
    if (!fs::exists(pinned)) return false;

    auto st = LoadKimiK3Topology(root, pw.top);
    if (!st.ok()) {
        log::error(absl::StrCat("[kimi_k3_load] topology: ", st.ToString()));
        return false;
    }

    auto load_raw_file = [&](const fs::path &p, std::vector<TensorRecord> &out) {
        if (!fs::exists(p)) return true;
        auto batch = ReadRecordBatchFromFile(p);
        if (!batch.ok()) return false;
        auto rows = PinnedSchema::UnpackBatch(*batch);
        if (!rows.ok()) return false;
        out = std::move(*rows);
        return true;
    };
    auto load_nvfp4_file = [&](const fs::path &p,
                               std::vector<DenseNvfp4Record> &out) {
        if (!fs::exists(p)) return true;
        auto batch = ReadRecordBatchFromFile(p);
        if (!batch.ok()) return false;
        auto rows = DenseNvfp4Schema::UnpackBatch(*batch);
        if (!rows.ok()) return false;
        out = std::move(*rows);
        return true;
    };

    // ── globals ──
    std::vector<TensorRecord> g_raw;
    std::vector<DenseNvfp4Record> g_nv;
    if (!load_raw_file(pinned / "global.arrow", g_raw)) return false;
    if (!load_nvfp4_file(pinned / "global_nvfp4.arrow", g_nv)) return false;
    {
        const auto *e = FindNvfp4(g_nv, "language_model.model.embed_tokens.weight");
        const auto *h = FindNvfp4(g_nv, "language_model.lm_head.weight");
        if (!e || !h) {
            log::error("[kimi_k3_load] global nvfp4 missing embed/head");
            return false;
        }
        if (!UploadNvfp4(*e, pw.embed, stream) ||
            !UploadNvfp4(*h, pw.head, stream)) {
            log::error("[kimi_k3_load] embed/head upload failed");
            return false;
        }
        const auto *fn = FindRaw(g_raw, "language_model.model.norm.weight");
        const auto *op =
            FindRaw(g_raw, "language_model.model.output_attn_res_proj.weight");
        const auto *on =
            FindRaw(g_raw, "language_model.model.output_attn_res_norm.weight");
        if (!fn || !op || !on) {
            log::error("[kimi_k3_load] global raw missing norm/out_res");
            return false;
        }
        pw.final_norm = UploadHalf(fn->data, stream);
        pw.out_res_proj = UploadHalf(op->data, stream);
        pw.out_res_norm = UploadHalf(on->data, stream);
        if (!pw.final_norm || !pw.out_res_proj || !pw.out_res_norm) return false;
    }

    const int L = pw.top.num_layers;
    pw.layers.assign(static_cast<size_t>(L), KimiK3LayerWeights{});
    pw.streamed.assign(static_cast<size_t>(L), KimiK3StreamedLayer{});

    // Bytes of indexer plane actually allocated (Kimi-K3.1-Pro only).
    std::uint64_t indexer_bytes = 0;

    for (int l = 0; l < L; ++l) {
        std::vector<TensorRecord> raw;
        std::vector<DenseNvfp4Record> nv;
        const std::string tag = "language_model.model.layers." +
                                std::to_string(l) + ".";
        if (!load_raw_file(pinned / ("layer_" + std::to_string(l) + ".arrow"),
                           raw))
            return false;
        if (!load_nvfp4_file(
                pinned / ("layer_" + std::to_string(l) + "_nvfp4.arrow"), nv))
            return false;

        KimiK3LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        lw.mla = pw.top.IsMla(l);
        auto H = [&](std::string_view suffix) -> __half * {
            const auto *r = FindRaw(raw, tag + std::string(suffix));
            if (!r) {
                log::error(absl::StrCat("[kimi_k3_load] missing ", tag,
                                        suffix));
                return nullptr;
            }
            __half *p = UploadHalf(r->data, stream);
            if (!p)
                log::error(absl::StrCat("[kimi_k3_load] half upload ", tag,
                                        suffix, " last_err=",
                                        cudaGetErrorString(cudaGetLastError())));
            return p;
        };
        auto F = [&](std::string_view suffix) -> float * {
            const auto *r = FindRaw(raw, tag + std::string(suffix));
            return r ? UploadF32(r->data, stream) : nullptr;
        };
        auto W = [&](std::string_view suffix,
                     lsx_nvfp4::LsxNvfp4Weight &w) -> bool {
            const auto *r = FindNvfp4(nv, tag + std::string(suffix));
            if (!r) {
                log::error(absl::StrCat("[kimi_k3_load] missing ", tag,
                                        suffix));
                return false;
            }
            if (!UploadNvfp4(*r, w, stream)) {
                log::error(absl::StrCat(
                    "[kimi_k3_load] upload ", tag, suffix, " rows=",
                    r->shape.empty() ? -1 : r->shape[0], " last_err=",
                    cudaGetErrorString(cudaGetLastError())));
                return false;
            }
            return true;
        };
        // Shape-checked BF16 -> FP16 upload (the Gated-DSA indexer plane). The
        // bytes are counted where they are allocated, so `resident_bytes` stays
        // exact for the plane that only the variant has.
        auto HS = [&](std::string_view suffix, int64_t rows, int64_t cols,
                      __half *&dst) -> bool {
            const auto *r = FindRaw(raw, tag + std::string(suffix));
            if (!r) {
                log::error(absl::StrCat("[kimi_k3_load] missing ", tag,
                                        suffix));
                return false;
            }
            const int64_t got_rows = r->shape.empty() ? 0 : r->shape[0];
            const int64_t got_cols =
                r->shape.size() < 2 ? int64_t{1} : r->shape[1];
            if (got_rows != rows || got_cols != cols) {
                log::error(absl::StrCat("[kimi_k3_load] indexer shape ", tag,
                                        suffix, " got ", got_rows, "x",
                                        got_cols, " want ", rows, "x", cols));
                return false;
            }
            dst = UploadHalf(r->data, stream);
            if (!dst) {
                log::error(absl::StrCat("[kimi_k3_load] indexer upload ", tag,
                                        suffix, " last_err=",
                                        cudaGetErrorString(cudaGetLastError())));
                return false;
            }
            indexer_bytes += static_cast<std::uint64_t>(r->data.size());
            return true;
        };

        lw.input_norm = H("input_layernorm.weight");
        lw.post_norm = H("post_attention_layernorm.weight");
        lw.attn_res_proj = H("self_attention_res_proj.weight");
        lw.attn_res_norm = H("self_attention_res_norm.weight");
        lw.mlp_res_proj = H("mlp_res_proj.weight");
        lw.mlp_res_norm = H("mlp_res_norm.weight");
        if (!lw.input_norm || !lw.post_norm || !lw.attn_res_proj ||
            !lw.attn_res_norm || !lw.mlp_res_proj || !lw.mlp_res_norm)
            return false;

        bool ok = true;
        if (lw.mla) {
            ok = W("self_attn.q_a_proj.weight", lw.mla_qa) &&
                 W("self_attn.q_b_proj.weight", lw.mla_qb) &&
                 W("self_attn.kv_a_proj_with_mqa.weight", lw.mla_kva) &&
                 W("self_attn.kv_b_proj.weight", lw.mla_kvb) &&
                 W("self_attn.o_proj.weight", lw.mla_o) &&
                 W("self_attn.g_proj.weight", lw.mla_g);
            lw.mla_qanorm = H("self_attn.q_a_layernorm.weight");
            lw.mla_kvanorm = H("self_attn.kv_a_layernorm.weight");
            ok = ok && lw.mla_qanorm && lw.mla_kvanorm;
            // Gated DSA (Kimi-K3.1-Pro): five FP16 indexer tensors per
            // full-attention layer, uploaded at the frozen geometry the index
            // declared. Any shape disagreement is a hard load failure -- a
            // mis-shaped indexer would select the wrong keys, not merely
            // produce a slightly wrong score.
            if (ok && pw.top.IsDsa(l)) {
                const auto &t = pw.top;
                const int64_t q_dim =
                    static_cast<int64_t>(t.index_heads) * t.index_head_dim;
                ok = HS("self_attn.indexer.wq_b.weight", q_dim, t.q_lora,
                        lw.idx_wqb) &&
                     HS("self_attn.indexer.wk.weight", t.index_head_dim,
                        t.hidden, lw.idx_wk) &&
                     HS("self_attn.indexer.k_norm.weight", t.index_head_dim, 1,
                        lw.idx_knorm_w) &&
                     HS("self_attn.indexer.k_norm.bias", t.index_head_dim, 1,
                        lw.idx_knorm_b) &&
                     HS("self_attn.indexer.weights_proj.weight", t.index_heads,
                        t.hidden, lw.idx_wproj);
            }
        } else {
            ok = W("self_attn.q_proj.weight", lw.kda_q) &&
                 W("self_attn.k_proj.weight", lw.kda_k) &&
                 W("self_attn.v_proj.weight", lw.kda_v) &&
                 W("self_attn.o_proj.weight", lw.kda_o) &&
                 W("self_attn.g_proj.weight", lw.kda_g);
            lw.kda_fa = H("self_attn.f_a_proj.weight");
            lw.kda_fb = H("self_attn.f_b_proj.weight");
            lw.kda_b = H("self_attn.b_proj.weight");
            lw.kda_conv_q = F("self_attn.q_conv1d.weight");
            lw.kda_conv_k = F("self_attn.k_conv1d.weight");
            lw.kda_conv_v = F("self_attn.v_conv1d.weight");
            lw.kda_alog = F("self_attn.A_log");
            lw.kda_dt = F("self_attn.dt_bias");
            lw.kda_onorm = F("self_attn.o_norm.weight");
            ok = ok && lw.kda_fa && lw.kda_fb && lw.kda_b && lw.kda_conv_q &&
                 lw.kda_conv_k && lw.kda_conv_v && lw.kda_alog && lw.kda_dt &&
                 lw.kda_onorm;
        }
        if (!ok) {
            log::error(absl::StrCat("[kimi_k3_load] layer ", l,
                                    " attention/proj upload failed"));
            return false;
        }

        if (l < pw.top.dense_layers) {
            ok = W("mlp.gate_proj.weight", lw.dense_gate) &&
                 W("mlp.up_proj.weight", lw.dense_up) &&
                 W("mlp.down_proj.weight", lw.dense_down);
        } else {
            lw.has_moe = true;
            lw.router_gate = H("block_sparse_moe.gate.weight");
            lw.router_bias = F("block_sparse_moe.gate.e_score_correction_bias");
            lw.latent_norm = H("block_sparse_moe.routed_expert_norm.weight");
            ok = lw.router_gate && lw.router_bias &&
                 (!pw.top.latent_norm || lw.latent_norm);
        }
        if (!ok) {
            log::error(absl::StrCat("[kimi_k3_load] layer ", l,
                                    " moe/router upload failed"));
            return false;
        }
    }

    // ── streamed plane ──
    const fs::path streamed_dir =
        fs::exists(base / "streamed") ? base / "streamed" : fs::path(root) / "streamed";
    for (int l = 0; l < L; ++l) {
        if (l < pw.top.dense_layers) continue;
        const auto path =
            streamed_dir / ("layer_" + std::to_string(l) + ".arrow");
        if (!fs::exists(path)) return false;
        auto views = LoadDenseNvfp4BatchViews(path);
        if (!views.ok()) {
            log::error(absl::StrCat("[kimi_k3_load] streamed views ", l, ": ",
                                    views.status().ToString()));
            return false;
        }
        auto &sl = pw.streamed[static_cast<size_t>(l)];
        sl.keepalive = views->keepalive;
        const std::string tag =
            "language_model.model.layers." + std::to_string(l) + ".";
        auto bind = [&](std::string_view suffix, KimiK3HostNvfp4 &dst) {
            const auto *v = FindView(*views, tag + std::string(suffix));
            if (!v) return false;
            dst.packed = v->packed;
            dst.scales = v->scales;
            dst.global_scale = v->global_scale;
            dst.rows = v->rank >= 1 ? static_cast<int>(v->shape[0]) : 0;
            dst.cols = v->rank >= 2 ? static_cast<int>(v->shape[1]) : 0;
            dst.packed_bytes = v->packed_bytes;
            dst.scale_bytes = v->scale_bytes;
            return dst.packed && dst.scales && dst.rows > 0 && dst.cols > 0;
        };
        sl.loaded =
            bind("block_sparse_moe.routed_expert_down_proj.weight",
                 sl.latent_down) &&
            bind("block_sparse_moe.routed_expert_up_proj.weight", sl.latent_up) &&
            bind("block_sparse_moe.shared_experts.gate_proj.weight",
                 sl.shared_gate) &&
            bind("block_sparse_moe.shared_experts.up_proj.weight",
                 sl.shared_up) &&
            bind("block_sparse_moe.shared_experts.down_proj.weight",
                 sl.shared_down);
        if (!sl.loaded) {
            log::error(absl::StrCat("[kimi_k3_load] streamed bind ", l));
            return false;
        }
    }

    // ── streamed staging (or pinned upload) ──
    int64_t max_staging = 0;
    for (const auto &sl : pw.streamed)
        if (sl.loaded) max_staging = std::max(max_staging, StagingBytesFor(sl));
    if (max_staging > 0) {
        if (cudaMalloc(&pw.stream_dev.buffer, max_staging) != cudaSuccess) {
            log::error(absl::StrCat("[kimi_k3_load] staging alloc ",
                                    max_staging));
            return false;
        }
        pw.stream_dev.bytes = max_staging;
    }

    // ── resident footprint ──
    std::uint64_t bytes = 0;
    auto add_w = [&](const lsx_nvfp4::LsxNvfp4Weight &w) {
        bytes += static_cast<std::uint64_t>(w.packed_bytes + w.scale_bytes);
    };
    add_w(pw.embed);
    add_w(pw.head);
    for (const auto &lw : pw.layers) {
        for (const auto *w : {&lw.mla_qa, &lw.mla_qb, &lw.mla_kva, &lw.mla_kvb,
                              &lw.mla_o, &lw.mla_g, &lw.kda_q, &lw.kda_k,
                              &lw.kda_v, &lw.kda_o, &lw.kda_g, &lw.dense_gate,
                              &lw.dense_up, &lw.dense_down})
            add_w(*w);
        // norms + AttnRes projections + KDA gates/conv (~small FP16/F32)
        bytes += static_cast<std::uint64_t>(pw.top.hidden) * 2 * 14;
        if (lw.has_moe) {
            bytes += static_cast<std::uint64_t>(pw.top.num_experts) *
                     pw.top.hidden * 2;  // router gate (BF16)
            bytes += static_cast<std::uint64_t>(pw.top.num_experts) * 4;
        }
        if (lw.kda_alog) {
            bytes += static_cast<std::uint64_t>(pw.top.kda_heads) *
                     pw.top.kda_head_dim * 4 * 2;  // dt_bias + conv (x3 below)
            bytes += static_cast<std::uint64_t>(pw.top.kda_heads) *
                     pw.top.kda_head_dim * 3 * pw.top.conv_kernel * 4;
            bytes += static_cast<std::uint64_t>(pw.top.kda_head_dim) * 4 * 2;
        }
    }
    bytes += static_cast<std::uint64_t>(pw.stream_dev.bytes);
    // The indexer plane is counted at its allocation sites (above), not
    // estimated: it exists only in the variant tree and must not be guessed.
    bytes += indexer_bytes;
    pw.resident_bytes = bytes;
    return true;
}

void FreeKimiK3PinnedWeights(KimiK3PinnedWeights &pw)
{
    pw.embed.Free();
    pw.head.Free();
    if (pw.final_norm) cudaFree(pw.final_norm);
    if (pw.out_res_proj) cudaFree(pw.out_res_proj);
    if (pw.out_res_norm) cudaFree(pw.out_res_norm);
    for (auto &lw : pw.layers) {
        auto fh = [](__half *p) { if (p) cudaFree(p); };
        auto ff = [](float *p) { if (p) cudaFree(p); };
        fh(lw.input_norm); fh(lw.post_norm); fh(lw.attn_res_proj);
        fh(lw.attn_res_norm); fh(lw.mlp_res_proj); fh(lw.mlp_res_norm);
        fh(lw.mla_qanorm); fh(lw.mla_kvanorm);
        fh(lw.idx_wqb); fh(lw.idx_wk); fh(lw.idx_knorm_w); fh(lw.idx_knorm_b);
        fh(lw.idx_wproj);
        fh(lw.kda_fa); fh(lw.kda_fb); fh(lw.kda_b);
        ff(lw.kda_conv_q); ff(lw.kda_conv_k); ff(lw.kda_conv_v);
        ff(lw.kda_alog); ff(lw.kda_dt); ff(lw.kda_onorm);
        fh(lw.router_gate); fh(lw.latent_norm); ff(lw.router_bias);
        for (auto *w : {&lw.mla_qa, &lw.mla_qb, &lw.mla_kva, &lw.mla_kvb, &lw.mla_o,
                        &lw.mla_g, &lw.kda_q, &lw.kda_k, &lw.kda_v, &lw.kda_o,
                        &lw.kda_g, &lw.dense_gate, &lw.dense_up, &lw.dense_down})
            w->Free();
    }
    if (pw.stream_dev.buffer) cudaFree(pw.stream_dev.buffer);
    pw = KimiK3PinnedWeights{};
}

cudaError_t StageKimiK3Layer(KimiK3PinnedWeights &pw, int layer,
                             cudaStream_t stream)
{
    if (layer < pw.top.dense_layers || layer >= pw.top.num_layers)
        return cudaSuccess;
    KimiK3StreamedLayer &sl = pw.streamed[static_cast<size_t>(layer)];
    if (!sl.loaded) return cudaErrorInvalidValue;

    uint8_t *base = pw.stream_dev.buffer;
    if (!base) return cudaErrorMemoryAllocation;
    struct Bind {
        KimiK3HostNvfp4 *src;
        lsx_nvfp4::LsxNvfp4Weight *dst;
    };
    const Bind binds[5] = {{&sl.latent_down, &pw.stream_dev.latent_down},
                           {&sl.latent_up, &pw.stream_dev.latent_up},
                           {&sl.shared_gate, &pw.stream_dev.shared_gate},
                           {&sl.shared_up, &pw.stream_dev.shared_up},
                           {&sl.shared_down, &pw.stream_dev.shared_down}};
    int64_t off = 0;
    for (const auto &b : binds) {
        uint8_t *p = nullptr;
        uint8_t *sc = nullptr;
        if (b.src->packed_bytes > 0) {
            p = base + off;
            cudaMemcpyAsync(p, b.src->packed,
                            static_cast<size_t>(b.src->packed_bytes),
                            cudaMemcpyHostToDevice, stream);
            off += b.src->packed_bytes;
            sc = base + off;
            cudaMemcpyAsync(sc, b.src->scales,
                            static_cast<size_t>(b.src->scale_bytes),
                            cudaMemcpyHostToDevice, stream);
            off += b.src->scale_bytes;
        }
        *b.dst = lsx_nvfp4::LsxNvfp4Weight{};
        b.dst->d_packed = p;
        b.dst->d_scales = sc;
        b.dst->global_scale = b.src->global_scale;
        b.dst->rows = b.src->rows;
        b.dst->cols = b.src->cols;
        b.dst->packed_bytes = b.src->packed_bytes;
        b.dst->scale_bytes = b.src->scale_bytes;
    }
    return cudaGetLastError();
}

size_t KimiK3ExpertWindowFloor(const KimiK3Topology &top)
{
    const int moe_layers = std::max(0, top.num_layers - top.dense_layers);
    const int top_k = std::max(0, top.top_k);
    const size_t floor =
        static_cast<size_t>(moe_layers) * static_cast<size_t>(top_k);
    return floor > 0 ? floor : 1;
}

size_t KimiK3ResolveExpertWindow(const KimiK3Topology &top, long requested)
{
    const size_t floor = KimiK3ExpertWindowFloor(top);
    if (requested <= 0) return floor;
    const size_t want = static_cast<size_t>(requested);
    if (want >= floor) return want;
    log::warn(absl::StrCat(
        "[kimi_k3_load] expert window ", want,
        " is below one token's working set (", floor,
        " = MoE layers x top-k); clamping up — below the floor every access "
        "misses and the store re-reads experts it already had"));
    return floor;
}

KimiK3ExpertDb LoadKimiK3ExpertIndex(const std::string &model_path,
                                     int num_layers, int experts_per_layer,
                                     int first_moe_layer, size_t max_open,
                                     bool warm_on_load)
{
    KimiK3ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.first_moe_layer = first_moe_layer;
    db.log_prefix = "[kimi_k3_load] ";
    db.mmap_fail_verb = "expert mmap fail ";
    db.emit_diag = false;
    db.log_batch_fails = true;
    db.warm_on_load = warm_on_load;
    // max_open arrives already resolved against the working-set floor by the
    // model (KimiK3ResolveExpertWindow); the store's own default applies only
    // when a caller passes 0.
    db.max_open = max_open ? max_open : 65536;
    db.Load(model_path, /*fallback=*/true);
    // Pinned staging ring (opt-in). Resolution is done HERE and not inside
    // InitStageRing so an explicit programmatic configuration is never
    // overridden by ambient environment; a failure (slots = 0, or pinned memory
    // unavailable) leaves every expert on the legacy copy path, which is what
    // the pipeline's per-expert fallback expects.
    db.ResolveStageRingFromEnv();
    if (db.stage_ring.slots > 0 && !db.InitStageRing()) {
        log::warn("[kimi_k3_load] staging ring requested but unavailable; "
                  "routed experts use the legacy copy path");
    }
    return db;
}

void EnsureKimiK3ExpertResident(KimiK3ExpertDb &db, int layer, int expert,
                                cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void FreeKimiK3ExpertIndex(KimiK3ExpertDb &db)
{
    // One line per session, and only if the ring staged anything: the counters
    // are the evidence that the ring engaged rather than silently falling back.
    db.LogStageStats();
    db.Free();
}

}  // namespace lsxcommon::kimi_k3
