#include "lsxcommon/beam_moe_load.h"

#include "beam_moe_constants.h"
#include "fast_load_internal.h"

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace lsxcommon::beam {
namespace {

using fastload::CvKind;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::PinnedStagingRing;
using fastload::RunUploadJobs;
using fastload::StreamDrain;
using fastload::UploadJob;
using fastload::UploadNvfp4Weight;

namespace fs = std::filesystem;

arrow::Status ReadWholeFile(const fs::path &path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return arrow::Status::IOError("cannot open ", path.string());
    out.assign(std::istreambuf_iterator<char>(f),
               std::istreambuf_iterator<char>());
    return arrow::Status::OK();
}

bool CheckNvfp4Shape(const lsx_nvfp4::LsxNvfp4Weight &w, int rows, int cols)
{
    return w.d_packed && w.rows == rows && w.cols == cols;
}

// Opens the neutral routed-expert store for a family whose dense prefix has no
// expert files (first_moe_layer = 3). The neutral LoadMoEExpertStore assumes
// every layer routes, so this is the family's own geometry pass over the same
// ExpertStore.
bool LoadBeamExpertStore(const std::string &model_path,
                         const BeamTopology &top,
                         moe::MoEExpertRuntime &store, std::string &error)
{
    store.Free();
    error.clear();
    const fs::path experts_dir =
        fs::path(model_path) / "experts";
    if (!fs::exists(experts_dir)) {
        error = "no experts dir under " +
                (fs::path(model_path)).string();
        return false;
    }
    store.num_layers = top.num_layers;
    store.num_experts = top.num_experts;
    store.records.resize(static_cast<size_t>(store.num_layers));

    moe::MoEExpertDb &db = store.experts;
    db.num_layers = store.num_layers;
    db.experts_per_layer = store.num_experts;
    db.first_moe_layer = top.num_layers > 0 &&
                                 top.mlp_layer_types.size() ==
                                     static_cast<size_t>(top.num_layers)
                             ? static_cast<int>(std::count(
                                   top.mlp_layer_types.begin(),
                                   top.mlp_layer_types.end(), "dense"))
                             : frozen::kFirstMoeLayer;
    db.max_open = moe::kMoEExpertMaxOpen;
    if (const char *env = std::getenv("LSX_BEAM_EXPERT_MAX_OPEN")) {
        const long v = std::atol(env);
        if (v > 0) db.max_open = static_cast<size_t>(v);
    }
    if (!db.Load(model_path, /*fallback=*/false)) {
        error = absl::StrCat("expert store load failed: ", model_path);
        return false;
    }

    for (int l = db.first_moe_layer; l < store.num_layers; ++l) {
        auto &recs = store.records[static_cast<size_t>(l)];
        recs.assign(static_cast<size_t>(store.num_experts), {});
        for (int e = 0; e < store.num_experts; ++e) {
            moe::EnsureMoEExpertView(db, l, e, nullptr);
            const size_t idx = static_cast<size_t>(l) *
                                   static_cast<size_t>(store.num_experts) +
                               static_cast<size_t>(e);
            if (idx >= db.entries.size() || !db.entries[idx].host_w1) {
                error = absl::StrCat("expert read failed: layer_", l, "/exp_", e);
                return false;
            }
            const moe::MoEExpertEntry &v = db.entries[idx];
            auto &r = recs[static_cast<size_t>(e)];
            r.w1_bytes = v.w1_bytes;
            r.w2_bytes = v.w2_bytes;
            r.w3_bytes = v.w3_bytes;
            r.s1_bytes = v.w1_scales_bytes;
            r.s2_bytes = v.w2_scales_bytes;
            r.s3_bytes = v.w3_scales_bytes;
            r.gs1 = v.w1_global_scale;
            r.gs2 = v.w2_global_scale;
            r.gs3 = v.w3_global_scale;
        }
        log::info(absl::StrCat("[beam_load] layer_", l, ": ",
                               store.num_experts, " experts mapped (views)"));
    }
    return true;
}

} // namespace

void BeamPinnedWeights::Free() { FreeBeamPinnedWeights(*this); }

void FreeBeamPinnedWeights(BeamPinnedWeights &pw)
{
    if (pw.d_embed) cudaFree(pw.d_embed);
    if (pw.d_head) cudaFree(pw.d_head);
    if (pw.d_final_norm) cudaFree(pw.d_final_norm);
    pw.d_embed = nullptr;
    pw.d_head = nullptr;
    pw.d_final_norm = nullptr;
    pw.embed_elems = 0;
    pw.head_elems = 0;
    for (auto &l : pw.layers) {
        l.attn.q.Free();
        l.attn.k.Free();
        l.attn.v.Free();
        l.attn.o.Free();
        l.attn.gate.Free();
        if (l.attn.d_arena) cudaFree(l.attn.d_arena);
        l.attn.d_arena = nullptr;
        l.attn.arena_bytes = 0;
        l.attn.d_input_norm = nullptr;
        l.attn.d_post_attn_norm = nullptr;
        l.attn.d_pre_ffn_norm = nullptr;
        l.attn.d_post_ffn_norm = nullptr;
        l.attn.d_q_norm = nullptr;
        l.attn.d_k_norm = nullptr;
        l.attn.d_gate_bias = nullptr;
        l.attn.loaded = false;
        l.mlp_gate.Free();
        l.mlp_up.Free();
        l.mlp_down.Free();
        l.d_router = nullptr;
        l.d_router_bias = nullptr;
        l.router_bytes = 0;
        l.loaded = false;
    }
    pw.layers.clear();
    pw.expert_store.Free();
    pw.resident_bytes = 0;
}

BeamTopology LoadBeamTopology(const std::string &model_path)
{
    BeamTopology top;
    const fs::path idx = fs::path(model_path) / "model.index.json";
    std::string json;
    auto st = ReadWholeFile(idx, json);
    if (!st.ok()) {
        top.error_msg = st.ToString();
        return top;
    }
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        top.error_msg = "index parse error: " + idx.string();
        return top;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        top.error_msg = "index is not an object";
        return top;
    }
    auto gi = [&](const char *k, int dflt) {
        int64_t v;
        if (root[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto gd = [&](const char *k, double dflt) {
        simdjson::ondemand::value x;
        if (root[k].get(x) != simdjson::SUCCESS) return dflt;
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };
    std::string_view format;
    if (root["format"].get_string().get(format) == simdjson::SUCCESS)
        top.error_msg = std::string(format);  // reused as a format carrier

    top.num_layers = gi("num_layers", 0);
    top.hidden_size = gi("hidden_size", 0);
    top.vocab_size = gi("vocab_size", 0);
    top.num_attention_heads = gi("num_attention_heads", 0);
    top.num_key_value_heads = gi("num_key_value_heads", 0);
    top.head_dim = gi("head_dim", 0);
    top.sliding_window = gi("sliding_window", 0);
    top.rms_norm_eps = gd("rms_norm_eps", 0.0);
    top.rope_theta = gd("rope_theta", 0.0);
    top.num_experts = gi("num_experts", 0);
    top.num_experts_per_tok = gi("num_experts_per_tok", 0);
    top.n_group = gi("n_group", 0);
    top.topk_group = gi("topk_group", 0);
    top.routed_scaling_factor = gd("routed_scaling_factor", 0.0);
    top.final_logit_softcapping = gd("final_logit_softcapping", 0.0);
    top.moe_intermediate_size = gi("moe_intermediate_size", 0);
    top.dense_intermediate_size = gi("dense_intermediate_size", 0);
    top.eos_token_id = gi("eos_token_id", -1);
    top.bos_token_id = gi("bos_token_id", -1);
    top.pad_token_id = gi("pad_token_id", -1);

    if (auto lt = root["layer_types"].get_array();
        lt.error() == simdjson::SUCCESS)
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) top.layer_types.emplace_back(sv);
        }
    if (auto mt = root["mlp_layer_types"].get_array();
        mt.error() == simdjson::SUCCESS)
        for (auto e : mt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) top.mlp_layer_types.emplace_back(sv);
        }
    if (auto es = root["eos_token_ids"].get_array();
        es.error() == simdjson::SUCCESS)
        for (auto e : es.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                top.eos_ids.push_back(static_cast<int>(v));
        }
    if (top.eos_ids.empty() && top.eos_token_id >= 0)
        top.eos_ids.push_back(top.eos_token_id);
    if (top.eos_ids.size() > 1) top.eos_token_id_alt = top.eos_ids[1];

    if (auto vr = root["vram"].get_object(); vr.error() == simdjson::SUCCESS) {
        auto v = vr.value();
        int64_t x = 0;
        if (v["min_bytes"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.min_bytes = x;
        if (v["optimal_bytes"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.optimal_bytes = x;
        if (v["max_bytes"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.max_bytes = x;
    }

    if (top.num_layers <= 0 || top.num_experts <= 0 ||
        static_cast<int>(top.layer_types.size()) != top.num_layers ||
        static_cast<int>(top.mlp_layer_types.size()) != top.num_layers) {
        top.error_msg = "index missing layers/experts/layer_types fields";
        return top;
    }
    return top;
}

KvCacheGeometry BeamKvGeometry(const BeamTopology &top, int64_t capacity)
{
    return KvCacheGeometry::Make(KvCacheLayout::kPosMajor, capacity,
                                 top.num_key_value_heads, top.head_dim,
                                 top.num_layers);
}

bool LoadBeamWeights(const std::string &model_path, BeamPinnedWeights &pw,
                     cudaStream_t stream, std::string &error)
{
    pw.Free();
    error.clear();

    const fs::path base = fs::path(model_path);
    const fs::path pinned_dir = base / "pinned";
    if (!fs::exists(pinned_dir)) {
        error = "no pinned dir under " + base.string();
        return false;
    }

    pw.top = LoadBeamTopology(model_path);
    const std::string fmt = pw.top.error_msg;
    pw.top.error_msg.clear();
    if (!fmt.empty() && fmt != frozen::kFormatTag) {
        error = absl::StrCat("unsupported index format '", fmt,
                             "' — convert via lsxforge --beam");
        return false;
    }
    if (pw.top.num_layers <= 0) {
        error = "topology load failed";
        return false;
    }
    if (!SyntheticGeometryAllowed() &&
        (pw.top.num_layers != frozen::kLayers ||
         pw.top.hidden_size != frozen::kHidden ||
         pw.top.num_experts != frozen::kNumExperts ||
         pw.top.num_attention_heads != frozen::kHeads ||
         pw.top.num_key_value_heads != frozen::kKvHeads ||
         pw.top.head_dim != frozen::kHeadDim ||
         pw.top.vocab_size != frozen::kVocab ||
         pw.top.moe_intermediate_size != frozen::kMoeIntermediate ||
         pw.top.dense_intermediate_size != frozen::kDenseIntermediate ||
         pw.top.num_experts_per_tok != frozen::kTopK ||
         pw.top.n_group != frozen::kNGroups ||
         pw.top.topk_group != frozen::kTopkGroup ||
         pw.top.sliding_window != frozen::kWindow)) {
        error = absl::StrCat("topology mismatches frozen passport: layers ",
                             pw.top.num_layers, "/",
                             static_cast<int>(frozen::kLayers), " hidden ",
                             pw.top.hidden_size, " experts ",
                             pw.top.num_experts, " window ",
                             pw.top.sliding_window);
        return false;
    }

    StreamDrain drain{stream};
    PinnedStagingRing ring;
    if (!ring.ok()) {
        error = "pinned staging ring init failed";
        return false;
    }
    std::uint64_t bytes_acc = 0;
    std::string job_error;

    // ── Globals: embed / lm_head stay FP16, the final norm is FP16. ──
    {
        auto raw = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw.ok()) {
            error = "global.arrow read failed: " + raw.status().ToString();
            return false;
        }
        auto *fn = FindTensorSuffix(*raw, "norm.weight");
        auto *emb = FindTensorSuffix(*raw, "embed_tokens.weight");
        auto *hd = FindTensorSuffix(*raw, "lm_head.weight");
        if (!fn || fn->rank != 1 || !emb || emb->rank != 2 || !hd ||
            hd->rank != 2) {
            error = "embed_tokens/lm_head/final norm missing in global.arrow";
            pw.Free();
            return false;
        }
        const int64_t hn = fn->shape[0];
        if (cudaMalloc(reinterpret_cast<void **>(&pw.d_final_norm),
                       static_cast<size_t>(hn) * sizeof(__half)) != cudaSuccess) {
            error = "final norm alloc failed";
            pw.Free();
            return false;
        }
        if (ring.UploadBf16AsFp16(fn->data, hn, pw.d_final_norm, stream,
                                  bytes_acc) != cudaSuccess) {
            error = "final norm upload failed";
            pw.Free();
            return false;
        }
        pw.embed_elems = emb->shape[0] * emb->shape[1];
        pw.head_elems = hd->shape[0] * hd->shape[1];
        if (cudaMalloc(reinterpret_cast<void **>(&pw.d_embed),
                       static_cast<size_t>(pw.embed_elems) * sizeof(__half)) !=
            cudaSuccess) {
            error = "embed table alloc failed";
            pw.Free();
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&pw.d_head),
                       static_cast<size_t>(pw.head_elems) * sizeof(__half)) !=
            cudaSuccess) {
            error = "head matrix alloc failed";
            pw.Free();
            return false;
        }
        cudaError_t e = ring.UploadBf16AsFp16(emb->data, pw.embed_elems,
                                              pw.d_embed, stream, bytes_acc);
        if (e == cudaSuccess)
            e = ring.UploadBf16AsFp16(hd->data, pw.head_elems, pw.d_head,
                                      stream, bytes_acc);
        if (e != cudaSuccess) {
            error = absl::StrCat("embed/head upload failed: ",
                                 cudaGetErrorString(e));
            pw.Free();
            return false;
        }
    }

    // ── Per-layer dense graph ──
    const int L = pw.top.num_layers;
    const int H = pw.top.hidden_size;
    const int NH = pw.top.num_attention_heads;
    const int NKV = pw.top.num_key_value_heads;
    const int HD = pw.top.head_dim;
    const int E = pw.top.num_experts;
    const int DI = pw.top.dense_intermediate_size;

    pw.layers.assign(static_cast<size_t>(L), BeamLayerWeights{});
    for (int l = 0; l < L; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        lw.is_moe = l < static_cast<int>(pw.top.mlp_layer_types.size()) &&
                    pw.top.mlp_layer_types[static_cast<size_t>(l)] == "sparse";

        auto raw = LoadTensorBatchViews(
            pinned_dir / absl::StrCat("layer_", l, ".arrow"));
        if (!raw.ok()) {
            error = absl::StrCat("layer_", l, ".arrow read failed");
            pw.Free();
            return false;
        }
        auto nv = LoadDenseNvfp4BatchViews(
            pinned_dir / absl::StrCat("layer_", l, "_nvfp4.arrow"));
        if (!nv.ok()) {
            error = absl::StrCat("layer_", l, "_nvfp4.arrow read failed");
            pw.Free();
            return false;
        }

        // Four sandwich norms, the per-head q/k norms and the per-head gate
        // bias (all FP16) live in one per-layer arena.
        auto *in_norm = FindTensorSuffix(*raw, "input_layernorm.weight");
        auto *post_attn = FindTensorSuffix(*raw, "post_attention_layernorm.weight");
        auto *pre_ffn = FindTensorSuffix(*raw, "pre_feedforward_layernorm.weight");
        auto *post_ffn = FindTensorSuffix(*raw, "post_feedforward_layernorm.weight");
        auto *q_norm = FindTensorSuffix(*raw, "self_attn.q_norm.weight");
        auto *k_norm = FindTensorSuffix(*raw, "self_attn.k_norm.weight");
        auto *gate_bias = FindTensorSuffix(*raw, "self_attn.gate_bias");
        auto *router = FindTensorSuffix(*raw, "mlp.gate.weight");
        auto *router_bias = FindTensorSuffix(*raw, "mlp.gate.e_score_correction_bias");
        if (!in_norm || !post_attn || !pre_ffn || !post_ffn || !q_norm ||
            !k_norm || !gate_bias) {
            error = absl::StrCat("layer_", l, " norms/qk-norms/gate-bias missing");
            pw.Free();
            return false;
        }
        if (gate_bias->rank != 1 || gate_bias->shape[0] != NH) {
            error = absl::StrCat("layer_", l, " gate_bias shape != [NH]");
            pw.Free();
            return false;
        }
        if (lw.is_moe && (!router || router->rank != 2 ||
                          router->shape[0] != E || router->shape[1] != H ||
                          !router_bias || router_bias->rank != 1 ||
                          router_bias->shape[0] != E)) {
            error = absl::StrCat("layer_", l, " router/bias missing or wrong shape");
            pw.Free();
            return false;
        }

        std::vector<UploadJob> jobs = {
            {in_norm, reinterpret_cast<void **>(&lw.attn.d_input_norm),
             CvKind::kBf16ToHalf},
            {post_attn, reinterpret_cast<void **>(&lw.attn.d_post_attn_norm),
             CvKind::kBf16ToHalf},
            {pre_ffn, reinterpret_cast<void **>(&lw.attn.d_pre_ffn_norm),
             CvKind::kBf16ToHalf},
            {post_ffn, reinterpret_cast<void **>(&lw.attn.d_post_ffn_norm),
             CvKind::kBf16ToHalf},
            {q_norm, reinterpret_cast<void **>(&lw.attn.d_q_norm),
             CvKind::kBf16ToHalf},
            {k_norm, reinterpret_cast<void **>(&lw.attn.d_k_norm),
             CvKind::kBf16ToHalf},
            {gate_bias, reinterpret_cast<void **>(&lw.attn.d_gate_bias),
             CvKind::kBf16ToHalf},
            {router, reinterpret_cast<void **>(&lw.d_router), CvKind::kF32Raw},
            {router_bias, reinterpret_cast<void **>(&lw.d_router_bias),
             CvKind::kF32Raw},
        };
        void *arena = nullptr;
        size_t arena_bytes = 0;
        cudaError_t e = RunUploadJobs(std::move(jobs), &arena, &arena_bytes,
                                      stream, ring, bytes_acc, &job_error);
        lw.attn.d_arena = static_cast<__half *>(arena);
        lw.attn.arena_bytes = arena_bytes;
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, " raw upload failed: ", job_error);
            pw.Free();
            return false;
        }
        lw.router_bytes = lw.is_moe
                              ? static_cast<int64_t>(E) * H * sizeof(float) +
                                    static_cast<int64_t>(E) * sizeof(float)
                              : 0;

        // Attention projections (NVFP4) + the output gate.
        const auto *q = FindNvfp4Suffix(*nv, "self_attn.q_proj.weight");
        const auto *k = FindNvfp4Suffix(*nv, "self_attn.k_proj.weight");
        const auto *v = FindNvfp4Suffix(*nv, "self_attn.v_proj.weight");
        const auto *o = FindNvfp4Suffix(*nv, "self_attn.o_proj.weight");
        const auto *g = FindNvfp4Suffix(*nv, "self_attn.gate_proj.weight");
        if (!q || !k || !v || !o || !g) {
            error = absl::StrCat("layer_", l, " attention NVFP4 records missing");
            pw.Free();
            return false;
        }
        e = UploadNvfp4Weight(q, lw.attn.q, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(k, lw.attn.k, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(v, lw.attn.v, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(o, lw.attn.o, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(g, lw.attn.gate, stream, ring, bytes_acc,
                                  &job_error);
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, " attention upload failed: ",
                                 job_error);
            pw.Free();
            return false;
        }
        if (!CheckNvfp4Shape(lw.attn.q, NH * HD, H) ||
            !CheckNvfp4Shape(lw.attn.k, NKV * HD, H) ||
            !CheckNvfp4Shape(lw.attn.v, NKV * HD, H) ||
            !CheckNvfp4Shape(lw.attn.o, H, NH * HD) ||
            !CheckNvfp4Shape(lw.attn.gate, NH * HD, H)) {
            error = absl::StrCat("layer_", l, " attention shape mismatch");
            pw.Free();
            return false;
        }
        lw.attn.loaded = true;

        if (lw.is_moe) {
            if (FindTensorSuffix(*raw, "mlp.gate_proj.weight") != nullptr) {
                error = absl::StrCat("layer_", l, " MoE layer has a dense MLP");
                pw.Free();
                return false;
            }
        } else {
            const auto *dg = FindNvfp4Suffix(*nv, "mlp.gate_proj.weight");
            const auto *du = FindNvfp4Suffix(*nv, "mlp.up_proj.weight");
            const auto *dd = FindNvfp4Suffix(*nv, "mlp.down_proj.weight");
            if (!dg || !du || !dd) {
                error = absl::StrCat("layer_", l, " dense MLP records missing");
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(dg, lw.mlp_gate, stream, ring, bytes_acc,
                                  &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(du, lw.mlp_up, stream, ring, bytes_acc,
                                      &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(dd, lw.mlp_down, stream, ring, bytes_acc,
                                      &job_error);
            if (e != cudaSuccess) {
                error = absl::StrCat("layer_", l, " dense MLP upload failed: ",
                                     job_error);
                pw.Free();
                return false;
            }
            if (!CheckNvfp4Shape(lw.mlp_gate, DI, H) ||
                !CheckNvfp4Shape(lw.mlp_up, DI, H) ||
                !CheckNvfp4Shape(lw.mlp_down, H, DI)) {
                error = absl::StrCat("layer_", l, " dense MLP shape mismatch");
                pw.Free();
                return false;
            }
        }
        lw.loaded = true;
    }

    // ── Routed-expert host store (streamed; never pinned) ──
    if (!LoadBeamExpertStore(model_path, pw.top, pw.expert_store, error)) {
        pw.Free();
        return false;
    }

    // Resident bytes: the pinned dense graph only. The streamed expert store is
    // host-backed and the device slot cache is sized by the session.
    pw.resident_bytes = bytes_acc;
    lsxcommon::log::info(absl::StrFormat(
        "[beam_load] dense graph resident: %.2f GiB, experts streamed",
        static_cast<double>(pw.resident_bytes) / (1024.0 * 1024.0 * 1024.0)));
    return true;
}

} // namespace lsxcommon::beam
