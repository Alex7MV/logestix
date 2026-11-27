#include "lsxcommon/minimax_moe_load.h"

#include "minimax_moe_constants.h"
#include "fast_load_internal.h"

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/expert_store.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace lsxcommon::minimax {
namespace {

using fastload::CvKind;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::PinnedStagingRing;
using fastload::RunUploadJobs;
using fastload::StreamDrain;
using fastload::UploadJob;
using fastload::UploadNvfp4Weight;

CvKind NormKind(const TensorView *t)
{
    return (t && t->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
}

arrow::Status ReadWholeFile(const std::filesystem::path &path,
                            std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return arrow::Status::IOError("cannot open ", path.string());
    out.assign(std::istreambuf_iterator<char>(f),
               std::istreambuf_iterator<char>());
    return arrow::Status::OK();
}

} // namespace

void MinimaxPinnedWeights::Free()
{
    if (d_embed) cudaFree(d_embed);
    if (d_head) cudaFree(d_head);
    d_embed = nullptr;
    d_head = nullptr;
    embed_elems = 0;
    head_elems = 0;
    if (d_final_norm) {
        cudaFree(d_final_norm);
        d_final_norm = nullptr;
    }
    for (auto &l : layers) {
        l.attn.q.Free();
        l.attn.k.Free();
        l.attn.v.Free();
        l.attn.o.Free();
        l.attn.index_q.Free();
        l.attn.index_k.Free();
        if (l.attn.d_arena) cudaFree(l.attn.d_arena);
        l.attn.d_arena = nullptr;
        l.attn.arena_bytes = 0;
        l.attn.d_input_norm = nullptr;
        l.attn.d_post_norm = nullptr;
        l.attn.d_q_norm = nullptr;
        l.attn.d_k_norm = nullptr;
        l.attn.d_index_q_norm = nullptr;
        l.attn.d_index_k_norm = nullptr;
        l.mlp_gate.Free();
        l.mlp_up.Free();
        l.mlp_down.Free();
        if (l.d_router) {
            cudaFree(l.d_router);
            l.d_router = nullptr;
        }
        l.router_bytes = 0;
        if (l.d_router_bias) {
            cudaFree(l.d_router_bias);
            l.d_router_bias = nullptr;
        }
        l.shared_gate.Free();
        l.shared_up.Free();
        l.shared_down.Free();
        l.loaded = false;
    }
    layers.clear();
    resident_bytes = 0;
}

void FreeMinimaxPinnedWeights(MinimaxPinnedWeights &pw) { pw.Free(); }

MinimaxTopology LoadMinimaxTopology(const std::string &model_path)
{
    MinimaxTopology top;
    namespace fs = std::filesystem;
    fs::path base = fs::path(model_path);
    const fs::path idx = base / "model.index.json";
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
    top.rotary_dim = gi("rotary_dim", 0);
    top.rms_norm_eps = gd("rms_norm_eps", 1e-6);
    top.rope_theta = gd("rope_theta", 5000000.0);
    top.num_experts = gi("num_experts", 0);
    top.num_experts_per_tok = gi("num_experts_per_tok", 0);
    top.moe_intermediate_size = gi("moe_intermediate_size", 0);
    top.dense_intermediate_size = gi("dense_intermediate_size", 0);
    top.shared_intermediate_size = gi("shared_intermediate_size", 0);
    top.routed_scaling_factor = gd("routed_scaling_factor", 2.0);
    top.swiglu_alpha = gd("swiglu_alpha", 1.702);
    top.swiglu_limit = gd("swiglu_limit", 7.0);
    top.index_n_heads = gi("index_n_heads", 0);
    top.index_head_dim = gi("index_head_dim", 0);
    top.index_block_size = gi("index_block_size", 0);
    top.index_topk_blocks = gi("index_topk_blocks", 0);
    top.index_local_blocks = gi("index_local_blocks", 1);
    top.eos_token_id = gi("eos_token_id", -1);
    top.bos_token_id = gi("bos_token_id", -1);
    top.pad_token_id = gi("pad_token_id", -1);

    if (auto lt = root["layer_types"].get_array(); lt.error() == simdjson::SUCCESS)
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) top.layer_types.emplace_back(sv);
        }
    if (auto mt = root["mlp_layer_types"].get_array();
        mt.error() == simdjson::SUCCESS)
        for (auto e : mt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) top.mlp_types.emplace_back(sv);
        }
    if (top.num_layers <= 0 || top.num_experts <= 0 ||
        static_cast<int>(top.layer_types.size()) != top.num_layers ||
        static_cast<int>(top.mlp_types.size()) != top.num_layers) {
        top.error_msg = "index missing layers/experts/dispatch fields";
        return top;
    }
    // Derived dispatch boundaries: every registered member cross-checks these
    // against its own frozen passport (M3: sparse from layer 3; M3.1-Flash:
    // sparse from layer 0). A value of num_layers means "never".
    auto first_of = [](const std::vector<std::string> &v,
                       const char *dense) -> int {
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i] != dense) return static_cast<int>(i);
        return static_cast<int>(v.size());
    };
    top.first_moe_layer = first_of(top.mlp_types, "dense");
    top.first_sparse_layer = first_of(top.layer_types, "dense_attention");
    return top;
}

bool LoadMinimaxDenseWeights(const std::string &model_path,
                             MinimaxPinnedWeights &pw, cudaStream_t stream,
                             std::string &error)
{
    namespace fs = std::filesystem;
    pw.Free();
    error.clear();

    const fs::path base = fs::path(model_path);
    const fs::path pinned_dir = base / "pinned";
    if (!fs::exists(pinned_dir)) {
        error = "no pinned dir under " + base.string();
        return false;
    }

    // Index format + geometry. LoadMinimaxTopology stashes the format string
    // in error_msg; clear it once consumed.
    pw.top = LoadMinimaxTopology(model_path);
    const std::string fmt = pw.top.error_msg;
    pw.top.error_msg.clear();
    if (!fmt.empty() && fmt != frozen::kFormatTag) {
        error = absl::StrCat("unsupported index format '", fmt,
                             "' — convert via lsxforge --minimax");
        return false;
    }
    if (pw.top.num_layers <= 0) {
        error = "topology load failed";
        return false;
    }
    if (!SyntheticGeometryAllowed() &&
        (pw.top.num_layers != frozen::kLayers ||
         pw.top.hidden_size != frozen::kHidden ||
         pw.top.num_experts != frozen::kNumExperts)) {
        error = absl::StrCat("topology mismatches frozen passport: layers ",
                             pw.top.num_layers, " hidden ", pw.top.hidden_size,
                             " experts ", pw.top.num_experts);
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

    // ── Globals ──
    {
        auto raw = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw.ok()) {
            error = "global.arrow read failed: " + raw.status().ToString();
            return false;
        }
        auto *fn = FindTensorSuffix(*raw, "norm.weight");
        if (!fn || fn->rank <= 0) {
            error = "final norm missing in global.arrow";
            pw.Free();
            return false;
        }
        const int64_t n = fn->shape[0];
        if (cudaMalloc(&pw.d_final_norm,
                       static_cast<size_t>(n) * sizeof(__half)) != cudaSuccess) {
            error = "final norm alloc failed";
            pw.Free();
            return false;
        }
        if (ring.UploadBf16AsFp16(fn->data, n, pw.d_final_norm, stream,
                                  bytes_acc) != cudaSuccess) {
            error = "final norm upload failed";
            pw.Free();
            return false;
        }
    }
    {
        auto raw = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw.ok()) {
            error = "global.arrow read failed: " + raw.status().ToString();
            pw.Free();
            return false;
        }
        auto *emb = FindTensorSuffix(*raw, "embed_tokens.weight");
        auto *hd = FindTensorSuffix(*raw, "lm_head.weight");
        if (!emb || emb->rank != 2 || !hd || hd->rank != 2) {
            error = "embed_tokens/lm_head missing in global.arrow";
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

    // ── Per-layer ──
    const int L = pw.top.num_layers;
    pw.layers.resize(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        const bool is_moe = pw.top.mlp_types[static_cast<size_t>(l)] == "sparse";
        const bool is_sparse_attn =
            pw.top.layer_types[static_cast<size_t>(l)] == "sparse_attention";
        lw.is_moe = is_moe;

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

        // Norms (FP16, Gemma offset form). One arena per layer.
        auto *in_norm = FindTensorSuffix(*raw, "input_layernorm.weight");
        auto *post_norm = FindTensorSuffix(*raw, "post_attention_layernorm.weight");
        auto *q_norm = FindTensorSuffix(*raw, "self_attn.q_norm.weight");
        auto *k_norm = FindTensorSuffix(*raw, "self_attn.k_norm.weight");
        auto *iq_norm = FindTensorSuffix(*raw, "self_attn.index_q_norm.weight");
        auto *ik_norm = FindTensorSuffix(*raw, "self_attn.index_k_norm.weight");
        std::vector<UploadJob> jobs = {
            {in_norm, reinterpret_cast<void **>(&lw.attn.d_input_norm),
             NormKind(in_norm)},
            {post_norm, reinterpret_cast<void **>(&lw.attn.d_post_norm),
             NormKind(post_norm)},
            {q_norm, reinterpret_cast<void **>(&lw.attn.d_q_norm),
             NormKind(q_norm)},
            {k_norm, reinterpret_cast<void **>(&lw.attn.d_k_norm),
             NormKind(k_norm)},
        };
        if (is_sparse_attn) {
            jobs.push_back({iq_norm,
                            reinterpret_cast<void **>(&lw.attn.d_index_q_norm),
                            NormKind(iq_norm)});
            jobs.push_back({ik_norm,
                            reinterpret_cast<void **>(&lw.attn.d_index_k_norm),
                            NormKind(ik_norm)});
        }
        void *arena = nullptr;
        size_t arena_bytes = 0;
        cudaError_t e = RunUploadJobs(std::move(jobs), &arena, &arena_bytes,
                                      stream, ring, bytes_acc, &job_error);
        lw.attn.d_arena = static_cast<__half *>(arena);
        lw.attn.arena_bytes = arena_bytes;
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, " norms: ", job_error);
            pw.Free();
            return false;
        }

        // Attention projections.
        e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "self_attn.q_proj.weight"),
                              lw.attn.q, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "self_attn.k_proj.weight"),
                                  lw.attn.k, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "self_attn.v_proj.weight"),
                                  lw.attn.v, stream, ring, bytes_acc, &job_error);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "self_attn.o_proj.weight"),
                                  lw.attn.o, stream, ring, bytes_acc, &job_error);
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, " attention: ", job_error);
            pw.Free();
            return false;
        }
        if (is_sparse_attn) {
            e = UploadNvfp4Weight(
                FindNvfp4Suffix(*nv, "self_attn.index_q_proj.weight"),
                lw.attn.index_q, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "self_attn.index_k_proj.weight"),
                    lw.attn.index_k, stream, ring, bytes_acc, &job_error);
            if (e != cudaSuccess || !lw.attn.index_q.d_packed ||
                !lw.attn.index_k.d_packed) {
                error = absl::StrCat("layer_", l, " indexer: ", job_error);
                pw.Free();
                return false;
            }
        }
        if (!lw.attn.q.d_packed || !lw.attn.k.d_packed || !lw.attn.v.d_packed ||
            !lw.attn.o.d_packed || !lw.attn.d_input_norm ||
            !lw.attn.d_post_norm || !lw.attn.d_q_norm || !lw.attn.d_k_norm) {
            error = absl::StrCat("layer_", l, ": attention weights incomplete");
            pw.Free();
            return false;
        }
        lw.attn.loaded = true;

        if (!is_moe) {
            // Dense SwiGLU-OAI MLP.
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mlp.gate_proj.weight"),
                                  lw.mlp_gate, stream, ring, bytes_acc,
                                  &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mlp.up_proj.weight"),
                                      lw.mlp_up, stream, ring, bytes_acc,
                                      &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mlp.down_proj.weight"),
                                      lw.mlp_down, stream, ring, bytes_acc,
                                      &job_error);
            if (e != cudaSuccess || !lw.mlp_gate.d_packed ||
                !lw.mlp_up.d_packed || !lw.mlp_down.d_packed) {
                error = absl::StrCat("layer_", l, " dense MLP: ", job_error);
                pw.Free();
                return false;
            }
            lw.loaded = true;
            continue;
        }

        // Router: F32 gate matrix + F32 routing bias, high precision.
        {
            auto rr = LoadTensorBatchViews(
                base / "router" / absl::StrCat("layer_", l, ".arrow"));
            if (!rr.ok()) {
                error = absl::StrCat("router/layer_", l, ".arrow read failed");
                pw.Free();
                return false;
            }
            auto *gate = FindTensorSuffix(*rr, "block_sparse_moe.gate.weight");
            auto *bias = FindTensorSuffix(*rr, "e_score_correction_bias");
            if (!gate || gate->rank != 2) {
                error = absl::StrCat("router/layer_", l, ": gate weight missing");
                pw.Free();
                return false;
            }
            if (!bias || bias->rank != 1) {
                error = absl::StrCat("router/layer_", l, ": bias missing");
                pw.Free();
                return false;
            }
            lw.router_bytes = gate->bytes;
            if (cudaMalloc(reinterpret_cast<void **>(&lw.d_router),
                           static_cast<size_t>(gate->bytes)) != cudaSuccess) {
                error = absl::StrCat("router/layer_", l, ": alloc failed");
                pw.Free();
                return false;
            }
            if (ring.UploadRaw(gate->data, lw.d_router,
                               static_cast<size_t>(gate->bytes), stream,
                               bytes_acc) != cudaSuccess) {
                error = absl::StrCat("router/layer_", l, ": upload failed");
                pw.Free();
                return false;
            }
            if (cudaMalloc(reinterpret_cast<void **>(&lw.d_router_bias),
                           static_cast<size_t>(bias->bytes)) != cudaSuccess) {
                error = absl::StrCat("router/layer_", l, ": bias alloc failed");
                pw.Free();
                return false;
            }
            if (ring.UploadRaw(bias->data, lw.d_router_bias,
                               static_cast<size_t>(bias->bytes), stream,
                               bytes_acc) != cudaSuccess) {
                error = absl::StrCat("router/layer_", l, ": bias upload failed");
                pw.Free();
                return false;
            }
        }

        // Ungated shared expert (plain SwiGLU-OAI MLP).
        {
            auto sv = LoadDenseNvfp4BatchViews(
                base / "shared" / absl::StrCat("layer_", l, "_nvfp4.arrow"));
            if (!sv.ok()) {
                error = absl::StrCat("shared/layer_", l,
                                     "_nvfp4.arrow read failed");
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(
                FindNvfp4Suffix(*sv, "shared_experts.gate_proj.weight"),
                lw.shared_gate, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(
                    FindNvfp4Suffix(*sv, "shared_experts.up_proj.weight"),
                    lw.shared_up, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(
                    FindNvfp4Suffix(*sv, "shared_experts.down_proj.weight"),
                    lw.shared_down, stream, ring, bytes_acc, &job_error);
            if (e != cudaSuccess || !lw.shared_gate.d_packed ||
                !lw.shared_up.d_packed || !lw.shared_down.d_packed) {
                error = absl::StrCat("layer_", l, " shared expert: ", job_error);
                pw.Free();
                return false;
            }
        }
        lw.loaded = true;
    }

    pw.resident_bytes = bytes_acc;
    return true;
}

void MinimaxExpertStore::Free()
{
    db.Free();
    records.clear();
    num_layers = 0;
    num_experts = 0;
    first_moe_layer = 0;
    loaded = false;
}

bool LoadMinimaxExpertStore(const std::string &model_path,
                            MinimaxExpertStore &out, std::string &error)
{
    namespace fs = std::filesystem;
    out.Free();
    error.clear();

    const fs::path base = fs::path(model_path);
    const fs::path experts_dir = base / "experts";
    if (!fs::exists(experts_dir)) {
        error = "no experts dir under " + base.string();
        return false;
    }

    const MinimaxTopology top = LoadMinimaxTopology(model_path);
    if (top.num_layers <= 0) {
        error = "topology load failed";
        return false;
    }
    // The dense prefix is data-driven (synthetic fixtures differ): the first
    // layer whose MLP dispatch is "sparse" owns the expert files.
    int first_moe = frozen::kFirstMoeLayer;
    for (int l = 0; l < top.num_layers; ++l) {
        if (top.mlp_types[static_cast<size_t>(l)] == "sparse") {
            first_moe = l;
            break;
        }
    }

    out.num_layers = top.num_layers;
    out.num_experts = top.num_experts;
    out.first_moe_layer = first_moe;
    out.records.assign(static_cast<size_t>(out.num_layers),
                       std::vector<MinimaxExpertRecord>(
                           static_cast<size_t>(out.num_experts)));

    MinimaxExpertDb &db = out.db;
    db.num_layers = out.num_layers;
    db.experts_per_layer = out.num_experts;
    db.first_moe_layer = first_moe;
    db.max_open = 4096;
    db.log_prefix = "[minimax-expert] ";
    if (const char *env = std::getenv("LSX_MINIMAX_EXPERT_MAX_OPEN")) {
        const long v = std::atol(env);
        if (v > 0) db.max_open = static_cast<size_t>(v);
    }
    if (!db.Load(model_path, /*fallback=*/true)) {
        error = absl::StrCat("expert store load failed: ", model_path);
        return false;
    }

    // Geometry pass: map each expert once (no payload retained), record sizes
    // and global scales, then trim the mapping window back to the LRU cap.
    cudaStream_t map_stream = nullptr;
    for (int l = first_moe; l < out.num_layers; ++l) {
        auto &recs = out.records[static_cast<size_t>(l)];
        for (int e = 0; e < out.num_experts; ++e) {
            db.EnsureResident(l, e, map_stream);
            const size_t idx = static_cast<size_t>(l) *
                                   static_cast<size_t>(out.num_experts) +
                               static_cast<size_t>(e);
            if (idx >= db.entries.size() || !db.entries[idx].host_w1) {
                error = absl::StrCat("expert read failed: layer_", l, "/exp_", e);
                return false;
            }
            const MinimaxExpertEntry &v = db.entries[idx];
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
    }
    out.loaded = true;
    return true;
}

int EnsureMinimaxExpertResident(MinimaxExpertStore &store,
                                gpu::ExpertDeviceCache &cache, int layer,
                                int eid, cudaStream_t stream,
                                std::string &error, bool *missed,
                                int *evicted_slot)
{
    if (missed) *missed = false;
    if (evicted_slot) *evicted_slot = -1;
    if (layer < 0 || layer >= store.num_layers || eid < 0 ||
        eid >= store.num_experts) {
        error = absl::StrCat("expert out of range: layer ", layer, " eid ", eid);
        return -1;
    }
    const int flat = layer * store.num_experts + eid;
    const int hit = cache.Lookup(flat);
    if (hit >= 0) return hit;

    int evicted_flat = -1;
    const int slot = cache.Insert(flat, evicted_slot, &evicted_flat);
    if (missed) *missed = true;
    if (slot < 0) {
        error = "expert cache insert failed";
        return -1;
    }

    store.db.EnsureResident(layer, eid, stream);
    const size_t idx = static_cast<size_t>(layer) *
                           static_cast<size_t>(store.num_experts) +
                       static_cast<size_t>(eid);
    if (idx >= store.db.entries.size() || !store.db.entries[idx].host_w1) {
        error = absl::StrCat("expert host view unavailable: layer ", layer,
                             " eid ", eid);
        return -1;
    }
    const MinimaxExpertEntry &v = store.db.entries[idx];

    // Device slot layout (fused): [w1|w3|w2] packed, then [s1|s3|s2] scales.
    uint8_t *dst_pack =
        cache.d_pack + static_cast<int64_t>(slot) * cache.pack_stride;
    uint8_t *dst_scale =
        cache.d_scale + static_cast<int64_t>(slot) * cache.scale_stride;
    struct PackSeg { const uint8_t *src; int64_t bytes; };
    const PackSeg psegs[3] = {{v.host_w1, v.w1_bytes},
                              {v.host_w3, v.w3_bytes},
                              {v.host_w2, v.w2_bytes}};
    const PackSeg ssegs[3] = {{v.host_w1_scales, v.w1_scales_bytes},
                              {v.host_w3_scales, v.w3_scales_bytes},
                              {v.host_w2_scales, v.w2_scales_bytes}};
    uint8_t *pk = dst_pack;
    for (const auto &seg : psegs) {
        if (cudaMemcpyAsync(pk, seg.src, static_cast<size_t>(seg.bytes),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            error = "expert H2D pack staging failed";
            return -1;
        }
        pk += seg.bytes;
    }
    uint8_t *sc = dst_scale;
    for (const auto &seg : ssegs) {
        if (cudaMemcpyAsync(sc, seg.src, static_cast<size_t>(seg.bytes),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            error = "expert H2D scale staging failed";
            return -1;
        }
        sc += seg.bytes;
    }
    return slot;
}

bool AllocateMinimaxExpertCache(MinimaxExpertStore &experts,
                                gpu::ExpertDeviceCache &cache, int slots,
                                int top_k, std::string &error)
{
    if (cache.d_base) {
        error = "expert cache already allocated";
        return false;
    }
    int64_t pack_stride = 0, scale_stride = 0;
    for (int l = 0; l < experts.num_layers; ++l) {
        const auto &recs = experts.records[static_cast<size_t>(l)];
        if (recs.empty()) continue;
        const auto &r = recs[0];
        pack_stride = std::max<int64_t>(
            pack_stride, r.w1_bytes + r.w2_bytes + r.w3_bytes);
        scale_stride = std::max<int64_t>(
            scale_stride, r.s1_bytes + r.s2_bytes + r.s3_bytes);
    }
    if (pack_stride <= 0) {
        error = "empty expert store";
        return false;
    }
    const int64_t per_slot = pack_stride + scale_stride;
    if (slots <= 0) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            error = "cudaMemGetInfo failed";
            return false;
        }
        const size_t reserve = size_t{2} << 30;
        const size_t usable = free_b > reserve ? free_b - reserve : 0;
        slots = static_cast<int>(usable / static_cast<size_t>(per_slot));
    }
    if (const char *env = std::getenv("LSX_MINIMAX_EXPERT_CACHE_SLOTS")) {
        const long v = std::atol(env);
        if (v > 0) slots = static_cast<int>(v);
    }
    const int floor_slots = std::max(top_k, 1) * 4;
    slots = std::max(slots, floor_slots);

    const size_t bytes =
        static_cast<size_t>(slots) * static_cast<size_t>(per_slot);
    void *base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        error = absl::StrCat("expert cache alloc failed (", bytes >> 20,
                             " MiB)");
        return false;
    }
    cache.d_base = static_cast<uint8_t *>(base);
    cache.d_pack = cache.d_base;
    cache.d_scale =
        cache.d_base + static_cast<int64_t>(slots) * pack_stride;
    cache.slots = slots;
    cache.pack_stride = pack_stride;
    cache.scale_stride = scale_stride;
    cache.total_experts = experts.num_layers * experts.num_experts;

    cache.h_slot_of = new int32_t[static_cast<size_t>(cache.total_experts)];
    for (int i = 0; i < cache.total_experts; ++i) cache.h_slot_of[i] = -1;
    cache.h_slot_owner = new int32_t[static_cast<size_t>(slots)];
    cache.h_lru = new int32_t[static_cast<size_t>(slots)];
    cache.h_prev = new int32_t[static_cast<size_t>(slots)];
    cache.h_next = new int32_t[static_cast<size_t>(slots)];
    cache.h_free = new int32_t[static_cast<size_t>(slots)];
    for (int i = 0; i < slots; ++i) {
        cache.h_slot_owner[i] = -1;
        cache.h_lru[i] = -1;
        cache.h_prev[i] = -1;
        cache.h_next[i] = -1;
        cache.h_free[i] = slots - 1 - i;
    }
    cache.h_free_top = slots;
    cache.h_lru_front = -1;
    cache.h_lru_back = -1;
    cache.layer_slots = 0;
    cache.h_rotor = nullptr;
    cache.num_layers = 0;
    cache.hits = 0;
    cache.misses = 0;
    return true;
}

void FreeMinimaxExpertCache(gpu::ExpertDeviceCache &cache)
{
    if (cache.d_base) cudaFree(cache.d_base);
    delete[] cache.h_slot_of;
    delete[] cache.h_slot_owner;
    delete[] cache.h_lru;
    delete[] cache.h_prev;
    delete[] cache.h_next;
    delete[] cache.h_free;
    delete[] cache.h_rotor;
    cache = {};
}

KvCacheGeometry MinimaxKvGeometry(const MinimaxTopology &top, int64_t capacity)
{
    return KvCacheGeometry::Make(KvCacheLayout::kPosMajor, capacity,
                                 top.num_key_value_heads, top.head_dim,
                                 top.num_layers);
}

} // namespace lsxcommon::minimax
