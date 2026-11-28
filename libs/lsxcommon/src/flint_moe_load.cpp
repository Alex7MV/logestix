#include "lsxcommon/flint_moe_load.h"
#include "lsxcommon/index_layout.h"

#include "flint_moe_constants.h"
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
#include <tuple>
#include <vector>

namespace lsxcommon::flint {
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

int64_t AlignUp16(int64_t v) { return (v + 15) / 16 * 16; }

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

} // namespace

void FlintPinnedWeights::Free() { FreeFlintPinnedWeights(*this); }

void FreeFlintPinnedWeights(FlintPinnedWeights &pw)
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
        l.attn.q_res.Free();
        l.attn.k_res.Free();
        l.attn.v_res.Free();
        l.attn.o_res.Free();
        if (l.attn.d_arena) cudaFree(l.attn.d_arena);
        l.attn.d_arena = nullptr;
        l.attn.arena_bytes = 0;
        l.attn.d_input_norm = nullptr;
        l.attn.d_post_norm = nullptr;
        l.attn.d_q_norm = nullptr;
        l.attn.d_k_norm = nullptr;
        l.attn.d_sink = nullptr;
        l.mlp_gate.Free();
        l.mlp_up.Free();
        l.mlp_down.Free();
        l.mlp_gate_res.Free();
        l.mlp_up_res.Free();
        l.mlp_down_res.Free();
        if (l.d_router) cudaFree(l.d_router);
        l.d_router = nullptr;
        l.router_bytes = 0;
        l.shared_gate.Free();
        l.shared_up.Free();
        l.shared_down.Free();
        l.shared_gate_res.Free();
        l.shared_up_res.Free();
        l.shared_down_res.Free();
        if (l.expert_slab) cudaFree(l.expert_slab);
        l.expert_slab = nullptr;
        if (l.d_row_table) cudaFree(l.d_row_table);
        l.d_row_table = nullptr;
        l.slab_bytes = 0;
        l.experts.clear();
        l.packed_gate.clear();
        l.packed_down.clear();
        l.packed_up.clear();
        l.scales_gate.clear();
        l.scales_down.clear();
        l.scales_up.clear();
        l.gs_gate.clear();
        l.gs_down.clear();
        l.gs_up.clear();
        l.loaded = false;
    }
    pw.layers.clear();
    pw.resident_bytes = 0;
}

FlintTopology LoadFlintTopology(const std::string &model_path)
{
    FlintTopology top;
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
    // Plain local: the topology struct has no field for a schema generation.
    int64_t layout = 0;
    if (root["layout"].get_int64().get(layout) != simdjson::SUCCESS)
        layout = 0;
    if (!lsxcommon::IndexLayoutSupported(layout)) {
        top.error_msg = lsxcommon::IndexLayoutError(layout);
        return top;
    }

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
    top.moe_intermediate_size = gi("moe_intermediate_size", 0);
    top.dense_intermediate_size = gi("dense_intermediate_size", 0);
    top.shared_intermediate_size = gi("shared_intermediate_size", 0);
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
        mt.error() == simdjson::SUCCESS) {
        int layer = 0;
        for (auto e : mt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS && sv == "dense")
                top.mlp_only_layers.push_back(layer);
            ++layer;
        }
    }
    if (auto nr = root["no_rope_layers"].get_array();
        nr.error() == simdjson::SUCCESS)
        for (auto e : nr.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                top.no_rope_layers.push_back(static_cast<int>(v));
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

    // VRAM block (conversion-time estimate; informational for the governor).
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
        static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = "index missing layers/experts/layer_types fields";
        return top;
    }
    return top;
}

KvCacheGeometry FlintKvGeometry(const FlintTopology &top, int64_t capacity)
{
    return KvCacheGeometry::Make(KvCacheLayout::kPosMajor, capacity,
                                 top.num_key_value_heads, top.head_dim,
                                 top.num_layers);
}

bool LoadFlintDenseWeights(const std::string &model_path, FlintPinnedWeights &pw,
                           cudaStream_t stream, std::string &error)
{
    pw.Free();
    error.clear();

    const fs::path base = fs::path(model_path);
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    if (!fs::exists(pinned_dir)) {
        error = "no pinned dir under " + base.string();
        return false;
    }

    // Index format + geometry. LoadFlintTopology stashes the format string in
    // error_msg; clear it once consumed.
    pw.top = LoadFlintTopology(model_path);
    // A topology that could not be parsed, or whose index declares a schema
    // generation this build does not read, leaves error_msg set (design §2.4.3:
    // the index names a layout, not a per-model format tag). This field used to
    // carry that tag and the check here compared it against the passport's;
    // reading it as a tag now would mangle a real layout refusal into
    // "unsupported index format '<sentence>'" and, worse, CLEAR every other
    // topology failure on the way past.
    if (!pw.top.error_msg.empty()) {
        error = pw.top.error_msg;
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
         pw.top.shared_intermediate_size != frozen::kSharedIntermediate ||
         pw.top.num_experts_per_tok != frozen::kTopK ||
         pw.top.sliding_window != frozen::kWindow)) {
        error = absl::StrCat("topology mismatches frozen passport: layers ",
                             pw.top.num_layers, "/", static_cast<int>(frozen::kLayers),
                             " hidden ", pw.top.hidden_size, " experts ",
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
    const int I = pw.top.moe_intermediate_size;
    const int DI = pw.top.dense_intermediate_size;
    const int SI = pw.top.shared_intermediate_size;

    pw.layers.assign(static_cast<size_t>(L), FlintLayerWeights{});
    for (int l = 0; l < L; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        // MLP dispatch comes from mlp_layer_types, not from the attention kind.
        const bool has_dense_mlp =
            std::find(pw.top.mlp_only_layers.begin(),
                      pw.top.mlp_only_layers.end(), l) !=
            pw.top.mlp_only_layers.end();
        lw.is_moe = !has_dense_mlp;

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

        // Norms (FP16, plain weight form) + the learned sink (FP16, exact for
        // its BF16 source in the logit range). One arena per layer.
        auto *in_norm = FindTensorSuffix(*raw, "input_layernorm.weight");
        auto *post_norm = FindTensorSuffix(*raw, "post_attention_layernorm.weight");
        auto *q_norm = FindTensorSuffix(*raw, "self_attn.q_norm.weight");
        auto *k_norm = FindTensorSuffix(*raw, "self_attn.k_norm.weight");
        auto *sink = FindTensorSuffix(*raw, "self_attn.softmax_offset");
        if (!sink || sink->rank != 1 || sink->shape[0] != NH) {
            error = absl::StrCat("layer_", l, " learned sink missing/shape != [NH]");
            pw.Free();
            return false;
        }
        std::vector<UploadJob> jobs = {
            {in_norm, reinterpret_cast<void **>(&lw.attn.d_input_norm),
             CvKind::kBf16ToHalf},
            {post_norm, reinterpret_cast<void **>(&lw.attn.d_post_norm),
             CvKind::kBf16ToHalf},
            {q_norm, reinterpret_cast<void **>(&lw.attn.d_q_norm),
             CvKind::kBf16ToHalf},
            {k_norm, reinterpret_cast<void **>(&lw.attn.d_k_norm),
             CvKind::kBf16ToHalf},
            {sink, reinterpret_cast<void **>(&lw.attn.d_sink),
             CvKind::kBf16ToHalf},
        };
        void *arena = nullptr;
        size_t arena_bytes = 0;
        cudaError_t e = RunUploadJobs(std::move(jobs), &arena, &arena_bytes,
                                      stream, ring, bytes_acc, &job_error);
        lw.attn.d_arena = static_cast<__half *>(arena);
        lw.attn.arena_bytes = arena_bytes;
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, " norms/sink: ", job_error);
            pw.Free();
            return false;
        }

        // Upload a projection plus, when the converter wrote one, its residual
        // plane (`<name>.res`).
        auto upload_plane = [&](const char *name,
                                lsx_nvfp4::LsxNvfp4Weight &base,
                                lsx_nvfp4::LsxNvfp4Weight &res) -> cudaError_t {
            cudaError_t ce = UploadNvfp4Weight(FindNvfp4Suffix(*nv, name), base,
                                               stream, ring, bytes_acc,
                                               &job_error);
            if (ce != cudaSuccess) return ce;
            const std::string res_name = std::string(name) + ".res";
            const auto *rn = FindNvfp4Suffix(*nv, res_name);
            if (!rn) return cudaSuccess;   // legacy single-plane tree
            return UploadNvfp4Weight(rn, res, stream, ring, bytes_acc,
                                     &job_error);
        };
        if (e == cudaSuccess)
            e = upload_plane("self_attn.q_proj.weight", lw.attn.q,
                             lw.attn.q_res);
        if (e == cudaSuccess)
            e = upload_plane("self_attn.k_proj.weight", lw.attn.k,
                             lw.attn.k_res);
        if (e == cudaSuccess)
            e = upload_plane("self_attn.v_proj.weight", lw.attn.v,
                             lw.attn.v_res);
        if (e == cudaSuccess)
            e = upload_plane("self_attn.o_proj.weight", lw.attn.o,
                             lw.attn.o_res);
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, " attention: ", job_error);
            pw.Free();
            return false;
        }
        if (!CheckNvfp4Shape(lw.attn.q, NH * HD, H) ||
            !CheckNvfp4Shape(lw.attn.k, NKV * HD, H) ||
            !CheckNvfp4Shape(lw.attn.v, NKV * HD, H) ||
            !CheckNvfp4Shape(lw.attn.o, H, NH * HD)) {
            error = absl::StrCat("layer_", l,
                                 " attention projection shape mismatch");
            pw.Free();
            return false;
        }

        if (has_dense_mlp) {
            e = upload_plane("mlp.gate_proj.weight", lw.mlp_gate,
                             lw.mlp_gate_res);
            if (e == cudaSuccess)
                e = upload_plane("mlp.up_proj.weight", lw.mlp_up,
                                 lw.mlp_up_res);
            if (e == cudaSuccess)
                e = upload_plane("mlp.down_proj.weight", lw.mlp_down,
                                 lw.mlp_down_res);
            if (e != cudaSuccess ||
                !CheckNvfp4Shape(lw.mlp_gate, DI, H) ||
                !CheckNvfp4Shape(lw.mlp_up, DI, H) ||
                !CheckNvfp4Shape(lw.mlp_down, H, DI)) {
                error = absl::StrCat("layer_", l, " dense MLP: ", job_error);
                pw.Free();
                return false;
            }
        } else {
            // Router: gate BF16 -> F32 on device. Never 4-bit.
            auto router = LoadTensorBatchViews(
                router_dir / absl::StrCat("layer_", l, ".arrow"));
            if (!router.ok()) {
                error = absl::StrCat("router/layer_", l, ".arrow read failed");
                pw.Free();
                return false;
            }
            auto *gate = FindTensorSuffix(*router, "mlp.gate.weight");
            if (!gate || gate->rank != 2) {
                error = absl::StrCat("layer_", l, " router gate missing");
                pw.Free();
                return false;
            }
            const int64_t rows = gate->shape[0];
            const int64_t cols = gate->shape[1];
            if (rows != E || cols != H) {
                error = absl::StrCat("layer_", l, " router shape != [E,H]");
                pw.Free();
                return false;
            }
            void *router_dst = nullptr;
            std::vector<UploadJob> rjobs = {
                {gate, &router_dst, CvKind::kBf16ToF32},
            };
            void *rarena = nullptr;
            size_t rarena_bytes = 0;
            e = RunUploadJobs(std::move(rjobs), &rarena, &rarena_bytes, stream,
                              ring, bytes_acc, &job_error);
            if (e != cudaSuccess) {
                error = absl::StrCat("layer_", l, " router: ", job_error);
                pw.Free();
                return false;
            }
            lw.d_router = static_cast<float *>(router_dst);
            lw.router_bytes = static_cast<int64_t>(rarena_bytes);

            // Ungated shared expert (plain SwiGLU).
            auto shared = LoadDenseNvfp4BatchViews(
                shared_dir / absl::StrCat("layer_", l, "_nvfp4.arrow"));
            if (!shared.ok()) {
                error = absl::StrCat("shared/layer_", l,
                                     "_nvfp4.arrow read failed");
                pw.Free();
                return false;
            }
            auto upload_shared = [&](const char *name,
                                     lsx_nvfp4::LsxNvfp4Weight &base,
                                     lsx_nvfp4::LsxNvfp4Weight &res)
                -> cudaError_t {
                cudaError_t ce = UploadNvfp4Weight(
                    FindNvfp4Suffix(*shared, name), base, stream, ring,
                    bytes_acc, &job_error);
                if (ce != cudaSuccess) return ce;
                const std::string res_name = std::string(name) + ".res";
                const auto *rn = FindNvfp4Suffix(*shared, res_name);
                if (!rn) return cudaSuccess;
                return UploadNvfp4Weight(rn, res, stream, ring, bytes_acc,
                                         &job_error);
            };
            e = upload_shared("shared_expert.gate_proj.weight", lw.shared_gate,
                              lw.shared_gate_res);
            if (e == cudaSuccess)
                e = upload_shared("shared_expert.up_proj.weight", lw.shared_up,
                                  lw.shared_up_res);
            if (e == cudaSuccess)
                e = upload_shared("shared_expert.down_proj.weight",
                                  lw.shared_down, lw.shared_down_res);
            if (e != cudaSuccess ||
                !CheckNvfp4Shape(lw.shared_gate, SI, H) ||
                !CheckNvfp4Shape(lw.shared_up, SI, H) ||
                !CheckNvfp4Shape(lw.shared_down, H, SI)) {
                error = absl::StrCat("layer_", l, " shared expert: ", job_error);
                pw.Free();
                return false;
            }
        }
        lw.attn.loaded = true;
        lw.loaded = true;
    }

    // ── Routed experts: one device slab per MoE layer (all resident). ──
    for (int l = 0; l < L; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        if (!lw.is_moe) continue;
        const fs::path dir = base / "experts" / absl::StrCat("layer_", l);
        if (!fs::exists(dir)) {
            error = absl::StrCat("missing experts dir ", dir.string());
            pw.Free();
            return false;
        }

        struct Staged {
            std::vector<uint8_t> w1, w2, w3, s1, s2, s3;
            FlintExpertRecord rec;
        };
        std::vector<Staged> staged(static_cast<size_t>(E));
        int64_t total = 0;
        for (int k = 0; k < E; ++k) {
            const fs::path path = dir / absl::StrCat("exp_", k, ".arrow");
            auto batch = ReadRecordBatchFromFile(path);
            if (!batch.ok()) {
                error = "read failed: " + path.string();
                pw.Free();
                return false;
            }
            auto pe = ExpertSchema::UnpackBatch(*batch, kNVFP4BlockSize);
            if (!pe.ok()) {
                error = "unpack failed: " + path.string();
                pw.Free();
                return false;
            }
            auto &st = staged[static_cast<size_t>(k)];
            st.w1 = pe->w1_packed;
            st.w2 = pe->w2_packed;
            st.w3 = pe->w3_packed;
            st.s1 = pe->w1_scales;
            st.s2 = pe->w2_scales;
            st.s3 = pe->w3_scales;
            const int64_t gate_elems = static_cast<int64_t>(I) * H;
            const int64_t down_elems = static_cast<int64_t>(H) * I;
            if (static_cast<int64_t>(st.w1.size()) != gate_elems / 2 ||
                static_cast<int64_t>(st.w3.size()) != gate_elems / 2 ||
                static_cast<int64_t>(st.w2.size()) != down_elems / 2) {
                error = absl::StrCat("expert ", k, " layer ", l,
                                     " packed size mismatch");
                pw.Free();
                return false;
            }
            auto &r = st.rec;
            r.w1_off = total;
            r.w1_bytes = static_cast<int64_t>(st.w1.size());
            total += r.w1_bytes;
            r.w2_off = total;
            r.w2_bytes = static_cast<int64_t>(st.w2.size());
            total += r.w2_bytes;
            r.w3_off = total;
            r.w3_bytes = static_cast<int64_t>(st.w3.size());
            total += r.w3_bytes;
            r.gs1 = pe->w1_global_scale;
            r.gs2 = pe->w2_global_scale;
            r.gs3 = pe->w3_global_scale;
            total = AlignUp16(total);
            r.s1_off = total;
            r.s1_bytes = static_cast<int64_t>(st.s1.size());
            total += r.s1_bytes;
            r.s2_off = total;
            r.s2_bytes = static_cast<int64_t>(st.s2.size());
            total += r.s2_bytes;
            r.s3_off = total;
            r.s3_bytes = static_cast<int64_t>(st.s3.size());
            total += r.s3_bytes;
        }

        void *slab = nullptr;
        const cudaError_t ae = cudaMalloc(&slab, static_cast<size_t>(total));
        if (ae != cudaSuccess) {
            error = absl::StrCat(
                "expert slab alloc failed (layer ", l, ", ",
                static_cast<unsigned long long>(total >> 20), " MiB): ",
                cudaGetErrorString(ae));
            pw.Free();
            return false;
        }
        lw.expert_slab = static_cast<uint8_t *>(slab);
        lw.slab_bytes = static_cast<size_t>(total);
        bytes_acc += static_cast<std::uint64_t>(total);
        for (int k = 0; k < E; ++k) {
            auto &st = staged[static_cast<size_t>(k)];
            const auto &r = st.rec;
            for (const auto [off, bytes, src] :
                 {std::tuple{r.w1_off, r.w1_bytes, &st.w1},
                  {r.w2_off, r.w2_bytes, &st.w2},
                  {r.w3_off, r.w3_bytes, &st.w3},
                  {r.s1_off, r.s1_bytes, &st.s1},
                  {r.s2_off, r.s2_bytes, &st.s2},
                  {r.s3_off, r.s3_bytes, &st.s3}}) {
                if (bytes == 0) continue;
                const cudaError_t cp = cudaMemcpy(
                    lw.expert_slab + off, src->data(), static_cast<size_t>(bytes),
                    cudaMemcpyHostToDevice);
                if (cp != cudaSuccess) {
                    error = "expert H2D copy failed";
                    pw.Free();
                    return false;
                }
            }
            lw.experts.push_back(r);
        }

        // Host pointer tables for the neutral batched expert GEMV.
        const size_t n = static_cast<size_t>(E);
        lw.packed_gate.assign(n, nullptr);
        lw.packed_down.assign(n, nullptr);
        lw.packed_up.assign(n, nullptr);
        lw.scales_gate.assign(n, nullptr);
        lw.scales_down.assign(n, nullptr);
        lw.scales_up.assign(n, nullptr);
        lw.gs_gate.assign(n, 1.0f);
        lw.gs_down.assign(n, 1.0f);
        lw.gs_up.assign(n, 1.0f);
        std::vector<lsx_nvfp4::LsxExpertPtrRow> table(3 * n);
        for (size_t k = 0; k < n; ++k) {
            const auto &r = lw.experts[k];
            lw.packed_gate[k] = lw.expert_slab + r.w1_off;
            lw.scales_gate[k] = lw.expert_slab + r.s1_off;
            lw.packed_down[k] = lw.expert_slab + r.w2_off;
            lw.scales_down[k] = lw.expert_slab + r.s2_off;
            lw.packed_up[k] = lw.expert_slab + r.w3_off;
            lw.scales_up[k] = lw.expert_slab + r.s3_off;
            lw.gs_gate[k] = r.gs1;
            lw.gs_down[k] = r.gs2;
            lw.gs_up[k] = r.gs3;
            table[k] = {lw.packed_gate[k], lw.scales_gate[k], r.gs1};
            table[n + k] = {lw.packed_down[k], lw.scales_down[k], r.gs2};
            table[2 * n + k] = {lw.packed_up[k], lw.scales_up[k], r.gs3};
        }
        if (cudaMalloc(reinterpret_cast<void **>(&lw.d_row_table),
                       sizeof(lsx_nvfp4::LsxExpertPtrRow) * table.size()) !=
            cudaSuccess) {
            error = absl::StrCat("layer ", l, ": expert pointer table alloc failed");
            pw.Free();
            return false;
        }
        if (cudaMemcpy(lw.d_row_table, table.data(),
                       sizeof(lsx_nvfp4::LsxExpertPtrRow) * table.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            error = absl::StrCat("layer ", l, ": expert pointer table upload failed");
            pw.Free();
            return false;
        }
    }

    // Any CUDA failure so far is sticky; surface it before claiming success.
    if (const cudaError_t sync = cudaStreamSynchronize(stream);
        sync != cudaSuccess) {
        error = absl::StrCat("weight upload drain failed: ",
                             cudaGetErrorString(sync));
        pw.Free();
        return false;
    }
    pw.resident_bytes = bytes_acc;
    log::info(absl::StrFormat(
        "flint loader: %d layers, %d experts/layer resident, %llu MiB", L, E,
        static_cast<unsigned long long>(bytes_acc >> 20)));
    return true;
}

} // namespace lsxcommon::flint
