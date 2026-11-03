#include "lsxcommon/step_moe_load.h"

#include "step_moe_constants.h"

#include "lsxcommon/log.h"
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

namespace lsxcommon::step {
namespace {

namespace fs = std::filesystem;

arrow::Status ReadWholeFile(const fs::path &path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return arrow::Status::IOError("cannot open ", path.string());
    out.assign(std::istreambuf_iterator<char>(f),
               std::istreambuf_iterator<char>());
    return arrow::Status::OK();
}

const TensorView *Find(const TensorBatchViews &t, const char *suffix)
{
    for (const auto &v : t.tensors) {
        if (v.name.size() >= std::strlen(suffix) &&
            v.name.compare(v.name.size() - std::strlen(suffix),
                           std::strlen(suffix), suffix) == 0)
            return &v;
    }
    return nullptr;
}

bool UploadBytes(const uint8_t *src, size_t bytes, void **dst, cudaStream_t s)
{
    if (cudaMalloc(dst, bytes) != cudaSuccess) return false;
    if (cudaMemcpyAsync(*dst, src, bytes, cudaMemcpyHostToDevice, s) !=
        cudaSuccess) {
        cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}

bool UploadNvfp4(const DenseNvfp4BatchViews &nv, const char *suffix,
                 lsx_nvfp4::LsxNvfp4Weight &w, cudaStream_t s)
{
    const DenseNvfp4View *v = nullptr;
    for (const auto &r : nv.records) {
        if (r.name.size() >= std::strlen(suffix) &&
            r.name.compare(r.name.size() - std::strlen(suffix),
                           std::strlen(suffix), suffix) == 0) {
            v = &r;
            break;
        }
    }
    if (!v || v->rank != 2) return false;
    w.rows = static_cast<int>(v->shape[0]);
    w.cols = static_cast<int>(v->shape[1]);
    w.global_scale = v->global_scale;
    w.packed_bytes = v->packed_bytes;
    w.scale_bytes = v->scale_bytes;
    void *p = nullptr;
    void *sc = nullptr;
    if (!UploadBytes(v->packed, static_cast<size_t>(v->packed_bytes), &p, s))
        return false;
    if (!UploadBytes(v->scales, static_cast<size_t>(v->scale_bytes), &sc, s)) {
        cudaFree(p);
        return false;
    }
    w.d_packed = static_cast<const uint8_t *>(p);
    w.d_scales = static_cast<const uint8_t *>(sc);
    return true;
}

bool CheckNvfp4(const lsx_nvfp4::LsxNvfp4Weight &w, int rows, int cols)
{
    return w.d_packed && w.rows == rows && w.cols == cols;
}

} // namespace

void StepPinnedWeights::Free() { FreeStepPinnedWeights(*this); }

void FreeStepPinnedWeights(StepPinnedWeights &pw)
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
        l.attn.g.Free();
        if (l.attn.d_q_norm) cudaFree(l.attn.d_q_norm);
        if (l.attn.d_k_norm) cudaFree(l.attn.d_k_norm);
        l.attn.d_q_norm = nullptr;
        l.attn.d_k_norm = nullptr;
        if (l.d_input_norm) cudaFree(l.d_input_norm);
        if (l.d_post_norm) cudaFree(l.d_post_norm);
        l.d_input_norm = nullptr;
        l.d_post_norm = nullptr;
        l.mlp_gate.Free();
        l.mlp_up.Free();
        l.mlp_down.Free();
        if (l.d_router) cudaFree(l.d_router);
        if (l.d_router_bias) cudaFree(l.d_router_bias);
        l.d_router = nullptr;
        l.d_router_bias = nullptr;
        l.router_bytes = 0;
        l.shared_gate.Free();
        l.shared_up.Free();
        l.shared_down.Free();
        l.loaded = false;
    }
    pw.layers.clear();
    moe::FreeMoEExpertCache(pw.expert_cache);
    pw.expert_cache_bytes = 0;
    pw.expert_store.Free();
    pw.resident_bytes = 0;
}

StepTopology LoadStepTopology(const std::string &model_path)
{
    StepTopology top;
    const fs::path idx = fs::path(model_path) / "logestix" / "model.index.json";
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
    top.rms_norm_eps = gd("rms_norm_eps", 1e-5);
    top.router_scaling_factor = gd("router_scaling_factor", 3.0);
    top.num_experts = gi("num_experts", 0);
    top.num_experts_per_tok = gi("num_experts_per_tok", 0);
    top.moe_intermediate_size = gi("moe_intermediate_size", 0);
    top.dense_intermediate_size = gi("dense_intermediate_size", 0);
    top.shared_intermediate_size = gi("shared_intermediate_size", 0);
    top.first_moe_layer = gi("first_moe_layer", 0);
    top.num_mtp_layers = gi("num_mtp_layers_stripped", 0);
    top.sliding_window = gi("sliding_window", 0);
    top.eos_token_id = gi("eos_token_id", -1);
    top.eos_token_id2 = gi("eos_token_id2", -1);
    top.bos_token_id = gi("bos_token_id", -1);
    top.pad_token_id = gi("pad_token_id", -1);
    top.skipped_ballast_tensors = gi("skipped_visual_tensors", 0);
    top.skipped_mtp_tensors = gi("skipped_mtp_tensors", 0);
    top.indexer_tensors = gi("indexer_tensors", 0);

    if (auto lt = root["mlp_layer_types"].get_array(); lt.error() == simdjson::SUCCESS)
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) top.mlp_types.emplace_back(sv);
        }
    if (auto lt = root["layer_types"].get_array(); lt.error() == simdjson::SUCCESS)
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS) top.layer_types.emplace_back(sv);
        }
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
        static_cast<int>(top.mlp_types.size()) != top.num_layers ||
        static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = "index missing layers/experts/dispatch fields";
        return top;
    }
    return top;
}

KvCacheGeometry StepKvGeometry(const StepTopology &top, int64_t capacity)
{
    return KvCacheGeometry::Make(KvCacheLayout::kPosMajor, capacity,
                                 top.num_key_value_heads, top.head_dim,
                                 top.num_layers);
}

namespace {

// Last routed-MoE layer + 1 (the dense prefix AND a trailing dense layer are
// excluded; the neutral store only skips a dense prefix).
int ExpertStoreLayers(const StepTopology &top)
{
    int last = -1;
    for (int l = 0; l < top.num_layers; ++l)
        if (top.mlp_types[static_cast<size_t>(l)] == "sparse") last = l;
    return last + 1;
}

bool LoadExpertStore(const std::string &model_path, StepPinnedWeights &pw,
                     std::string &error)
{
    moe::MoEExpertRuntime &store = pw.expert_store;
    const StepTopology &top = pw.top;
    const int expert_layers = ExpertStoreLayers(top);
    store.num_layers = expert_layers;
    store.num_experts = top.num_experts;
    store.records.assign(
        static_cast<size_t>(std::max(expert_layers, top.num_layers)),
        std::vector<moe::MoEExpertRecord>(static_cast<size_t>(top.num_experts)));

    moe::MoEExpertDb &db = store.experts;
    db.num_layers = expert_layers;
    db.experts_per_layer = top.num_experts;
    db.first_moe_layer = top.first_moe_layer;
    db.zero_pad_expert_names = false;
    db.max_open = moe::kMoEExpertMaxOpen;
    if (const char *env = std::getenv("LSX_STEP_EXPERT_MAX_OPEN")) {
        const long v = std::atol(env);
        if (v > 0) db.max_open = static_cast<size_t>(v);
    }
    db.log_prefix = "[step-expert] ";
    if (!db.Load(model_path, false)) {
        error = "expert store load failed: " + model_path;
        return false;
    }
    // Geometry pass: byte sizes (one slot layout) and the per-tensor global
    // scales. Each expert carries its own three NVFP4 global scales.
    for (int l = top.first_moe_layer; l < expert_layers; ++l) {
        if (top.mlp_types[static_cast<size_t>(l)] != "sparse") continue;
        auto &recs = store.records[static_cast<size_t>(l)];
        for (int e = 0; e < top.num_experts; ++e) {
            moe::EnsureMoEExpertView(db, l, e, nullptr);
            const size_t idx = static_cast<size_t>(l) *
                                   static_cast<size_t>(top.num_experts) +
                               static_cast<size_t>(e);
            if (idx >= db.entries.size() || !db.entries[idx].host_w1) {
                error = absl::StrCat("expert read failed: layer_", l, "/exp_", e);
                return false;
            }
            const auto &v = db.entries[idx];
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
    // Fused w13 global scales for the neutral split route.
    const int flats = expert_layers * top.num_experts;
    std::vector<float> hw(static_cast<size_t>(flats) * 2, 1.0f);
    for (int l = top.first_moe_layer; l < expert_layers; ++l)
        for (int e = 0; e < top.num_experts; ++e) {
            const auto &r = store.records[static_cast<size_t>(l)]
                                        [static_cast<size_t>(e)];
            const size_t at = (static_cast<size_t>(l) * top.num_experts + e) * 2;
            hw[at] = r.gs1;
            hw[at + 1] = r.gs3;
        }
    if (cudaMalloc(&store.d_seg_wgs, hw.size() * sizeof(float)) != cudaSuccess)
        return false;
    if (cudaMemcpy(store.d_seg_wgs, hw.data(), hw.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return true;
}

int ResolveCacheSlots(int64_t slot_bytes)
{
    if (const char *env = std::getenv("LSX_STEP_EXPERT_CACHE_SLOTS")) {
        const long v = std::atol(env);
        if (v > 0) return static_cast<int>(v);
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 512;
    // Reserve the KV cache at the supported context plus the per-step scratch
    // and the CUDA/graph overhead.
    const int64_t reserve = (int64_t{6} << 30);
    if (static_cast<int64_t>(free_b) <= reserve) return 64;
    const int64_t usable = static_cast<int64_t>(free_b) - reserve;
    int64_t slots = slot_bytes > 0 ? usable / slot_bytes : 64;
    slots = std::clamp<int64_t>(slots, 64, 4096);
    return static_cast<int>(slots);
}

} // namespace

bool LoadStepDenseWeights(const std::string &model_path, StepPinnedWeights &pw,
                          cudaStream_t stream, std::string &error)
{
    pw.Free();
    error.clear();
    pw.top = LoadStepTopology(model_path);
    const std::string fmt = pw.top.error_msg;
    pw.top.error_msg.clear();
    if (!fmt.empty() && fmt != frozen::kFormatTag) {
        error = absl::StrCat("unsupported index format '", fmt,
                             "' — convert via lsxforge --step");
        return false;
    }
    if (pw.top.num_layers <= 0) {
        error = "model.index.json unreadable";
        return false;
    }
    const StepTopology &top = pw.top;
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const fs::path base = fs::path(model_path) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";

    std::uint64_t accounted = 0;

    // ── globals ─────────────────────────────────────────────────────────────
    {
        auto raw = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw.ok()) {
            error = "global.arrow read failed: " + raw.status().ToString();
            return false;
        }
        auto *fn = Find(*raw, "norm.weight");
        auto *emb = Find(*raw, "embed_tokens.weight");
        auto *hd = Find(*raw, "lm_head.weight");
        if (!fn || fn->rank != 1 || !emb || emb->rank != 2 || !hd ||
            hd->rank != 2) {
            error = "embed_tokens/lm_head/final norm missing in global.arrow";
            pw.Free();
            return false;
        }
        if (!UploadBytes(fn->data, static_cast<size_t>(fn->bytes),
                         reinterpret_cast<void **>(&pw.d_final_norm), stream) ||
            !UploadBytes(emb->data, static_cast<size_t>(emb->bytes),
                         reinterpret_cast<void **>(&pw.d_embed), stream) ||
            !UploadBytes(hd->data, static_cast<size_t>(hd->bytes),
                         reinterpret_cast<void **>(&pw.d_head), stream)) {
            error = "global upload failed";
            pw.Free();
            return false;
        }
        pw.embed_elems = emb->shape[0] * emb->shape[1];
        pw.head_elems = hd->shape[0] * hd->shape[1];
        accounted += fn->bytes + emb->bytes + hd->bytes;
    }

    pw.layers.resize(static_cast<size_t>(top.num_layers));
    for (int l = 0; l < top.num_layers; ++l) {
        StepLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        lw.is_moe = top.mlp_types[static_cast<size_t>(l)] == "sparse";

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

        // Norms (F32, the checkpoint's norm_dtype).
        auto *in_norm = Find(*raw, "input_layernorm.weight");
        auto *post_norm = Find(*raw, "post_attention_layernorm.weight");
        auto *q_norm = Find(*raw, "self_attn.q_norm.weight");
        auto *k_norm = Find(*raw, "self_attn.k_norm.weight");
        if (!in_norm || !post_norm || !q_norm || !k_norm) {
            error = absl::StrCat("layer_", l, " norms missing");
            pw.Free();
            return false;
        }
        if (!UploadBytes(in_norm->data, static_cast<size_t>(in_norm->bytes),
                         reinterpret_cast<void **>(&lw.d_input_norm), stream) ||
            !UploadBytes(post_norm->data, static_cast<size_t>(post_norm->bytes),
                         reinterpret_cast<void **>(&lw.d_post_norm), stream) ||
            !UploadBytes(q_norm->data, static_cast<size_t>(q_norm->bytes),
                         reinterpret_cast<void **>(&lw.attn.d_q_norm), stream) ||
            !UploadBytes(k_norm->data, static_cast<size_t>(k_norm->bytes),
                         reinterpret_cast<void **>(&lw.attn.d_k_norm), stream)) {
            error = absl::StrCat("layer_", l, " norm upload failed");
            pw.Free();
            return false;
        }
        accounted += in_norm->bytes + post_norm->bytes + q_norm->bytes +
                     k_norm->bytes;

        // Attention projections (+ the per-head gate).
        const struct {
            const char *suffix;
            lsx_nvfp4::LsxNvfp4Weight *w;
            int rows, cols;
        } kAttn[] = {
            {"self_attn.q_proj.weight", &lw.attn.q, NH * HD, H},
            {"self_attn.k_proj.weight", &lw.attn.k, NKV * HD, H},
            {"self_attn.v_proj.weight", &lw.attn.v, NKV * HD, H},
            {"self_attn.o_proj.weight", &lw.attn.o, H, NH * HD},
            {"self_attn.g_proj.weight", &lw.attn.g, NH, H},
        };
        for (const auto &a : kAttn) {
            if (!UploadNvfp4(*nv, a.suffix, *a.w, stream) ||
                !CheckNvfp4(*a.w, a.rows, a.cols)) {
                error = absl::StrCat("layer_", l, " ", a.suffix, " mismatch");
                pw.Free();
                return false;
            }
            accounted += a.w->packed_bytes + a.w->scale_bytes;
        }

        if (!lw.is_moe) {
            const int DI = top.dense_intermediate_size;
            if (!UploadNvfp4(*nv, "mlp.gate_proj.weight", lw.mlp_gate, stream) ||
                !UploadNvfp4(*nv, "mlp.up_proj.weight", lw.mlp_up, stream) ||
                !UploadNvfp4(*nv, "mlp.down_proj.weight", lw.mlp_down, stream) ||
                !CheckNvfp4(lw.mlp_gate, DI, H) ||
                !CheckNvfp4(lw.mlp_up, DI, H) ||
                !CheckNvfp4(lw.mlp_down, H, DI)) {
                error = absl::StrCat("layer_", l, " dense MLP mismatch");
                pw.Free();
                return false;
            }
            accounted += lw.mlp_gate.packed_bytes + lw.mlp_gate.scale_bytes +
                         lw.mlp_up.packed_bytes + lw.mlp_up.scale_bytes +
                         lw.mlp_down.packed_bytes + lw.mlp_down.scale_bytes;
        } else {
            auto router = LoadTensorBatchViews(
                router_dir / absl::StrCat("layer_", l, ".arrow"));
            if (!router.ok()) {
                error = absl::StrCat("router/layer_", l, ".arrow read failed");
                pw.Free();
                return false;
            }
            auto *gate = Find(*router, "moe.gate.weight");
            auto *bias = Find(*router, "router_bias");
            if (!gate || !bias || gate->rank != 2) {
                error = absl::StrCat("layer_", l, " router tensors missing");
                pw.Free();
                return false;
            }
            if (!UploadBytes(gate->data, static_cast<size_t>(gate->bytes),
                             reinterpret_cast<void **>(&lw.d_router), stream) ||
                !UploadBytes(bias->data, static_cast<size_t>(bias->bytes),
                             reinterpret_cast<void **>(&lw.d_router_bias),
                             stream)) {
                error = absl::StrCat("layer_", l, " router upload failed");
                pw.Free();
                return false;
            }
            lw.router_bytes = static_cast<int64_t>(gate->bytes + bias->bytes);
            accounted += gate->bytes + bias->bytes;

            auto shared = LoadDenseNvfp4BatchViews(
                shared_dir / absl::StrCat("layer_", l, "_nvfp4.arrow"));
            if (!shared.ok()) {
                error = absl::StrCat("shared/layer_", l, " read failed");
                pw.Free();
                return false;
            }
            const int SI = top.shared_intermediate_size;
            if (!UploadNvfp4(*shared, "share_expert.gate_proj.weight",
                             lw.shared_gate, stream) ||
                !UploadNvfp4(*shared, "share_expert.up_proj.weight",
                             lw.shared_up, stream) ||
                !UploadNvfp4(*shared, "share_expert.down_proj.weight",
                             lw.shared_down, stream) ||
                !CheckNvfp4(lw.shared_gate, SI, H) ||
                !CheckNvfp4(lw.shared_up, SI, H) ||
                !CheckNvfp4(lw.shared_down, H, SI)) {
                error = absl::StrCat("layer_", l, " shared expert mismatch");
                pw.Free();
                return false;
            }
            accounted += lw.shared_gate.packed_bytes +
                         lw.shared_gate.scale_bytes + lw.shared_up.packed_bytes +
                         lw.shared_up.scale_bytes +
                         lw.shared_down.packed_bytes +
                         lw.shared_down.scale_bytes;
        }
        lw.loaded = true;
    }
    {
        cudaError_t ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            error = absl::StrCat("dense upload fence: ", cudaGetErrorString(ce));
            pw.Free();
            return false;
        }
    }

    // ── routed experts (streamed) ───────────────────────────────────────────
    if (!LoadExpertStore(model_path, pw, error)) {
        pw.Free();
        return false;
    }
    {
        size_t max_pack = 0, max_scale = 0;
        for (const auto &layer : pw.expert_store.records)
            for (const auto &r : layer) {
                max_pack = std::max<size_t>(
                    max_pack, static_cast<size_t>(r.w1_bytes + r.w2_bytes +
                                                  r.w3_bytes));
                max_scale = std::max<size_t>(
                    max_scale, static_cast<size_t>(r.s1_bytes + r.s2_bytes +
                                                   r.s3_bytes));
            }
        const int64_t slot_bytes = static_cast<int64_t>((max_pack + 255) & ~255ull) +
                                   static_cast<int64_t>((max_scale + 255) & ~255ull);
        const int slots = ResolveCacheSlots(slot_bytes);
        if (!moe::AllocateMoEExpertCache(pw.expert_store, slots, pw.expert_cache,
                                         error)) {
            pw.Free();
            return false;
        }
        pw.expert_cache_bytes = static_cast<int64_t>(pw.expert_cache.DeviceBytes());
        lsxcommon::log::info(absl::StrFormat(
            "[step-load] expert cache: %d slots x %.2f MiB = %lld MiB "
            "(LSX_STEP_EXPERT_CACHE_SLOTS overrides)",
            slots, static_cast<double>(slot_bytes) / (1024.0 * 1024.0),
            static_cast<long long>(pw.expert_cache_bytes >> 20)));
    }

    pw.resident_bytes = accounted;
    return true;
}

} // namespace lsxcommon::step
