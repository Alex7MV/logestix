#include "lsxcommon/inkling_moe_base_load.h"
#include "lsxcommon/index_layout.h"
#include "lsxcommon/config_json.h"

#include "inkling_moe_base_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string_view>

namespace lsxcommon::inkling::base {
namespace {
// Dedup Ф5: numeric config field reader is the shared lsxcommon::config body.
using lsxcommon::config::GetDouble;
inline bool get_d(simdjson::ondemand::object &o, const char *k, double &out)
{
    return GetDouble(o, k, out);
}

namespace fs = std::filesystem;
namespace frozen = lsxcommon::inkling::base::frozen;

bool get_i(simdjson::ondemand::object &o, const char *k, int &out)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return false;
    out = static_cast<int>(v);
    return true;
}

}  // namespace

arrow::Status LoadTopology(const std::string &model_path, Topology &out)
{
    out = Topology{};
    const fs::path idx = fs::path(model_path) / "model.index.json";
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

    int64_t layout_v = 0;
    if (root["layout"].get_int64().get(layout_v) == simdjson::SUCCESS)
        out.layout = layout_v;
    if (!lsxcommon::IndexLayoutSupported(out.layout)) {
        out.error_msg = lsxcommon::IndexLayoutError(out.layout);
        return arrow::Status::Invalid("inkling loader: expected format ",
                                      frozen::kFormatTag, ", got ", std::to_string(out.layout));
    }
    std::string_view s;
    if (root["source_model_type"].get_string().get(s) == simdjson::SUCCESS)
        out.source_model_type.assign(s.data(), s.size());

    get_i(root, "num_layers", out.num_layers);
    get_i(root, "hidden_size", out.hidden);
    get_i(root, "vocab_size", out.vocab);
    get_i(root, "unpadded_vocab_size", out.unpadded_vocab);
    get_i(root, "num_attention_heads", out.num_heads);
    get_i(root, "num_key_value_heads", out.num_kv_heads);
    get_i(root, "swa_num_key_value_heads", out.swa_kv_heads);
    get_i(root, "head_dim", out.head_dim);
    get_i(root, "d_rel", out.d_rel);
    get_i(root, "rel_extent", out.rel_extent);
    get_i(root, "sliding_window_size", out.sliding_window);
    get_i(root, "conv_kernel", out.conv_kernel);
    get_i(root, "dense_mlp_layers", out.dense_mlp_layers);
    get_i(root, "dense_intermediate_size", out.dense_intermediate);
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
    get_i(root, "skipped_audio_tensors", out.skipped_audio);
    get_i(root, "skipped_mtp_tensors", out.skipped_mtp);
    get_d(root, "attn_scale_denom", out.attn_scale_denom);
    get_d(root, "rms_norm_eps", out.rms_eps);
    get_d(root, "log_scaling_alpha", out.log_scaling_alpha);
    get_d(root, "logits_mup_div", out.logits_mup_div);
    get_d(root, "routed_scaling_factor", out.route_scale);

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

    if (out.num_layers <= 0 || out.hidden <= 0 || out.num_experts <= 0 ||
        out.swa_kv_heads <= 0 ||
        static_cast<int>(out.full_layers.size() + out.sliding_layers.size()) !=
            out.num_layers) {
        out.error_msg = "index is missing the layer partition";
        return arrow::Status::Invalid("inkling index: ", out.error_msg);
    }
    return arrow::Status::OK();
}

bool TopologyMatchesFrozen(const Topology &top)
{
    if (SyntheticGeometryAllowed()) return true;
    if (top.num_layers != frozen::kLayers || top.hidden != frozen::kHidden ||
        top.vocab != frozen::kVocab ||
        top.unpadded_vocab != frozen::kBaseVocab ||
        top.num_heads != frozen::kHeads ||
        top.num_kv_heads != frozen::kKvHeadsFull ||
        top.swa_kv_heads != frozen::kKvHeadsSliding ||
        top.head_dim != frozen::kHeadDim || top.d_rel != frozen::kDRel ||
        top.rel_extent != frozen::kRelExtent ||
        top.sliding_window != frozen::kSlidingWindow ||
        top.conv_kernel != frozen::kConvKernel ||
        top.dense_mlp_layers != frozen::kDenseMlpLayers ||
        top.dense_intermediate != frozen::kDenseIntermediate ||
        top.num_experts != frozen::kExperts || top.top_k != frozen::kTopK ||
        top.moe_intermediate != frozen::kMoeIntermediate ||
        top.shared_experts != frozen::kSharedExperts ||
        top.logits_mup_div != frozen::kLogitsMupDiv ||
        static_cast<int>(top.full_layers.size()) != frozen::kFullLayerCount)
        return false;
    for (size_t i = 0; i < top.full_layers.size(); ++i)
        if (top.full_layers[i] != frozen::kFullLayers[i]) return false;
    return true;
}

namespace {

// ── upload helpers ─────────────────────────────────────────────────────────
const TensorView *FindRaw(const TensorBatchViews &v, std::string_view n)
{
    for (const auto &t : v.tensors)
        if (t.name == n) return &t;
    return nullptr;
}

const DenseNvfp4View *FindNvfp4(const DenseNvfp4BatchViews &v, std::string_view n)
{
    for (const auto &t : v.records)
        if (t.name == n) return &t;
    return nullptr;
}

// BF16 -> F32. NOT `__ushort_as_half`: BF16 and FP16 share a width but not an
// exponent bias (127 vs 15), so reinterpreting the bits turns every weight into
// a value near 1-2. That single mistake corrupted every norm and conv tap in
// the sibling port while the synthetic fixtures stayed self-consistent.
inline float Bf16BitsToFloat(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// BF16 (or F32) host view -> FP16 device tensor.
__half *UploadHalf(const TensorView &t, cudaStream_t s, std::uint64_t &acc)
{
    const size_t n = t.bytes / (t.dtype == "F32" ? 4 : 2);
    std::vector<__half> h(n);
    if (t.dtype == "F32") {
        const float *f = reinterpret_cast<const float *>(t.data);
        for (size_t i = 0; i < n; ++i) h[i] = __float2half(f[i]);
    } else {
        for (size_t i = 0; i < n; ++i) {
            const uint16_t b =
                static_cast<uint16_t>(t.data[2 * i] | (t.data[2 * i + 1] << 8));
            h[i] = __float2half(Bf16BitsToFloat(b));
        }
    }
    void *d = nullptr;
    if (cudaMalloc(&d, n * sizeof(__half)) != cudaSuccess) return nullptr;
    cudaMemcpyAsync(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice, s);
    acc += n * sizeof(__half);
    return static_cast<__half *>(d);
}

// BF16/F32 host view -> FP32 device tensor (the reference keeps the rel bank,
// the conv taps and the router at full precision).
float *UploadF32(const TensorView &t, cudaStream_t s, std::uint64_t &acc)
{
    const size_t n = t.bytes / (t.dtype == "F32" ? 4 : 2);
    std::vector<float> h(n);
    if (t.dtype == "F32") {
        std::memcpy(h.data(), t.data, n * 4);
    } else {
        const uint16_t *b = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < n; ++i) h[i] = Bf16BitsToFloat(b[i]);
    }
    void *d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return nullptr;
    cudaMemcpyAsync(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice, s);
    acc += n * sizeof(float);
    return static_cast<float *>(d);
}

std::string LayerName(int layer, const char *rest)
{
    return absl::StrCat("model.llm.layers.", layer, ".", rest);
}

// One dense NVFP4 projection: packed and scales are SEPARATE Arrow columns,
// i.e. two unrelated host addresses — uploading them as one blob and pointing
// scales at packed + packed_bytes reads garbage (the xing loader's defect).
bool UploadNvfp4(const DenseNvfp4View &v, DenseNvfp4 &w, cudaStream_t s,
                 std::uint64_t &acc)
{
    if (v.packed_bytes <= 0 || v.scale_bytes <= 0 || v.rank != 2) return false;
    void *dp = nullptr;
    if (cudaMalloc(&dp, static_cast<size_t>(v.packed_bytes)) != cudaSuccess)
        return false;
    cudaMemcpyAsync(dp, v.packed, static_cast<size_t>(v.packed_bytes),
                    cudaMemcpyHostToDevice, s);
    void *ds = nullptr;
    if (cudaMalloc(&ds, static_cast<size_t>(v.scale_bytes)) != cudaSuccess) {
        cudaFree(dp);
        return false;
    }
    cudaMemcpyAsync(ds, v.scales, static_cast<size_t>(v.scale_bytes),
                    cudaMemcpyHostToDevice, s);
    w.d_packed = static_cast<const uint8_t *>(dp);
    w.d_scales = static_cast<const uint8_t *>(ds);
    w.global_scale = v.global_scale;
    w.rows = static_cast<int>(v.shape[0]);
    w.cols = static_cast<int>(v.shape[1]);
    w.packed_bytes = v.packed_bytes;
    w.scale_bytes = v.scale_bytes;
    acc += static_cast<std::uint64_t>(v.packed_bytes + v.scale_bytes);
    return true;
}

bool RequireNvfp4(const DenseNvfp4BatchViews &v, int layer, const char *rest,
                  DenseNvfp4 &out, cudaStream_t s, std::uint64_t &acc,
                  const char *tag)
{
    const std::string name = LayerName(layer, rest);
    const DenseNvfp4View *t = FindNvfp4(v, name);
    if (t == nullptr) {
        log::error(absl::StrFormat("[inkling_base_load] missing %s (%s)", name,
                                   tag));
        return false;
    }
    if (!UploadNvfp4(*t, out, s, acc)) {
        log::error(absl::StrFormat("[inkling_base_load] upload failed for %s",
                                   name));
        return false;
    }
    return true;
}

bool RequireRaw(const TensorBatchViews &v, int layer, const char *rest,
                const char *tag, std::function<void(const TensorView &)> take)
{
    const std::string name = LayerName(layer, rest);
    const TensorView *t = FindRaw(v, name);
    if (t == nullptr) {
        log::error(absl::StrFormat("[inkling_base_load] missing tensor %s (%s)",
                                   name, tag));
        return false;
    }
    take(*t);
    return true;
}

}  // namespace

size_t ExpertWindowFloor(const Topology &top)
{
    const int moe_layers = top.num_layers - top.dense_mlp_layers;
    return static_cast<size_t>(moe_layers) * static_cast<size_t>(top.top_k);
}

size_t ResolveExpertWindow(const Topology &top, long requested)
{
    const size_t floor = ExpertWindowFloor(top);
    if (requested <= 0) return floor;
    return std::max<size_t>(static_cast<size_t>(requested), floor);
}

ExpertDb LoadExpertIndex(const std::string &model_path, int num_layers,
                         int first_moe_layer, int experts_per_layer,
                         size_t max_open)
{
    ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.first_moe_layer = first_moe_layer;
    db.max_open = max_open;
    db.scale_block_size = 16;
    db.log_batch_fails = true;
    db.log_prefix = "[inkling_base]";
    db.mmap_fail_verb = "inkling base expert mmap fail ";
    // The real tree has 16384 expert files; warming them all at Load is a
    // pure startup tax (the LRU fills from real accesses within two tokens).
    db.warm_on_load = false;
    if (!db.Load(model_path, /*fallback=*/false))
        log::error("[inkling_base_load] routed-expert store failed to load");
    return db;
}

void FreeExpertIndex(ExpertDb &db) { db.Free(); }

void FreePinnedWeights(PinnedWeights &pw)
{
    auto release = [](auto *p) {
        if (p)
            cudaFree(const_cast<std::remove_const_t<
                     std::remove_pointer_t<decltype(p)>> *>(p));
    };
    release(pw.embed);
    release(pw.head);
    pw.embed = nullptr;
    pw.head = nullptr;
    pw.embed_rows = 0;
    pw.embed_cols = 0;
    release(pw.embed_norm);
    release(pw.final_norm);
    pw.embed_norm = nullptr;
    pw.final_norm = nullptr;
    for (auto &l : pw.layers) {
        for (auto *w : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wr, &l.dense_gate,
                        &l.dense_up, &l.dense_down})
            w->Free();
        for (auto &se : l.shared) {
            se.gate.Free();
            se.up.Free();
            se.down.Free();
        }
        l.shared.clear();
        release(l.attn_norm);
        release(l.mlp_norm);
        release(l.q_norm);
        release(l.k_norm);
        release(l.rel_proj);
        release(l.kconv_w);
        release(l.vconv_w);
        release(l.attn_conv_w);
        release(l.mlp_conv_w);
        release(l.router_w);
        release(l.router_bias);
        release(l.router_gs);
        release(l.mlp_gs);
        release(l.conv_state);
        l = LayerWeights{};
    }
    pw.layers.clear();
    pw.resident_bytes = 0;
}

bool LoadPinnedWeights(const std::string &root, PinnedWeights &pw,
                       cudaStream_t stream)
{
    const fs::path base = fs::path(root);
    const fs::path pinned = base / "pinned";
    const Topology &top = pw.top;
    const int H = top.hidden;
    const int k = top.conv_kernel;
    std::uint64_t resident = 0;

    auto g_raw = LoadTensorBatchViews(pinned / "global.arrow");
    if (!g_raw.ok()) {
        log::error(absl::StrFormat("[inkling_base_load] global: %s",
                                   g_raw.status().ToString()));
        return false;
    }

    // ── globals ─────────────────────────────────────────────────────────────
    const TensorView *emb = FindRaw(*g_raw, "model.llm.embed.weight");
    const TensorView *hd = FindRaw(*g_raw, "model.llm.unembed.weight");
    if (emb == nullptr || hd == nullptr || emb->rank != 2 || hd->rank != 2) {
        log::error("[inkling_base_load] embed/unembed absent from the raw plane");
        return false;
    }
    pw.embed_rows = static_cast<int>(emb->shape[0]);
    pw.embed_cols = static_cast<int>(emb->shape[1]);
    pw.embed = UploadHalf(*emb, stream, resident);
    pw.head = UploadHalf(*hd, stream, resident);
    if (pw.embed == nullptr || pw.head == nullptr) {
        log::error("[inkling_base_load] embed/head upload failed");
        return false;
    }
    auto need_global_half = [&](const char *name, __half *&dst) {
        const TensorView *t = FindRaw(*g_raw, name);
        if (!t) {
            log::error(absl::StrFormat("[inkling_base_load] missing %s", name));
            return false;
        }
        dst = UploadHalf(*t, stream, resident);
        return dst != nullptr;
    };
    if (!need_global_half("model.llm.embed_norm.weight", pw.embed_norm) ||
        !need_global_half("model.llm.norm.weight", pw.final_norm))
        return false;

    // ── per layer ───────────────────────────────────────────────────────────
    pw.layers.resize(static_cast<size_t>(top.num_layers));
    for (int l = 0; l < top.num_layers; ++l) {
        LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        lw.full = top.IsFull(l);
        const int kv_dim = top.KvDimOf(l);

        const fs::path raw_path = absl::StrFormat("layer_%d.arrow", l);
        auto raw = LoadTensorBatchViews(pinned / raw_path);
        if (!raw.ok()) {
            log::error(absl::StrFormat("[inkling_base_load] %s: %s",
                                       raw_path.string(),
                                       raw.status().ToString()));
            return false;
        }
        const fs::path nv_path = absl::StrFormat("layer_%d_nvfp4.arrow", l);
        auto nv = LoadDenseNvfp4BatchViews(pinned / nv_path);
        if (!nv.ok()) {
            log::error(absl::StrFormat("[inkling_base_load] %s: %s",
                                       nv_path.string(),
                                       nv.status().ToString()));
            return false;
        }

        if (!RequireNvfp4(*nv, l, "attn.wq_du.weight", lw.wq, stream, resident,
                          "wq") ||
            !RequireNvfp4(*nv, l, "attn.wk_dv.weight", lw.wk, stream, resident,
                          "wk") ||
            !RequireNvfp4(*nv, l, "attn.wv_dv.weight", lw.wv, stream, resident,
                          "wv") ||
            !RequireNvfp4(*nv, l, "attn.wo_ud.weight", lw.wo, stream, resident,
                          "wo") ||
            !RequireNvfp4(*nv, l, "attn.wr_du.weight", lw.wr, stream, resident,
                          "wr"))
            return false;

        auto half_into = [&](const char *rest, __half *&dst, const char *tag) {
            return RequireRaw(*raw, l, rest, tag, [&](const TensorView &t) {
                dst = UploadHalf(t, stream, resident);
            }) && dst != nullptr;
        };
        auto f32_into = [&](const char *rest, float *&dst, const char *tag) {
            return RequireRaw(*raw, l, rest, tag, [&](const TensorView &t) {
                dst = UploadF32(t, stream, resident);
            }) && dst != nullptr;
        };

        if (!half_into("attn_norm.weight", lw.attn_norm, "attn norm") ||
            !half_into("mlp_norm.weight", lw.mlp_norm, "mlp norm") ||
            !half_into("attn.q_norm.weight", lw.q_norm, "q norm") ||
            !half_into("attn.k_norm.weight", lw.k_norm, "k norm"))
            return false;
        if (!f32_into("attn.rel_logits_proj.proj", lw.rel_proj, "rel bank") ||
            !f32_into("attn.k_sconv.weight", lw.kconv_w, "k conv") ||
            !f32_into("attn.v_sconv.weight", lw.vconv_w, "v conv") ||
            !f32_into("attn_sconv.weight", lw.attn_conv_w, "attn conv") ||
            !f32_into("mlp_sconv.weight", lw.mlp_conv_w, "mlp conv"))
            return false;

        if (top.IsDenseMlp(l)) {
            if (!RequireNvfp4(*nv, l, "mlp.gate_proj.weight", lw.dense_gate,
                              stream, resident, "dense gate") ||
                !RequireNvfp4(*nv, l, "mlp.up_proj.weight", lw.dense_up, stream,
                              resident, "dense up") ||
                !RequireNvfp4(*nv, l, "mlp.down_proj.weight", lw.dense_down,
                              stream, resident, "dense down"))
                return false;
            if (!f32_into("mlp.global_scale", lw.mlp_gs, "mlp global scale"))
                return false;
        } else {
            // Router: FP32 weight ([E+S, H]) and FP32 selection bias ([E]).
            // NVFP4 rounding on the argmax surface silently swaps the expert
            // set, so both stay high precision.
            if (!f32_into("mlp.gate.weight", lw.router_w, "router weight") ||
                !f32_into("mlp.gate.bias", lw.router_bias, "router bias") ||
                !f32_into("mlp.gate.global_scale", lw.router_gs, "router scale"))
                return false;
            lw.shared.resize(static_cast<size_t>(top.shared_experts));
            for (int s = 0; s < top.shared_experts; ++s) {
                auto &se = lw.shared[static_cast<size_t>(s)];
                const std::string gate_name =
                    absl::StrFormat("mlp.shared_experts.%d.gate_proj.weight", s);
                const std::string up_name =
                    absl::StrFormat("mlp.shared_experts.%d.up_proj.weight", s);
                const std::string down_name =
                    absl::StrFormat("mlp.shared_experts.%d.down_proj.weight", s);
                if (!RequireNvfp4(*nv, l, gate_name.c_str(), se.gate, stream,
                                  resident, "shared gate") ||
                    !RequireNvfp4(*nv, l, up_name.c_str(), se.up, stream,
                                  resident, "shared up") ||
                    !RequireNvfp4(*nv, l, down_name.c_str(), se.down, stream,
                                  resident, "shared down"))
                    return false;
            }
        }

        // Four recurrent convolutions: k and v (kv_dim channels each) then the
        // post-attention and post-MLP ones (hidden channels each).
        const size_t floats = 2 * static_cast<size_t>(k) * kv_dim +
                              2 * static_cast<size_t>(k) * H;
        float *state = nullptr;
        if (cudaMalloc(&state, floats * sizeof(float)) != cudaSuccess)
            return false;
        if (cudaMemsetAsync(state, 0, floats * sizeof(float), stream) !=
            cudaSuccess)
            return false;
        lw.conv_state = state;
        lw.conv_state_bytes = static_cast<int64_t>(floats * sizeof(float));
        resident += floats * sizeof(float);
    }

    pw.resident_bytes = resident;
    cudaStreamSynchronize(stream);
    return true;
}

}  // namespace lsxcommon::inkling::base