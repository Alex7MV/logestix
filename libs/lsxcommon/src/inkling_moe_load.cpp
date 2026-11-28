#include "lsxcommon/inkling_moe_load.h"
#include "lsxcommon/index_layout.h"
#include "lsxcommon/config_json.h"

#include "inkling_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include "lsxcommon/layer_codec_map.h"
#include "lsxstorage/artifact.h"

#include "lsxcommon/synthetic_geometry.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

namespace lsxcommon::inkling {
namespace {
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

}  // namespace

arrow::Status LoadInklingTopology(const std::string &model_path,
                                  InklingTopology &out)
{
    out = InklingTopology{};
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
        return arrow::Status::Invalid(out.error_msg);
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
    // NO per-layer codec keys are read here (design §2.4.3 rule 5). Which
    // expert layers are NVFP4 and which are MXFP8 comes from
    // artifact.fb::layer_codec, projected onto these flags by the model after
    // BeginLoad — one source of truth, and the index cannot contradict it.
    // A tree converted before the MXFP8 plane existed has every expert plane in
    // NVFP4 (i.e. layers 40-47 were requantized); treat the missing field as
    // all-false rather than failing, so old artifacts still load.
    // No per-layer evidence in the index (rule 5 removed those keys), so derive
    // it from the BYTES: the same classification artifact.fb records, pointed at
    // a tree that may not have an artifact yet (a fixture, or the moment between
    // conversion and emit). Doing it here keeps the topology self-sufficient for
    // every caller — including the tests that parse it directly, with no model
    // to project for them — and the model may still override it from
    // artifact.fb, which is the authority once one exists.
    if (out.expert_nvfp4.empty() || out.expert_fp8.empty()) {
        const auto scan = lsx::storage::ScanArtifactTree(model_path);
        if (!scan.layer_codec.empty())
            lsxcommon::ProjectLayerCodecs(scan.layer_codec, &out.expert_nvfp4,
                                          &out.expert_fp8);
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

namespace {

// ── upload helpers ─────────────────────────────────────────────────────────
const TensorView *FindRaw(const TensorBatchViews &v, std::string_view n)
{
    for (const auto &t : v.tensors)
        if (t.name == n) return &t;
    return nullptr;
}

// BF16 -> F32. NOT `__ushort_as_half`: BF16 and FP16 share a width but not an
// exponent bias (127 vs 15), so reinterpreting the bits turns every weight into
// a value near 1-2 (BF16 1.0 = 0x3F80 reads as FP16 1.875). That single mistake
// corrupted every norm, clip scale and conv tap and made the real-weight output
// garbage while the synthetic fixtures (whose BF16 payload the test also wrote
// through the same path) stayed self-consistent.
inline float Bf16BitsToFloat(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// BF16 (or F32) host view -> FP16 device tensor.
__half *UploadHalf(const TensorView &t, cudaStream_t s)
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
    return static_cast<__half *>(d);
}

// BF16/F32 host view -> FP32 device tensor (the reference keeps these at full
// precision: the L2-clip scales, the relative bank and the router).
float *UploadF32(const TensorView &t, cudaStream_t s)
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
    return static_cast<float *>(d);
}

std::string LayerName(int layer, const char *rest)
{
    return absl::StrCat("model.llm.layers.", layer, ".", rest);
}

// Uploads one dense projection as an FP16 table. `UploadHalf` converts BF16 ->
// FP16, which is lossless for these magnitudes (BF16 and FP16 both hold 8+
// mantissa bits and every weight here is in 1e-3..1e1).
bool RequireDenseF16(const TensorBatchViews &v, int layer, const char *rest,
                     InklingDenseF16 &out, cudaStream_t s, const char *tag)
{
    const std::string name = LayerName(layer, rest);
    const TensorView *t = FindRaw(v, name);
    if (t == nullptr) {
        log::error(absl::StrFormat("[inkling_load] missing tensor %s (%s)", name,
                                   tag));
        return false;
    }
    if (t->rank != 2) {
        log::error(absl::StrFormat("[inkling_load] %s is not 2-D", name));
        return false;
    }
    out.rows = static_cast<int>(t->shape[0]);
    out.cols = static_cast<int>(t->shape[1]);
    out.d = UploadHalf(*t, s);
    if (out.d == nullptr) {
        log::error(absl::StrFormat("[inkling_load] upload failed for %s", name));
        return false;
    }
    return true;
}

bool RequireRaw(const TensorBatchViews &v, int layer, const char *rest,
                std::string_view want_dtype, const char *tag,
                std::function<void(const TensorView &)> take)
{
    const std::string name = LayerName(layer, rest);
    const TensorView *t = FindRaw(v, name);
    if (t == nullptr) {
        log::error(absl::StrFormat("[inkling_load] missing tensor %s (%s)", name,
                                   tag));
        return false;
    }
    if (t->dtype != want_dtype) {
        log::error(absl::StrFormat("[inkling_load] %s has dtype %s, expected %s",
                                   name, std::string(t->dtype),
                                   std::string(want_dtype)));
        return false;
    }
    take(*t);
    return true;
}

}  // namespace

size_t InklingExpertWindowFloor(const InklingTopology &top)
{
    // One token touches top_k experts in every MoE layer; the layers are walked
    // in ascending order once per token, so a window below this makes every
    // touch a re-mmap.
    return static_cast<size_t>(top.num_layers) * static_cast<size_t>(top.top_k);
}

size_t InklingResolveExpertWindow(const InklingTopology &top, long requested)
{
    const size_t floor = InklingExpertWindowFloor(top);
    if (requested <= 0) return floor;
    return std::max<size_t>(static_cast<size_t>(requested), floor);
}

InklingExpertDb LoadInklingExpertIndex(const std::string &model_path,
                                       int num_layers, int experts_per_layer,
                                       size_t max_open, bool warm_on_load)
{
    InklingExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.first_moe_layer = 0;  // every layer of this model is sparse
    db.max_open = max_open;
    // The MXFP8 expert layers are 1 byte per element (E4M3), not two nibbles per
    // byte, so ExpertSchema's per-16 count check — which assumes packed nibbles —
    // rejects them by 2x. StageExperts and the dequant path both verify the exact
    // plane sizes against the topology instead, which is a strictly stronger
    // check, so the schema heuristic is disabled here (`0` = no count check; it
    // is the same switch the pre-repack source stores use).
    db.scale_block_size = 0;
    // A plane that fails to parse is silent at info level, which hid the first
    // MXFP8 store failure for a whole run; keep the reason.
    db.log_batch_fails = true;
    db.log_prefix = "[inkling]";
    db.mmap_fail_verb = "inkling expert mmap fail ";
    if (!db.Load(model_path, /*fallback=*/false))
        log::error("[inkling_load] routed-expert store failed to load");
    (void)warm_on_load;
    return db;
}

void EnsureInklingExpertResident(InklingExpertDb &db, int layer, int expert,
                                 cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void FreeInklingExpertIndex(InklingExpertDb &db) { db.Free(); }

void FreeInklingPinnedWeights(InklingPinnedWeights &pw)
{
    auto release = [](auto *p) {
        if (p) cudaFree(const_cast<std::remove_const_t<std::remove_pointer_t<decltype(p)>> *>(p));
    };
    release(pw.embed);
    release(pw.head);
    pw.embed_rows = 0;
    pw.embed_cols = 0;
    release(pw.embed_norm);
    release(pw.final_norm);
    release(pw.attn_queries);
    release(pw.mlp_queries);
    release(pw.out_query);
    delete[] pw.attn_query_offset;
    pw.attn_query_offset = nullptr;
    for (auto &l : pw.layers) {
        for (auto *w : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wr, &l.attn_gate,
                        &l.ksconv_proj, &l.vsconv_proj}) {
            if (w->d) cudaFree(w->d);
            *w = InklingDenseF16{};
        }
        release(l.attn_norm);
        release(l.q_in_norm);
        release(l.k_in_norm);
        release(l.v_in_norm);
        release(l.mlp_norm);
        release(l.sink_proj);
        release(l.q_clip);
        release(l.rel_clip);
        release(l.rel_proj);
        release(l.kconv_w);
        release(l.vconv_w);
        release(l.kconv_static);
        release(l.vconv_static);
        release(l.kconv_proj_w);
        release(l.vconv_proj_w);
        release(l.attn_conv_w);
        release(l.mlp_conv_w);
        release(l.router_w);
        release(l.router_bias);
        release(l.conv_state);
    }
    pw.layers.clear();
    pw.resident_bytes = 0;
}

bool LoadInklingPinnedWeights(const std::string &root, InklingPinnedWeights &pw,
                              cudaStream_t stream)
{
    namespace fs = std::filesystem;
    const fs::path base = fs::path(root);
    const fs::path pinned = base / "pinned";
    const InklingTopology &top = pw.top;
    const int H = top.hidden;
    const int kv_dim = top.num_kv_heads * top.head_dim;
    const int k = top.conv_kernel;

    auto g_raw = LoadTensorBatchViews(pinned / "global.arrow");
    if (!g_raw.ok()) {
        log::error(absl::StrFormat("[inkling_load] global: %s",
                                   g_raw.status().ToString()));
        return false;
    }

    // ── globals ─────────────────────────────────────────────────────────────
    // embed/unembed are BF16 in the artifact (the converter keeps them out of
    // the NVFP4 plane) and are uploaded as FP16 tables.
    const TensorView *emb = FindRaw(*g_raw, "model.llm.embed.weight");
    const TensorView *hd = FindRaw(*g_raw, "model.llm.unembed.weight");
    if (emb == nullptr || hd == nullptr || emb->rank != 2 || hd->rank != 2) {
        log::error("[inkling_load] embed/unembed absent from the BF16 plane");
        return false;
    }
    pw.embed_rows = static_cast<int>(emb->shape[0]);
    pw.embed_cols = static_cast<int>(emb->shape[1]);
    pw.embed = UploadHalf(*emb, stream);
    pw.head = UploadHalf(*hd, stream);
    if (pw.embed == nullptr || pw.head == nullptr) {
        log::error("[inkling_load] embed/head upload failed");
        return false;
    }
    auto need_global_half = [&](const char *name, __half *&dst) {
        const TensorView *t = FindRaw(*g_raw, name);
        if (!t) {
            log::error(absl::StrFormat("[inkling_load] missing %s", name));
            return false;
        }
        dst = UploadHalf(*t, stream);
        return dst != nullptr;
    };
    if (!need_global_half("model.llm.embed_norm.weight", pw.embed_norm) ||
        !need_global_half("model.llm.norm.weight", pw.final_norm))
        return false;
    {
        const TensorView *t = FindRaw(*g_raw, "model.llm.attn_res.output_query");
        if (!t) {
            log::error("[inkling_load] missing output_query");
            return false;
        }
        pw.out_query = UploadF32(*t, stream);
        if (!pw.out_query) return false;
    }

    // AttnRes query banks. The converter re-indexed the attention bank to the
    // owning layer, so both banks are named by layer; the attention bank has
    // 3 rows on the full layers and 1 elsewhere.
    pw.attn_query_offset = new int[static_cast<size_t>(top.num_layers)]();
    size_t total_q = 0;
    for (int l = 0; l < top.num_layers; ++l) {
        pw.attn_query_offset[l] = static_cast<int>(total_q);
        if (l > 0) total_q += static_cast<size_t>(top.IsFull(l) ? 3 : 1);
    }
    {
        std::vector<float> all_q;
        all_q.reserve(total_q * static_cast<size_t>(H));
        for (int l = 1; l < top.num_layers; ++l) {
            const std::string real = absl::StrFormat(
                "model.llm.attn_res.attn_queries.%d", l);
            const TensorView *t = FindRaw(*g_raw, real);
            if (!t) {
                log::error(absl::StrFormat("[inkling_load] missing %s", real));
                return false;
            }
            const size_t n = t->bytes / (t->dtype == "F32" ? 4 : 2);
            const size_t rows = n / static_cast<size_t>(H);
            const size_t want = top.IsFull(l) ? 3u : 1u;
            if (rows != want) {
                log::error(absl::StrFormat(
                    "[inkling_load] %s has %zu rows, expected %zu (full=%d)",
                    real, rows, want, top.IsFull(l) ? 1 : 0));
                return false;
            }
            std::vector<float> tmp(n);
            if (t->dtype == "F32") {
                std::memcpy(tmp.data(), t->data, n * 4);
            } else {
                const uint16_t *b = reinterpret_cast<const uint16_t *>(t->data);
                for (size_t i = 0; i < n; ++i)
                    tmp[i] = Bf16BitsToFloat(b[i]);
            }
            all_q.insert(all_q.end(), tmp.begin(), tmp.end());
        }
        if (all_q.size() != total_q * static_cast<size_t>(H)) {
            log::error("[inkling_load] attn query bank size mismatch");
            return false;
        }
        void *d = nullptr;
        if (cudaMalloc(&d, all_q.size() * sizeof(float)) != cudaSuccess) {
            log::error("[inkling_load] attn query bank allocation failed");
            return false;
        }
        cudaMemcpyAsync(d, all_q.data(), all_q.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        pw.attn_queries = static_cast<float *>(d);
    }
    {
        std::vector<float> all_m;
        all_m.reserve(static_cast<size_t>(top.num_layers) * H);
        for (int l = 0; l < top.num_layers; ++l) {
            const std::string real = absl::StrFormat(
                "model.llm.attn_res.mlp_queries.%d", l);
            const TensorView *t = FindRaw(*g_raw, real);
            if (!t) {
                log::error(absl::StrFormat("[inkling_load] missing %s", real));
                return false;
            }
            const size_t n = t->bytes / (t->dtype == "F32" ? 4 : 2);
            if (n != static_cast<size_t>(H)) {
                log::error(absl::StrFormat("[inkling_load] %s has %zu rows", real,
                                           n));
                return false;
            }
            std::vector<float> tmp(n);
            if (t->dtype == "F32") {
                std::memcpy(tmp.data(), t->data, n * 4);
            } else {
                const uint16_t *b = reinterpret_cast<const uint16_t *>(t->data);
                for (size_t i = 0; i < n; ++i)
                    tmp[i] = Bf16BitsToFloat(b[i]);
            }
            all_m.insert(all_m.end(), tmp.begin(), tmp.end());
        }
        void *d = nullptr;
        if (cudaMalloc(&d, all_m.size() * sizeof(float)) != cudaSuccess) return false;
        cudaMemcpyAsync(d, all_m.data(), all_m.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        pw.mlp_queries = static_cast<float *>(d);
    }

    // ── per layer ───────────────────────────────────────────────────────────
    pw.layers.resize(static_cast<size_t>(top.num_layers));
    std::uint64_t resident = 0;
    for (int l = 0; l < top.num_layers; ++l) {
        InklingLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        lw.full = top.IsFull(l);

        const fs::path raw_path = absl::StrFormat("layer_%d.arrow", l);
        auto raw = LoadTensorBatchViews(pinned / raw_path);
        if (!raw.ok()) {
            log::error(absl::StrFormat("[inkling_load] %s: %s", raw_path.string(),
                                       raw.status().ToString()));
            return false;
        }
        if (!RequireDenseF16(*raw, l, "attn.wq_du.weight", lw.wq, stream, "wq") ||
            !RequireDenseF16(*raw, l, "attn.wk_dv.weight", lw.wk, stream, "wk") ||
            !RequireDenseF16(*raw, l, "attn.wv_dv.weight", lw.wv, stream, "wv") ||
            !RequireDenseF16(*raw, l, "attn.wo_ud.weight", lw.wo, stream, "wo") ||
            !RequireDenseF16(*raw, l, "attn.wr_du.weight", lw.wr, stream, "wr") ||
            !RequireDenseF16(*raw, l, "attn.gate_attn.weight", lw.attn_gate,
                             stream, "attn gate"))
            return false;
        if (lw.full &&
            (!RequireDenseF16(*raw, l, "attn.k_sconv.weight_proj",
                              lw.ksconv_proj, stream, "k conv proj") ||
             !RequireDenseF16(*raw, l, "attn.v_sconv.weight_proj",
                              lw.vsconv_proj, stream, "v conv proj")))
            return false;
        for (const auto &w : {lw.wq, lw.wk, lw.wv, lw.wo, lw.wr, lw.attn_gate,
                              lw.ksconv_proj, lw.vsconv_proj})
            resident += static_cast<std::uint64_t>(w.rows) *
                        static_cast<std::uint64_t>(w.cols) * sizeof(__half);

        auto half_into = [&](const char *rest, __half *&dst, const char *tag) {
            return RequireRaw(*raw, l, rest, "BF16", tag, [&](const TensorView &t) {
                dst = UploadHalf(t, stream);
            }) && dst != nullptr;
        };
        auto f32_into = [&](const char *rest, float *&dst, const char *tag) {
            return RequireRaw(*raw, l, rest, "BF16", tag, [&](const TensorView &t) {
                dst = UploadF32(t, stream);
            }) && dst != nullptr;
        };

        if (lw.full) {
            if (!half_into("q_input_norm.weight", lw.q_in_norm, "q input norm") ||
                !half_into("k_input_norm.weight", lw.k_in_norm, "k input norm") ||
                !half_into("v_input_norm.weight", lw.v_in_norm, "v input norm"))
                return false;
        } else {
            if (!half_into("attn_norm.weight", lw.attn_norm, "attn norm"))
                return false;
        }
        if (!half_into("mlp_norm.weight", lw.mlp_norm, "mlp norm")) return false;
        if (!half_into("attn.attn_sink_proj.weight", lw.sink_proj, "sink proj"))
            return false;
        if (!f32_into("attn.q_l2_clip.scale", lw.q_clip, "q clip")) return false;
        if (!f32_into("attn.rel_l2_clip.scale", lw.rel_clip, "rel clip"))
            return false;
        if (!f32_into("attn.rel_logits_proj.proj", lw.rel_proj, "rel bank"))
            return false;
        if (!f32_into("attn_sconv.weight", lw.attn_conv_w, "attn conv")) return false;
        if (!f32_into("mlp_sconv.weight", lw.mlp_conv_w, "mlp conv")) return false;
        if (lw.full) {
            if (!f32_into("attn.k_sconv.static_w", lw.kconv_static, "k static"))
                return false;
            if (!f32_into("attn.v_sconv.static_w", lw.vconv_static, "v static"))
                return false;
        } else {
            if (!f32_into("attn.k_sconv.weight", lw.kconv_w, "k conv")) return false;
            if (!f32_into("attn.v_sconv.weight", lw.vconv_w, "v conv")) return false;
        }
        {
            // Router: FP32 weight and FP32 selection bias.
            const TensorView *rw = FindRaw(*raw, LayerName(l, "mlp.gate.weight"));
            const TensorView *rb = FindRaw(*raw, LayerName(l, "mlp.gate.bias"));
            if (!rw || !rb) {
                log::error(absl::StrFormat(
                    "[inkling_load] layer %d: router tensors missing", l));
                return false;
            }
            lw.router_w = UploadF32(*rw, stream);
            lw.router_bias = UploadF32(*rb, stream);
            if (!lw.router_w || !lw.router_bias) return false;
            resident += static_cast<std::uint64_t>(rw->bytes) +
                        static_cast<std::uint64_t>(rb->bytes);
        }

        // Four recurrent convolutions: k and v (kv_dim channels each) then the
        // post-attention and post-MLP ones (hidden channels each).
        const size_t floats = 2 * static_cast<size_t>(k) * kv_dim +
                              2 * static_cast<size_t>(k) * H;
        float *state = nullptr;
        if (cudaMalloc(&state, floats * sizeof(float)) != cudaSuccess) return false;
        if (cudaMemsetAsync(state, 0, floats * sizeof(float), stream) != cudaSuccess)
            return false;
        lw.conv_state = state;
        lw.conv_state_bytes = static_cast<int64_t>(floats * sizeof(float));
        resident += floats * sizeof(float);

    }

    // Small tensors: count them from the uploaded shapes.
    resident += 2ull * static_cast<std::uint64_t>(pw.embed_rows) *
                static_cast<std::uint64_t>(pw.embed_cols) * sizeof(__half);
    resident += static_cast<std::uint64_t>(
        (total_q + static_cast<size_t>(top.num_layers) + 1) *
            static_cast<size_t>(H) * sizeof(float));
    resident += static_cast<std::uint64_t>(2 * H * sizeof(__half));
    pw.resident_bytes = resident;
    cudaStreamSynchronize(stream);
    return true;
}

}  // namespace lsxcommon::inkling
