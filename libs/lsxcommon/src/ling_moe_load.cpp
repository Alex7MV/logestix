// Ling-3.1-flash loader (Model Ingestion Protocol step 3). Dense side is
// uploaded once as NVFP4/FP16; the 34 816 routed experts stream from the
// neutral mmap ExpertStore through a device slot LRU.
#include "lsxcommon/ling_moe_load.h"

#include "ling_moe_constants.h"

#include "lsxcommon/fast_load.h"
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
#include <sstream>

namespace lsxcommon::ling {
namespace {

namespace fs = std::filesystem;

cudaError_t UploadRaw(const uint8_t *src, size_t bytes, const void **dst,
                      std::size_t &accounted)
{
    if (bytes == 0) return cudaSuccess;
    void *p = nullptr;
    const cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) return err;
    const cudaError_t cp = cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice);
    if (cp != cudaSuccess) {
        cudaFree(p);
        return cp;
    }
    *dst = p;
    accounted += bytes;
    return cudaSuccess;
}

cudaError_t UploadBf16AsHalf(const TensorView &v, __half **dst,
                             std::size_t &accounted, cudaStream_t stream)
{
    int64_t numel = 1;
    for (int64_t d = 0; d < v.rank; ++d) numel *= v.shape[d];
    void *p = nullptr;
    const size_t bytes = static_cast<size_t>(numel) * 2;
    const cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) return err;
    if (cudaMemcpy(p, v.data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p);
        return cudaGetLastError();
    }
    const cudaError_t conv = fastload::Bf16ToHalfInPlace(
        static_cast<uint16_t *>(p), numel, stream);
    if (conv != cudaSuccess) {
        cudaFree(p);
        return conv;
    }
    *dst = static_cast<__half *>(p);
    accounted += bytes;
    return cudaSuccess;
}

cudaError_t UploadNvfp4(const DenseNvfp4View &r, lsx_nvfp4::LsxNvfp4Weight &w,
                        std::size_t &accounted)
{
    if (r.packed_bytes == 0) return cudaSuccess;
    const void *p = nullptr;
    const void *s = nullptr;
    cudaError_t e = UploadRaw(r.packed, static_cast<size_t>(r.packed_bytes), &p,
                              accounted);
    if (e != cudaSuccess) return e;
    e = UploadRaw(r.scales, static_cast<size_t>(r.scale_bytes), &s, accounted);
    if (e != cudaSuccess) {
        cudaFree(const_cast<void *>(p));
        return e;
    }
    w.d_packed = static_cast<const uint8_t *>(p);
    w.d_scales = static_cast<const uint8_t *>(s);
    w.packed_bytes = r.packed_bytes;
    w.scale_bytes = r.scale_bytes;
    w.global_scale = r.global_scale;
    w.rows = r.rank >= 2 ? static_cast<int>(r.shape[0]) : 0;
    w.cols = r.rank >= 2 ? static_cast<int>(r.shape[1]) : 0;
    return cudaSuccess;
}

cudaError_t UploadNvfp4(const DenseNvfp4Record &r,
                        lsx_nvfp4::LsxNvfp4Weight &w,
                        std::size_t &accounted)
{
    const void *p = nullptr;
    const void *s = nullptr;
    cudaError_t e = UploadRaw(r.packed.data(), r.packed.size(), &p, accounted);
    if (e != cudaSuccess) return e;
    e = UploadRaw(r.scales.data(), r.scales.size(), &s, accounted);
    if (e != cudaSuccess) {
        cudaFree(const_cast<void *>(p));
        return e;
    }
    w.d_packed = static_cast<const uint8_t *>(p);
    w.d_scales = static_cast<const uint8_t *>(s);
    w.packed_bytes = static_cast<int64_t>(r.packed.size());
    w.scale_bytes = static_cast<int64_t>(r.scales.size());
    w.global_scale = r.global_scale;
    w.rows = r.shape.size() > 0 ? static_cast<int>(r.shape[0]) : 0;
    w.cols = r.shape.size() > 1 ? static_cast<int>(r.shape[1]) : 0;
    return cudaSuccess;
}

const TensorView *Find(const TensorBatchViews &b, std::string_view suffix)
{
    for (const auto &v : b.tensors) {
        if (v.name.size() >= suffix.size() &&
            v.name.compare(v.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &v;
    }
    return nullptr;
}

const DenseNvfp4View *Find(const DenseNvfp4BatchViews &b,
                           std::string_view suffix)
{
    for (const auto &r : b.records) {
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &r;
    }
    return nullptr;
}

const DenseNvfp4Record *Find(const std::vector<DenseNvfp4Record> &v,
                             std::string_view suffix)
{
    for (const auto &r : v) {
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &r;
    }
    return nullptr;
}

void ReadIntArray(simdjson::ondemand::object &root, const char *key,
                  std::vector<int> &out)
{
    if (auto arr = root[key].get_array(); arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            int64_t x = 0;
            if (v.get_int64().get(x) == simdjson::SUCCESS)
                out.push_back(static_cast<int>(x));
        }
    }
}

arrow::Status ReadWholeFile(const fs::path &path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return arrow::Status::IOError("cannot open ", path.string());
    out.assign(std::istreambuf_iterator<char>(f),
               std::istreambuf_iterator<char>());
    return arrow::Status::OK();
}

} // namespace

LingIndex LoadLingIndex(const std::string &model_path)
{
    LingIndex top;
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
    std::string_view format;
    if (root["format"].get_string().get(format) == simdjson::SUCCESS)
        top.format = std::string(format);
    auto gi = [&](const char *k, int dflt) {
        int64_t v;
        if (root[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto gd = [&](const char *k, double dflt) {
        simdjson::ondemand::value x;
        if (root[k].get(x) != simdjson::SUCCESS) return dflt;
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS)
            return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };
    top.num_layers = gi("num_layers", 0);
    top.hidden = gi("hidden_size", 0);
    top.vocab = gi("vocab_size", 0);
    top.heads = gi("num_attention_heads", 0);
    top.kv_heads = gi("num_key_value_heads", 0);
    top.head_dim = gi("head_dim", 0);
    top.qk_head_dim = gi("qk_head_dim", 0);
    top.qk_nope = gi("qk_nope_head_dim", 0);
    top.qk_rope = gi("qk_rope_head_dim", 0);
    top.v_head = gi("v_head_dim", 0);
    top.q_lora = gi("q_lora_rank", 0);
    top.kv_lora = gi("kv_lora_rank", 0);
    top.conv_kernel = gi("short_conv_kernel_size", 0);
    top.num_experts = gi("num_experts", 0);
    top.top_k = gi("num_experts_per_tok", 0);
    top.n_group = gi("n_group", 0);
    top.topk_group = gi("topk_group", 0);
    top.moe_intermediate = gi("moe_intermediate_size", 0);
    top.shared_intermediate = gi("shared_intermediate_size", 0);
    top.dense_intermediate = gi("dense_intermediate_size", 0);
    top.first_moe_layer = gi("first_moe_layer", 0);
    top.total_experts = gi("total_experts", 0);
    top.rms_eps = gd("rms_norm_eps", 0.0);
    top.rope_theta = gd("rope_theta", 0.0);
    top.routed_scale = gd("routed_scaling", 0.0);
    top.eos_token_id = gi("eos_token_id", -1);
    top.pad_token_id = gi("pad_token_id", -1);
    auto read_farray = [&](const char *key, std::vector<float> &out) {
        if (auto arr = root[key].get_array(); arr.error() == simdjson::SUCCESS) {
            for (auto v : arr.value()) {
                double d = 0.0;
                if (v.get_double().get(d) != simdjson::SUCCESS) {
                    int64_t i = 0;
                    if (v.get_int64().get(i) == simdjson::SUCCESS)
                        d = static_cast<double>(i);
                }
                out.push_back(static_cast<float>(d));
            }
        }
    };
    read_farray("expert_swiglu_limit_list", top.expert_swiglu_limit);
    read_farray("share_expert_swiglu_limit_list", top.shared_swiglu_limit);
    ReadIntArray(root, "mla_layers", top.mla_layers);
    ReadIntArray(root, "kda_layers", top.kda_layers);
    ReadIntArray(root, "moe_layers", top.moe_layers);
    ReadIntArray(root, "dense_layers", top.dense_layers);
    if (auto vr = root["vram"].get_object(); vr.error() == simdjson::SUCCESS) {
        auto v = vr.value();
        int64_t x = 0;
        if (v["min_bytes"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.min_bytes = x;
        if (v["optimal_bytes"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.optimal_bytes = x;
        if (v["max_bytes"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.max_bytes = x;
        if (v["context_tokens"].get_int64().get(x) == simdjson::SUCCESS)
            top.vram.context_tokens = x;
    }
    if (top.num_layers <= 0 || top.num_experts <= 0 || top.hidden <= 0 ||
        static_cast<int>(top.mla_layers.size()) +
                static_cast<int>(top.kda_layers.size()) !=
            top.num_layers) {
        top.error_msg = "index missing layers/experts/layer tables";
        return top;
    }
    return top;
}

void LingPinnedWeights::Free() { FreeLingPinnedWeights(*this); }

void FreeLingPinnedWeights(LingPinnedWeights &pw)
{
    auto drop = [](const void *p) {
        if (p) cudaFree(const_cast<void *>(p));
    };
    drop(pw.d_embed);
    drop(pw.d_head);
    drop(pw.d_final_norm);
    for (auto &l : pw.layers) {
        drop(l.d_input_norm);
        drop(l.d_post_norm);
        for (auto *w : {&l.kda_q, &l.kda_k, &l.kda_v, &l.kda_o, &l.kda_f,
                        &l.kda_g, &l.mla_q_a, &l.mla_q_b, &l.mla_kv_a,
                        &l.mla_kv_b, &l.mla_dense, &l.dense_gate, &l.dense_up,
                        &l.dense_down, &l.shared_gate, &l.shared_up,
                        &l.shared_down})
            w->Free();
        drop(l.d_conv_q);
        drop(l.d_conv_k);
        drop(l.d_conv_v);
        drop(l.d_a_log);
        drop(l.d_dt_bias);
        drop(l.d_kda_b);
        drop(l.d_kda_o_norm);
        drop(l.d_q_a_norm);
        drop(l.d_kv_a_norm);
        drop(l.d_mla_g);
        drop(l.d_router);
        drop(l.d_router_bias);
        l = LingLayerWeights{};
    }
    pw.layers.clear();
    pw.expert_store.Free();
    pw.resident_bytes = 0;
}

namespace {

bool LoadExpertStore(const std::string &model_path, LingPinnedWeights &pw,
                     std::string &error)
{
    moe::MoEExpertRuntime &store = pw.expert_store;
    const LingIndex &top = pw.top;
    store.num_layers = top.num_layers;
    store.num_experts = top.num_experts;
    store.records.assign(
        static_cast<size_t>(top.num_layers),
        std::vector<moe::MoEExpertRecord>(static_cast<size_t>(top.num_experts)));

    moe::MoEExpertDb &db = store.experts;
    db.num_layers = top.num_layers;
    db.experts_per_layer = top.num_experts;
    db.first_moe_layer = top.first_moe_layer;
    db.zero_pad_expert_names = false;
    db.max_open = moe::kMoEExpertMaxOpen;
    if (const char *env = std::getenv("LSX_LING_MAX_OPEN")) {
        const long v = std::atol(env);
        if (v > 0) db.max_open = static_cast<size_t>(v);
    }
    db.log_prefix = "[ling-expert] ";
    if (!db.Load(model_path, false)) {
        error = "expert store load failed: " + model_path;
        return false;
    }

    // Every expert carries its OWN three global scales (the per-tensor
    // quantization range differs), so the geometry pass must visit all of
    // them; the byte sizes are identical but the scales are not.
    for (int l : top.moe_layers) {
        auto &recs = store.records[static_cast<size_t>(l)];
        for (int e = 0; e < top.num_experts; ++e) {
            moe::EnsureMoEExpertView(db, l, e, nullptr);
            const size_t idx = static_cast<size_t>(l) *
                                   static_cast<size_t>(top.num_experts) +
                               static_cast<size_t>(e);
            if (idx >= db.entries.size() || !db.entries[idx].host_w1) {
                error = absl::StrCat("expert read failed: layer_", l, "/exp_",
                                     e);
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
    const int flats = top.num_layers * top.num_experts;
    std::vector<float> hw(static_cast<size_t>(flats) * 2, 1.0f);
    for (int l : top.moe_layers)
        for (int e = 0; e < top.num_experts; ++e) {
            const auto &r = store.records[static_cast<size_t>(l)]
                                        [static_cast<size_t>(e)];
            const size_t at =
                (static_cast<size_t>(l) * top.num_experts + e) * 2;
            hw[at] = r.gs1;
            hw[at + 1] = r.gs3;
        }
    if (cudaMalloc(&store.d_seg_wgs, hw.size() * sizeof(float)) !=
        cudaSuccess)
        return false;
    if (cudaMemcpy(store.d_seg_wgs, hw.data(), hw.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return true;
}

} // namespace

bool LoadLingDenseWeights(const std::string &model_path, LingPinnedWeights &pw,
                          cudaStream_t stream, std::string &error)
{
    pw.Free();
    error.clear();
    pw.top = LoadLingIndex(model_path);
    if (!pw.top.error_msg.empty()) {
        error = pw.top.error_msg;
        return false;
    }
    const LingIndex &top = pw.top;
    const fs::path base = fs::path(model_path) / "logestix";

    std::size_t accounted = 0;

    // ── globals ─────────────────────────────────────────────────────────────
    {
        auto raw = LoadTensorBatchViews(base / "pinned" / "global.arrow");
        if (!raw.ok()) {
            error = "global.arrow read failed: " + raw.status().ToString();
            return false;
        }
        const auto *emb = Find(*raw, "word_embeddings.weight");
        const auto *hd = Find(*raw, "lm_head.weight");
        const auto *fn = Find(*raw, "norm.weight");
        if (!emb || !hd || !fn) {
            error = "global.arrow missing embed/lm_head/final norm";
            return false;
        }
        pw.embed_elems = emb->shape[0] * emb->shape[1];
        pw.head_elems = hd->shape[0] * hd->shape[1];
        if (UploadBf16AsHalf(*emb, &pw.d_embed, accounted, stream) !=
                cudaSuccess ||
            UploadBf16AsHalf(*hd, &pw.d_head, accounted, stream) !=
                cudaSuccess ||
            UploadBf16AsHalf(*fn, &pw.d_final_norm, accounted, stream) !=
                cudaSuccess) {
            error = "global weight upload failed";
            return false;
        }
    }

    pw.layers.resize(static_cast<size_t>(top.num_layers));
    for (int l = 0; l < top.num_layers; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        lw.is_mla = std::find(top.mla_layers.begin(), top.mla_layers.end(),
                              l) != top.mla_layers.end();
        lw.is_moe = std::find(top.moe_layers.begin(), top.moe_layers.end(),
                              l) != top.moe_layers.end();
        if (l >= 0 && l < static_cast<int>(top.expert_swiglu_limit.size()))
            lw.expert_swiglu_limit = top.expert_swiglu_limit[static_cast<size_t>(l)];
        if (l >= 0 && l < static_cast<int>(top.shared_swiglu_limit.size()))
            lw.shared_swiglu_limit = top.shared_swiglu_limit[static_cast<size_t>(l)];
        const std::string lp = absl::StrCat("model.layers.", l, ".");

        auto raw = LoadTensorBatchViews(
            base / "pinned" / absl::StrCat("layer_", l, ".arrow"));
        if (!raw.ok()) {
            error = absl::StrCat("cannot read pinned/layer_", l, ".arrow");
            return false;
        }
        const auto *inorm = Find(*raw, "input_layernorm.weight");
        const auto *pnorm = Find(*raw, "post_attention_layernorm.weight");
        if (!inorm || !pnorm ||
            UploadBf16AsHalf(*inorm, &lw.d_input_norm, accounted, stream) !=
                cudaSuccess ||
            UploadBf16AsHalf(*pnorm, &lw.d_post_norm, accounted, stream) !=
                cudaSuccess) {
            error = absl::StrCat("layer norm missing/upload failed: layer ", l);
            return false;
        }

        auto nv = LoadDenseNvfp4BatchViews(
            base / "pinned" / absl::StrCat("layer_", l, "_nvfp4.arrow"));
        if (!nv.ok()) {
            error = absl::StrCat("cannot read layer_", l, "_nvfp4.arrow");
            return false;
        }
        auto need_nv = [&](const char *suffix, lsx_nvfp4::LsxNvfp4Weight &w)
            -> bool {
            const auto *r = Find(*nv, suffix);
            if (!r || UploadNvfp4(*r, w, accounted) != cudaSuccess) {
                error = absl::StrCat("missing/nvfp4 upload failed: ", lp,
                                     suffix);
                return false;
            }
            return true;
        };

        if (lw.is_mla) {
            if (!need_nv("attention.q_a_proj.weight", lw.mla_q_a) ||
                !need_nv("attention.q_b_proj.weight", lw.mla_q_b) ||
                !need_nv("attention.kv_a_proj_with_mqa.weight",
                         lw.mla_kv_a) ||
                !need_nv("attention.kv_b_proj.weight", lw.mla_kv_b) ||
                !need_nv("attention.dense.weight", lw.mla_dense))
                return false;
            const auto *qan = Find(*raw, "q_a_layernorm.weight");
            const auto *kvan = Find(*raw, "kv_a_layernorm.weight");
            const auto *g = Find(*raw, "attention.g_proj.weight");
            if (!qan || !kvan || !g ||
                UploadBf16AsHalf(*qan, &lw.d_q_a_norm, accounted, stream) !=
                    cudaSuccess ||
                UploadBf16AsHalf(*kvan, &lw.d_kv_a_norm, accounted, stream) !=
                    cudaSuccess ||
                UploadBf16AsHalf(*g, &lw.d_mla_g, accounted, stream) !=
                    cudaSuccess) {
                error = absl::StrCat("MLA norm/gate upload failed: layer ", l);
                return false;
            }
        } else {
            if (!need_nv("attention.q_proj.weight", lw.kda_q) ||
                !need_nv("attention.k_proj.weight", lw.kda_k) ||
                !need_nv("attention.v_proj.weight", lw.kda_v) ||
                !need_nv("attention.o_proj.weight", lw.kda_o) ||
                !need_nv("attention.f_proj.weight", lw.kda_f) ||
                !need_nv("attention.g_proj.weight", lw.kda_g))
                return false;
            const auto *cq = Find(*raw, "q_conv1d.weight");
            const auto *ck = Find(*raw, "k_conv1d.weight");
            const auto *cv = Find(*raw, "v_conv1d.weight");
            const auto *alog = Find(*raw, "A_log");
            const auto *dtb = Find(*raw, "dt_bias");
            const auto *bp = Find(*raw, "b_proj.weight");
            const auto *onorm = Find(*raw, "o_norm.weight");
            const void *alog_dev = nullptr;
            const void *dtb_dev = nullptr;
            if (!cq || !ck || !cv || !alog || !dtb || !bp || !onorm) {
                error = absl::StrCat("KDA tensor missing: layer ", l);
                return false;
            }
            if (UploadBf16AsHalf(*cq, &lw.d_conv_q, accounted, stream) !=
                    cudaSuccess ||
                UploadBf16AsHalf(*ck, &lw.d_conv_k, accounted, stream) !=
                    cudaSuccess ||
                UploadBf16AsHalf(*cv, &lw.d_conv_v, accounted, stream) !=
                    cudaSuccess ||
                UploadBf16AsHalf(*bp, &lw.d_kda_b, accounted, stream) !=
                    cudaSuccess ||
                UploadBf16AsHalf(*onorm, &lw.d_kda_o_norm, accounted,
                                 stream) != cudaSuccess ||
                UploadRaw(alog->data, alog->bytes, &alog_dev,
                          accounted) != cudaSuccess ||
                UploadRaw(dtb->data, dtb->bytes, &dtb_dev,
                          accounted) != cudaSuccess) {
                error = absl::StrCat("KDA upload failed: layer ", l);
                return false;
            }
            lw.d_a_log = static_cast<float *>(
                const_cast<void *>(alog_dev));
            lw.d_dt_bias = static_cast<float *>(
                const_cast<void *>(dtb_dev));
        }

        if (lw.is_moe) {
            auto router = LoadTensorBatchViews(
                base / "router" / absl::StrCat("layer_", l, ".arrow"));
            if (!router.ok()) {
                error = absl::StrCat("cannot read router/layer_", l, ".arrow");
                return false;
            }
            const auto *gate = Find(*router, "mlp.gate.weight");
            const auto *bias = Find(*router, "mlp.gate.expert_bias");
            const void *gate_dev = nullptr;
            const void *bias_dev = nullptr;
            if (!gate || !bias ||
                UploadRaw(gate->data, gate->bytes, &gate_dev, accounted) !=
                    cudaSuccess ||
                UploadRaw(bias->data, bias->bytes, &bias_dev, accounted) !=
                    cudaSuccess) {
                error = absl::StrCat("router upload failed: layer ", l);
                return false;
            }
            lw.d_router =
                static_cast<__half *>(const_cast<void *>(gate_dev));
            lw.d_router_bias =
                static_cast<float *>(const_cast<void *>(bias_dev));
            lw.router_bytes = static_cast<int64_t>(gate->bytes);

            auto shared_batch = ReadRecordBatchFromFile(
                base / "shared" / absl::StrCat("layer_", l, "_nvfp4.arrow"));
            if (!shared_batch.ok()) {
                error = absl::StrCat("cannot read shared/layer_", l,
                                     "_nvfp4.arrow");
                return false;
            }
            auto shared = DenseNvfp4Schema::UnpackBatch(*shared_batch);
            if (!shared.ok()) {
                error = absl::StrCat("cannot unpack shared/layer_", l);
                return false;
            }
            auto need_shared = [&](const char *suffix,
                                   lsx_nvfp4::LsxNvfp4Weight &w) -> bool {
                const auto *r = Find(*shared, suffix);
                if (!r || UploadNvfp4(*r, w, accounted) != cudaSuccess) {
                    error = absl::StrCat("shared upload failed: ", lp, suffix);
                    return false;
                }
                return true;
            };
            if (!need_shared("shared_experts.gate_proj.weight",
                             lw.shared_gate) ||
                !need_shared("shared_experts.up_proj.weight", lw.shared_up) ||
                !need_shared("shared_experts.down_proj.weight",
                             lw.shared_down))
                return false;
        } else {
            if (!need_nv("mlp.gate_proj.weight", lw.dense_gate) ||
                !need_nv("mlp.up_proj.weight", lw.dense_up) ||
                !need_nv("mlp.down_proj.weight", lw.dense_down))
                return false;
        }
        lw.loaded = true;
    }

    if (!LoadExpertStore(model_path, pw, error)) return false;
    pw.resident_bytes = accounted;
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        error = "weight upload stream sync failed";
        return false;
    }
    log::info(absl::StrFormat(
        "[ling-load] %d layers (%d MLA), %d experts/layer, %s pinned",
        top.num_layers, static_cast<int>(top.mla_layers.size()),
        top.num_experts,
        lsxcommon::FormatBytesHuman(static_cast<uint64_t>(accounted))));
    return true;
}

KvCacheGeometry LingKvGeometry(const LingIndex &top, int64_t capacity)
{
    return KvCacheGeometry::Make(
        KvCacheLayout::kPosMajor, capacity, top.heads,
        top.qk_nope + top.qk_rope + top.v_head,
        static_cast<int64_t>(top.mla_layers.size()));
}

} // namespace lsxcommon::ling
