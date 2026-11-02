// Mistral-Large-4-1T-A52B loader (Model Ingestion Protocol stage 3). Uploads
// the dense graph once (NVFP4 projections + BF16 edges/norms/router/embeddings)
// and points the routed experts at the neutral mmap expert store, which streams
// them through a device slot LRU.

#include "lsxcommon/mistral_moe_large4_load.h"

#include "mistral_moe_large4_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>

#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace lsxcommon {
namespace mistral_large4 {
namespace {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::mistral_large4::frozen;

cudaError_t UploadBytes(const uint8_t *src, size_t n, const uint8_t **dst,
                        std::size_t &accounted)
{
    if (n == 0) return cudaSuccess;
    void *p = nullptr;
    const cudaError_t err = cudaMalloc(&p, n);
    if (err != cudaSuccess) return err;
    const cudaError_t cp = cudaMemcpy(p, src, n, cudaMemcpyHostToDevice);
    if (cp != cudaSuccess)
    {
        cudaFree(p);
        return cp;
    }
    *dst = static_cast<const uint8_t *>(p);
    accounted += n;
    return cudaSuccess;
}

const TensorView *Find(const TensorBatchViews &b, std::string_view name)
{
    for (const auto &v : b.tensors)
        if (v.name == name) return &v;
    return nullptr;
}

cudaError_t UploadBf16(const TensorView &v, Large4Bf16Weight &w,
                       std::size_t &accounted)
{
    const uint8_t *raw = nullptr;
    const cudaError_t e = UploadBytes(v.data, v.bytes, &raw, accounted);
    if (e != cudaSuccess) return e;
    w.data = reinterpret_cast<const __nv_bfloat16 *>(raw);
    w.rows = v.rank > 0 ? v.shape[0] : 0;
    w.cols = v.rank > 1 ? v.shape[1] : 1;
    w.bytes = static_cast<int64_t>(v.bytes);
    return cudaSuccess;
}

cudaError_t UploadBf16Raw(const TensorView &v, const __nv_bfloat16 **dst,
                          std::size_t &accounted)
{
    const uint8_t *raw = nullptr;
    const cudaError_t e = UploadBytes(v.data, v.bytes, &raw, accounted);
    if (e != cudaSuccess) return e;
    *dst = reinterpret_cast<const __nv_bfloat16 *>(raw);
    return cudaSuccess;
}

cudaError_t UploadNvfp4(const DenseNvfp4View &v, lsx_nvfp4::LsxNvfp4Weight &w,
                        std::size_t &accounted)
{
    const uint8_t *p = nullptr, *s = nullptr;
    cudaError_t e = UploadBytes(v.packed, static_cast<size_t>(v.packed_bytes),
                                &p, accounted);
    if (e != cudaSuccess) return e;
    e = UploadBytes(v.scales, static_cast<size_t>(v.scale_bytes), &s, accounted);
    if (e != cudaSuccess) return e;
    w.d_packed = p;
    w.packed_bytes = v.packed_bytes;
    w.d_scales = s;
    w.scale_bytes = v.scale_bytes;
    w.global_scale = v.global_scale;
    w.rows = v.rank > 0 ? static_cast<int>(v.shape[0]) : 0;
    w.cols = v.rank > 1 ? static_cast<int>(v.shape[1]) : 1;
    return cudaSuccess;
}

const DenseNvfp4View *FindDense(const DenseNvfp4BatchViews &b,
                                std::string_view name)
{
    for (const auto &v : b.records)
        if (v.name == name) return &v;
    return nullptr;
}

struct IndexInfo {
    Large4Dims dims;
    std::vector<int> rope_layers;
    std::vector<int> kv_source;
    std::vector<int> indexer_layers;
    std::vector<int> moe_layers;
    std::vector<int> dense_layers;
    std::string format;
    std::string error_msg;
};

bool ReadIntArray(simdjson::dom::element o, const char *key,
                  std::vector<int> &into)
{
    simdjson::dom::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return false;
    for (auto v : arr)
    {
        int64_t x = 0;
        if (v.get_int64().get(x) != simdjson::SUCCESS) return false;
        into.push_back(static_cast<int>(x));
    }
    return true;
}

IndexInfo ReadIndex(const fs::path &p)
{
    IndexInfo ix;
    std::ifstream f(p);
    if (!f)
    {
        ix.error_msg = "cannot open " + p.string();
        return ix;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(text).get(doc) != simdjson::SUCCESS)
    {
        ix.error_msg = "invalid JSON: " + p.string();
        return ix;
    }
    std::string_view fmt;
    if (doc["format"].get_string().get(fmt) != simdjson::SUCCESS)
    {
        ix.error_msg = "index has no format tag";
        return ix;
    }
    ix.format = std::string(fmt);
    if (ix.format != frozen::kFormat)
    {
        ix.error_msg = absl::StrCat("unexpected format tag '", ix.format,
                                    "' (expected ", frozen::kFormat, ")");
        return ix;
    }
    const auto geti = [&doc](const char *k, int dflt) {
        int64_t v = 0;
        return doc[k].get_int64().get(v) == simdjson::SUCCESS
                   ? static_cast<int>(v)
                   : dflt;
    };
    ix.dims.layers = geti("num_layers", 0);
    ix.dims.experts = geti("experts_per_layer", 0);
    ix.dims.hidden = geti("hidden_size", 0);
    ix.dims.heads = geti("num_attention_heads", 0);
    ix.dims.kv_lora = geti("kv_lora_rank", 0);
    ix.dims.qk_nope = geti("qk_nope_head_dim", 0);
    ix.dims.qk_rope = geti("qk_rope_head_dim", 0);
    ix.dims.v_head = geti("v_head_dim", 0);
    ix.dims.moe_inter = geti("moe_intermediate_size", 0);
    ix.dims.dense_inter = geti("dense_intermediate_size", 0);
    ix.dims.top_k = geti("num_experts_per_tok", 0);
    ix.dims.index_heads = geti("index_n_heads", 0);
    ix.dims.index_head_dim = geti("index_head_dim", 0);
    ix.dims.index_topk = geti("index_topk", 0);
    ix.dims.qk_head = ix.dims.qk_nope + ix.dims.qk_rope;
    ix.dims.vocab = geti("vocab_size", 0);

    if (ix.dims.layers <= 0 || ix.dims.experts <= 0 || ix.dims.hidden <= 0)
    {
        ix.error_msg = "index missing num_layers/experts_per_layer/hidden_size";
        return ix;
    }
    if (!ReadIntArray(doc, "rope_layers", ix.rope_layers) ||
        !ReadIntArray(doc, "kv_source_layers", ix.kv_source) ||
        !ReadIntArray(doc, "indexer_layers", ix.indexer_layers) ||
        !ReadIntArray(doc, "moe_layers", ix.moe_layers) ||
        !ReadIntArray(doc, "dense_layers", ix.dense_layers))
    {
        ix.error_msg = "index missing the per-layer topology arrays";
        return ix;
    }
    const auto n = static_cast<size_t>(ix.dims.layers);
    if (ix.rope_layers.size() != n || ix.kv_source.size() != n ||
        ix.indexer_layers.size() != n || ix.moe_layers.size() != n)
    {
        ix.error_msg = "index per-layer arrays do not match num_layers";
        return ix;
    }
    if (!SyntheticGeometryAllowed())
    {
        const bool ok =
            ix.dims.layers == frozen::kLayers &&
            ix.dims.experts == frozen::kNumRouted &&
            ix.dims.hidden == frozen::kHidden && ix.dims.heads == frozen::kHeads &&
            ix.dims.kv_lora == frozen::kKvLora &&
            ix.dims.qk_nope == frozen::kQkNope &&
            ix.dims.qk_rope == frozen::kQkRope &&
            ix.dims.v_head == frozen::kVHead &&
            ix.dims.moe_inter == frozen::kMoeIntermediate &&
            ix.dims.dense_inter == frozen::kDenseIntermediate &&
            ix.dims.top_k == frozen::kTopK &&
            ix.dims.index_heads == frozen::kIndexHeads &&
            ix.dims.index_head_dim == frozen::kIndexHeadDim &&
            ix.dims.index_topk == frozen::kIndexTopk &&
            ix.dims.vocab == frozen::kVocab;
        bool tables_ok = ok;
        for (int i = 0; tables_ok && i < ix.dims.layers; ++i)
            tables_ok =
                ix.rope_layers[static_cast<size_t>(i)] ==
                    frozen::kRopeLayer[static_cast<size_t>(i)] &&
                ix.kv_source[static_cast<size_t>(i)] ==
                    frozen::kKvSource[static_cast<size_t>(i)] &&
                ix.indexer_layers[static_cast<size_t>(i)] ==
                    frozen::kIndexerFull[static_cast<size_t>(i)] &&
                ix.moe_layers[static_cast<size_t>(i)] ==
                    frozen::kMoeLayer[static_cast<size_t>(i)];
        if (!tables_ok)
            ix.error_msg =
                "frozen geometry mismatch between the converted index and the "
                "Mistral-Large-4 passport";
    }
    return ix;
}

} // namespace

void Large4LayerWeights::Free()
{
    for (lsx_nvfp4::LsxNvfp4Weight *w :
         {&wq, &wo, &indexer_wq_b, &shared_gate, &shared_up, &shared_down,
          &dense_gate, &dense_up, &dense_down})
    {
        if (w->d_packed) cudaFree(const_cast<uint8_t *>(w->d_packed));
        if (w->d_scales) cudaFree(const_cast<uint8_t *>(w->d_scales));
        w->d_packed = nullptr;
        w->d_scales = nullptr;
    }
    for (Large4Bf16Weight *w : {&wkv_a, &wkv_b, &indexer_wk, &indexer_weights_proj})
    {
        if (w->data) cudaFree(const_cast<__nv_bfloat16 *>(w->data));
        *w = Large4Bf16Weight{};
    }
    for (const __nv_bfloat16 **p :
         {&input_ln, &post_ln, &kv_a_norm, &ssmax_s, &ssmax_b, &idx_q_norm,
          &idx_q_norm_bias, &idx_k_norm, &idx_k_norm_bias, &idx_ssmax_s,
          &idx_ssmax_b, &router})
        if (*p)
        {
            cudaFree(const_cast<__nv_bfloat16 *>(*p));
            *p = nullptr;
        }
    router_bytes = 0;
}

void Large4PinnedWeights::Free() { FreeMistralLarge4Weights(*this); }

void FreeMistralLarge4Weights(Large4PinnedWeights &pw)
{
    auto drop = [](const void *p) {
        if (p) cudaFree(const_cast<void *>(p));
    };
    drop(pw.embed.data);
    drop(pw.head.data);
    drop(pw.final_norm);
    drop(pw.lerope_alpha);
    for (auto &l : pw.layers) l.Free();
    pw.layers.clear();
    pw.expert_store.Free();
    pw.resident_bytes = 0;
}

cudaError_t LoadMistralLarge4Weights(const std::string &model_path,
                                     Large4PinnedWeights &pw,
                                     cudaStream_t stream, std::string &error)
{
    (void)stream;
    pw.model_path = model_path;
    const fs::path base = fs::path(model_path) / "logestix";

    const IndexInfo ix = ReadIndex(base / "model.index.json");
    if (!ix.error_msg.empty())
    {
        error = ix.error_msg;
        return cudaErrorInvalidValue;
    }
    pw.dims = ix.dims;
    pw.dims.rms_eps = frozen::kRmsEps;
    pw.dims.attn_scale = frozen::kAttnScale;
    pw.dims.logit_softcap = frozen::kLogitSoftcap;

    std::size_t accounted = 0;

    // ── globals ────────────────────────────────────────────────────────────
    auto global = LoadTensorBatchViews(base / "pinned" / "global.arrow");
    if (!global.ok())
    {
        error = "cannot read pinned/global.arrow: " + global.status().ToString();
        return cudaErrorInvalidValue;
    }
    const auto *embed = Find(*global, "tok_embeddings.weight");
    const auto *head = Find(*global, "output.weight");
    const auto *final_norm = Find(*global, "norm.weight");
    const auto *alpha = Find(*global, "lerope.alpha");
    if (!embed || !head || !final_norm || !alpha)
    {
        error = "pinned/global.arrow missing tok_embeddings/output/norm/lerope";
        return cudaErrorInvalidValue;
    }
    if (UploadBf16(*embed, pw.embed, accounted) != cudaSuccess ||
        UploadBf16(*head, pw.head, accounted) != cudaSuccess ||
        UploadBf16Raw(*final_norm, &pw.final_norm, accounted) != cudaSuccess ||
        UploadBf16Raw(*alpha, &pw.lerope_alpha, accounted) != cudaSuccess)
    {
        error = "global weight upload failed";
        return cudaErrorMemoryAllocation;
    }
    pw.lerope_alpha_host.resize(static_cast<size_t>(alpha->bytes / 2));
    {
        // Host mirror of the BF16 alpha tensor (rope tables are built on the
        // host at request time).
        std::vector<uint16_t> bits(pw.lerope_alpha_host.size());
        std::memcpy(bits.data(), alpha->data, alpha->bytes);
        for (size_t i = 0; i < bits.size(); ++i)
        {
            const uint32_t u = static_cast<uint32_t>(bits[i]) << 16;
            float v;
            std::memcpy(&v, &u, sizeof(v));
            pw.lerope_alpha_host[i] = v;
        }
    }
    if (pw.head.rows != pw.dims.vocab || pw.embed.cols != pw.dims.hidden ||
        pw.embed.rows != pw.dims.vocab)
    {
        error = "global weight shapes disagree with the converted index";
        return cudaErrorInvalidValue;
    }

    // ── per-layer weights ──────────────────────────────────────────────────
    pw.layers.resize(static_cast<size_t>(pw.dims.layers));
    for (int l = 0; l < pw.dims.layers; ++l)
    {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        lw.uses_rope = ix.rope_layers[static_cast<size_t>(l)] != 0;
        lw.kv_source = ix.kv_source[static_cast<size_t>(l)];
        lw.owns_kv = lw.kv_source == l;
        lw.full_indexer = ix.indexer_layers[static_cast<size_t>(l)] != 0;
        lw.is_moe = ix.moe_layers[static_cast<size_t>(l)] != 0;

        const std::string p = absl::StrCat("layers.", l, ".");
        auto raw = LoadTensorBatchViews(
            base / "pinned" / absl::StrCat("layer_", l, ".arrow"));
        if (!raw.ok())
        {
            error = absl::StrCat("cannot read pinned/layer_", l, ".arrow");
            return cudaErrorInvalidValue;
        }
        auto dense = LoadDenseNvfp4BatchViews(
            base / "pinned" / absl::StrCat("layer_", l, "_nvfp4.arrow"));
        if (!dense.ok())
        {
            error = absl::StrCat("cannot read pinned/layer_", l,
                                 "_nvfp4.arrow");
            return cudaErrorInvalidValue;
        }
        const auto need = [&](const char *suffix) -> const TensorView * {
            const auto *v = Find(*raw, absl::StrCat(p, suffix));
            if (!v) error = absl::StrCat("missing tensor ", p, suffix);
            return v;
        };
        const auto need_dense = [&](const char *suffix)
            -> const DenseNvfp4View * {
            const auto *v = FindDense(*dense, absl::StrCat(p, suffix));
            if (!v) error = absl::StrCat("missing NVFP4 tensor ", p, suffix);
            return v;
        };

        const auto *inl = need("attention_norm.weight");
        const auto *pol = need("ffn_norm.weight");
        const auto *ss_s = need("attention.ssmax_scaling.s");
        const auto *ss_b = need("attention.ssmax_scaling.b");
        const auto *wkvb = need("attention.wkv_b.weight");
        const auto *wq = need_dense("attention.wq.weight");
        const auto *wo = need_dense("attention.wo.weight");
        if (!inl || !pol || !ss_s || !ss_b || !wkvb || !wq || !wo)
            return cudaErrorInvalidValue;
        if (UploadBf16Raw(*inl, &lw.input_ln, accounted) != cudaSuccess ||
            UploadBf16Raw(*pol, &lw.post_ln, accounted) != cudaSuccess ||
            UploadBf16Raw(*ss_s, &lw.ssmax_s, accounted) != cudaSuccess ||
            UploadBf16Raw(*ss_b, &lw.ssmax_b, accounted) != cudaSuccess ||
            UploadBf16(*wkvb, lw.wkv_b, accounted) != cudaSuccess ||
            UploadNvfp4(*wq, lw.wq, accounted) != cudaSuccess ||
            UploadNvfp4(*wo, lw.wo, accounted) != cudaSuccess)
        {
            error = "layer attention upload failed";
            return cudaErrorMemoryAllocation;
        }

        if (lw.owns_kv)
        {
            const auto *wkva = need("attention.wkv_a_with_mqa.weight");
            const auto *kvn = need("attention.kv_a_norm.weight");
            if (!wkva || !kvn) return cudaErrorInvalidValue;
            if (UploadBf16(*wkva, lw.wkv_a, accounted) != cudaSuccess ||
                UploadBf16Raw(*kvn, &lw.kv_a_norm, accounted) != cudaSuccess)
            {
                error = "kv projection upload failed";
                return cudaErrorMemoryAllocation;
            }
        }

        if (lw.full_indexer)
        {
            const auto *wqb = need_dense("attention.indexer.wq_b.weight");
            const auto *wk = need("attention.indexer.wk.weight");
            const auto *wp = need("attention.indexer.weights_proj.weight");
            const auto *qn = need("attention.indexer.q_input_norm.weight");
            const auto *qnb = need("attention.indexer.q_input_norm.bias");
            const auto *kn = need("attention.indexer.k_norm.weight");
            const auto *knb = need("attention.indexer.k_norm.bias");
            const auto *iss = need("attention.indexer.ssmax_scaling.s");
            const auto *isb = need("attention.indexer.ssmax_scaling.b");
            if (!wqb || !wk || !wp || !qn || !qnb || !kn || !knb || !iss ||
                !isb)
                return cudaErrorInvalidValue;
            if (UploadNvfp4(*wqb, lw.indexer_wq_b, accounted) != cudaSuccess ||
                UploadBf16(*wk, lw.indexer_wk, accounted) != cudaSuccess ||
                UploadBf16(*wp, lw.indexer_weights_proj, accounted) !=
                    cudaSuccess ||
                UploadBf16Raw(*qn, &lw.idx_q_norm, accounted) != cudaSuccess ||
                UploadBf16Raw(*qnb, &lw.idx_q_norm_bias, accounted) !=
                    cudaSuccess ||
                UploadBf16Raw(*kn, &lw.idx_k_norm, accounted) != cudaSuccess ||
                UploadBf16Raw(*knb, &lw.idx_k_norm_bias, accounted) !=
                    cudaSuccess ||
                UploadBf16Raw(*iss, &lw.idx_ssmax_s, accounted) != cudaSuccess ||
                UploadBf16Raw(*isb, &lw.idx_ssmax_b, accounted) != cudaSuccess)
            {
                error = "indexer upload failed";
                return cudaErrorMemoryAllocation;
            }
        }

        if (lw.is_moe)
        {
            const auto *gate = need("gate.weight");
            const auto *sg = need_dense("shared_experts.w1.weight");
            const auto *sd = need_dense("shared_experts.w2.weight");
            const auto *su = need_dense("shared_experts.w3.weight");
            if (!gate || !sg || !sd || !su) return cudaErrorInvalidValue;
            if (UploadBf16Raw(*gate, &lw.router, accounted) != cudaSuccess ||
                UploadNvfp4(*sg, lw.shared_gate, accounted) != cudaSuccess ||
                UploadNvfp4(*sd, lw.shared_down, accounted) != cudaSuccess ||
                UploadNvfp4(*su, lw.shared_up, accounted) != cudaSuccess)
            {
                error = "router/shared-expert upload failed";
                return cudaErrorMemoryAllocation;
            }
            lw.router_bytes = gate->bytes;
        }
        else
        {
            const auto *dg = need_dense("feed_forward.w1.weight");
            const auto *dd = need_dense("feed_forward.w2.weight");
            const auto *du = need_dense("feed_forward.w3.weight");
            if (!dg || !dd || !du) return cudaErrorInvalidValue;
            if (UploadNvfp4(*dg, lw.dense_gate, accounted) != cudaSuccess ||
                UploadNvfp4(*dd, lw.dense_down, accounted) != cudaSuccess ||
                UploadNvfp4(*du, lw.dense_up, accounted) != cudaSuccess)
            {
                error = "dense MLP upload failed";
                return cudaErrorMemoryAllocation;
            }
        }
    }

    // ── routed experts (neutral mmap store) ────────────────────────────────
    {
        moe::MoEExpertRuntime &store = pw.expert_store;
        const int E = pw.dims.experts;
        store.num_layers = pw.dims.layers;
        store.num_experts = E;
        store.records.assign(static_cast<size_t>(pw.dims.layers),
                             std::vector<moe::MoEExpertRecord>(
                                 static_cast<size_t>(E)));
        moe::MoEExpertDb &db = store.experts;
        db.num_layers = pw.dims.layers;
        db.experts_per_layer = E;
        db.first_moe_layer = frozen::kMoeFirstLayer;
        db.zero_pad_expert_names = false;
        db.max_open = moe::kMoEExpertMaxOpen;
        if (const char *env = std::getenv("LSX_MISTRAL_LARGE4_MAX_OPEN"))
        {
            const long v = std::atol(env);
            if (v > 0) db.max_open = static_cast<size_t>(v);
        }
        db.log_prefix = "[mistral-large4-expert] ";
        if (!db.Load(model_path, /*fallback=*/false))
        {
            error = "expert store load failed: " + model_path;
            return cudaErrorInvalidValue;
        }
        for (int l = frozen::kMoeFirstLayer; l < pw.dims.layers; ++l)
        {
            auto &recs = store.records[static_cast<size_t>(l)];
            for (int e = 0; e < E; ++e)
            {
                moe::EnsureMoEExpertView(db, l, e, nullptr);
                const size_t idx = static_cast<size_t>(l) * E +
                                   static_cast<size_t>(e);
                if (idx >= db.entries.size() || !db.entries[idx].host_w1)
                {
                    error = absl::StrCat("expert read failed: layer_", l,
                                         "/exp_", e);
                    return cudaErrorInvalidValue;
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
    }

    pw.resident_bytes = accounted;
    log::info(absl::StrCat("[mistral-large4] resident weights ",
                           static_cast<long long>(accounted), " B (",
                           pw.dims.layers, " layers)"));
    return cudaSuccess;
}

} // namespace mistral_large4
} // namespace lsxcommon
