// Mistral-Small-4-119B loader (Model Ingestion Protocol step 3). Uploads the
// text decoder's dense weights once: MLA projections (BF16), the plain norms,
// the router (BF16) and the embeddings/head (BF16). The routed experts live in
// the neutral host store and stream through a device slot LRU.

#include "lsxcommon/mistral_moe_load.h"
#include "fast_load_internal.h"

#include "mistral_moe_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/strip.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace lsxcommon::mistral_moe {
namespace {
// Dedup Ф5: byte upload is the shared lsxcommon::fastload body.
using lsxcommon::fastload::UploadBytes;


namespace fs = std::filesystem;
namespace frozen = lsxcommon::mistral_moe::frozen;

const lsxcommon::TensorView *Find(const lsxcommon::TensorBatchViews &b,
                                  std::string_view name)
{
    for (const auto &v : b.tensors)
        if (v.name == name) return &v;
    return nullptr;
}

cudaError_t UploadBf16Weight(const lsxcommon::TensorView &v,
                             MistralBf16Weight &w, std::size_t &accounted)
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

cudaError_t UploadBf16Raw(const lsxcommon::TensorView &v,
                          const __nv_bfloat16 **dst, std::size_t &accounted)
{
    const uint8_t *raw = nullptr;
    const cudaError_t e = UploadBytes(v.data, v.bytes, &raw, accounted);
    if (e != cudaSuccess) return e;
    *dst = reinterpret_cast<const __nv_bfloat16 *>(raw);
    return cudaSuccess;
}

cudaError_t UploadNvfp4(const lsxcommon::PackedExpert &pe,
                        lsx_nvfp4::LsxNvfp4Weight &w1,
                        lsx_nvfp4::LsxNvfp4Weight &w2,
                        lsx_nvfp4::LsxNvfp4Weight &w3, std::size_t &accounted)
{
    auto one = [&](const std::vector<uint8_t> &packed,
                   const std::vector<uint8_t> &scales,
                   const std::vector<int64_t> &shape, float gs,
                   lsx_nvfp4::LsxNvfp4Weight &out) -> cudaError_t {
        const uint8_t *p = nullptr, *s = nullptr;
        cudaError_t e = UploadBytes(packed.data(), packed.size(), &p, accounted);
        if (e != cudaSuccess) return e;
        e = UploadBytes(scales.data(), scales.size(), &s, accounted);
        if (e != cudaSuccess) return e;
        out.d_packed = p;
        out.packed_bytes = static_cast<int64_t>(packed.size());
        out.d_scales = s;
        out.scale_bytes = static_cast<int64_t>(scales.size());
        out.global_scale = gs;
        out.rows = shape.size() > 0 ? static_cast<int>(shape[0]) : 0;
        out.cols = shape.size() > 1 ? static_cast<int>(shape[1]) : 1;
        return cudaSuccess;
    };
    cudaError_t e = one(pe.w1_packed, pe.w1_scales, pe.w1_shape,
                        pe.w1_global_scale, w1);
    if (e != cudaSuccess) return e;
    e = one(pe.w2_packed, pe.w2_scales, pe.w2_shape, pe.w2_global_scale, w2);
    if (e != cudaSuccess) return e;
    return one(pe.w3_packed, pe.w3_scales, pe.w3_shape, pe.w3_global_scale, w3);
}

long long JsonInt(const std::string &text, const std::string &key,
                  long long fallback)
{
    const size_t k = text.find(absl::StrCat("\"", key, "\""));
    if (k == std::string::npos) return fallback;
    const size_t colon = text.find(':', k);
    if (colon == std::string::npos) return fallback;
    return std::strtoll(
        std::string(absl::StripAsciiWhitespace(text.substr(colon + 1, 24)))
            .c_str(),
        nullptr, 10);
}

struct IndexInfo {
    long long num_layers = 0;
    long long experts_per_layer = 0;
    long long hidden = 0;
    long long heads = 0;
    long long q_lora = 0;
    long long kv_lora = 0;
    long long qk_nope = 0;
    long long qk_rope = 0;
    long long v_head = 0;
    long long moe_inter = 0;
    long long top_k = 0;
    std::string format;
    std::string error_msg;
};

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
    ix.format = mistral_load_topology(p.string()).format;
    ix.num_layers = JsonInt(text, "num_layers", 0);
    ix.experts_per_layer = JsonInt(text, "experts_per_layer", 0);
    ix.hidden = JsonInt(text, "hidden_size", 0);
    ix.heads = JsonInt(text, "num_attention_heads", 0);
    ix.q_lora = JsonInt(text, "q_lora_rank", 0);
    ix.kv_lora = JsonInt(text, "kv_lora_rank", 0);
    ix.qk_nope = JsonInt(text, "qk_nope_head_dim", 0);
    ix.qk_rope = JsonInt(text, "qk_rope_head_dim", 0);
    ix.v_head = JsonInt(text, "v_head_dim", 0);
    ix.moe_inter = JsonInt(text, "moe_intermediate_size", 0);
    ix.top_k = JsonInt(text, "num_experts_per_tok", 0);
    if (ix.format != "logestix-mistral-nvfp4")
    {
        ix.error_msg = absl::StrCat("unexpected format tag '", ix.format,
                                    "' (expected logestix-mistral-nvfp4)");
        return ix;
    }
    if (ix.num_layers <= 0 || ix.experts_per_layer <= 0)
    {
        ix.error_msg = "index missing num_layers/experts_per_layer";
        return ix;
    }
    if (!SyntheticGeometryAllowed() &&
        (ix.num_layers != frozen::kLayers ||
         ix.experts_per_layer != frozen::kNumRouted))
    {
        ix.error_msg = absl::StrCat(
            "frozen geometry mismatch: layers=", ix.num_layers, " experts=",
            ix.experts_per_layer);
    }
    return ix;
}

} // namespace

void MistralPinnedWeights::Free() { FreeMistralWeights(*this); }

void FreeMistralWeights(MistralPinnedWeights &pw)
{
    auto drop = [](const void *p) {
        if (p) cudaFree(const_cast<void *>(p));
    };
    drop(pw.embed.data);
    drop(pw.head.data);
    drop(pw.final_norm);
    for (auto &l : pw.layers)
    {
        for (MistralBf16Weight *w :
             {&l.wq_a, &l.wq_b, &l.wkv_a, &l.wkv_b, &l.wo})
            drop(w->data);
        drop(l.q_a_norm);
        drop(l.kv_a_norm);
        drop(l.input_ln);
        drop(l.post_ln);
        drop(l.router);
        drop(l.shared_gate.d_packed);
        drop(l.shared_gate.d_scales);
        drop(l.shared_up.d_packed);
        drop(l.shared_up.d_scales);
        drop(l.shared_down.d_packed);
        drop(l.shared_down.d_scales);
        l = MistralLayerWeights{};
    }
    pw.layers.clear();
    pw.expert_store.Free();
    pw.resident_bytes = 0;
}

cudaError_t LoadMistralWeights(const std::string &model_path,
                               MistralPinnedWeights &pw, cudaStream_t stream,
                               std::string &error)
{
    (void)stream;
    pw.model_path = model_path;
    const fs::path base = fs::path(model_path);

    const IndexInfo ix = ReadIndex(base / "model.index.json");
    if (!ix.error_msg.empty())
    {
        error = ix.error_msg;
        return cudaErrorInvalidValue;
    }
    pw.num_layers = static_cast<int>(ix.num_layers);
    pw.num_experts = static_cast<int>(ix.experts_per_layer);
    pw.hidden = static_cast<int>(ix.hidden);
    pw.heads = static_cast<int>(ix.heads);
    pw.q_lora = static_cast<int>(ix.q_lora);
    pw.kv_lora = static_cast<int>(ix.kv_lora);
    pw.qk_nope = static_cast<int>(ix.qk_nope);
    pw.qk_rope = static_cast<int>(ix.qk_rope);
    pw.v_head = static_cast<int>(ix.v_head);
    pw.moe_inter = static_cast<int>(ix.moe_inter);
    pw.top_k = static_cast<int>(ix.top_k);

    std::size_t accounted = 0;

    // ── globals (embed / head / final norm, all BF16) ──────────────────────
    auto global = lsxcommon::LoadTensorBatchViews(base / "pinned" / "global.arrow");
    if (!global.ok())
    {
        error = "cannot read pinned/global.arrow: " + global.status().ToString();
        return cudaErrorInvalidValue;
    }
    const auto *embed = Find(*global, "tok_embeddings.weight");
    const auto *head = Find(*global, "output.weight");
    const auto *final_norm = Find(*global, "norm.weight");
    if (!embed || !head || !final_norm)
    {
        error = "pinned/global.arrow missing tok_embeddings/output/norm";
        return cudaErrorInvalidValue;
    }
    if (UploadBf16Weight(*embed, pw.embed, accounted) != cudaSuccess ||
        UploadBf16Weight(*head, pw.head, accounted) != cudaSuccess ||
        UploadBf16Raw(*final_norm, &pw.final_norm, accounted) != cudaSuccess)
    {
        error = "global weight upload failed";
        return cudaErrorMemoryAllocation;
    }

    // ── shared experts (NVFP4, one record per layer) ───────────────────────
    auto shared_batch =
        lsxcommon::ReadRecordBatchFromFile(base / "pinned" / "shared_experts.arrow");
    if (!shared_batch.ok())
    {
        error = "cannot read pinned/shared_experts.arrow: " +
                shared_batch.status().ToString();
        return cudaErrorInvalidValue;
    }
    auto shared = lsxcommon::SharedExpertSchema::UnpackBatch(*shared_batch);
    if (!shared.ok())
    {
        error = "cannot unpack pinned/shared_experts.arrow";
        return cudaErrorInvalidValue;
    }

    // ── per-layer dense weights ────────────────────────────────────────────
    pw.layers.resize(static_cast<size_t>(pw.num_layers));
    for (int l = 0; l < pw.num_layers; ++l)
    {
        auto batch = lsxcommon::LoadTensorBatchViews(
            base / "pinned" / absl::StrCat("layer_", l, ".arrow"));
        if (!batch.ok())
        {
            error = absl::StrCat("cannot read pinned/layer_", l, ".arrow");
            return cudaErrorInvalidValue;
        }
        const std::string p = absl::StrCat("layers.", l, ".");
        auto &lw = pw.layers[static_cast<size_t>(l)];
        const auto need = [&](const char *suffix) -> const lsxcommon::TensorView * {
            const auto *v = Find(*batch, absl::StrCat(p, suffix));
            if (!v) error = absl::StrCat("missing tensor ", p, suffix);
            return v;
        };
        const auto *wq_a = need("attention.wq_a.weight");
        const auto *wq_b = need("attention.wq_b.weight");
        const auto *wkv_a = need("attention.wkv_a_with_mqa.weight");
        const auto *wkv_b = need("attention.wkv_b.weight");
        const auto *wo = need("attention.wo.weight");
        const auto *qan = need("attention.q_a_norm.weight");
        const auto *kvan = need("attention.kv_a_norm.weight");
        const auto *inl = need("attention_norm.weight");
        const auto *pol = need("ffn_norm.weight");
        const auto *router = need("gate.weight");
        if (!wq_a || !wq_b || !wkv_a || !wkv_b || !wo || !qan || !kvan ||
            !inl || !pol || !router)
            return cudaErrorInvalidValue;
        if (UploadBf16Weight(*wq_a, lw.wq_a, accounted) != cudaSuccess ||
            UploadBf16Weight(*wq_b, lw.wq_b, accounted) != cudaSuccess ||
            UploadBf16Weight(*wkv_a, lw.wkv_a, accounted) != cudaSuccess ||
            UploadBf16Weight(*wkv_b, lw.wkv_b, accounted) != cudaSuccess ||
            UploadBf16Weight(*wo, lw.wo, accounted) != cudaSuccess ||
            UploadBf16Raw(*qan, &lw.q_a_norm, accounted) != cudaSuccess ||
            UploadBf16Raw(*kvan, &lw.kv_a_norm, accounted) != cudaSuccess ||
            UploadBf16Raw(*inl, &lw.input_ln, accounted) != cudaSuccess ||
            UploadBf16Raw(*pol, &lw.post_ln, accounted) != cudaSuccess ||
            UploadBf16Raw(*router, &lw.router, accounted) != cudaSuccess)
        {
            error = "layer weight upload failed";
            return cudaErrorMemoryAllocation;
        }
        lw.router_bytes = router->bytes;
    }

    // ── shared expert NVFP4 rows ───────────────────────────────────────────
    if (shared->size() < static_cast<size_t>(pw.num_layers))
    {
        error = "shared_experts.arrow has fewer rows than layers";
        return cudaErrorInvalidValue;
    }
    for (int l = 0; l < pw.num_layers; ++l)
    {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        const cudaError_t e = UploadNvfp4(
            (*shared)[static_cast<size_t>(l)], lw.shared_gate,
            lw.shared_down, lw.shared_up, accounted);
        if (e != cudaSuccess)
        {
            error = "shared expert upload failed";
            return e;
        }
        // The Arrow schema carries no shape; the geometry comes from the index.
        // w1/w3: [I, H]; w2: [H, I].
        lw.shared_gate.rows = pw.moe_inter;
        lw.shared_gate.cols = pw.hidden;
        lw.shared_up.rows = pw.moe_inter;
        lw.shared_up.cols = pw.hidden;
        lw.shared_down.rows = pw.hidden;
        lw.shared_down.cols = pw.moe_inter;
    }

    // ── routed experts (host pinned store, streamed on demand) ────────────
    if (!lsxcommon::moe::LoadMoEExpertStore(model_path, pw.expert_store, error))
        return cudaErrorInvalidValue;

    pw.resident_bytes = accounted;
    return cudaSuccess;
}

} // namespace lsxcommon::mistral_moe
