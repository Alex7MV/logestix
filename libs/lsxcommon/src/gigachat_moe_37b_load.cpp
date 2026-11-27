// GigaChat4-37B-A4B loader-stage passport (index parse + frozen-geometry gate)
// and the resident device weight graph. Mirrors the GigaChat3.5 loader contour
// (pinned staging ring, NVFP4 batch views, mmap-only Arrow reads) and adds the
// DSA indexer weights and the non-zero-centred norm variant.
#include "lsxcommon/gigachat_moe_37b_load.h"
#include "lsxcommon/synthetic_geometry.h"

#include "fast_load_internal.h"
#include "gigachat_moe_37b_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

#include <simdjson.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

namespace lsxcommon::gigachat4 {

bool LoadGigaChat4Index(const std::string &model_path, Gc4Index &out,
                        std::string &error)
{
    return LoadGigaChat4IndexWithFormat(model_path, frozen::kFormat, out, error);
}

bool LoadGigaChat4IndexWithFormat(const std::string &model_path,
                                 const std::string &expected_format,
                                 Gc4Index &out, std::string &error)
{
    namespace fs = std::filesystem;
    const fs::path index_path =
        fs::path(model_path) / "model.index.json";
    std::ifstream f(index_path, std::ios::binary);
    if (!f)
    {
        error = "model.index.json not found in " + model_path;
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (json.empty())
    {
        error = "index file is empty";
        return false;
    }

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
    {
        error = "cannot parse model.index.json";
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
    {
        error = "index root is not an object";
        return false;
    }

    auto as_string = [](simdjson::ondemand::value v, std::string &s) -> bool {
        std::string_view sv;
        if (v.get_string().get(sv) != simdjson::SUCCESS) return false;
        s.assign(sv);
        return true;
    };
    auto as_num = [](simdjson::ondemand::value v, double &d) -> bool {
        if (v.get_double().get(d) == simdjson::SUCCESS) return true;
        int64_t i = 0;
        if (v.get_int64().get(i) == simdjson::SUCCESS)
        {
            d = static_cast<double>(i);
            return true;
        }
        return false;
    };

    bool have_format = false;
    for (auto field : root)
    {
        std::string_view key;
        if (field.unescaped_key().get(key) != simdjson::SUCCESS) continue;
        if (key == "format")
        {
            have_format = as_string(field.value(), out.format);
            continue;
        }
        if (key == "model")
        {
            as_string(field.value(), out.model);
            continue;
        }
        if (key == "full_attention_layers")
        {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) == simdjson::SUCCESS)
                for (auto e : arr)
                {
                    int64_t v = 0;
                    if (e.get_int64().get(v) == simdjson::SUCCESS)
                        out.full_attention_layers.push_back(static_cast<int>(v));
                }
            continue;
        }
        if (key == "layer_types")
        {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) == simdjson::SUCCESS)
                for (auto e : arr)
                {
                    std::string_view sv;
                    if (e.get_string().get(sv) == simdjson::SUCCESS)
                        out.layer_types.emplace_back(sv);
                }
            continue;
        }
        if (key == "vram" || key == "deferred")
        {
            // Rendered as objects; vram is re-read from the raw text below
            // (simdjson ondemand object re-reads are order-sensitive).
            continue;
        }
        double d = 0.0;
        if (!as_num(field.value(), d)) continue;
        const int di = static_cast<int>(d);
        if (key == "num_layers") out.num_layers = di;
        else if (key == "hidden_size") out.hidden_size = di;
        else if (key == "vocab_size") out.vocab_size = di;
        else if (key == "num_attention_heads") out.num_attention_heads = di;
        else if (key == "num_key_value_heads") out.num_key_value_heads = di;
        else if (key == "num_experts") out.num_experts = di;
        else if (key == "num_experts_per_tok") out.num_experts_per_tok = di;
        else if (key == "moe_intermediate_size") out.moe_intermediate_size = di;
        else if (key == "dense_intermediate_size") out.dense_intermediate_size = di;
        else if (key == "first_k_dense_replace") out.first_k_dense_replace = di;
        else if (key == "q_lora_rank") out.q_lora_rank = di;
        else if (key == "kv_lora_rank") out.kv_lora_rank = di;
        else if (key == "qk_nope_head_dim") out.qk_nope_head_dim = di;
        else if (key == "qk_rope_head_dim") out.qk_rope_head_dim = di;
        else if (key == "v_head_dim") out.v_head_dim = di;
        else if (key == "rms_norm_eps") out.rms_norm_eps = d;
        else if (key == "router_scaling_factor") out.router_scaling_factor = d;
        else if (key == "swiglu_limit") out.swiglu_limit = d;
        else if (key == "logit_softcap") out.logit_softcap = d;
        else if (key == "attn_scale") out.attn_scale = d;
        else if (key == "index_heads") out.index_heads = di;
        else if (key == "index_head_dim") out.index_head_dim = di;
        else if (key == "index_topk") out.index_topk = di;
        else if (key == "index_rope") out.index_rope = di;
        else if (key == "bos_token_id") out.bos_token_id = di;
        else if (key == "eos_token_id") out.eos_token_id = di;
        else if (key == "pad_token_id") out.pad_token_id = di;
        else if (key == "total_pinned_tensors") out.total_pinned_tensors = di;
    }

    if (!have_format)
    {
        error = "index missing 'format' tag";
        return false;
    }
    if (out.format != expected_format)
    {
        error = "unknown format tag: " + out.format;
        return false;
    }
    ParseVramBlock(json, out.vram, error);
    if (!error.empty())
    {
        // A missing vram block is tolerated by the loader (older converters).
        error.clear();
    }
    if (out.num_layers <= 0 || out.hidden_size <= 0)
    {
        error = "index missing required geometry";
        return false;
    }
    if (out.index_topk <= 0 || out.index_heads <= 0)
    {
        error = "index missing the DSA indexer geometry";
        return false;
    }
    return true;
}

Gc4FrozenExpect Gc4DefaultExpect()
{
    Gc4FrozenExpect e;
    e.format = frozen::kFormat;
    const Gc4IngestSpec s = Gc4DefaultIngestSpec();
    e.geo = s.frozen;
    e.full_attention_layers = s.full_attention_layers;
    e.gate_rank = frozen::kGateRank;
    return e;
}

bool ValidateGigaChat4Geometry(const Gc4Index &idx, std::string &error)
{
    return ValidateGigaChat4GeometryAgainst(idx, Gc4DefaultExpect(), error);
}

bool ValidateGigaChat4GeometryAgainst(const Gc4Index &idx,
                                     const Gc4FrozenExpect &expect,
                                     std::string &error)
{
    if (SyntheticGeometryAllowed()) return true;
    const Gc4FrozenGeometry &g = expect.geo;
    auto fail = [&](const char *field, long long expected, long long actual) {
        error = absl::StrFormat("frozen geometry mismatch: %s config=%lld tensor=%lld",
                                field, expected, actual);
        return false;
    };
    if (idx.num_layers != g.layers) return fail("num_layers", g.layers, idx.num_layers);
    if (idx.hidden_size != g.hidden) return fail("hidden_size", g.hidden, idx.hidden_size);
    if (idx.vocab_size != g.vocab) return fail("vocab_size", g.vocab, idx.vocab_size);
    if (idx.num_attention_heads != g.heads)
        return fail("num_attention_heads", g.heads, idx.num_attention_heads);
    if (idx.num_key_value_heads != g.kv_heads)
        return fail("num_key_value_heads", g.kv_heads, idx.num_key_value_heads);
    if (idx.num_experts != g.num_experts)
        return fail("num_experts", g.num_experts, idx.num_experts);
    if (idx.num_experts_per_tok != g.top_k)
        return fail("num_experts_per_tok", g.top_k, idx.num_experts_per_tok);
    if (idx.moe_intermediate_size != g.moe_intermediate)
        return fail("moe_intermediate_size", g.moe_intermediate,
                    idx.moe_intermediate_size);
    if (idx.first_k_dense_replace != g.first_k_dense)
        return fail("first_k_dense_replace", g.first_k_dense,
                    idx.first_k_dense_replace);
    if (idx.index_heads != g.index_heads)
        return fail("index_heads", g.index_heads, idx.index_heads);
    if (idx.index_head_dim != g.index_head_dim)
        return fail("index_head_dim", g.index_head_dim, idx.index_head_dim);
    if (idx.index_topk != g.index_topk)
        return fail("index_topk", g.index_topk, idx.index_topk);
    if (!expect.full_attention_layers.empty())
    {
        const int n = static_cast<int>(expect.full_attention_layers.size());
        if (static_cast<int>(idx.full_attention_layers.size()) != n)
            return fail("full_attention_layers", n,
                        static_cast<long long>(idx.full_attention_layers.size()));
        for (int i = 0; i < n; ++i)
            if (idx.full_attention_layers[static_cast<size_t>(i)] !=
                expect.full_attention_layers[static_cast<size_t>(i)])
                return fail("full_attention_layers[i]",
                            expect.full_attention_layers[static_cast<size_t>(i)],
                            idx.full_attention_layers[static_cast<size_t>(i)]);
    }
    error.clear();
    return true;
}

// ── Device weight graph ─────────────────────────────────────────────────────

namespace {

using fastload::CvKind;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::PinnedStagingRing;
using fastload::UploadJob;

bool UploadRaw(const TensorView *t, CvKind kind, const uint8_t **dst,
               const std::string &what, cudaStream_t stream,
               PinnedStagingRing &ring, std::uint64_t &acc, std::string &error)
{
    if (!t)
    {
        error = "missing tensor: " + what;
        return false;
    }
    void *out = nullptr;
    std::vector<UploadJob> jobs = {{t, &out, kind, nullptr, nullptr}};
    void *arena = nullptr;
    size_t arena_bytes = 0;
    std::string e2;
    const cudaError_t e = fastload::RunUploadJobs(
        std::move(jobs), &arena, &arena_bytes, stream, ring, acc, &e2);
    if (e != cudaSuccess)
    {
        error = "upload failed: " + what + ": " + e2;
        return false;
    }
    *dst = static_cast<const uint8_t *>(out);
    return true;
}

bool UploadNvfp4(const DenseNvfp4BatchViews &nv, std::string_view suffix,
                 lsx_nvfp4::LsxNvfp4Weight &w, const std::string &what,
                 cudaStream_t stream, PinnedStagingRing &ring,
                 std::uint64_t &acc, std::string &error)
{
    const DenseNvfp4View *r = FindNvfp4Suffix(nv, suffix);
    if (!r)
    {
        error = "missing nvfp4 tensor: " + what;
        return false;
    }
    std::string e2;
    const cudaError_t e =
        fastload::UploadNvfp4Weight(r, w, stream, ring, acc, &e2);
    if (e != cudaSuccess)
    {
        error = "nvfp4 upload failed: " + what + ": " + e2;
        return false;
    }
    return true;
}

// Gated-norm triple (weight + rank-16 gate up/down). Shape contract shared with
// the 3.5 loader: weight [dim]; gate_up [rank, dim]; gate_down [dim, rank].
bool UploadNormTriple(const TensorBatchViews &raw, const std::string &prefix,
                      gigachat::GcGatedNorm &n, int expect_rank,
                      cudaStream_t stream, PinnedStagingRing &ring,
                      std::uint64_t &acc, std::string &error)
{
    const TensorView *w = FindTensorSuffix(raw, prefix + ".weight");
    const TensorView *gu =
        FindTensorSuffix(raw, prefix + ".gate_up_projection.weight");
    const TensorView *gd =
        FindTensorSuffix(raw, prefix + ".gate_down_projection.weight");
    if (!w || !gu || !gd)
    {
        error = "missing gated norm: " + prefix;
        return false;
    }
    if (w->rank != 1 || gu->rank != 2 || gd->rank != 2)
    {
        error = "gated norm tensor rank mismatch: " + prefix;
        return false;
    }
    const int64_t dim = w->shape[0];
    const int64_t rank = gu->shape[0];
    if (gu->shape[1] != dim || gd->shape[0] != dim || gd->shape[1] != rank)
    {
        error = absl::StrFormat(
            "gated norm shape mismatch for %s: weight[%lld] "
            "gate_up[%lld,%lld] gate_down[%lld,%lld]",
            prefix, dim, gu->shape[0], gu->shape[1], gd->shape[0],
            gd->shape[1]);
        return false;
    }
    if (!SyntheticGeometryAllowed() && rank != expect_rank)
    {
        error = absl::StrFormat(
            "gated norm gate rank %lld != frozen %d for %s", rank,
            expect_rank, prefix);
        return false;
    }
    if (!UploadRaw(w, CvKind::kBf16Raw, &n.d_weight, prefix + ".weight", stream,
                   ring, acc, error) ||
        !UploadRaw(gu, CvKind::kBf16Raw, &n.d_gate_up,
                   prefix + ".gate_up_projection.weight", stream, ring, acc,
                   error) ||
        !UploadRaw(gd, CvKind::kBf16Raw, &n.d_gate_down,
                   prefix + ".gate_down_projection.weight", stream, ring, acc,
                   error))
        return false;
    n.dim = static_cast<int>(dim);
    n.rank = static_cast<int>(rank);
    return true;
}

bool LayerIsFullAttention(const Gc4Index &idx, int layer)
{
    if (!idx.layer_types.empty() &&
        static_cast<int>(idx.layer_types.size()) == idx.num_layers)
        return idx.layer_types[static_cast<std::size_t>(layer)] ==
               "full_attention";
    return std::find(idx.full_attention_layers.begin(),
                     idx.full_attention_layers.end(),
                     layer) != idx.full_attention_layers.end();
}

} // namespace

bool LoadGigaChat4Weights(const std::string &model_path, const Gc4Index &idx,
                          Gc4Weights &w, cudaStream_t stream, std::string &error)
{
    return LoadGigaChat4WeightsWithExpect(model_path, idx, w, stream,
                                          Gc4DefaultExpect(), error);
}

bool LoadGigaChat4WeightsWithExpect(const std::string &model_path,
                                   const Gc4Index &idx, Gc4Weights &w,
                                   cudaStream_t stream,
                                   const Gc4FrozenExpect &expect,
                                   std::string &error)
{
    namespace fs = std::filesystem;
    const fs::path base = fs::path(model_path);
    const int gate_rank = expect.gate_rank;
    PinnedStagingRing ring;
    if (!ring.ok())
    {
        error = "pinned staging ring allocation failed";
        return false;
    }
    fastload::StreamDrain drain{stream};
    std::uint64_t acc = 0;

    w.hidden = idx.hidden_size;
    w.vocab = idx.vocab_size;

    auto global = LoadTensorBatchViews(base / "pinned" / "global.arrow");
    if (!global.ok())
    {
        error = "cannot open pinned/global.arrow: " + global.status().ToString();
        return false;
    }
    {
        const TensorView *emb =
            FindTensorSuffix(*global, "model.embed_tokens.weight");
        const size_t emb_bytes =
            static_cast<size_t>(w.vocab) * static_cast<size_t>(w.hidden) * 2;
        if (!emb || static_cast<size_t>(emb->bytes) != emb_bytes)
        {
            error = "missing/malformed tensor: model.embed_tokens.weight";
            w.Free();
            return false;
        }
        void *pinned = nullptr;
        if (cudaHostAlloc(&pinned, emb_bytes, cudaHostAllocDefault) !=
            cudaSuccess)
        {
            error = "embed pinned host allocation failed";
            w.Free();
            return false;
        }
        std::memcpy(pinned, emb->data, emb_bytes);
        w.h_embed = static_cast<const uint8_t *>(pinned);
        w.h_embed_pinned = pinned;
    }
    if (!UploadRaw(FindTensorSuffix(*global, "lm_head.weight"),
                   CvKind::kBf16Raw, &w.d_lm_head, "lm_head.weight", stream,
                   ring, acc, error) ||
        !UploadNormTriple(*global, "model.norm", w.final_norm, gate_rank, stream,
                          ring, acc, error))
    {
        w.Free();
        return false;
    }

    w.layers.resize(static_cast<std::size_t>(idx.num_layers));
    for (int l = 0; l < idx.num_layers; ++l)
    {
        gigachat::GcLayerWeights &lw = w.layers[static_cast<std::size_t>(l)];
        lw.layer = l;
        lw.full_attention = LayerIsFullAttention(idx, l);
        lw.moe = l >= idx.first_k_dense_replace;

        const auto nv_path = base / "pinned" /
                             absl::StrFormat("layer_%03d_nvfp4.arrow", l);
        const auto raw_path =
            base / "pinned" / absl::StrFormat("layer_%03d.arrow", l);
        auto nv = LoadDenseNvfp4BatchViews(nv_path);
        auto raw = LoadTensorBatchViews(raw_path);
        if (!nv.ok() || !raw.ok())
        {
            error = absl::StrFormat("cannot open layer %d arrow files", l);
            w.Free();
            return false;
        }
        const std::string lp = absl::StrFormat("layers.%d.", l);

        if (!UploadNormTriple(*raw, lp + "input_layernorm", lw.input_norm, gate_rank,
                              stream, ring, acc, error) ||
            !UploadNormTriple(*raw, lp + "post_attention_layernorm",
                              lw.post_attn_norm, gate_rank, stream, ring, acc,
                              error) ||
            !UploadNormTriple(*raw, lp + "post_self_attn_layernorm",
                              lw.post_self_attn_norm, gate_rank, stream, ring,
                              acc, error) ||
            !UploadNormTriple(*raw, lp + "post_feedforward_layernorm",
                              lw.post_ffw_norm, gate_rank, stream, ring, acc,
                              error))
        {
            w.Free();
            return false;
        }

        if (lw.full_attention)
        {
            if (!UploadNormTriple(*raw, lp + "self_attn.q_a_layernorm",
                                  lw.q_a_norm, gate_rank, stream, ring, acc,
                                  error) ||
                !UploadNormTriple(*raw, lp + "self_attn.kv_a_layernorm",
                                  lw.kv_a_norm, gate_rank, stream, ring, acc,
                                  error) ||
                !UploadNvfp4(*nv, lp + "self_attn.q_a_proj.weight", lw.q_a,
                             "q_a_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "self_attn.q_b_proj.weight", lw.q_b,
                             "q_b_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "self_attn.kv_a_proj_with_mqa.weight",
                             lw.kv_a, "kv_a_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "self_attn.kv_b_proj.weight", lw.kv_b,
                             "kv_b_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "self_attn.o_proj.weight", lw.o_proj,
                             "o_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "self_attn.gate_proj.weight",
                             lw.attn_gate, "attn gate_proj", stream, ring, acc,
                             error))
            {
                w.Free();
                return false;
            }
            // DSA lightning indexer.
            if (!UploadNvfp4(*nv, lp + "self_attn.indexer.wq_b.weight",
                             lw.indexer.wq_b, "indexer.wq_b", stream, ring, acc,
                             error) ||
                !UploadRaw(FindTensorSuffix(*raw, lp + "self_attn.indexer.wk.weight"),
                           CvKind::kBf16Raw, &lw.indexer.d_wk, "indexer.wk",
                           stream, ring, acc, error) ||
                !UploadRaw(
                    FindTensorSuffix(*raw, lp + "self_attn.indexer.weights_proj.weight"),
                    CvKind::kBf16Raw, &lw.indexer.d_weights_proj,
                    "indexer.weights_proj", stream, ring, acc, error) ||
                !UploadRaw(
                    FindTensorSuffix(*raw, lp + "self_attn.indexer.k_norm.weight"),
                    CvKind::kBf16Raw, &lw.indexer.d_k_norm, "indexer.k_norm",
                    stream, ring, acc, error))
            {
                w.Free();
                return false;
            }
        }
        else
        {
            const TensorView *conv =
                FindTensorSuffix(*raw, lp + "self_attn.conv1d.weight");
            const TensorView *alog = FindTensorSuffix(*raw, lp + "self_attn.A_log");
            const TensorView *dtb = FindTensorSuffix(*raw, lp + "self_attn.dt_bias");
            const TensorView *ba =
                FindTensorSuffix(*raw, lp + "self_attn.in_proj_ba.weight");
            const TensorView *gnorm =
                FindTensorSuffix(*raw, lp + "self_attn.norm.weight");
            if (!conv || !alog || !dtb || !ba || !gnorm)
            {
                error = absl::StrFormat("missing GDN tensor(s) on layer %d", l);
                w.Free();
                return false;
            }
            if (!UploadNvfp4(*nv, lp + "self_attn.in_proj_qkvz.weight",
                             lw.gdn_in_qkvz, "in_proj_qkvz", stream, ring, acc,
                             error) ||
                !UploadNvfp4(*nv, lp + "self_attn.out_proj.weight",
                             lw.gdn_out_proj, "gdn out_proj", stream, ring, acc,
                             error) ||
                !UploadRaw(conv, CvKind::kBf16ToF32, &lw.d_conv1d,
                           "conv1d.weight", stream, ring, acc, error) ||
                !UploadRaw(alog, CvKind::kBf16ToF32, &lw.d_A_log, "A_log",
                           stream, ring, acc, error) ||
                !UploadRaw(dtb, CvKind::kBf16ToF32, &lw.d_dt_bias, "dt_bias",
                           stream, ring, acc, error) ||
                !UploadRaw(ba, CvKind::kBf16Raw, &lw.d_gdn_in_ba,
                           "in_proj_ba.weight", stream, ring, acc, error) ||
                !UploadRaw(gnorm, CvKind::kBf16Raw, &lw.d_gdn_norm,
                           "gdn norm.weight", stream, ring, acc, error))
            {
                w.Free();
                return false;
            }
            lw.conv_dim = static_cast<int>(conv->shape[0]);
            lw.conv_kernel = static_cast<int>(conv->shape[2]);
            lw.gdn_value_heads = static_cast<int>(alog->shape[0]);
            lw.gdn_ba_rows = static_cast<int>(ba->shape[0]);
            lw.gdn_norm_dim = static_cast<int>(gnorm->shape[0]);
        }

        if (lw.moe)
        {
            const TensorView *router =
                FindTensorSuffix(*raw, lp + "mlp.gate.weight");
            const TensorView *rbias =
                FindTensorSuffix(*raw, lp + "mlp.gate.e_score_correction_bias");
            if (!router || !rbias)
            {
                error = absl::StrFormat("missing router tensor(s) on layer %d", l);
                w.Free();
                return false;
            }
            // The checkpoint stores e_score_correction_bias as BF16; the device
            // router consumes F32. Accept F32 too for older fixtures.
            CvKind bias_kind;
            if (rbias->dtype == "F32")
                bias_kind = CvKind::kF32Raw;
            else if (rbias->dtype == "BF16")
                bias_kind = CvKind::kBf16ToF32;
            else
            {
                error = absl::StrFormat(
                    "unsupported e_score_correction_bias dtype on layer %d", l);
                w.Free();
                return false;
            }
            if (!UploadRaw(router, CvKind::kBf16Raw, &lw.d_router,
                           "mlp.gate.weight", stream, ring, acc, error) ||
                !UploadRaw(rbias, bias_kind, &lw.d_router_bias,
                           "e_score_correction_bias", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "mlp.shared_experts.gate_proj.weight",
                             lw.shared_gate, "shared gate_proj", stream, ring,
                             acc, error) ||
                !UploadNvfp4(*nv, lp + "mlp.shared_experts.up_proj.weight",
                             lw.shared_up, "shared up_proj", stream, ring, acc,
                             error) ||
                !UploadNvfp4(*nv, lp + "mlp.shared_experts.down_proj.weight",
                             lw.shared_down, "shared down_proj", stream, ring,
                             acc, error))
            {
                w.Free();
                return false;
            }
            lw.num_experts = static_cast<int>(router->shape[0]);
        }
        else
        {
            if (!UploadNvfp4(*nv, lp + "mlp.gate_proj.weight", lw.mlp_gate,
                             "mlp gate_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "mlp.up_proj.weight", lw.mlp_up,
                             "mlp up_proj", stream, ring, acc, error) ||
                !UploadNvfp4(*nv, lp + "mlp.down_proj.weight", lw.mlp_down,
                             "mlp down_proj", stream, ring, acc, error))
            {
                w.Free();
                return false;
            }
        }
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess)
    {
        error = "weight upload stream sync failed";
        w.Free();
        return false;
    }
    w.resident_bytes = acc;
    error.clear();
    return true;
}

} // namespace lsxcommon::gigachat4
