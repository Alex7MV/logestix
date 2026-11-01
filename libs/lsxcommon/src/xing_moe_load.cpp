// Xing4.0-29B-A4B loader (Model Ingestion Protocol step 3). Uploads the whole
// converted model into device memory once: MLA projections, both mHC blocks,
// the noaux_tc router and — unlike every other MoE family in the repo — ALL
// routed experts.
//
// Why no expert cache: 64 experts x 38 MoE layers x 3 x 1024 x 3584 is ~15 GiB
// at NVFP4, which fits a 32 GiB Blackwell card next to a 2048-token KV cache.
// Keeping the experts resident removes the per-token H2D staging and the slot
// LRU entirely; routing only computes offsets into the layer slab. The cost is
// that this family cannot run with a small % of VRAM reserved — the Load fails
// loudly instead of silently falling back to streaming.

#include "lsxcommon/xing_moe_load.h"

#include "xing_moe_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>

#include <tuple>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace lsxcommon::xing {
namespace {

namespace fs = std::filesystem;

int64_t AlignUp(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

cudaError_t UploadBytes(const uint8_t *src, size_t n, const uint8_t **dst,
                        std::size_t &accounted)
{
    if (n == 0) return cudaSuccess;
    void *p = nullptr;
    const cudaError_t err = cudaMalloc(&p, n);
    if (err != cudaSuccess) return err;
    const cudaError_t cp =
        cudaMemcpy(p, src, n, cudaMemcpyHostToDevice);
    if (cp != cudaSuccess)
    {
        cudaFree(p);
        return cp;
    }
    *dst = static_cast<const uint8_t *>(p);
    accounted += n;
    return cudaSuccess;
}

// Dense NVFP4 record -> device. packed and scales are SEPARATE Arrow columns,
// i.e. two unrelated host addresses: uploading them as one blob and pointing
// `scales` at `packed + packed_bytes` reads whatever follows the nibbles in the
// packed buffer. That produced garbage E4M3 scales (0xf7ef instead of 0x38) and
// an all-inf logit row with no error anywhere - the same split the shared
// fastload::UploadNvfp4Weight does.
cudaError_t UploadNvfp4(const lsxcommon::DenseNvfp4View &v,
                         XingNvfp4Weight &w, cudaStream_t stream,
                         std::size_t &accounted)
{
    const cudaError_t err = UploadBytes(v.packed,
                                        static_cast<size_t>(v.packed_bytes),
                                        &w.packed, accounted);
    if (err != cudaSuccess) return err;
    const cudaError_t esc = UploadBytes(v.scales,
                                        static_cast<size_t>(v.scale_bytes),
                                        &w.scales, accounted);
    if (esc != cudaSuccess) return esc;
    w.global_scale = v.global_scale;
    w.rows = v.rank > 0 ? v.shape[0] : 0;
    w.cols = v.rank > 1 ? v.shape[1] : 1;
    w.packed_bytes = v.packed_bytes;
    w.scale_bytes = v.scale_bytes;
    (void)stream;
    return cudaSuccess;
}

// Raw BF16/F32 record -> device.
cudaError_t UploadRaw(const lsxcommon::TensorView &v, const uint8_t **dst,
                      cudaStream_t stream, std::size_t &accounted)
{
    (void)stream;
    return UploadBytes(v.data, v.bytes, dst, accounted);
}

// Typed variant: the caller knows the record is BF16 (norms, mHC, router), so
// the device pointer is exposed with its real dtype instead of `uint8_t*`.
cudaError_t UploadBf16(const lsxcommon::TensorView &v,
                       const __nv_bfloat16 **dst, cudaStream_t stream,
                       std::size_t &accounted)
{
    const uint8_t *raw = nullptr;
    const cudaError_t e = UploadRaw(v, &raw, stream, accounted);
    if (e != cudaSuccess) return e;
    *dst = reinterpret_cast<const __nv_bfloat16 *>(raw);
    return cudaSuccess;
}

struct IndexInfo {
    long long num_layers = 0;
    long long experts_per_layer = 0;
    long long hc_mult = 0;
    std::vector<long long> dense_layers;
    std::vector<long long> moe_layers;
    std::vector<long long> mtp_layers;
    std::string format;
    std::string error_msg;
};

long long JsonInt(const std::string &text, const std::string &key,
                  long long fallback)
{
    const size_t k = text.find(absl::StrCat("\"", key, "\""));
    if (k == std::string::npos) return fallback;
    const size_t colon = text.find(':', k);
    if (colon == std::string::npos) return fallback;
    const std::string tail{
        std::string(absl::StripAsciiWhitespace(text.substr(colon + 1, 24)))};
    return std::strtoll(tail.c_str(), nullptr, 10);
}

void ScanList(const std::string &text, const char *key,
              std::vector<long long> &into)
{
    const size_t k = text.find(absl::StrCat("\"", key, "\""));
    if (k == std::string::npos) return;
    const size_t open = text.find('[', k);
    if (open == std::string::npos) return;
    const size_t close = text.find(']', open);
    if (close == std::string::npos) return;
    for (const auto &part :
         absl::StrSplit(text.substr(open + 1, close - open - 1), ','))
    {
        const std::string t{
            std::string(absl::StripAsciiWhitespace(part))};
        if (!t.empty()) into.push_back(std::strtoll(t.c_str(), nullptr, 10));
    }
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
    ix.format = xing_load_topology(p.string()).format;
    ix.num_layers = JsonInt(text, "num_layers", 0);
    ix.experts_per_layer = JsonInt(text, "experts_per_layer", 0);
    ix.hc_mult = JsonInt(text, "hc_mult", 0);
    ScanList(text, "dense_layers", ix.dense_layers);
    ScanList(text, "moe_layers", ix.moe_layers);
    ScanList(text, "mtp_layers", ix.mtp_layers);
    if (ix.format != "logestix-xing-nvfp4")
    {
        ix.error_msg =
            absl::StrCat("unexpected format tag '", ix.format,
                         "' (expected logestix-xing-nvfp4)");
        return ix;
    }
    if (ix.num_layers <= 0 || ix.experts_per_layer <= 0)
    {
        ix.error_msg = "index missing num_layers/experts_per_layer";
        return ix;
    }
    if (!SyntheticGeometryAllowed())
    {
        if (ix.num_layers != frozen::kLayers ||
            ix.experts_per_layer != frozen::kNumRouted ||
            ix.hc_mult != frozen::kHcMult)
        {
            ix.error_msg = absl::StrCat(
                "frozen geometry mismatch: layers=", ix.num_layers,
                " experts=", ix.experts_per_layer, " hc_mult=", ix.hc_mult);
            return ix;
        }
        std::vector<long long> want_dense(frozen::kMoeFirstLayer);
        for (int i = 0; i < frozen::kMoeFirstLayer; ++i) want_dense[i] = i;
        std::vector<long long> want_moe;
        for (int i = frozen::kMoeFirstLayer; i < frozen::kLayers; ++i)
            want_moe.push_back(i);
        if (ix.dense_layers != want_dense || ix.moe_layers != want_moe)
        {
            ix.error_msg = "dense/moe layer split disagrees with the passport";
            return ix;
        }
    }
    return ix;
}

// Loads one mHC block out of the layer's raw batch.
bool LoadHyperConn(const lsxcommon::TensorBatchViews &batch,
                   const std::string &prefix, XingHyperConn &hc,
                   cudaStream_t stream, std::size_t &accounted)
{
    const lsxcommon::TensorView *fn = nullptr;
    const lsxcommon::TensorView *base = nullptr;
    const lsxcommon::TensorView *scale = nullptr;
    for (const auto &v : batch.tensors)
    {
        if (v.name == prefix + ".hc_fn") fn = &v;
        else if (v.name == prefix + ".hc_base") base = &v;
        else if (v.name == prefix + ".hc_scale") scale = &v;
    }
    if (!fn || !base || !scale)
    {
        log::error(absl::StrCat("xing loader: incomplete mHC block ", prefix));
        return false;
    }
    if (fn->rank != 2)
    {
        log::error(absl::StrCat("xing loader: hc_fn must be 2-D, got rank ",
                                fn->rank));
        return false;
    }
    const uint8_t *fn_dev = nullptr, *base_dev = nullptr, *scale_dev = nullptr;
    if (UploadRaw(*fn, &fn_dev, stream, accounted) != cudaSuccess) return false;
    if (UploadRaw(*base, &base_dev, stream, accounted) != cudaSuccess) return false;
    if (UploadRaw(*scale, &scale_dev, stream, accounted) != cudaSuccess)
        return false;
    hc.hc_fn = reinterpret_cast<const __nv_bfloat16 *>(fn_dev);
    hc.hc_base = reinterpret_cast<const __nv_bfloat16 *>(base_dev);
    hc.hc_scale = reinterpret_cast<const float *>(scale_dev);
    hc.hc_fn_rows = fn->shape[0];
    hc.hc_fn_cols = fn->shape[1];
    return true;
}

} // namespace

void XingPinnedWeights::Free()
{
    FreeXingWeights(*this);
}

void FreeXingWeights(XingPinnedWeights &pw)
{
    auto drop = [](const void *p) {
        if (p) cudaFree(const_cast<void *>(p));
    };
    drop(pw.embed.packed);
    drop(pw.head.packed);
    drop(pw.final_norm);
    for (auto &l : pw.layers)
    {
        for (XingNvfp4Weight *w :
             {&l.q_a, &l.q_b, &l.kv_a, &l.kv_b, &l.o_proj,
              &l.dense_gate, &l.dense_up, &l.dense_down, &l.shared_gate,
              &l.shared_up, &l.shared_down})
            drop(w->packed);
        drop(l.q_a_ln);
        drop(l.kv_a_ln);
        drop(l.input_ln);
        drop(l.post_attn_ln);
        drop(l.attn_hc.hc_fn);
        drop(l.attn_hc.hc_base);
        drop(l.attn_hc.hc_scale);
        drop(l.ffn_hc.hc_fn);
        drop(l.ffn_hc.hc_base);
        drop(l.ffn_hc.hc_scale);
        drop(l.e_score_bias);
        drop(l.router);
        l = XingLayerWeights{};
    }
    pw.layers.clear();
    for (auto *s : pw.expert_slabs)
        if (s) cudaFree(s);
    pw.expert_slabs.clear();
    pw.expert_packed_ptrs.clear();
    pw.expert_scale_ptrs.clear();
    pw.expert_gs.clear();
    pw.slab_bytes.clear();
    pw.expert_records.clear();
    pw.resident_bytes = 0;
}

cudaError_t LoadXingWeights(const std::string &model_path,
                            XingPinnedWeights &pw, cudaStream_t stream,
                            std::string &error)
{
    namespace frozen = lsxcommon::xing::frozen;
    error.clear();
    pw.Free();
    pw.model_path = model_path;

    const fs::path base = fs::path(model_path) / "logestix";
    const IndexInfo ix = ReadIndex(base / "model.index.json");
    if (!ix.error_msg.empty())
    {
        error = ix.error_msg;
        return cudaErrorInvalidValue;
    }
    pw.num_layers = static_cast<int>(ix.num_layers);
    pw.num_experts = static_cast<int>(ix.experts_per_layer);
    pw.first_moe_layer =
        ix.dense_layers.empty()
            ? 0
            : static_cast<int>(*ix.dense_layers.rbegin()) + 1;

    std::size_t acc = 0;
    auto fail = [&](const std::string &what, cudaError_t err) {
        error = what;
        pw.Free();
        return err;
    };

    // ── globals ────────────────────────────────────────────────────────────
    {
        auto g = lsxcommon::LoadDenseNvfp4BatchViews(base / "pinned" /
                                                     "global_nvfp4.arrow");
        if (!g.ok())
            return fail("global_nvfp4.arrow: " + g.status().ToString(),
                        cudaErrorInvalidValue);
        for (const auto &rec : g->records)
        {
            XingNvfp4Weight *dst = nullptr;
            if (rec.name == "model.embed_tokens.weight") dst = &pw.embed;
            else if (rec.name == "lm_head.weight") dst = &pw.head;
            if (!dst)
                return fail("unexpected global record " + std::string(rec.name),
                            cudaErrorInvalidValue);
            const cudaError_t e = UploadNvfp4(rec, *dst, stream, acc);
            if (e != cudaSuccess) return fail("global upload failed", e);
        }
        auto n = lsxcommon::LoadTensorBatchViews(base / "pinned" / "global.arrow");
        if (!n.ok())
            return fail("global.arrow: " + n.status().ToString(),
                        cudaErrorInvalidValue);
        for (const auto &v : n->tensors)
        {
            if (v.name != "model.norm.weight")
                return fail("unexpected pinned global " + std::string(v.name),
                            cudaErrorInvalidValue);
            const cudaError_t e = UploadBf16(v, &pw.final_norm, stream, acc);
            if (e != cudaSuccess) return fail("final norm upload failed", e);
        }
        if (!pw.embed.valid() || !pw.head.valid() || !pw.final_norm)
            return fail("globals incomplete (embed/head/final norm)",
                        cudaErrorInvalidValue);
    }

    // ── layers ─────────────────────────────────────────────────────────────
    pw.layers.assign(static_cast<size_t>(pw.num_layers), XingLayerWeights{});
    for (int l = 0; l < pw.num_layers; ++l)
    {
        auto &ly = pw.layers[static_cast<size_t>(l)];
        ly.is_moe = l >= pw.first_moe_layer;

        auto raw = lsxcommon::LoadTensorBatchViews(
            base / "pinned" / absl::StrCat("layer_", l, ".arrow"));
        if (!raw.ok())
            return fail(absl::StrCat("layer_", l, ".arrow: ",
                                     raw.status().ToString()),
                        cudaErrorInvalidValue);
        auto q = lsxcommon::LoadDenseNvfp4BatchViews(
            base / "pinned" / absl::StrCat("layer_", l, "_nvfp4.arrow"));
        if (!q.ok())
            return fail(absl::StrCat("layer_", l, "_nvfp4.arrow: ",
                                     q.status().ToString()),
                        cudaErrorInvalidValue);

        const std::string pfx = absl::StrCat("model.layers.", l, ".");
        for (const auto &v : raw->tensors)
        {
            const std::string name{v.name};
            const std::string rel = name.substr(pfx.size());
            if (rel == "self_attn.q_a_layernorm.weight")
                UploadBf16(v, &ly.q_a_ln, stream, acc);
            else if (rel == "self_attn.kv_a_layernorm.weight")
                UploadBf16(v, &ly.kv_a_ln, stream, acc);
            else if (rel == "input_layernorm.weight")
                UploadBf16(v, &ly.input_ln, stream, acc);
            else if (rel == "post_attention_layernorm.weight")
                UploadBf16(v, &ly.post_attn_ln, stream, acc);
            else if (rel == "mlp.gate.weight")
            {
                UploadBf16(v, &ly.router, stream, acc);
                ly.router_size = v.bytes;
            }
            else if (rel == "mlp.gate.e_score_correction_bias")
            {
                const uint8_t *dev = nullptr;
                if (UploadRaw(v, &dev, stream, acc) != cudaSuccess)
                    return fail(absl::StrCat("layer_", l, ": router bias upload"),
                                cudaErrorMemoryAllocation);
                ly.e_score_bias = reinterpret_cast<const float *>(dev);
            }
            else if (absl::StartsWith(rel, "attn_hc."))
            {
                if (!LoadHyperConn(*raw, pfx + "attn_hc", ly.attn_hc, stream, acc))
                    return fail(absl::StrCat("layer_", l, ": bad attn_hc block"),
                                cudaErrorInvalidValue);
            }
            else if (absl::StartsWith(rel, "ffn_hc."))
            {
                if (!LoadHyperConn(*raw, pfx + "ffn_hc", ly.ffn_hc, stream, acc))
                    return fail(absl::StrCat("layer_", l, ": bad ffn_hc block"),
                                cudaErrorInvalidValue);
            }
        }

        for (const auto &rec : q->records)
        {
            const std::string rel{std::string(rec.name)};
            const size_t off = pfx.size();
            const std::string tail = rel.substr(off);
            XingNvfp4Weight *dst = nullptr;
            if (tail == "self_attn.q_a_proj.weight") dst = &ly.q_a;
            else if (tail == "self_attn.q_b_proj.weight") dst = &ly.q_b;
            else if (tail == "self_attn.kv_a_proj_with_mqa.weight") dst = &ly.kv_a;
            else if (tail == "self_attn.kv_b_proj.weight") dst = &ly.kv_b;
            else if (tail == "self_attn.o_proj.weight") dst = &ly.o_proj;
            else if (tail == "mlp.gate_proj.weight") dst = &ly.dense_gate;
            else if (tail == "mlp.up_proj.weight") dst = &ly.dense_up;
            else if (tail == "mlp.down_proj.weight") dst = &ly.dense_down;
            else if (tail == "mlp.shared_experts.gate_proj.weight")
                dst = &ly.shared_gate;
            else if (tail == "mlp.shared_experts.up_proj.weight")
                dst = &ly.shared_up;
            else if (tail == "mlp.shared_experts.down_proj.weight")
                dst = &ly.shared_down;
            if (!dst)
                return fail(absl::StrCat("unexpected quantized tensor in layer_",
                                         l, ": ", tail),
                            cudaErrorInvalidValue);
            const cudaError_t e = UploadNvfp4(rec, *dst, stream, acc);
            if (e != cudaSuccess)
                return fail(absl::StrCat("layer_", l, ": upload failed (",
                                         cudaGetErrorString(e), ")"),
                            e);
        }

        if (!ly.attn_hc.valid() || !ly.ffn_hc.valid() || !ly.input_ln ||
            !ly.post_attn_ln || !ly.q_a_ln || !ly.kv_a_ln || !ly.q_a.valid() ||
            !ly.q_b.valid() || !ly.kv_a.valid() || !ly.kv_b.valid() ||
            !ly.o_proj.valid())
            return fail(absl::StrCat("layer_", l, ": incomplete attention/mHC"),
                        cudaErrorInvalidValue);
        if (ly.is_moe)
        {
            if (!ly.router || !ly.e_score_bias || !ly.shared_gate.valid() ||
                !ly.shared_up.valid() || !ly.shared_down.valid())
                return fail(absl::StrCat("layer_", l, ": incomplete MoE block"),
                            cudaErrorInvalidValue);
        }
        else if (!ly.dense_gate.valid() || !ly.dense_up.valid() ||
                 !ly.dense_down.valid())
        {
            return fail(absl::StrCat("layer_", l, ": incomplete dense MLP"),
                        cudaErrorInvalidValue);
        }
    }

    // ── routed experts: one device slab per MoE layer ──────────────────────
    pw.expert_slabs.assign(static_cast<size_t>(pw.num_layers), nullptr);
    pw.slab_bytes.assign(static_cast<size_t>(pw.num_layers), 0);
    pw.expert_records.assign(static_cast<size_t>(pw.num_layers), {});
    pw.expert_packed_ptrs.assign(static_cast<size_t>(pw.num_layers), {});
    pw.expert_scale_ptrs.assign(static_cast<size_t>(pw.num_layers), {});
    pw.expert_gs.assign(static_cast<size_t>(pw.num_layers), {});
    for (int l = pw.first_moe_layer; l < pw.num_layers; ++l)
    {
        const fs::path dir = base / "experts" / absl::StrCat("layer_", l);
        if (!fs::exists(dir))
            return fail(absl::StrCat("missing experts dir ", dir.string()),
                        cudaErrorInvalidValue);

        struct Staged {
            std::vector<uint8_t> w1, w2, w3, s1, s2, s3;
            XingExpertRecord rec;
        };
        // Two packed pointers and one scale pointer per expert: gate rows for
        // the gate GEMV, down rows (w2) for the down GEMV, and the up rows'
        // scale tile (s3) because gate and up share the gate/up scale split.
        std::vector<Staged> staged(static_cast<size_t>(pw.num_experts));
        size_t total = 0;
        for (int e = 0; e < pw.num_experts; ++e)
        {
            const fs::path path = dir / absl::StrCat("exp_", e, ".arrow");
            auto batch = lsxcommon::ReadRecordBatchFromFile(path);
            if (!batch.ok())
                return fail("read failed: " + path.string(),
                            cudaErrorInvalidValue);
            auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch, kNVFP4BlockSize);
            if (!pe.ok())
                return fail("unpack failed: " + path.string(),
                            cudaErrorInvalidValue);
            auto &st = staged[static_cast<size_t>(e)];
            st.w1 = pe->w1_packed;
            st.w2 = pe->w2_packed;
            st.w3 = pe->w3_packed;
            st.s1 = pe->w1_scales;
            st.s2 = pe->w2_scales;
            st.s3 = pe->w3_scales;
            auto &r = st.rec;
            r.w1_off = total; r.w1_bytes = static_cast<int64_t>(st.w1.size());
            total += static_cast<size_t>(r.w1_bytes);
            r.w2_off = total; r.w2_bytes = static_cast<int64_t>(st.w2.size());
            total += static_cast<size_t>(r.w2_bytes);
            r.w3_off = total; r.w3_bytes = static_cast<int64_t>(st.w3.size());
            total += static_cast<size_t>(r.w3_bytes);
            r.gs1 = pe->w1_global_scale;
            r.gs2 = pe->w2_global_scale;
            r.gs3 = pe->w3_global_scale;
            total = static_cast<size_t>(AlignUp(static_cast<int64_t>(total), 16));
            r.s1_off = static_cast<int64_t>(total);
            r.s1_bytes = static_cast<int64_t>(st.s1.size());
            total += static_cast<size_t>(r.s1_bytes);
            r.s2_off = static_cast<int64_t>(total);
            r.s2_bytes = static_cast<int64_t>(st.s2.size());
            total += static_cast<size_t>(r.s2_bytes);
            r.s3_off = static_cast<int64_t>(total);
            r.s3_bytes = static_cast<int64_t>(st.s3.size());
            total += static_cast<size_t>(r.s3_bytes);
        }

        void *slab = nullptr;
        cudaError_t e = cudaMalloc(&slab, total);
        if (e != cudaSuccess)
            return fail(absl::StrCat("expert slab alloc failed (layer ", l, "): ",
                                     cudaGetErrorString(e)),
                        e);
        pw.expert_slabs[static_cast<size_t>(l)] = static_cast<uint8_t *>(slab);
        pw.slab_bytes[static_cast<size_t>(l)] = total;
        acc += total;
        for (int k = 0; k < pw.num_experts; ++k)
        {
            auto &st = staged[static_cast<size_t>(k)];
            auto &r = st.rec;
            const uint8_t *dst = pw.expert_slabs[static_cast<size_t>(l)];
            for (const auto [off, bytes, src] :
                 {std::tuple{r.w1_off, r.w1_bytes, &st.w1},
                  {r.w2_off, r.w2_bytes, &st.w2},
                  {r.w3_off, r.w3_bytes, &st.w3},
                  {r.s1_off, r.s1_bytes, &st.s1},
                  {r.s2_off, r.s2_bytes, &st.s2},
                  {r.s3_off, r.s3_bytes, &st.s3}})
            {
                if (bytes == 0) continue;
                const cudaError_t cp = cudaMemcpy(
                    const_cast<uint8_t *>(dst + off), src->data(),
                    static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
                if (cp != cudaSuccess)
                    return fail("expert H2D copy failed", cp);
            }
            pw.expert_records[static_cast<size_t>(l)]
                .push_back(std::move(st.rec));
        }
    }

    // HOST pointer tables for the model-neutral expert-batch GEMV: gate/up rows
    // (w1 + its scale tile) first, down rows (w2 + s3) second.
    for (int l = pw.first_moe_layer; l < pw.num_layers; ++l)
    {
        const size_t slot = static_cast<size_t>(l);
        const size_t n = static_cast<size_t>(pw.num_experts);
        const uint8_t *slab = pw.expert_slabs[slot];
        auto &pp = pw.expert_packed_ptrs[slot];
        auto &sp = pw.expert_scale_ptrs[slot];
        auto &gv = pw.expert_gs[slot];
        // Three views per expert, in the repo's w1/w2/w3 order:
        // [gate (w1) | down (w2) | up (w3)].
        pp.assign(n * 3, nullptr);
        sp.assign(n * 3, nullptr);
        gv.assign(n * 3, 1.0f);
        const auto &recs = pw.expert_records[slot];
        for (size_t k = 0; k < recs.size() && k < n; ++k)
        {
            const auto &r = recs[k];
            pp[k] = slab + r.w1_off;          // w1: gate rows
            pp[n + k] = slab + r.w2_off;      // w2: down rows
            pp[2 * n + k] = slab + r.w3_off;  // w3: up rows
            sp[k] = slab + r.s1_off;
            sp[n + k] = slab + r.s2_off;
            sp[2 * n + k] = slab + r.s3_off;
            gv[k] = r.gs1;
            gv[n + k] = r.gs2;
            gv[2 * n + k] = r.gs3;
        }
    }

    pw.resident_bytes = acc;
    log::info(absl::StrFormat(
        "xing loader: %d layers, %d experts/layer, resident=%llu MiB",
        pw.num_layers, pw.num_experts,
        static_cast<unsigned long long>(acc >> 20)));
    return cudaSuccess;
}

} // namespace lsxcommon::xing