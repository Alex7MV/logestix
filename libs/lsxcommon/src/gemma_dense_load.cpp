#include "lsxcommon/gemma_dense_load.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"
#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace gemma_dense {

namespace fs = std::filesystem;

namespace {

using fastload::PinnedStagingRing;

const lsxcommon::TensorView *FindTensor(
    const std::vector<lsxcommon::TensorView> &tensors,
    const std::string &suffix)
{
    for (const auto &t : tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &t;
    }
    return nullptr;
}

// Zero-copy: the returned views point into the mapping and own it through
// their keepalive, so the loader uploads straight from the page cache.
arrow::Result<lsxcommon::TensorBatchViews> LoadArrowFile(const std::string &path)
{
    return lsxcommon::LoadTensorBatchViews(path);
}

// BF16 host bits -> F32 host (exact: BF16 is a truncated F32).
std::vector<float> Bf16BytesToF32(const std::uint8_t *data, int64_t numel)
{
    std::vector<float> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        std::uint32_t wide =
            static_cast<std::uint32_t>(data[static_cast<size_t>(i) * 2]) |
            (static_cast<std::uint32_t>(
                 data[static_cast<size_t>(i) * 2 + 1])
             << 8);
        wide <<= 16;
        std::memcpy(&out[static_cast<size_t>(i)], &wide, 4);
    }
    return out;
}

cudaError_t UploadF32(const float *host, int64_t n, float **d_out,
                      cudaStream_t stream, PinnedStagingRing &ring,
                      std::uint64_t &bytes_acc)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d = nullptr;
    cudaError_t e = cudaMalloc(&d, bytes);
    if (e != cudaSuccess) return e;
    e = ring.UploadRaw(host, d, bytes, stream, bytes_acc);
    if (e != cudaSuccess) {
        cudaFree(d);
        return e;
    }
    *d_out = d;
    return cudaSuccess;
}

cudaError_t LoadGlobal(const fs::path &pinned_dir, GemmaDensePinnedWeights &pw,
                       cudaStream_t stream, std::uint64_t &bytes_acc)
{
    auto tensors_res = LoadArrowFile((pinned_dir / "global.arrow").string());
    if (!tensors_res.ok()) {
        log::error("[gemma_dense_load] ERROR loading global.arrow: " +
                   tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    const auto &tensors = tensors_res->tensors;

    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // Tied NVFP4 embed / lm_head table (global_nvfp4.arrow, the shared
    // 6-column DenseNvfp4 schema).
    auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
    if (!nv_res.ok()) {
        log::error("[gemma_dense_load] ERROR loading global_nvfp4.arrow: " +
                   nv_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    bool have_embed = false;
    for (const auto &r : nv_res->records) {
        if (r.name.find("embed_tokens.weight") == std::string::npos) continue;
        const cudaError_t e = fastload::UploadNvfp4Weight(
            &r, pw.embed, stream, ring, bytes_acc, nullptr);
        if (e != cudaSuccess) {
            log::error(std::string("[gemma_dense_load] nvfp4 embed upload "
                                   "failed: ") +
                       cudaGetErrorString(e));
            return e;
        }
        pw.vocab_size = pw.embed.rows;
        pw.hidden = pw.embed.cols;
        have_embed = true;
        break;
    }
    if (!have_embed) {
        log::error("[gemma_dense_load] embed_tokens.weight missing in "
                   "global_nvfp4.arrow");
        return cudaErrorInvalidValue;
    }

    auto *norm = FindTensor(tensors, "model.norm.weight");
    if (!norm || norm->rank != 1 || norm->shape[0] <= 0 ||
        norm->bytes < static_cast<size_t>(norm->shape[0]) * 2) {
        log::error("[gemma_dense_load] model.norm.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }
    {
        std::vector<lsxcommon::TensorView> views(1);
        views[0].name = norm->name;
        views[0].rank = norm->rank;
        views[0].shape = norm->shape;
        views[0].bytes = norm->bytes;
        views[0].data = norm->data;
        std::vector<fastload::UploadJob> jobs = {
            {&views[0], reinterpret_cast<void **>(&pw.d_final_norm),
             fastload::CvKind::kBf16ToHalf, nullptr, nullptr}};
        void *arena = nullptr;
        size_t arena_bytes = 0;
        const cudaError_t e = fastload::RunUploadJobs(
            std::move(jobs), &arena, &arena_bytes, stream, ring, bytes_acc,
            nullptr);
        if (e != cudaSuccess) return e;
        // The arena holds exactly the final norm; Free() releases it whole.
        pw.final_norm_arena = arena;
        pw.final_norm_arena_bytes = arena_bytes;
    }
    return cudaSuccess;
}

cudaError_t LoadLayer(const std::vector<lsxcommon::TensorView> &tensors,
                      GemmaDenseLayerWeights &lw, float *h_layer_scalar,
                      cudaStream_t stream, std::uint64_t &bytes_acc)
{
    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // NVFP4 triple detection: "<name>.packed" + ".scales" + ".global_scale".
    struct Nvfp4Recs {
        const lsxcommon::TensorView *packed = nullptr;
        const lsxcommon::TensorView *scales = nullptr;
        const lsxcommon::TensorView *gs = nullptr;
    };
    auto gather = [&](std::string_view base) {
        Nvfp4Recs r;
        for (const auto &t : tensors) {
            const std::string suffix = std::string(base) + ".packed";
            if (t.name.size() > suffix.size() &&
                t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                               suffix) == 0)
                r.packed = &t;
        }
        for (const auto &t : tensors) {
            const std::string suffix = std::string(base) + ".scales";
            if (t.name.size() > suffix.size() &&
                t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                               suffix) == 0)
                r.scales = &t;
        }
        for (const auto &t : tensors) {
            const std::string suffix = std::string(base) + ".global_scale";
            if (t.name.size() > suffix.size() &&
                t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                               suffix) == 0)
                r.gs = &t;
        }
        return r;
    };
    auto complete = [](const Nvfp4Recs &r) {
        return r.packed && r.scales && r.gs;
    };
    auto upload_nv = [&](const Nvfp4Recs &r,
                         lsx_nvfp4::LsxNvfp4Weight &w) -> cudaError_t {
        if (!complete(r)) return cudaErrorInvalidValue;
        w.rows = r.packed->rank >= 2 ? static_cast<int>(r.packed->shape[0]) : 0;
        w.cols = r.packed->rank >= 2 ? static_cast<int>(r.packed->shape[1]) : 0;
        if (w.rows <= 0 || w.cols <= 0) return cudaErrorInvalidValue;
        std::memcpy(&w.global_scale, r.gs->data, sizeof(float));
        w.packed_bytes = static_cast<int64_t>(r.packed->bytes);
        w.scale_bytes = static_cast<int64_t>(r.scales->bytes);
        void *d1 = nullptr;
        cudaError_t e = cudaMalloc(&d1, r.packed->bytes);
        if (e != cudaSuccess) return e;
        void *d2 = nullptr;
        e = cudaMalloc(&d2, r.scales->bytes);
        if (e != cudaSuccess) {
            cudaFree(d1);
            return e;
        }
        e = ring.UploadRaw(r.packed->data, d1, r.packed->bytes,
                           stream, bytes_acc);
        if (e == cudaSuccess)
            e = ring.UploadRaw(r.scales->data, d2, r.scales->bytes,
                               stream, bytes_acc);
        if (e != cudaSuccess) {
            cudaFree(d1);
            cudaFree(d2);
            return e;
        }
        w.d_packed = static_cast<const std::uint8_t *>(d1);
        w.d_scales = static_cast<const std::uint8_t *>(d2);
        return cudaSuccess;
    };

    const Nvfp4Recs q = gather("self_attn.q_proj.weight");
    const Nvfp4Recs k = gather("self_attn.k_proj.weight");
    const Nvfp4Recs v = gather("self_attn.v_proj.weight");
    const Nvfp4Recs o = gather("self_attn.o_proj.weight");
    const Nvfp4Recs gate = gather("mlp.gate_proj.weight");
    const Nvfp4Recs up = gather("mlp.up_proj.weight");
    const Nvfp4Recs down = gather("mlp.down_proj.weight");

    if (!complete(q) || !complete(k) || !complete(o) || !complete(gate) ||
        !complete(up) || !complete(down)) {
        log::error("[gemma_dense_load] layer is missing an NVFP4 projection "
                   "triple (packed + scales + global_scale)");
        return cudaErrorInvalidValue;
    }
    cudaError_t e = upload_nv(q, lw.attn_q);
    if (e == cudaSuccess) e = upload_nv(k, lw.attn_k);
    // v is optional: a full-attention layer ties V to K and ships no triple.
    if (e == cudaSuccess && complete(v)) e = upload_nv(v, lw.attn_v);
    if (e == cudaSuccess) e = upload_nv(o, lw.attn_o);
    if (e == cudaSuccess) e = upload_nv(gate, lw.mlp_gate);
    if (e == cudaSuccess) e = upload_nv(up, lw.mlp_up);
    if (e == cudaSuccess) e = upload_nv(down, lw.mlp_down);
    if (e != cudaSuccess) return e;

    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    auto *pre_ffn = FindTensor(tensors, "pre_feedforward_layernorm.weight");
    auto *post_ffn = FindTensor(tensors, "post_feedforward_layernorm.weight");
    auto *q_norm = FindTensor(tensors, "self_attn.q_norm.weight");
    auto *k_norm = FindTensor(tensors, "self_attn.k_norm.weight");
    if (!in_norm || !post_norm || !pre_ffn || !post_ffn || !q_norm ||
        !k_norm) {
        log::error("[gemma_dense_load] layer missing norm tensors");
        return cudaErrorInvalidValue;
    }

    std::vector<lsxcommon::TensorView> views;
    views.reserve(6);
    auto make_job = [&](const lsxcommon::TensorView *t, void **dst) {
        if (!t) return fastload::UploadJob{nullptr, dst,
                                           fastload::CvKind::kBf16ToHalf,
                                           nullptr, nullptr};
        views.push_back(lsxcommon::TensorView{});
        auto &v = views.back();
        v.name = t->name;
        v.rank = t->rank;
        v.shape = t->shape;
        v.bytes = t->bytes;
        v.data = t->data;
        return fastload::UploadJob{&views.back(), dst,
                                   fastload::CvKind::kBf16ToHalf, nullptr,
                                   nullptr};
    };
    std::vector<fastload::UploadJob> jobs = {
        make_job(in_norm, reinterpret_cast<void **>(&lw.d_input_norm)),
        make_job(post_norm, reinterpret_cast<void **>(&lw.d_post_norm)),
        make_job(pre_ffn, reinterpret_cast<void **>(&lw.d_pre_ffn)),
        make_job(post_ffn, reinterpret_cast<void **>(&lw.d_post_ffn)),
        make_job(q_norm, reinterpret_cast<void **>(&lw.d_q_norm)),
        make_job(k_norm, reinterpret_cast<void **>(&lw.d_k_norm)),
    };
    void *arena = nullptr;
    size_t arena_bytes = 0;
    e = fastload::RunUploadJobs(std::move(jobs), &arena, &arena_bytes, stream,
                                ring, bytes_acc, nullptr);
    lw.d_arena = arena;
    lw.arena_bytes = arena_bytes;
    if (e != cudaSuccess) return e;

    // layer_scalar: BF16 in the real checkpoint, F32 in synthetic fixtures.
    if (const auto *sc = FindTensor(tensors, "layer_scalar");
        sc && sc->rank == 1 &&
        ((sc->dtype == "F32" && sc->bytes >= sizeof(float)) ||
         (sc->dtype == "BF16" && sc->bytes >= 2))) {
        if (sc->dtype == "F32") {
            e = UploadF32(reinterpret_cast<const float *>(sc->data), 1,
                          &lw.d_layer_scalar, stream, ring, bytes_acc);
            if (h_layer_scalar)
                *h_layer_scalar =
                    *reinterpret_cast<const float *>(sc->data);
        } else {
            const auto f32 = Bf16BytesToF32(sc->data, 1);
            e = UploadF32(f32.data(), 1, &lw.d_layer_scalar, stream, ring,
                          bytes_acc);
            if (h_layer_scalar) *h_layer_scalar = f32[0];
        }
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace

void GemmaDenseLayerWeights::Free()
{
    if (d_arena) cudaFree(d_arena);
    if (d_layer_scalar) cudaFree(d_layer_scalar);
    attn_q.Free();
    attn_k.Free();
    attn_v.Free();
    attn_o.Free();
    mlp_gate.Free();
    mlp_up.Free();
    mlp_down.Free();
    *this = GemmaDenseLayerWeights{};
}

void GemmaDensePinnedWeights::Free()
{
    if (final_norm_arena) cudaFree(final_norm_arena);
    final_norm_arena = nullptr;
    final_norm_arena_bytes = 0;
    d_final_norm = nullptr;
    embed.Free();
    vocab_size = 0;
    hidden = 0;
    resident_bytes = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
    h_layer_scalar.clear();
}

cudaError_t LoadGemmaDensePinnedWeights(const std::string &model_path,
                                        GemmaDensePinnedWeights &pw,
                                        cudaStream_t stream)
{
    pw.Free();

    fs::path pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir) && fs::exists(fs::path(model_path) / "pinned"))
        pinned_dir = fs::path(model_path) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[gemma_dense_load] pinned dir not found under " +
                   model_path);
        return cudaErrorInvalidValue;
    }

    std::uint64_t bytes_acc = 0;
    cudaError_t e = LoadGlobal(pinned_dir, pw, stream, bytes_acc);
    if (e != cudaSuccess) return e;

    int max_layer = -1;
    for (const auto &entry : fs::directory_iterator(pinned_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("layer_", 0) != 0) continue;
        if (name.size() < 7 || name.substr(name.size() - 6) != ".arrow")
            continue;
        max_layer = std::max(max_layer, std::atoi(name.c_str() + 6));
    }
    if (max_layer < 0) {
        log::error("[gemma_dense_load] no layer_*.arrow files in " +
                   pinned_dir.string());
        return cudaErrorInvalidValue;
    }
    pw.layers.assign(static_cast<size_t>(max_layer + 1), GemmaDenseLayerWeights{});

    pw.h_layer_scalar.resize(pw.layers.size(), 1.0f);
    for (int l = 0; l <= max_layer; ++l) {
        const auto path = pinned_dir / absl::StrCat("layer_", l, ".arrow");
        if (!fs::exists(path)) continue;
        auto tensors_res = LoadArrowFile(path.string());
        if (!tensors_res.ok()) {
            log::error("[gemma_dense_load] ERROR loading " + path.string() +
                       ": " + tensors_res.status().ToString());
            return cudaErrorInvalidValue;
        }
        e = LoadLayer(tensors_res->tensors, pw.layers[static_cast<size_t>(l)],
                      &pw.h_layer_scalar[static_cast<size_t>(l)], stream,
                      bytes_acc);
        if (e != cudaSuccess) return e;
    }

    pw.resident_bytes = bytes_acc;

    // Topology from the index. Mandatory for generation; a parse failure is
    // logged and surfaces through ValidateGeometry.
    auto idx_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(idx_path))
        idx_path = fs::path(model_path) / "model.index.json";
    if (fs::exists(idx_path)) {
        pw.top = load_gemma_dense_topology(idx_path.string());
        if (!pw.top.error_msg.empty())
            log::error("[gemma_dense_load] index topology parse failed: " +
                       pw.top.error_msg);
    }
    return cudaSuccess;
}

cudaError_t GemmaDenseKvCache::Allocate(const GemmaDensePinnedWeights &pw,
                                        int max_total_in)
{
    Free();
    if (!pw.top.error_msg.empty() || pw.layers.empty() || max_total_in <= 0)
        return cudaErrorInvalidValue;
    dk.resize(pw.layers.size(), nullptr);
    dv.resize(pw.layers.size(), nullptr);
    geo.assign(pw.layers.size(), KvCacheGeometry{});
    std::uint64_t bytes = 0;
    for (size_t l = 0; l < pw.layers.size(); ++l) {
        // Pos-major (the layout the ingestion protocol ships unless a measured
        // end-to-end number says otherwise); per-layer geometry because the
        // sliding and global layers store different KV widths.
        const int64_t kv_heads = pw.top.KvHeads(static_cast<int>(l));
        const int64_t head_dim = pw.top.HeadDim(static_cast<int>(l));
        if (kv_heads <= 0 || head_dim <= 0) {
            Free();
            return cudaErrorInvalidValue;
        }
        geo[l] = KvCacheGeometry::Make(KvCacheLayout::kPosMajor, max_total_in,
                                       kv_heads, head_dim, 1);
        const size_t sz = static_cast<size_t>(geo[l].LayerStride()) *
                          sizeof(__half);
        if (cudaMalloc(&dk[l], sz) != cudaSuccess ||
            cudaMalloc(&dv[l], sz) != cudaSuccess) {
            Free();
            return cudaErrorMemoryAllocation;
        }
        bytes += 2 * sz;
    }
    max_total = max_total_in;
    cache_bytes = bytes;
    return cudaSuccess;
}

void GemmaDenseKvCache::Free()
{
    for (auto p : dk)
        if (p) cudaFree(p);
    for (auto p : dv)
        if (p) cudaFree(p);
    dk.clear();
    dv.clear();
    geo.clear();
    max_total = 0;
    cache_bytes = 0;
}

} // namespace gemma_dense
} // namespace lsxcommon
