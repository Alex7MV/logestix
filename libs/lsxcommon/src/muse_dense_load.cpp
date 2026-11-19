#include "lsxcommon/muse_dense_load.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"
#include "lsxcommon/muse_dense_convert.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace muse_dense {

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

cudaError_t LoadGlobal(const fs::path &pinned_dir, MuseDensePinnedWeights &pw,
                       cudaStream_t stream, std::uint64_t &bytes_acc)
{
    auto tensors_res = LoadArrowFile((pinned_dir / "global.arrow").string());
    if (!tensors_res.ok()) {
        log::error("[muse_dense_load] ERROR loading global.arrow: " +
                   tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    const auto &tensors = tensors_res->tensors;

    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // FP16 globals: embed + UNTIED lm_head + final norm. All three are raw
    // BF16 records converted at upload; none is quantized.
    auto *embed = FindTensor(tensors, "embed_tokens.weight");
    auto *head = FindTensor(tensors, "lm_head.weight");
    auto *norm = FindTensor(tensors, "model.language_model.norm.weight");
    if (!embed || embed->rank != 2 || embed->shape[0] <= 0 ||
        embed->shape[1] <= 0) {
        log::error("[muse_dense_load] embed_tokens.weight missing/malformed "
                   "in global.arrow");
        return cudaErrorInvalidValue;
    }
    if (!head || head->rank != 2 || head->shape[0] != embed->shape[0] ||
        head->shape[1] != embed->shape[1]) {
        log::error("[muse_dense_load] lm_head.weight missing or shape-mismatched "
                   "in global.arrow (the head is untied but vocab/hidden must "
                   "match the embed)");
        return cudaErrorInvalidValue;
    }
    if (!norm || norm->rank != 1 || norm->shape[0] != embed->shape[1]) {
        log::error("[muse_dense_load] final norm missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }

    std::vector<lsxcommon::TensorView> views;
    views.reserve(3);
    auto make_job = [&](const lsxcommon::TensorView *t, void **dst) {
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
        make_job(embed, reinterpret_cast<void **>(&pw.d_embed)),
        make_job(head, reinterpret_cast<void **>(&pw.d_lm_head)),
        make_job(norm, reinterpret_cast<void **>(&pw.d_final_norm)),
    };
    void *arena = nullptr;
    size_t arena_bytes = 0;
    const cudaError_t e = fastload::RunUploadJobs(
        std::move(jobs), &arena, &arena_bytes, stream, ring, bytes_acc, nullptr);
    if (e != cudaSuccess) return e;
    pw.global_arena = arena;
    pw.global_arena_bytes = arena_bytes;
    pw.vocab_size = static_cast<int>(embed->shape[0]);
    pw.hidden = static_cast<int>(embed->shape[1]);
    return cudaSuccess;
}

cudaError_t LoadLayer(const std::vector<lsxcommon::TensorView> &tensors,
                      MuseDenseLayerWeights &lw, cudaStream_t stream,
                      std::uint64_t &bytes_acc)
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
    const Nvfp4Recs ag = gather("self_attn.gate_proj.weight");
    const Nvfp4Recs gate = gather("mlp.gate_proj.weight");
    const Nvfp4Recs up = gather("mlp.up_proj.weight");
    const Nvfp4Recs down = gather("mlp.down_proj.weight");

    // ALL EIGHT triples are mandatory on EVERY layer — a missing attention
    // gate would silently decode without the sigmoid output gate.
    if (!complete(q) || !complete(k) || !complete(v) || !complete(o) ||
        !complete(ag) || !complete(gate) || !complete(up) || !complete(down)) {
        log::error("[muse_dense_load] layer is missing an NVFP4 projection "
                   "triple (packed + scales + global_scale)");
        return cudaErrorInvalidValue;
    }
    cudaError_t e = upload_nv(q, lw.attn_q);
    if (e == cudaSuccess) e = upload_nv(k, lw.attn_k);
    if (e == cudaSuccess) e = upload_nv(v, lw.attn_v);
    if (e == cudaSuccess) e = upload_nv(o, lw.attn_o);
    if (e == cudaSuccess) e = upload_nv(ag, lw.attn_gate);
    if (e == cudaSuccess) e = upload_nv(gate, lw.mlp_gate);
    if (e == cudaSuccess) e = upload_nv(up, lw.mlp_up);
    if (e == cudaSuccess) e = upload_nv(down, lw.mlp_down);
    if (e != cudaSuccess) return e;

    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_attn = FindTensor(tensors, "post_attention_layernorm.weight");
    auto *pre_ffn = FindTensor(tensors, "pre_feedforward_layernorm.weight");
    auto *post_ffn = FindTensor(tensors, "post_feedforward_layernorm.weight");
    if (!in_norm || !post_attn || !pre_ffn || !post_ffn) {
        log::error("[muse_dense_load] layer missing norm tensors");
        return cudaErrorInvalidValue;
    }

    std::vector<lsxcommon::TensorView> views;
    views.reserve(4);
    auto make_job = [&](const lsxcommon::TensorView *t, void **dst) {
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
        make_job(post_attn, reinterpret_cast<void **>(&lw.d_post_attn_norm)),
        make_job(pre_ffn, reinterpret_cast<void **>(&lw.d_pre_ffn)),
        make_job(post_ffn, reinterpret_cast<void **>(&lw.d_post_ffn)),
    };
    void *arena = nullptr;
    size_t arena_bytes = 0;
    e = fastload::RunUploadJobs(std::move(jobs), &arena, &arena_bytes, stream,
                                ring, bytes_acc, nullptr);
    lw.d_arena = arena;
    lw.arena_bytes = arena_bytes;
    return e;
}

} // namespace

void MuseDenseLayerWeights::Free()
{
    if (d_arena) cudaFree(d_arena);
    attn_q.Free();
    attn_k.Free();
    attn_v.Free();
    attn_o.Free();
    attn_gate.Free();
    mlp_gate.Free();
    mlp_up.Free();
    mlp_down.Free();
    *this = MuseDenseLayerWeights{};
}

void MuseDensePinnedWeights::Free()
{
    if (global_arena) cudaFree(global_arena);
    global_arena = nullptr;
    global_arena_bytes = 0;
    d_embed = nullptr;
    d_lm_head = nullptr;
    d_final_norm = nullptr;
    vocab_size = 0;
    hidden = 0;
    resident_bytes = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
}

cudaError_t LoadMuseDensePinnedWeights(const std::string &model_path,
                                       MuseDensePinnedWeights &pw,
                                       cudaStream_t stream)
{
    pw.Free();

    fs::path pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir) && fs::exists(fs::path(model_path) / "pinned"))
        pinned_dir = fs::path(model_path) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[muse_dense_load] pinned dir not found under " +
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
        log::error("[muse_dense_load] no layer_*.arrow files in " +
                   pinned_dir.string());
        return cudaErrorInvalidValue;
    }
    pw.layers.assign(static_cast<size_t>(max_layer + 1), MuseDenseLayerWeights{});

    for (int l = 0; l <= max_layer; ++l) {
        const auto path = pinned_dir / absl::StrCat("layer_", l, ".arrow");
        // Strict: a gap in the layer range is a hard error, never a silently
        // zero-weight layer.
        if (!fs::exists(path)) {
            log::error("[muse_dense_load] missing " + path.string());
            return cudaErrorInvalidValue;
        }
        auto tensors_res = LoadArrowFile(path.string());
        if (!tensors_res.ok()) {
            log::error("[muse_dense_load] ERROR loading " + path.string() +
                       ": " + tensors_res.status().ToString());
            return cudaErrorInvalidValue;
        }
        e = LoadLayer(tensors_res->tensors, pw.layers[static_cast<size_t>(l)],
                      stream, bytes_acc);
        if (e != cudaSuccess) return e;
    }

    pw.resident_bytes = bytes_acc;

    // Topology from the index. Mandatory for generation; a parse failure is
    // logged and surfaces through ValidateGeometry.
    auto idx_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(idx_path))
        idx_path = fs::path(model_path) / "model.index.json";
    if (fs::exists(idx_path)) {
        pw.top = load_muse_dense_topology(idx_path.string());
        if (!pw.top.error_msg.empty())
            log::error("[muse_dense_load] index topology parse failed: " +
                       pw.top.error_msg);
    }
    return cudaSuccess;
}

cudaError_t MuseDenseKvCache::Allocate(const MuseDensePinnedWeights &pw,
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
        // end-to-end number says otherwise). UNIFORM geometry: every layer
        // stores num_key_value_heads x head_dim; the sliding/full difference
        // is the attention window applied by the kernel, not the cache shape.
        const int64_t kv_heads = pw.top.num_key_value_heads;
        const int64_t head_dim = pw.top.head_dim;
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

void MuseDenseKvCache::Free()
{
    for (auto *p : dk) if (p) cudaFree(p);
    for (auto *p : dv) if (p) cudaFree(p);
    dk.clear();
    dv.clear();
    geo.clear();
    max_total = 0;
    cache_bytes = 0;
}

} // namespace muse_dense
} // namespace lsxcommon