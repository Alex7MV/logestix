#include "lsxcommon/granite_dense_load.h"

#include "lsxcommon/granite_dense_convert.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/schema.h"

#include <arrow/api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"

namespace lsxcommon {
namespace granite {

void GraniteLayerWeights::Free()
{
    if (d_arena) {
        cudaFree(d_arena);
        d_arena = nullptr;
        arena_bytes = 0;
        d_input_norm = nullptr;
        d_post_norm = nullptr;
        q.Free(); k.Free(); v.Free(); o.Free();
        mlp_gate.Free(); mlp_up.Free(); mlp_down.Free(); mlp_gate_up.Free();
        norm_size = 0;
        loaded = false;
        return;
    }
    cudaFree(d_input_norm); d_input_norm = nullptr;
    cudaFree(d_post_norm);  d_post_norm = nullptr;
    q.Free(); k.Free(); v.Free(); o.Free();
    mlp_gate.Free(); mlp_up.Free(); mlp_down.Free(); mlp_gate_up.Free();
    norm_size = 0;
    loaded = false;
}

void GranitePinnedWeights::Free()
{
    embed.Free();
    head.Free();
    cudaFree(d_final_norm); d_final_norm = nullptr;
    final_norm_size = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
    resident_bytes = 0;
}

namespace {

using fastload::CvKind;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::RunUploadJobs;
using fastload::StreamDrain;
using fastload::UploadJob;
using fastload::UploadNvfp4Weight;

CvKind NormKind(const TensorView *t)
{
    return (t && t->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
}

cudaError_t LoadLayer(const TensorBatchViews &raw,
                      const DenseNvfp4BatchViews &nv, GraniteLayerWeights &lw,
                      cudaStream_t stream, fastload::PinnedStagingRing &ring,
                      std::uint64_t &bytes_acc)
{
    auto *in_norm = FindTensorSuffix(raw, "input_layernorm.weight");
    if (in_norm && in_norm->rank > 0)
        lw.norm_size = static_cast<int>(in_norm->shape[0]);
    auto *post_norm = FindTensorSuffix(raw, "post_attention_layernorm.weight");

    std::vector<UploadJob> jobs = {
        {in_norm, reinterpret_cast<void **>(&lw.d_input_norm),
         NormKind(in_norm)},
        {post_norm, reinterpret_cast<void **>(&lw.d_post_norm),
         NormKind(post_norm)},
    };
    cudaError_t e = RunUploadJobs(std::move(jobs), &lw.d_arena,
                                  &lw.arena_bytes, stream, ring, bytes_acc,
                                  nullptr);
    if (e != cudaSuccess) return e;

    e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "self_attn.q_proj.weight"), lw.q,
                          stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "self_attn.k_proj.weight"),
                              lw.k, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "self_attn.v_proj.weight"),
                              lw.v, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "self_attn.o_proj.weight"),
                              lw.o, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "mlp.gate_proj.weight"),
                              lw.mlp_gate, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "mlp.up_proj.weight"),
                              lw.mlp_up, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "mlp.down_proj.weight"),
                              lw.mlp_down, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "mlp.gate_up_proj.weight"),
                              lw.mlp_gate_up, stream, ring, bytes_acc, nullptr);
    if (e != cudaSuccess) return e;

    if (lw.mlp_gate_up.d_packed) {
        // Fused [2I, K] record: derive non-owning gate/up row-slices. NVFP4
        // [N,K] nibble rows are contiguous, so the byte split at I rows is
        // exact for both packed data and per-16 scales.
        const int64_t I = lw.mlp_gate_up.rows / 2;
        const int64_t gate_packed = lw.mlp_gate_up.packed_bytes / 2;
        const int64_t gate_scales = lw.mlp_gate_up.scale_bytes / 2;
        auto make_view = [&](lsx_nvfp4::LsxNvfp4Weight &w,
                             const uint8_t *packed, const uint8_t *scales) {
            w.d_packed = packed;
            w.d_scales = scales;
            w.global_scale = lw.mlp_gate_up.global_scale;
            w.rows = static_cast<int>(I);
            w.cols = lw.mlp_gate_up.cols;
            w.packed_bytes = gate_packed;
            w.scale_bytes = gate_scales;
            w.view_only = true;
        };
        make_view(lw.mlp_gate, lw.mlp_gate_up.d_packed,
                  lw.mlp_gate_up.d_scales);
        make_view(lw.mlp_up, lw.mlp_gate_up.d_packed + gate_packed,
                  lw.mlp_gate_up.d_scales + gate_scales);
    }

    const bool attn_ok =
        lw.q.d_packed && lw.k.d_packed && lw.v.d_packed && lw.o.d_packed;
    const bool mlp_ok = lw.mlp_gate.d_packed && lw.mlp_up.d_packed &&
                        lw.mlp_down.d_packed;
    lw.loaded = attn_ok && mlp_ok && lw.d_input_norm && lw.d_post_norm;
    if (!lw.loaded)
        log::error(
            "[granite_load] layer weights incomplete; layer marked not loaded");
    return cudaSuccess;
}

} // namespace

bool LoadGranitePinnedWeights(const std::string &output_root,
                              GranitePinnedWeights &pw, cudaStream_t stream)
{
    namespace fs = std::filesystem;

    pw.Free();
    const auto t0 = std::chrono::steady_clock::now();

    auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir))
        pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[granite_load] no pinned dir under " + output_root);
        return false;
    }

    auto idx = fs::path(output_root) / "logestix" / "model.index.json";
    if (fs::exists(idx)) {
        pw.top = load_granite_topology(idx.string());
        if (!pw.top.error_msg.empty()) {
            log::error("[granite_load] topology error in " + idx.string() +
                       ": " + pw.top.error_msg);
            return false;
        }
    }

    StreamDrain drain{stream};

    fastload::PinnedStagingRing ring;
    if (!ring.ok()) return false;
    std::uint64_t &bytes_acc = pw.resident_bytes;
    bytes_acc = 0;

    // Global raw (final norm).
    if (fs::exists(pinned_dir / "global.arrow")) {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw_res.ok()) {
            log::error("[granite_load] global.arrow read failed: " +
                       raw_res.status().ToString());
            pw.Free();
            return false;
        }
        auto *fn = FindTensorSuffix(*raw_res, "norm.weight");
        if (fn && fn->rank > 0) {
            const int64_t n = fn->shape[0];
            pw.final_norm_size = static_cast<int>(n);
            cudaError_t e = cudaMalloc(
                &pw.d_final_norm, static_cast<size_t>(n) * sizeof(__half));
            if (e != cudaSuccess) { pw.Free(); return false; }
            if (fn->dtype == "F32") {
                std::vector<__half> h(static_cast<size_t>(n));
                const float *src = reinterpret_cast<const float *>(fn->data);
                for (int64_t i = 0; i < n; ++i)
                    h[static_cast<size_t>(i)] = __float2half(src[i]);
                e = cudaMemcpyAsync(pw.d_final_norm, h.data(),
                                    static_cast<size_t>(n) * sizeof(__half),
                                    cudaMemcpyHostToDevice, stream);
            } else {
                e = ring.UploadBf16AsFp16(fn->data, n, pw.d_final_norm, stream,
                                          bytes_acc);
            }
            if (e != cudaSuccess) { pw.Free(); return false; }
        }
    }

    // Global NVFP4 (embed + head).
    if (fs::exists(pinned_dir / "global_nvfp4.arrow")) {
        auto nv_res =
            LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
        if (!nv_res.ok()) {
            log::error("[granite_load] global_nvfp4.arrow read failed: " +
                       nv_res.status().ToString());
            pw.Free();
            return false;
        }
        cudaError_t e = UploadNvfp4Weight(
            FindNvfp4Suffix(*nv_res, "embed_tokens.weight"), pw.embed, stream,
            ring, bytes_acc, nullptr);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv_res, "lm_head.weight"),
                                  pw.head, stream, ring, bytes_acc, nullptr);
        if (e != cudaSuccess) { pw.Free(); return false; }
    }

    int num_layers = pw.top.num_layers;
    if (num_layers <= 0) {
        for (auto &entry : fs::directory_iterator(pinned_dir)) {
            auto fn = entry.path().filename().string();
            if (fn.rfind("layer_", 0) == 0 &&
                fn.find("_nvfp4") == std::string::npos) {
                int l = 0;
                const std::string layer_str = fn.substr(6, fn.find('.') - 6);
                if (!absl::SimpleAtoi(layer_str, &l)) continue;
                num_layers = std::max(num_layers, l + 1);
            }
        }
    }

    pw.layers.resize(static_cast<size_t>(num_layers));
    for (int l = 0; l < num_layers; ++l) {
        std::string raw_name = absl::StrCat("layer_", l, ".arrow");
        std::string nv_name = absl::StrCat("layer_", l, "_nvfp4.arrow");
        if (!fs::exists(pinned_dir / raw_name)) continue;

        auto raw = LoadTensorBatchViews(pinned_dir / raw_name);
        if (!raw.ok()) {
            log::error("[granite_load] " + raw_name + " read failed: " +
                       raw.status().ToString());
            continue;
        }
        DenseNvfp4BatchViews nv;
        if (fs::exists(pinned_dir / nv_name)) {
            auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / nv_name);
            if (nv_res.ok()) nv = std::move(*nv_res);
        }

        cudaError_t lerr = LoadLayer(*raw, nv, pw.layers[static_cast<size_t>(l)],
                                     stream, ring, bytes_acc);
        if (lerr != cudaSuccess) {
            pw.Free();
            return false;
        }
    }

    pw.resident_bytes = bytes_acc;
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    log::info(absl::StrCat("[granite_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms"));
    return true;
}

} // namespace granite
} // namespace lsxcommon
