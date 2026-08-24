#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/log.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"

namespace lsxcommon::qwen {

// QwenNvfp4Weight::Free is defined inline in the header.

void QwenLayerWeights::Free() {
    if (d_arena) {
        cudaFree(d_arena);
        d_arena = nullptr;
        arena_bytes = 0;
        d_input_norm = nullptr; d_post_norm = nullptr;
        d_conv1d = nullptr; d_A_log = nullptr; d_dt_bias = nullptr;
        d_attn_norm = nullptr;
        d_q_norm = nullptr; d_k_norm = nullptr;
        qkv.Free(); z.Free(); b.Free(); a.Free(); out.Free();
        q.Free(); k.Free(); v.Free(); o.Free();
        mlp_gate.Free(); mlp_up.Free(); mlp_down.Free();
        norm_size = attn_norm_size = 0;
        conv_dim = conv_kernel = 0;
        q_norm_size = k_norm_size = 0;
        loaded = false;
        return;
    }
    cudaFree(d_input_norm); d_input_norm = nullptr;
    cudaFree(d_post_norm);  d_post_norm = nullptr;
    qkv.Free(); z.Free(); b.Free(); a.Free(); out.Free();
    cudaFree(d_conv1d);    d_conv1d = nullptr;
    cudaFree(d_A_log);     d_A_log = nullptr;
    cudaFree(d_dt_bias);   d_dt_bias = nullptr;
    cudaFree(d_attn_norm); d_attn_norm = nullptr;
    q.Free(); k.Free(); v.Free(); o.Free();
    cudaFree(d_q_norm); d_q_norm = nullptr;
    cudaFree(d_k_norm); d_k_norm = nullptr;
    mlp_gate.Free(); mlp_up.Free(); mlp_down.Free();
    loaded = false;
}

void QwenPinnedWeights::Free() {
    embed.Free();
    head.Free();
    cudaFree(d_final_norm); d_final_norm = nullptr;
    for (auto &l : layers) l.Free();
    layers.clear();
    resident_bytes = 0;
}

std::vector<uint8_t> ReadFileBytes(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    return buf;
}

namespace {

using fastload::CvKind;
using fastload::OutElemSize;
using fastload::UploadJob;
using fastload::kArenaAlign;
using fastload::AlignUp;
using fastload::StreamDrain;
using fastload::RunUploadJobs;
using fastload::UploadNvfp4Weight;

CvKind NormKind(const TensorView *t)
{
    return (t && t->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
}

// Dtype-aware kind for F32-destined tensors: honor BF16-on-disk records
// (e.g. A_log in real Qwen checkpoints) instead of reinterpreting bits.
CvKind CvKindFromDtype(const TensorView *t, CvKind f32_kind)
{
    return (t && t->dtype == "BF16") ? CvKind::kBf16ToF32 : f32_kind;
}

cudaError_t LoadLayer(const TensorBatchViews &raw,
                      const DenseNvfp4BatchViews &nv,
                      QwenLayerWeights &lw, cudaStream_t stream,
                      fastload::PinnedStagingRing &ring,
                      std::uint64_t &bytes_acc, bool is_linear)
{
    lw.is_linear = is_linear;

    auto *in_norm = fastload::FindTensorSuffix(raw, "input_layernorm.weight");
    if (in_norm && in_norm->rank > 0)
        lw.norm_size = static_cast<int>(in_norm->shape[0]);
    auto *post_norm = fastload::FindTensorSuffix(raw, "post_attention_layernorm.weight");

    std::vector<UploadJob> jobs = {
        {in_norm,   reinterpret_cast<void **>(&lw.d_input_norm),
                    NormKind(in_norm)},
        {post_norm, reinterpret_cast<void **>(&lw.d_post_norm),
                    NormKind(post_norm)},
    };

    if (is_linear) {
        auto *conv = fastload::FindTensorSuffix(raw, "linear_attn.conv1d.weight");
        if (conv && conv->rank > 0) {
            // nn.Conv1d weight is [C, 1, K] on disk; legacy fixtures store
            // [C, K]. The squeezed [C, K] layout is contiguous either way.
            lw.conv_dim = static_cast<int>(conv->shape[0]);
            lw.conv_kernel = conv->rank >= 3
                ? static_cast<int>(conv->shape[2])
                : static_cast<int>(conv->shape[conv->rank - 1]);
        }
        jobs.push_back({conv,
                        reinterpret_cast<void **>(&lw.d_conv1d),
                        CvKind::kBf16ToHalf});
        jobs.push_back({fastload::FindTensorSuffix(raw, "linear_attn.A_log"),
                        reinterpret_cast<void **>(&lw.d_A_log),
                        CvKindFromDtype(fastload::FindTensorSuffix(raw, "linear_attn.A_log"),
                                        CvKind::kF32Raw)});
        auto *dt = fastload::FindTensorSuffix(raw, "linear_attn.dt_bias");
        // dt_bias is BF16 on disk; kernels consume it as FP32.
        CvKind dt_kind = (!dt || dt->dtype == "BF16" || dt->dtype.empty())
                             ? CvKind::kBf16ToF32 : CvKind::kF32Raw;
        jobs.push_back({dt, reinterpret_cast<void **>(&lw.d_dt_bias), dt_kind});
        auto *an = fastload::FindTensorSuffix(raw, "linear_attn.norm.weight");
        if (an && an->rank > 0)
            lw.attn_norm_size = static_cast<int>(an->shape[0]);
        jobs.push_back({an, reinterpret_cast<void **>(&lw.d_attn_norm),
                        NormKind(an)});
    } else {
        auto *qn = fastload::FindTensorSuffix(raw, "self_attn.q_norm.weight");
        if (qn && qn->rank > 0)
            lw.q_norm_size = static_cast<int>(qn->shape[0]);
        auto *kn = fastload::FindTensorSuffix(raw, "self_attn.k_norm.weight");
        if (kn && kn->rank > 0)
            lw.k_norm_size = static_cast<int>(kn->shape[0]);
        jobs.push_back({qn, reinterpret_cast<void **>(&lw.d_q_norm),
                        NormKind(qn)});
        jobs.push_back({kn, reinterpret_cast<void **>(&lw.d_k_norm),
                        NormKind(kn)});
    }

    cudaError_t e = fastload::RunUploadJobs(std::move(jobs), &lw.d_arena,
                                  &lw.arena_bytes, stream, ring, bytes_acc, nullptr);
    if (e != cudaSuccess) return e;

    if (is_linear) {
        e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "linear_attn.in_proj_qkv.weight"), lw.qkv, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "linear_attn.in_proj_z.weight"), lw.z, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "linear_attn.in_proj_b.weight"), lw.b, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "linear_attn.in_proj_a.weight"), lw.a, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "linear_attn.out_proj.weight"), lw.out, stream, ring, bytes_acc, nullptr);
    } else {
        e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "self_attn.q_proj.weight"), lw.q, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "self_attn.k_proj.weight"), lw.k, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "self_attn.v_proj.weight"), lw.v, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "self_attn.o_proj.weight"), lw.o, stream, ring, bytes_acc, nullptr);
    }
    if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "mlp.gate_proj.weight"), lw.mlp_gate, stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "mlp.up_proj.weight"),   lw.mlp_up,   stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(nv, "mlp.down_proj.weight"), lw.mlp_down, stream, ring, bytes_acc, nullptr);
    if (e != cudaSuccess) return e;

    // Mark the layer usable only when its essential weights actually uploaded;
    // missing records previously left loaded=true and failed later as cryptic
    // GEMM null-pointer errors.
    const bool attn_ok = is_linear
        ? (lw.qkv.d_packed && lw.out.d_packed)
        : (lw.q.d_packed && lw.k.d_packed && lw.v.d_packed && lw.o.d_packed);
    const bool mlp_ok = lw.mlp_gate.d_packed && lw.mlp_up.d_packed &&
                        lw.mlp_down.d_packed;
    lw.loaded = attn_ok && mlp_ok && lw.d_input_norm && lw.d_post_norm;
    if (!lw.loaded)
        log::error("[qwen_load] layer weights incomplete; layer marked not loaded");
    return cudaSuccess;
}

} // namespace

bool LoadQwenPinnedWeights(const std::string &output_root,
                           QwenPinnedWeights &pw, cudaStream_t stream)
{
    namespace fs = std::filesystem;

    // Idempotent re-load: drop any previously uploaded set first (Free() is
    // safe on a nulled/empty set), so calling Load twice without an explicit
    // Free() in between cannot leak the old device allocations.
    pw.Free();

    const auto t0 = std::chrono::steady_clock::now();

    auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir))
        pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[qwen_load] no pinned dir under " + output_root);
        return false;
    }

    auto idx = fs::path(output_root) / "logestix" / "model.index.json";
    if (fs::exists(idx)) {
        pw.top = load_qwen_topology(idx.string());
        if (!pw.top.error_msg.empty()) {
            log::error("[qwen_load] topology error in " + idx.string()
                       + ": " + pw.top.error_msg);
            return false;
        }
    }

    // Drains the stream on every error exit below so pinned staging never
    // dies while a DMA transfer is still in flight.
    StreamDrain drain{stream};

    fastload::PinnedStagingRing ring;
    if (!ring.ok()) return false;
    std::uint64_t &bytes_acc = pw.resident_bytes;
    bytes_acc = 0;

    // Global raw (final norm).
    if (fs::exists(pinned_dir / "global.arrow")) {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw_res.ok()) {
            log::error("[qwen_load] global.arrow read failed: " +
                       raw_res.status().ToString());
            pw.Free();
            return false;
        }
        auto *fn = fastload::FindTensorSuffix(*raw_res, "norm.weight");
        if (fn && fn->rank > 0) {
            int64_t n = fn->shape[0];
            pw.final_norm_size = static_cast<int>(n);
            cudaError_t e = cudaMalloc(&pw.d_final_norm,
                                        static_cast<size_t>(n) * sizeof(__half));
            if (e != cudaSuccess) { pw.Free(); return false; }
            if (fn->dtype == "F32") {
                std::vector<__half> h(static_cast<size_t>(n));
                const float *src =
                    reinterpret_cast<const float *>(fn->data);
                for (int64_t i = 0; i < n; ++i)
                    h[static_cast<size_t>(i)] = __float2half(src[i]);
                e = cudaMemcpyAsync(pw.d_final_norm, h.data(),
                                    static_cast<size_t>(n) * sizeof(__half),
                                    cudaMemcpyHostToDevice, stream);
            } else {
                e = ring.UploadBf16AsFp16(fn->data, n, pw.d_final_norm,
                                          stream, bytes_acc);
            }
            if (e != cudaSuccess) { pw.Free(); return false; }
        }
    }
    // Global NVFP4 (embed + head).
    if (fs::exists(pinned_dir / "global_nvfp4.arrow")) {
        auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
        if (!nv_res.ok()) {
            log::error("[qwen_load] global_nvfp4.arrow read failed: " +
                       nv_res.status().ToString());
            pw.Free();
            return false;
        }
        cudaError_t e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv_res, "embed_tokens.weight"),
                                          pw.embed, stream, ring, bytes_acc, nullptr);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv_res, "lm_head.weight"),
                                  pw.head, stream, ring, bytes_acc, nullptr);
        if (e != cudaSuccess) { pw.Free(); return false; }
    }

    // Per-layer.
    int num_layers = pw.top.num_layers;
    if (num_layers <= 0) {
        // Infer from files present.
        for (auto &entry : fs::directory_iterator(pinned_dir)) {
            auto fn = entry.path().filename().string();
            if (fn.rfind("layer_", 0) == 0 && fn.find("_nvfp4") == std::string::npos) {
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
        std::string nv_name  = absl::StrCat("layer_", l, "_nvfp4.arrow");
        if (!fs::exists(pinned_dir / raw_name)) continue;

        auto raw = LoadTensorBatchViews(pinned_dir / raw_name);
        if (!raw.ok()) {
            log::error("[qwen_load] " + raw_name + " read failed: " +
                       raw.status().ToString());
            continue;
        }
        DenseNvfp4BatchViews nv;
        if (fs::exists(pinned_dir / nv_name)) {
            auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / nv_name);
            if (nv_res.ok()) nv = std::move(*nv_res);
        }

        bool is_linear = true;
        if (l < static_cast<int>(pw.top.layer_types.size()))
            is_linear = pw.top.layer_types[static_cast<size_t>(l)] == "linear_attention";

        cudaError_t lerr = LoadLayer(*raw, nv, pw.layers[static_cast<size_t>(l)],
                                     stream, ring, bytes_acc, is_linear);
        if (lerr != cudaSuccess) {
            pw.Free();
            return false;
        }
    }

    // Resident footprint = every byte uploaded through the ring / host paths
    // (global norm + embed/head + per-layer arenas + NVFP4 blocks).
    pw.resident_bytes = bytes_acc;

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double gib =
        static_cast<double>(pw.resident_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double gibs = (ms > 0.0) ? (gib / (ms / 1000.0)) : 0.0;
    log::info(absl::StrCat("[qwen_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms (", gibs,
                           " GiB/s)"));

    return true;
}

} // namespace lsxcommon::qwen