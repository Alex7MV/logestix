#include "lsxcommon/phi_dense_load.h"

#include "lsxcommon/phi_dense_convert.h"
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
namespace phi {

void PhiLayerWeights::Free()
{
    if (d_arena_bf16) {
        cudaFree(d_arena_bf16);
        d_arena_bf16 = nullptr;
        arena_bytes_bf16 = 0;
        qkv_bf16 = gemv::DenseWeight{};
        o_bf16 = gemv::DenseWeight{};
        mlp_gate_up_bf16 = gemv::DenseWeight{};
        mlp_down_bf16 = gemv::DenseWeight{};
    }
    if (d_arena) {
        cudaFree(d_arena);
        d_arena = nullptr;
        arena_bytes = 0;
        d_input_norm = nullptr;
        d_post_norm = nullptr;
        qkv.Free(); q.Free(); k.Free(); v.Free(); o.Free();
        mlp_gate.Free(); mlp_up.Free(); mlp_down.Free(); mlp_gate_up.Free();
        qkv_res.Free(); o_res.Free();
        mlp_gate_up_res.Free(); mlp_down_res.Free();
        norm_size = 0;
        loaded = false;
        return;
    }
    cudaFree(d_input_norm); d_input_norm = nullptr;
    cudaFree(d_post_norm);  d_post_norm = nullptr;
    qkv.Free();
    q.Free(); k.Free(); v.Free(); o.Free();
    mlp_gate.Free(); mlp_up.Free(); mlp_down.Free(); mlp_gate_up.Free();
    qkv_res.Free(); o_res.Free();
    mlp_gate_up_res.Free(); mlp_down_res.Free();
    norm_size = 0;
    loaded = false;
}

void PhiPinnedWeights::Free()
{
    cudaFree(d_embed_f16); d_embed_f16 = nullptr;
    cudaFree(d_head_f16);  d_head_f16 = nullptr;
    embed_rows = embed_cols = head_rows = head_cols = 0;
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
                      const DenseNvfp4BatchViews &nv,
                      const PhiTopology &top, PhiLayerWeights &lw,
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

    // Fused q/k/v record (phi stores q/k/v pre-fused in the source).
    e = UploadNvfp4Weight(FindNvfp4Suffix(nv, "self_attn.qkv_proj.weight"),
                          lw.qkv, stream, ring, bytes_acc, nullptr);
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

    // ── The high-precision encoding (the `bf16` codec profile) ─────────────
    //
    // A bf16 tree has no NVFP4 records at all, so every upload above was a no-op
    // and lw.qkv.d_packed stayed null. The same four projections are then read
    // from the `raw` planes as BF16 and kept device-resident AS BF16
    // (CvKind::kBf16Raw is a passthrough: the value is stored, so there is
    // nothing to reconstruct). The inference pipeline dispatches to
    // gemv::RunLxGemmVec for these.
    if (!lw.qkv.d_packed) {   // an NVFP4 tree keeps the NVFP4 path only
        struct DenseJob {
            const char *suffix;
            gemv::DenseWeight *out;
        };
        const DenseJob candidates[] = {
            {"self_attn.qkv_proj.weight", &lw.qkv_bf16},
            {"self_attn.o_proj.weight", &lw.o_bf16},
            {"mlp.gate_up_proj.weight", &lw.mlp_gate_up_bf16},
            {"mlp.down_proj.weight", &lw.mlp_down_bf16},
        };
        // The job and its owner travel together: RunUploadJobs is free to skip
        // nothing, but a `continue` here would silently shift every later
        // pointer, and the shape patch below must describe the SAME tensor.
        std::vector<UploadJob> jobs_dense;
        std::vector<const DenseJob *> owners;
        for (const auto &c : candidates) {
            const auto *t = FindTensorSuffix(raw, c.suffix);
            if (t == nullptr || t->rank < 2) continue;
            jobs_dense.push_back({t, const_cast<void **>(&c.out->d_data),
                                  CvKind::kBf16Raw});
            owners.push_back(&c);
        }
        if (!jobs_dense.empty()) {
            e = RunUploadJobs(std::move(jobs_dense), &lw.d_arena_bf16,
                              &lw.arena_bytes_bf16, stream, ring, bytes_acc,
                              nullptr);
            if (e != cudaSuccess) return e;
            // rows/cols come from the tensor's own shape, so the GEMV cannot be
            // handed a transposed view of a [N,K] plane.
            for (const auto *owner : owners) {
                const auto *t = FindTensorSuffix(raw, owner->suffix);
                owner->out->rows = static_cast<int>(t->shape[0]);
                owner->out->cols = static_cast<int>(t->shape[1]);
                owner->out->dtype = gemv::LxDtype::kBF16;
            }
        }
    }

    // Residual planes (optional: absent on a legacy 4-bit-only tree). They are
    // uploaded with the same helper; a missing record leaves the weight null
    // and the decode simply skips the accumulate pass.
    auto upload_res = [&](const char *suffix, lsx_nvfp4::LsxNvfp4Weight &w) {
        const auto *view = FindNvfp4Suffix(nv, suffix);
        if (view == nullptr) return cudaSuccess;
        return UploadNvfp4Weight(view, w, stream, ring, bytes_acc, nullptr);
    };
    if ((e = upload_res("self_attn.qkv_proj.weight.res", lw.qkv_res)) !=
        cudaSuccess) return e;
    if ((e = upload_res("self_attn.o_proj.weight.res", lw.o_res)) != cudaSuccess)
        return e;
    if ((e = upload_res("mlp.gate_up_proj.weight.res", lw.mlp_gate_up_res)) !=
        cudaSuccess) return e;
    if ((e = upload_res("mlp.down_proj.weight.res", lw.mlp_down_res)) !=
        cudaSuccess) return e;

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

    if (lw.qkv.d_packed) {
        // q/k/v as non-owning row-slices of the fused [NH*HD + 2*NKV*HD, K]
        // record (NVFP4 rows are contiguous, so the byte split is exact).
        const int qn = top.num_attention_heads * top.head_dim;
        const int kn = top.num_key_value_heads * top.head_dim;
        const int64_t total_rows = static_cast<int64_t>(qn) + 2 * kn;
        const int64_t pbytes = lw.qkv.packed_bytes;
        const int64_t sbytes = lw.qkv.scale_bytes;
        const int64_t qp = pbytes * qn / total_rows;
        const int64_t kp = pbytes * kn / total_rows;
        const int64_t qs = sbytes * qn / total_rows;
        const int64_t ks = sbytes * kn / total_rows;
        auto view = [&](lsx_nvfp4::LsxNvfp4Weight &w, const uint8_t *p,
                        const uint8_t *s, int rows, int64_t pb, int64_t sb) {
            w.d_packed = p;
            w.d_scales = s;
            w.global_scale = lw.qkv.global_scale;
            w.rows = rows;
            w.cols = lw.qkv.cols;
            w.packed_bytes = pb;
            w.scale_bytes = sb;
            w.view_only = true;
        };
        view(lw.q, lw.qkv.d_packed, lw.qkv.d_scales, qn, qp, qs);
        view(lw.k, lw.qkv.d_packed + qp, lw.qkv.d_scales + qs, kn, kp, ks);
        view(lw.v, lw.qkv.d_packed + qp + kp, lw.qkv.d_scales + qs + ks, kn, kp,
             ks);
    }
    // "Loaded" is an ENCODING-AGNOSTIC statement: the projections this layer
    // needs are resident, in whichever encoding the artifact holds. Testing only
    // d_packed is what left every layer of a bf16 tree marked incomplete
    // (defect A3.13) while the weights were in fact there.
    const bool attn_ok =
        (lw.o.d_packed && lw.qkv.d_packed) ||
        (lw.o_bf16.valid() && lw.qkv_bf16.valid());
    const bool mlp_ok =
        (lw.mlp_gate.d_packed && lw.mlp_up.d_packed && lw.mlp_down.d_packed) ||
        (lw.mlp_gate_up_bf16.valid() && lw.mlp_down_bf16.valid());
    lw.loaded = attn_ok && mlp_ok && lw.d_input_norm && lw.d_post_norm;
    if (!lw.loaded)
        log::error(
            "[phi_load] layer weights incomplete; layer marked not loaded");
    return cudaSuccess;
}

} // namespace

bool LoadPhiPinnedWeights(const std::string &output_root,
                          PhiPinnedWeights &pw, cudaStream_t stream)
{
    namespace fs = std::filesystem;

    pw.Free();
    const auto t0 = std::chrono::steady_clock::now();

    auto pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir))
        pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[phi_load] no pinned dir under " + output_root);
        return false;
    }

    auto idx = fs::path(output_root) / "model.index.json";
    if (fs::exists(idx)) {
        pw.top = load_phi_topology(idx.string());
        if (!pw.top.error_msg.empty()) {
            log::error("[phi_load] topology error in " + idx.string() +
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
            log::error("[phi_load] global.arrow read failed: " +
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

    // Edges from the raw batch (FP16 path): embed + lm_head stay high
    // precision. A legacy tree keeps them in global_nvfp4.arrow instead, and
    // the block below then populates the NVFP4 members.
    if (fs::exists(pinned_dir / "global.arrow")) {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw_res.ok()) {
            log::error("[phi_load] global.arrow read failed: " +
                       raw_res.status().ToString());
            pw.Free();
            return false;
        }
        auto upload_edge = [&](const char *suffix, __half **dst, int &rows,
                               int &cols) -> cudaError_t {
            auto *t = FindTensorSuffix(*raw_res, suffix);
            if (t == nullptr || t->rank < 2) return cudaErrorNotSupported;
            rows = static_cast<int>(t->shape[0]);
            cols = static_cast<int>(t->shape[1]);
            const int64_t n = static_cast<int64_t>(rows) * cols;
            cudaError_t e = cudaMalloc(reinterpret_cast<void **>(dst),
                                       static_cast<size_t>(n) * sizeof(__half));
            if (e != cudaSuccess) return e;
            if (t->dtype == "F32") {
                std::vector<__half> h(static_cast<size_t>(n));
                const float *src = reinterpret_cast<const float *>(t->data);
                for (int64_t i = 0; i < n; ++i)
                    h[static_cast<size_t>(i)] = __float2half(src[i]);
                return cudaMemcpyAsync(*dst, h.data(),
                                       static_cast<size_t>(n) * sizeof(__half),
                                       cudaMemcpyHostToDevice, stream);
            }
            return ring.UploadBf16AsFp16(t->data, n, *dst, stream, bytes_acc);
        };
        cudaError_t ee = upload_edge("embed_tokens.weight", &pw.d_embed_f16,
                                     pw.embed_rows, pw.embed_cols);
        if (ee == cudaSuccess)
            ee = upload_edge("lm_head.weight", &pw.d_head_f16, pw.head_rows,
                             pw.head_cols);
        if (ee != cudaSuccess && ee != cudaErrorNotSupported) {
            pw.Free();
            return false;
        }
        if (pw.d_embed_f16 || pw.d_head_f16)
            log::info(absl::StrCat("[phi_load] edges FP16: embed ",
                                   pw.embed_rows, "x", pw.embed_cols,
                                   ", head ", pw.head_rows, "x",
                                   pw.head_cols));
    }

    // Global NVFP4 (embed + head) -- legacy trees only.
    if (!pw.d_embed_f16 && !pw.d_head_f16 &&
        fs::exists(pinned_dir / "global_nvfp4.arrow")) {
        auto nv_res =
            LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
        if (!nv_res.ok()) {
            log::error("[phi_load] global_nvfp4.arrow read failed: " +
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
            log::error("[phi_load] " + raw_name + " read failed: " +
                       raw.status().ToString());
            continue;
        }
        DenseNvfp4BatchViews nv;
        if (fs::exists(pinned_dir / nv_name)) {
            auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / nv_name);
            if (nv_res.ok()) nv = std::move(*nv_res);
        }

        cudaError_t lerr = LoadLayer(*raw, nv, pw.top,
                                     pw.layers[static_cast<size_t>(l)],
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
    log::info(absl::StrCat("[phi_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms"));
    return true;
}

} // namespace phi
} // namespace lsxcommon
