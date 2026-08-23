#include "lsxcommon/kimi_inference_load.h"
#include "lsxcommon/kimi_convert.h"
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
#include <string>
#include <vector>
#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"

namespace fs = std::filesystem;

namespace lsxcommon {
namespace kimi {

namespace {

using fastload::PinnedStagingRing;

constexpr size_t kArenaAlign = 256;

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};

const TensorView *FindTensor(const TensorBatchViews &tensors,
                             std::string_view suffix)
{
    for (const auto &t : tensors.tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(),
                           suffix.size(), suffix) == 0)
            return &t;
    }
    return nullptr;
}

cudaError_t LoadGlobal(const fs::path &pinned_dir, KimiPinnedWeights &pw,
                       cudaStream_t stream, std::uint64_t &bytes_acc,
                       PinnedStagingRing &ring)
{
    auto tensors_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
    if (!tensors_res.ok()) {
        log::error("[kimi_load] ERROR loading global.arrow: "
                   + tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    auto &tensors = *tensors_res;

    auto *embed = FindTensor(tensors, "embed_tokens.weight");
    if (embed && embed->rank >= 2 &&
        embed->bytes >=
            static_cast<size_t>(embed->shape[0]) *
                static_cast<size_t>(embed->shape[1]) * 2) {
        pw.vocab_size = static_cast<int>(embed->shape[0]);
        pw.hidden = static_cast<int>(embed->shape[1]);
        cudaError_t e = cudaMalloc(&pw.d_embed,
            static_cast<size_t>(pw.vocab_size) * pw.hidden * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(embed->data,
                                  static_cast<int64_t>(pw.vocab_size) *
                                      pw.hidden,
                                  pw.d_embed, stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[kimi_load] embed upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    } else {
        log::error("[kimi_load] embed_tokens.weight missing/malformed in "
                   "global.arrow — model cannot run without it");
        return cudaErrorInvalidValue;
    }
    auto *head = FindTensor(tensors, "lm_head.weight");
    if (head && head->rank >= 2 && head != embed &&
        head->bytes >=
            static_cast<size_t>(head->shape[0]) *
                static_cast<size_t>(head->shape[1]) * 2) {
        int64_t numel = head->shape[0] * head->shape[1];
        cudaError_t e = cudaMalloc(&pw.d_head, static_cast<size_t>(numel) * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(head->data, numel, pw.d_head,
                                  stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[kimi_load] head upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    }
    auto *norm = FindTensor(tensors, "norm.weight");
    if (!norm) norm = FindTensor(tensors, "model.norm.weight");
    if (norm && norm->rank == 1 && norm->shape[0] > 0 &&
        norm->bytes >= static_cast<size_t>(norm->shape[0]) * 2) {
        cudaError_t e = cudaMalloc(&pw.d_final_norm,
                                   static_cast<size_t>(norm->shape[0]) * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(norm->data, norm->shape[0],
                                  pw.d_final_norm, stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[kimi_load] model.norm upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    } else {
        log::error("[kimi_load] norm.weight missing/malformed in "
                   "global.arrow — final RMSNorm cannot run without it");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

struct Fp16Job {
    const TensorView *t;
    __half **dst;
    int64_t *rows;   // optional row capture (2D only)
    int64_t *cols;   // optional col capture (2D only)
};

struct JobShape {
    bool is_2d = false;
    size_t numel = 0;
};

cudaError_t ValidateJob(const Fp16Job &j, JobShape &sh)
{
    if (j.t->rank == 1 && j.t->shape[0] > 0) {
        sh.is_2d = false;
        sh.numel = static_cast<size_t>(j.t->shape[0]);
    } else if (j.t->rank == 2 && j.t->shape[0] > 0 && j.t->shape[1] > 0) {
        sh.is_2d = true;
        sh.numel = static_cast<size_t>(j.t->shape[0]) *
                   static_cast<size_t>(j.t->shape[1]);
    } else {
        log::error("[kimi_load] malformed tensor '" + std::string(j.t->name) + "'");
        return cudaErrorInvalidValue;
    }
    if (j.t->bytes < sh.numel * 2) {
        log::error("[kimi_load] malformed tensor '" + std::string(j.t->name) + "'");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// Validates every job, issues ONE arena allocation sized to the layer's FP16
// footprint and uploads each tensor through the pinned ring into its
// 256-byte-aligned arena view.
cudaError_t UploadJobs(std::vector<Fp16Job> jobs, KimiLayerWeights &lw,
                       cudaStream_t stream, PinnedStagingRing &ring,
                       std::uint64_t &bytes_acc)
{
    std::vector<JobShape> shapes(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        cudaError_t e = ValidateJob(jobs[i], shapes[i]);
        if (e != cudaSuccess) return e;
    }

    size_t total = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        total = AlignUp(total, kArenaAlign) + shapes[i].numel * 2;
    }

    void *arena = nullptr;
    if (total > 0) {
        cudaError_t e = cudaMalloc(&arena, total);
        if (e != cudaSuccess) {
            log::error("[kimi_load] arena alloc failed: " +
                       std::string(cudaGetErrorString(e)));
            return e;
        }
    }
    lw.d_arena = arena;
    lw.arena_bytes = total;

    size_t off = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto &j = jobs[i];
        if (!j.t) continue;
        off = AlignUp(off, kArenaAlign);
        __half *view = static_cast<__half *>(arena) + off / 2;
        *j.dst = view;
        if (shapes[i].is_2d && j.rows)
            { *j.rows = j.t->shape[0]; *j.cols = j.t->shape[1]; }
        cudaError_t e = ring.UploadBf16AsFp16(j.t->data,
                            static_cast<int64_t>(shapes[i].numel),
                            view, stream, bytes_acc);
        if (e != cudaSuccess) return e;
        off += shapes[i].numel * 2;
    }
    return cudaSuccess;
}

} // anonymous namespace

cudaError_t LoadSingleKimiLayer(const TensorBatchViews &tensors,
                                KimiLayerWeights &lw, cudaStream_t stream,
                                std::uint64_t &bytes_acc,
                                PinnedStagingRing &ring)
{
    // Required-for-type validation (uniform grid: layer N type by content).
    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[kimi_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }
    const bool moe = FindTensor(tensors, "mlp.gate.weight") != nullptr;
    if (!moe && !FindTensor(tensors, "mlp.gate_proj.weight")) {
        log::error("[kimi_load] dense layer missing mlp.gate_proj");
        return cudaErrorInvalidValue;
    }

    std::vector<Fp16Job> jobs = {
        {in_norm,   &lw.d_input_norm,     nullptr, nullptr},
        {post_norm, &lw.d_post_attn_norm, nullptr, nullptr},
        {FindTensor(tensors, "self_attn.q_a_proj.weight"),
                    &lw.d_q_a_proj,       &lw.q_a_rows, &lw.q_a_cols},
        {FindTensor(tensors, "self_attn.q_a_layernorm.weight"),
                    &lw.d_q_a_norm,       nullptr, nullptr},
        {FindTensor(tensors, "self_attn.q_b_proj.weight"),
                    &lw.d_q_b_proj,       &lw.q_b_rows, &lw.q_b_cols},
        {FindTensor(tensors, "self_attn.kv_a_proj_with_mqa.weight"),
                    &lw.d_kv_a_proj,      &lw.kv_a_rows, &lw.kv_a_cols},
        {FindTensor(tensors, "self_attn.kv_a_layernorm.weight"),
                    &lw.d_kv_a_norm,      nullptr, nullptr},
        {FindTensor(tensors, "self_attn.kv_b_proj.weight"),
                    &lw.d_kv_b_proj,      &lw.kv_b_rows, &lw.kv_b_cols},
        {FindTensor(tensors, "self_attn.o_proj.weight"),
                    &lw.d_o_proj,         &lw.o_rows, &lw.o_cols},
    };

    if (moe) {
        jobs.push_back({FindTensor(tensors, "mlp.gate.weight"),
                        &lw.d_router_gate, nullptr, nullptr});
    } else {
        jobs.push_back({FindTensor(tensors, "mlp.gate_proj.weight"),
                        &lw.d_mlp_gate, nullptr, nullptr});
        jobs.push_back({FindTensor(tensors, "mlp.down_proj.weight"),
                        &lw.d_mlp_down, nullptr, nullptr});
        jobs.push_back({FindTensor(tensors, "mlp.up_proj.weight"),
                        &lw.d_mlp_up, nullptr, nullptr});
    }

    cudaError_t e = UploadJobs(std::move(jobs), lw, stream, ring, bytes_acc);
    if (e != cudaSuccess) return e;

    auto *router = FindTensor(tensors, "mlp.gate.weight");
    if (router) {
        auto *gate_bias =
            FindTensor(tensors, "mlp.gate.e_score_correction_bias");
        if (gate_bias && gate_bias->rank == 1 &&
            gate_bias->bytes >=
                static_cast<size_t>(gate_bias->shape[0]) * 4) {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias,
                               static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data,
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }
    }
    return cudaSuccess;
}

void KimiLayerWeights::Free()
{
    if (h_gate_bias) cudaFreeHost(h_gate_bias);
    h_gate_bias = nullptr;
    gate_bias_n = 0;
    if (d_arena) {
        cudaFree(d_arena);
        d_arena = nullptr;
        arena_bytes = 0;
        d_input_norm = d_post_attn_norm = nullptr;
        d_q_a_proj = d_q_a_norm = d_q_b_proj = nullptr;
        d_kv_a_proj = d_kv_a_norm = d_kv_b_proj = d_o_proj = nullptr;
        d_router_gate = nullptr;
        d_mlp_gate = d_mlp_down = d_mlp_up = nullptr;
        q_a_rows = q_a_cols = q_b_rows = q_b_cols = 0;
        kv_a_rows = kv_a_cols = kv_b_rows = kv_b_cols = 0;
        o_rows = o_cols = 0;
        return;
    }
    if (d_input_norm)     cudaFree(d_input_norm);
    if (d_post_attn_norm) cudaFree(d_post_attn_norm);
    if (d_q_a_proj)       cudaFree(d_q_a_proj);
    if (d_q_a_norm)       cudaFree(d_q_a_norm);
    if (d_q_b_proj)       cudaFree(d_q_b_proj);
    if (d_kv_a_proj)      cudaFree(d_kv_a_proj);
    if (d_kv_a_norm)      cudaFree(d_kv_a_norm);
    if (d_kv_b_proj)      cudaFree(d_kv_b_proj);
    if (d_o_proj)         cudaFree(d_o_proj);
    if (d_router_gate)    cudaFree(d_router_gate);
    if (d_mlp_gate)       cudaFree(d_mlp_gate);
    if (d_mlp_down)       cudaFree(d_mlp_down);
    if (d_mlp_up)         cudaFree(d_mlp_up);
    d_input_norm = d_post_attn_norm = nullptr;
    d_q_a_proj = d_q_a_norm = d_q_b_proj = nullptr;
    d_kv_a_proj = d_kv_a_norm = d_kv_b_proj = d_o_proj = nullptr;
    d_router_gate = nullptr;
    d_mlp_gate = d_mlp_down = d_mlp_up = nullptr;
}

void KimiPinnedWeights::Free()
{
    if (d_embed)      cudaFree(d_embed);
    if (d_head)       cudaFree(d_head);
    if (d_final_norm) cudaFree(d_final_norm);
    d_embed = d_head = d_final_norm = nullptr;
    vocab_size = 0;
    hidden = 0;
    resident_bytes = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
    shared_host.clear();
    shared_keepalive.reset();
}

cudaError_t LoadKimiPinnedWeights(const std::string &model_path,
                                  KimiPinnedWeights &pw,
                                  cudaStream_t stream)
{
    namespace fs = std::filesystem;

    // Idempotent re-load: drop any previously uploaded set first.
    pw.Free();

    const auto t0 = std::chrono::steady_clock::now();

    // Accept both the logestix root itself and its container directory.
    fs::path pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir) && fs::exists(fs::path(model_path) / "pinned"))
        pinned_dir = fs::path(model_path) / "pinned";
    fs::path index_path = fs::path(model_path) / "logestix" / "model.index.json";

    auto topo = kimi_load_topology(index_path.string());
    if (!topo.error_msg.empty()) {
        log::error("[kimi_load] topology error: " + topo.error_msg);
        return cudaErrorInvalidValue;
    }
    if (topo.num_layers <= 0) {
        log::error("[kimi_load] index has no layer count: " +
                   index_path.string());
        return cudaErrorInvalidValue;
    }
    pw.layers.resize(static_cast<size_t>(topo.num_layers));

    // Drains the stream on every error exit below so pinned staging never
    // dies while a DMA transfer is still in flight.
    StreamDrain drain{stream};

    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    auto err = LoadGlobal(pinned_dir, pw, stream, pw.resident_bytes, ring);
    if (err != cudaSuccess) return err;

    for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
        auto fn = absl::StrCat("layer_", l, ".arrow");
        auto layer_path = pinned_dir / fn;
        if (!fs::exists(layer_path)) {
            log::error("[kimi_load] required layer file missing: " + fn);
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto tensors_res = LoadTensorBatchViews(layer_path);
        if (!tensors_res.ok()) {
            log::error("[kimi_load] ERROR loading " + fn + ": "
                       + tensors_res.status().ToString());
            pw.Free();
            return cudaErrorInvalidValue;
        }
        cudaError_t lerr =
            LoadSingleKimiLayer(*tensors_res, pw.layers[static_cast<size_t>(l)],
                                stream, pw.resident_bytes, ring);
        if (lerr != cudaSuccess) {
            pw.Free();
            return lerr;
        }
    }

    // Shared experts: one NVFP4-packed shared expert per MoE layer in
    // pinned/shared_experts.arrow. Zero-copy mmap views kept host-side.
    pw.shared_host.assign(pw.layers.size(), PackedExpertView{});
    auto se_path = pinned_dir / "shared_experts.arrow";
    if (fs::exists(se_path)) {
        auto se_res = LoadSharedExpertBatchViews(se_path);
        if (se_res.ok()) {
            pw.shared_keepalive = se_res->keepalive;
            auto &sh = se_res->layers;
            for (size_t l = 0;
                 l < sh.size() && l < pw.layers.size(); ++l) {
                pw.shared_host[l] = sh[l];
            }
        } else {
            log::warn("[kimi_load] shared_experts.arrow read failed: " +
                      se_res.status().ToString());
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double gib =
        static_cast<double>(pw.resident_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double gibs = (ms > 0.0) ? (gib / (ms / 1000.0)) : 0.0;
    log::info(absl::StrCat("[kimi_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms (", gibs,
                           " GiB/s)"));

    return cudaSuccess;
}

// ── On-demand routed-expert index ──────────────────────────────────────────

KimiExpertDb load_kimi_expert_index(const std::string &model_path,
                                    int num_layers, int experts_per_layer)
{
    KimiExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.log_prefix = "[kimi_load] ";
    db.mmap_fail_verb = "expert mmap fail ";
    db.emit_diag = false;
    db.log_batch_fails = true;
    db.Load(model_path, /*fallback=*/true);
    return db;
}

void ensure_kimi_expert_resident(KimiExpertDb &db, int layer, int expert,
                                 cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_kimi_expert_index(KimiExpertDb &db) { db.Free(); }

} // namespace kimi
} // namespace lsxcommon
