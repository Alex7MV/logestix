#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/log.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include "fast_load_internal.h"

namespace fs = std::filesystem;

namespace lsxcommon {
namespace kimi {

namespace {

using fastload::PinnedStagingRing;


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

    auto *embed = fastload::FindTensorSuffix(tensors, "embed_tokens.weight");
    if (embed && embed->rank >= 2 &&
        embed->bytes >=
            static_cast<size_t>(embed->shape[0]) *
                static_cast<size_t>(embed->shape[1]) * 2) {
        pw.vocab_size = static_cast<int>(embed->shape[0]);
        pw.hidden = static_cast<int>(embed->shape[1]);
        cudaError_t e = cudaMalloc(&pw.d_embed,
            static_cast<size_t>(pw.vocab_size) * pw.hidden * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadRaw(embed->data, pw.d_embed,
                                  static_cast<size_t>(pw.vocab_size) *
                                      pw.hidden * 2,
                                  stream, bytes_acc);
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
    auto *head = fastload::FindTensorSuffix(tensors, "lm_head.weight");
    if (head && head->rank >= 2 && head != embed &&
        head->bytes >=
            static_cast<size_t>(head->shape[0]) *
                static_cast<size_t>(head->shape[1]) * 2) {
        int64_t numel = head->shape[0] * head->shape[1];
        cudaError_t e = cudaMalloc(&pw.d_head, static_cast<size_t>(numel) * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadRaw(head->data, pw.d_head,
                                  static_cast<size_t>(numel) * 2,
                                  stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[kimi_load] head upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    }
    auto *norm = fastload::FindTensorSuffix(tensors, "norm.weight");
    if (!norm) norm = fastload::FindTensorSuffix(tensors, "model.norm.weight");
    if (norm && norm->rank == 1 && norm->shape[0] > 0 &&
        norm->bytes >= static_cast<size_t>(norm->shape[0]) * 2) {
        cudaError_t e = cudaMalloc(&pw.d_final_norm,
                                   static_cast<size_t>(norm->shape[0]) * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadRaw(norm->data, pw.d_final_norm,
                                  static_cast<size_t>(norm->shape[0]) * 2,
                                  stream, bytes_acc);
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

} // anonymous namespace

cudaError_t LoadSingleKimiLayer(
    const TensorBatchViews &tensors, KimiLayerWeights &lw,
    cudaStream_t stream, std::uint64_t &bytes_acc, PinnedStagingRing &ring,
    const absl::flat_hash_map<std::string, float> &mla_gs)
{
    // Required-for-type validation (uniform grid: layer N type by content).
    auto *in_norm = fastload::FindTensorSuffix(tensors, "input_layernorm.weight");
    auto *post_norm = fastload::FindTensorSuffix(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[kimi_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }
    const bool moe = fastload::FindTensorSuffix(tensors, "mlp.gate.weight") != nullptr;
    if (!moe && !fastload::FindTensorSuffix(tensors, "mlp.gate_proj.weight")) {
        log::error("[kimi_load] dense layer missing mlp.gate_proj");
        return cudaErrorInvalidValue;
    }

    using fastload::CvKind;
    auto bf16job = [&](const TensorView *t, void **dst,
                       int64_t *rows = nullptr, int64_t *cols = nullptr) {
        return fastload::UploadJob{t, dst, CvKind::kBf16Raw, rows, cols};
    };
    // Round-2: MLA projections arrive as NVFP4 packed/scale U8 pairs (the
    // converter quantized them); upload raw through the ring. Strict: both
    // records are required for every projection.
    auto nvfp4pair = [&](const char *proj, uint8_t **d_pk, uint8_t **d_sc,
                         float *gs_out, int64_t *pk_bytes,
                         int64_t *sc_bytes, int64_t *rows, int64_t *cols)
        -> cudaError_t {
        (void)rows;
        (void)cols;
        const auto *pk = fastload::FindTensorSuffix(
            tensors, absl::StrCat(proj, ".weight_packed"));
        const auto *sc = fastload::FindTensorSuffix(
            tensors, absl::StrCat(proj, ".weight_scale"));
        if (!pk || pk->bytes == 0 || !sc || sc->bytes == 0) {
            log::error(absl::StrCat("[kimi_load] MLA NVFP4 records missing: ",
                                    proj));
            return cudaErrorInvalidValue;
        }
        // Logical weight shape [rows, cols] derives from the packed payload:
        // packed bytes = rows*cols/2; the record shape is 1-D {packed}.
        if (rows && cols) {
            // rank-2 packed records (r2 stores 1-D {bytes}); derive cols
            // from the scale record when possible: scale bytes = rows*cols/16
            // -> cols = scale_bytes*16/rows with rows = packed*2/scale/16...
            // Keep it simple: r2 stores shape {packed_bytes}; the geometry
            // is compile-time frozen, so rows/cols come from the callers'
            // constants — capture the packed-byte counts only.
            (void)rows;
            (void)cols;
        }
        *pk_bytes = static_cast<int64_t>(pk->bytes);
        *sc_bytes = static_cast<int64_t>(sc->bytes);
        // Global scale from the index (mla_nvfp4[layer][base]); the
        // index keys are the classifier bases, e.g. "q_a_proj".
        const char *slash = std::strchr(proj, '.');
        std::string base_key(slash ? slash + 1 : proj);
        auto it = mla_gs.find(base_key);
        if (it == mla_gs.end()) {
            log::error(absl::StrCat("[kimi_load] no global scale for ",
                                    base_key));
            return cudaErrorInvalidValue;
        }
        *gs_out = it->second;
        // Device allocations are individual (not arena views): the arena
        // uploader only handles the fixed-size UploadJob table.
        if (cudaMalloc(d_pk, pk->bytes) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        if (cudaMalloc(d_sc, sc->bytes) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        cudaError_t e = ring.UploadRaw(pk->data, *d_pk, pk->bytes, stream,
                                       bytes_acc);
        if (e != cudaSuccess) return e;
        return ring.UploadRaw(sc->data, *d_sc, sc->bytes, stream, bytes_acc);
    };
    std::vector<fastload::UploadJob> jobs = {
        bf16job(in_norm,   reinterpret_cast<void **>(&lw.d_input_norm)),
        bf16job(post_norm, reinterpret_cast<void **>(&lw.d_post_attn_norm)),
    };
    {
        cudaError_t e = nvfp4pair(
            "self_attn.q_a_proj", &lw.d_q_a_packed, &lw.d_q_a_scale, &lw.q_a_gs,
            &lw.q_a_packed_bytes, &lw.q_a_scale_bytes, &lw.q_a_rows,
            &lw.q_a_cols);
        if (e != cudaSuccess) return e;
        e = nvfp4pair("self_attn.q_b_proj", &lw.d_q_b_packed, &lw.d_q_b_scale,
                      &lw.q_b_gs, &lw.q_b_packed_bytes, &lw.q_b_scale_bytes,
                      &lw.q_b_rows, &lw.q_b_cols);
        if (e != cudaSuccess) return e;
        e = nvfp4pair("self_attn.kv_a_proj_with_mqa", &lw.d_kv_a_packed,
                      &lw.d_kv_a_scale, &lw.kv_a_gs, &lw.kv_a_packed_bytes,
                      &lw.kv_a_scale_bytes, &lw.kv_a_rows, &lw.kv_a_cols);
        if (e != cudaSuccess) return e;
        e = nvfp4pair("self_attn.kv_b_proj", &lw.d_kv_b_packed,
                      &lw.d_kv_b_scale, &lw.kv_b_gs, &lw.kv_b_packed_bytes,
                      &lw.kv_b_scale_bytes, &lw.kv_b_rows, &lw.kv_b_cols);
        if (e != cudaSuccess) return e;
        e = nvfp4pair("self_attn.o_proj", &lw.d_o_packed, &lw.d_o_scale,
                      &lw.o_gs, &lw.o_packed_bytes, &lw.o_scale_bytes,
                      &lw.o_rows, &lw.o_cols);
        if (e != cudaSuccess) return e;
    }
    jobs.push_back(
        bf16job(fastload::FindTensorSuffix(tensors,
                                           "self_attn.q_a_layernorm.weight"),
                reinterpret_cast<void **>(&lw.d_q_a_norm)));
    jobs.push_back(
        bf16job(fastload::FindTensorSuffix(tensors,
                                           "self_attn.kv_a_layernorm.weight"),
                reinterpret_cast<void **>(&lw.d_kv_a_norm)));

    if (moe) {
        jobs.push_back(bf16job(
            fastload::FindTensorSuffix(tensors, "mlp.gate.weight"),
            reinterpret_cast<void **>(&lw.d_router_gate)));
    } else {
        jobs.push_back(bf16job(
            fastload::FindTensorSuffix(tensors, "mlp.gate_proj.weight"),
            reinterpret_cast<void **>(&lw.d_mlp_gate)));
        jobs.push_back(bf16job(
            fastload::FindTensorSuffix(tensors, "mlp.down_proj.weight"),
            reinterpret_cast<void **>(&lw.d_mlp_down)));
        jobs.push_back(bf16job(
            fastload::FindTensorSuffix(tensors, "mlp.up_proj.weight"),
            reinterpret_cast<void **>(&lw.d_mlp_up)));
    }

    void *kimi_arena = nullptr;
    size_t kimi_arena_bytes = 0;
    cudaError_t e = fastload::RunUploadJobs(std::move(jobs), &kimi_arena,
                                            &kimi_arena_bytes, stream, ring,
                                            bytes_acc, nullptr);
    lw.d_arena = kimi_arena;
    lw.arena_bytes = kimi_arena_bytes;
    if (e != cudaSuccess) return e;

    auto *router = fastload::FindTensorSuffix(tensors, "mlp.gate.weight");
    if (router) {
        auto *gate_bias =
            fastload::FindTensorSuffix(tensors, "mlp.gate.e_score_correction_bias");
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
    if (d_kv_b_bf16) cudaFree(d_kv_b_bf16);
    d_kv_b_bf16 = nullptr;
    if (d_arena) {
        cudaFree(d_arena);
        d_arena = nullptr;
        arena_bytes = 0;
        d_input_norm = d_post_attn_norm = nullptr;
        d_q_a_norm = d_kv_a_norm = nullptr;
        d_q_a_packed = d_q_a_scale = nullptr;
        d_q_b_packed = d_q_b_scale = nullptr;
        d_kv_a_packed = d_kv_a_scale = nullptr;
        d_kv_b_packed = d_kv_b_scale = nullptr;
        d_o_packed = d_o_scale = nullptr;
        d_router_gate = nullptr;
        d_mlp_gate = d_mlp_down = d_mlp_up = nullptr;
        q_a_rows = q_a_cols = q_b_rows = q_b_cols = 0;
        kv_a_rows = kv_a_cols = kv_b_rows = kv_b_cols = 0;
        o_rows = o_cols = 0;
        return;
    }
    if (d_input_norm)     cudaFree(d_input_norm);
    if (d_post_attn_norm) cudaFree(d_post_attn_norm);
    if (d_q_a_norm)       cudaFree(d_q_a_norm);
    if (d_kv_a_norm)      cudaFree(d_kv_a_norm);
    if (d_q_a_packed)     cudaFree(d_q_a_packed);
    if (d_q_a_scale)      cudaFree(d_q_a_scale);
    if (d_q_b_packed)     cudaFree(d_q_b_packed);
    if (d_q_b_scale)      cudaFree(d_q_b_scale);
    if (d_kv_a_packed)    cudaFree(d_kv_a_packed);
    if (d_kv_a_scale)     cudaFree(d_kv_a_scale);
    if (d_kv_b_packed)    cudaFree(d_kv_b_packed);
    if (d_kv_b_scale)     cudaFree(d_kv_b_scale);
    if (d_o_packed)       cudaFree(d_o_packed);
    if (d_o_scale)        cudaFree(d_o_scale);
    if (d_router_gate)    cudaFree(d_router_gate);
    if (d_mlp_gate)       cudaFree(d_mlp_gate);
    if (d_mlp_down)       cudaFree(d_mlp_down);
    if (d_mlp_up)         cudaFree(d_mlp_up);
    d_input_norm = d_post_attn_norm = nullptr;
    d_q_a_norm = d_kv_a_norm = nullptr;
    d_q_a_packed = d_q_a_scale = nullptr;
    d_q_b_packed = d_q_b_scale = nullptr;
    d_kv_a_packed = d_kv_a_scale = nullptr;
    d_kv_b_packed = d_kv_b_scale = nullptr;
    d_o_packed = d_o_scale = nullptr;
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
    if (!topo.format.empty() && topo.format != "logestix-kimi-r2") {
        log::error("[kimi_load] format '" + topo.format +
                   "' is pre-r2: MLA weights require NVFP4 — "
                   "reconvert required");
        return cudaErrorInvalidValue;
    }
    if (topo.num_layers <= 0) {
        log::error("[kimi_load] index has no layer count: " +
                   index_path.string());
        return cudaErrorInvalidValue;
    }
    pw.layers.resize(static_cast<size_t>(topo.num_layers));

    // Round-2: per-layer MLA global scales live in the index
    // (mla_nvfp4[layer][base]); the packed/scale records do not carry them.
    using MlaGsMap = absl::flat_hash_map<int, absl::flat_hash_map<std::string, float>>;
    MlaGsMap mla_gs;
    {
        std::ifstream f(index_path);
        if (!f) {
            log::error("[kimi_load] cannot reopen index for mla_nvfp4: " +
                       index_path.string());
            return cudaErrorInvalidValue;
        }
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        simdjson::ondemand::parser parser;
        simdjson::padded_string pad(json.data(), json.size());
        simdjson::ondemand::document doc;
        if (parser.iterate(pad).get(doc) == simdjson::SUCCESS) {
            simdjson::ondemand::object mla;
            if (doc["mla_nvfp4"].get_object().get(mla) == simdjson::SUCCESS) {
                for (simdjson::ondemand::field lf : mla) {
                    std::string_view lsv;
                    if (lf.unescaped_key().get(lsv) != simdjson::SUCCESS)
                        continue;
                    int layer = std::atoi(std::string(lsv).c_str());
                    simdjson::ondemand::object bases;
                    if (lf.value().get_object().get(bases) !=
                        simdjson::SUCCESS)
                        continue;
                    for (simdjson::ondemand::field bf : bases) {
                        std::string_view bsv;
                        double gs = 1.0;
                        if (bf.unescaped_key().get(bsv) != simdjson::SUCCESS)
                            continue;
                        if (bf.value().get_double().get(gs) !=
                            simdjson::SUCCESS)
                            continue;
                        mla_gs[layer][std::string(bsv)] =
                            static_cast<float>(gs);
                    }
                }
            }
        }
        if (mla_gs.empty()) {
            log::error("[kimi_load] index has no mla_nvfp4 global scales "
                       "(pre-r2 conversion?) — reconvert required");
            return cudaErrorInvalidValue;
        }
    }

    // Drains the stream on every error exit below so pinned staging never
    // dies while a DMA transfer is still in flight.
    fastload::StreamDrain drain{stream};

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
                                stream, pw.resident_bytes, ring,
                                mla_gs[static_cast<size_t>(l)]);
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
    // ── Pin the routed-expert store into locked host memory ──────────────
    // File-backed mmap sources carry a driver VMA-lookup penalty on every
    // H2D copy and vanish under munmap while an async copy is in flight.
    // Per expert the blobs are laid out as packed [w1|w3|w2] then scales
    // [s1|s3|s2] — the order the slot upload streams. Entries without
    // weights (dense layer 0 of the 61-layer passport) are skipped.
    // Disable with LSX_EXPERT_PIN=0 (falls back to mmap views).
    //
    // Chunked: the driver fails a single ~530 GiB cudaHostAlloc after a
    // 12-minute direct-reclaim storm (probed: mlock succeeds, chunked
    // 16 GiB cudaHostAlloc succeeds in ~280 s), so the store pins as a
    // vector of 16 GiB chunks; an expert block never spans two chunks.
    // Each chunk alloc self-reclaims its 16 GiB from the page cache the
    // warm mmaps left behind — bounded per-alloc work, no storm — and the
    // copy pass then reads the blobs from the still-warm cache (RAM speed,
    // no disk round-trip).
    // Expert paths + slot bookkeeping are built unconditionally: the pin
    // pass and the mmap fallback share the same table geometry.
    {
        const int total = num_layers * experts_per_layer;
        db.entries.assign(static_cast<size_t>(total), KimiExpertEntry());
        db.mmaps.assign(static_cast<size_t>(total), nullptr);
        db.batches.assign(static_cast<size_t>(total), nullptr);
        db.paths.resize(static_cast<size_t>(total));
        namespace fs = std::filesystem;
        fs::path exp_dir = fs::path(model_path) / "logestix" / "experts";
        if (!fs::exists(exp_dir)
            && fs::exists(fs::path(model_path) / "experts"))
            exp_dir = fs::path(model_path) / "experts";
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int exp = 0; exp < experts_per_layer; ++exp) {
                const size_t idx =
                    static_cast<size_t>(layer)
                        * static_cast<size_t>(experts_per_layer)
                    + static_cast<size_t>(exp);
                db.paths[idx] =
                    (exp_dir / ("layer_" + std::to_string(layer))
                     / ("exp_" + std::to_string(exp) + ".arrow"))
                        .string();
            }
        }
    }
    const bool pin_enabled =
        std::getenv("LSX_EXPERT_PIN") == nullptr
        || std::getenv("LSX_EXPERT_PIN")[0] != '0';
    if (pin_enabled) {
        constexpr int64_t kPinChunkBytes = int64_t{16} << 30;
        auto aln = [](uintptr_t v) { return (v + 255) / 256 * 256; };
        auto pin_t0 = std::chrono::steady_clock::now();
        // Parallel parse (23040 NVMe files are latency-bound sequentially):
        // fills entries/mmaps for the pin plan. When pinning succeeds the
        // mmaps are dropped immediately (blobs live in locked host chunks);
        // on fallback the mmap table doubles as the legacy view source.
        db.LoadParallelParse(/*n_threads=*/0);
        // Phase 1 — plan (no IO): span sizes, chunk assignment, offsets.
        struct PinPlan {
            KimiExpertEntry *en;
            uint32_t chunk;
            int64_t off_a, off_b, off_c, off_d;   // span starts in the chunk
        };
        std::vector<PinPlan> plan;
        std::vector<const uint8_t *> src_w1, src_w3, src_s1, src_s3,
            src_w2, src_s2;   // captured mmap views (freed after copy)
        {
            int64_t cur = 0;
            uint32_t chunk_idx = 0;
            for (auto &en : db.entries) {
                if (!en.host_w1 || !en.host_w2 || !en.host_w3
                    || !en.host_w1_scales || !en.host_w2_scales
                    || !en.host_w3_scales)
                    continue;
                en.blob_a_bytes = aln(en.w1_bytes) + en.w3_bytes;
                en.blob_b_bytes = aln(en.w1_scales_bytes)
                                + en.w3_scales_bytes;
                en.blob_c_bytes = en.w2_bytes;
                en.blob_d_bytes = en.w2_scales_bytes;
                const int64_t block = aln(en.blob_a_bytes)
                                    + aln(en.blob_b_bytes)
                                    + aln(en.blob_c_bytes)
                                    + aln(en.blob_d_bytes);
                if (cur != 0 && cur + block > kPinChunkBytes) {
                    ++chunk_idx;
                    cur = 0;
                }
                PinPlan pp{};
                pp.en = &en;
                pp.chunk = chunk_idx;
                pp.off_a = cur;
                pp.off_b = pp.off_a + aln(en.blob_a_bytes);
                pp.off_c = pp.off_b + aln(en.blob_b_bytes);
                pp.off_d = pp.off_c + aln(en.blob_c_bytes);
                src_w1.push_back(en.host_w1);
                src_w3.push_back(en.host_w3);
                src_s1.push_back(en.host_w1_scales);
                src_s3.push_back(en.host_w3_scales);
                src_w2.push_back(en.host_w2);
                src_s2.push_back(en.host_w2_scales);
                plan.push_back(pp);
                cur = pp.off_d + aln(en.blob_d_bytes);
            }
        }
        // Phase 2 — chunk allocations in parallel (each cudaHostAlloc
        // zero-fills 16 GiB at ~8 GiB/s single-threaded; 6 workers cut the
        // wall time ~5x on a 542 GiB store).
        const int n_chunks = static_cast<int>(plan.empty() ? 0
                                                          : plan.back().chunk
                                                                + 1);
        {
            std::atomic<int> next{0};
            auto alloc_worker = [&]() {
                for (;;) {
                    const int i = next.fetch_add(1);
                    if (i >= n_chunks) return;
                    // Anonymous mmap + cudaHostRegister: the driver-serializing
                    // cudaHostAlloc zerofill (measured 332 s for 34 x 16 GiB,
                    // 1.6 GB/s) both dominated the load and evicted the
                    // parse's page cache, turning the copy pass into a disk
                    // re-read. Register takes ~1.6 s per chunk and leaves
                    // pages unfaulted — the copy pass faults them with data
                    // in a single write.
                    void *chunk = mmap(nullptr,
                                       static_cast<size_t>(kPinChunkBytes),
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS
                                           | MAP_NORESERVE,
                                       -1, 0);
                    if (chunk == MAP_FAILED) {
                        db.pinned_pools[static_cast<size_t>(i)] = nullptr;
                        continue;
                    }
                    cudaError_t ce = cudaHostRegister(
                        chunk, static_cast<size_t>(kPinChunkBytes),
                        cudaHostRegisterDefault);
                    if (ce != cudaSuccess) {
                        cudaGetLastError();
                        munmap(chunk, static_cast<size_t>(kPinChunkBytes));
                        db.pinned_pools[static_cast<size_t>(i)] = nullptr;
                        continue;
                    }
                    db.pinned_pools[static_cast<size_t>(i)] = chunk;
                    db.pinned_registered[static_cast<size_t>(i)] = true;
                    db.pinned_chunk_sizes[static_cast<size_t>(i)] =
                        static_cast<size_t>(kPinChunkBytes);
                }
            };
            db.pinned_pools.assign(static_cast<size_t>(n_chunks), nullptr);
            db.pinned_registered.assign(static_cast<size_t>(n_chunks), false);
            db.pinned_chunk_sizes.assign(static_cast<size_t>(n_chunks), 0);
            db.pinned_bytes = static_cast<std::size_t>(n_chunks)
                            * static_cast<std::size_t>(kPinChunkBytes);
            const unsigned hw = std::thread::hardware_concurrency();
            const int n_threads = std::clamp<unsigned>(hw / 16u, 4u, 8u);
            std::vector<std::thread> pool;
            for (int t = 1; t < n_threads; ++t) pool.emplace_back(alloc_worker);
            alloc_worker();
            for (auto &th : pool) th.join();
            bool all_ok = true;
            for (void *c : db.pinned_pools)
                if (!c) all_ok = false;
            if (!all_ok) {
                log::warn("[kimi_load] expert pinning stopped — chunk alloc "
                          "failed; store stays file-backed");
                for (size_t ci = 0; ci < db.pinned_pools.size(); ++ci) {
                    if (!db.pinned_pools[ci]) continue;
                    if (db.pinned_registered[ci]) {
                        cudaHostUnregister(db.pinned_pools[ci]);
                        munmap(db.pinned_pools[ci],
                               db.pinned_chunk_sizes[ci]);
                    } else {
                        cudaFreeHost(db.pinned_pools[ci]);
                    }
                }
                db.pinned_pools.clear();
                db.pinned_registered.clear();
                db.pinned_chunk_sizes.clear();
                db.pinned_bytes = 0;
            }
        }
        const auto alloc_t1 = std::chrono::steady_clock::now();
        log::info("[kimi_load] pin: chunk alloc done in " +
                  std::to_string(std::chrono::duration_cast<
                                     std::chrono::milliseconds>(
                                     alloc_t1 - pin_t0).count()) + " ms");
        // Phase 3 — parallel span copies from the captured mmap views;
        // entry pointers re-pointed afterwards (single-threaded).
        if (!db.pinned_pools.empty()) {
            std::atomic<size_t> next{0};
            auto copy_worker = [&]() {
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= plan.size()) return;
                    const PinPlan &pp = plan[i];
                    auto *base = static_cast<uint8_t *>(
                        db.pinned_pools[pp.chunk]);
                    std::memcpy(base + pp.off_a, src_w1[i],
                                static_cast<std::size_t>(
                                    pp.en->w1_bytes));
                    std::memcpy(base + pp.off_a + aln(pp.en->w1_bytes),
                                src_w3[i],
                                static_cast<std::size_t>(
                                    pp.en->w3_bytes));
                    std::memcpy(base + pp.off_b, src_s1[i],
                                static_cast<std::size_t>(
                                    pp.en->w1_scales_bytes));
                    std::memcpy(base + pp.off_b
                                    + aln(pp.en->w1_scales_bytes),
                                src_s3[i],
                                static_cast<std::size_t>(
                                    pp.en->w3_scales_bytes));
                    std::memcpy(base + pp.off_c, src_w2[i],
                                static_cast<std::size_t>(
                                    pp.en->w2_bytes));
                    std::memcpy(base + pp.off_d, src_s2[i],
                                static_cast<std::size_t>(
                                    pp.en->w2_scales_bytes));
                }
            };
            const unsigned hw = std::thread::hardware_concurrency();
            const int n_threads = std::clamp<unsigned>(hw / 8u, 8u, 24u);
            std::vector<std::thread> pool;
            for (int t = 1; t < n_threads; ++t) pool.emplace_back(copy_worker);
            copy_worker();
            for (auto &th : pool) th.join();
            for (size_t i = 0; i < plan.size(); ++i) {
                PinPlan &pp = plan[i];
                auto *base = static_cast<uint8_t *>(
                    db.pinned_pools[pp.chunk]);
                KimiExpertEntry &en = *pp.en;
                en.pinned_blob = base + pp.off_a;
                en.host_w1 = base + pp.off_a;
                en.host_w3 = base + pp.off_a + aln(en.w1_bytes);
                en.host_w1_scales = base + pp.off_b;
                en.host_w3_scales =
                    base + pp.off_b + aln(en.w1_scales_bytes);
                en.host_w2 = base + pp.off_c;
                en.host_w2_scales = base + pp.off_d;
            }
            const int pinned_cnt = static_cast<int>(plan.size());
            auto pin_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - pin_t0).count();
            log::info("[kimi_load] pinned routed-expert store: "
                      + std::to_string(pinned_cnt) + " experts, "
                      + std::to_string(n_chunks) + " x 16 GiB chunks in "
                      + std::to_string(pin_ms) + " ms");
        } else {
            log::warn("[kimi_load] expert pinning unavailable — "
                      "keeping file-backed mmap sources");
        }
    }
    if (!pin_enabled || db.pinned_pools.empty()) {
        // Fallback (pin disabled or impossible): the classic sequential
        // Load parses the files again into the pre-sized tables and warms
        // the LRU-bounded mmap view cache.
        db.entries.assign(db.paths.size(), KimiExpertEntry());
        db.mmaps.assign(db.paths.size(), nullptr);
        db.batches.assign(db.paths.size(), nullptr);
        db.lru.clear();
        db.Load(model_path, /*fallback=*/false);
    }
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
