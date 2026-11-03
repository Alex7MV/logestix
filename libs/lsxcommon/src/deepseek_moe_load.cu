#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/fast_load.h"
#include "fast_load_internal.h"
#include "deepseek_moe_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <chrono>
#include <tuple>
#include "lsxcommon/log.h"
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

namespace {

inline const TensorView *find_tensor_suffix(
    const TensorBatchViews &tensors, std::string_view suffix)
{
    return lsxcommon::fastload::FindTensorSuffix(tensors, suffix);
}

const TensorView *find_tensor_substr(const TensorBatchViews &tensors,
                                     std::string_view substr)
{
    for (const auto &t : tensors.tensors) {
        if (t.name.find(substr) != std::string_view::npos)
            return &t;
    }
    return nullptr;
}

// kArenaAlign / AlignUp / StreamDrain come from fast_load_internal.h
// (shared across all loaders).

// NOTE (model-unification audit): unlike the other five loaders, DeepSeek
// keeps its typed ArenaJob/UploadLayerJobs engine family-local. It uploads
// F8_E4M3/F8_E8M0 payloads byte-verbatim (byte-count-based, not element-based)
// and captures dimensions through int* fields wired directly into
// kernel-visible LayerWeights — generalizing fastload::UploadJob for this
// would either change kernel-facing types or bloat the shared struct.
// One planned upload into the per-layer arena. BF16 jobs upload through the
// staging ring with an on-device BF16→FP16 conversion; raw jobs upload bytes
// verbatim (F32 / F8_E4M3 / F8_E8M0 payloads). The typed constructors route
// the job: __half** ⇒ FP16 conversion, everything else ⇒ raw bytes.
struct ArenaJob {
    const TensorView *t;
    void **dst;
    bool to_fp16;
    int64_t *bytes_cap = nullptr;   // optional: uploaded byte count
    int *rows_cap = nullptr;        // optional: shape[0] when 2D
    int *cols_cap = nullptr;        // optional: shape[1] when 2D

    ArenaJob(const TensorView *t_, __half **dst_)
        : t(t_), dst(reinterpret_cast<void **>(dst_)), to_fp16(true) {}
    ArenaJob(const TensorView *t_, float **dst_)
        : t(t_), dst(reinterpret_cast<void **>(dst_)), to_fp16(false) {}
    ArenaJob(const TensorView *t_, uint8_t **dst_,
             int64_t *bytes_cap_ = nullptr, int *rows_cap_ = nullptr,
             int *cols_cap_ = nullptr)
        : t(t_), dst(reinterpret_cast<void **>(dst_)), to_fp16(false),
          bytes_cap(bytes_cap_), rows_cap(rows_cap_), cols_cap(cols_cap_) {}
    ArenaJob(const TensorView *t_, const uint8_t **dst_,
             int64_t *bytes_cap_ = nullptr, int *rows_cap_ = nullptr,
             int *cols_cap_ = nullptr)
        : t(t_), dst(reinterpret_cast<void **>(
                     const_cast<uint8_t **>(dst_))),
          to_fp16(false), bytes_cap(bytes_cap_), rows_cap(rows_cap_),
          cols_cap(cols_cap_) {}
};

cudaError_t ValidateArenaJob(const ArenaJob &j, size_t &upload_bytes)
{
    if (j.to_fp16) {
        if (j.t->rank == 1 && j.t->shape[0] > 0)
            upload_bytes = static_cast<size_t>(j.t->shape[0]) * 2;
        else if (j.t->rank == 2 && j.t->shape[0] > 0 && j.t->shape[1] > 0)
            upload_bytes = static_cast<size_t>(j.t->shape[0])
                         * static_cast<size_t>(j.t->shape[1]) * 2;
        else {
            log::error("[load_pinned] malformed BF16 tensor '" +
                       std::string(j.t->name) + "'");
            return cudaErrorInvalidValue;
        }
        if (j.t->bytes < upload_bytes) {
            log::error("[load_pinned] truncated BF16 tensor '" +
                       std::string(j.t->name) + "'");
            return cudaErrorInvalidValue;
        }
    } else {
        upload_bytes = j.t->bytes;
        if (upload_bytes == 0) {
            log::error("[load_pinned] empty byte tensor '" +
                       std::string(j.t->name) + "'");
            return cudaErrorInvalidValue;
        }
    }
    return cudaSuccess;
}

// Validates every job, issues ONE arena allocation sized to the layer's total
// upload footprint and streams each tensor through the pinned staging ring
// into its 256-byte-aligned arena view. On success lw.d_arena/arena_bytes own
// the block and every *dst points inside it. Any failure aborts the whole
// layer (caller frees the weights).
cudaError_t UploadLayerJobs(std::vector<ArenaJob> jobs, LayerWeights &lw,
                            cudaStream_t stream,
                            lsxcommon::fastload::PinnedStagingRing &ring,
                            std::uint64_t &bytes_acc)
{
    std::vector<size_t> sz(jobs.size(), 0);
    size_t total = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        cudaError_t e = ValidateArenaJob(jobs[i], sz[i]);
        if (e != cudaSuccess) return e;
        total = fastload::AlignUp(total, fastload::kArenaAlign) + sz[i];
    }

    void *arena = nullptr;
    if (total > 0) {
        cudaError_t e = cudaMalloc(&arena, total);
        if (e != cudaSuccess) {
            log::error(std::string("[load_pinned] layer arena alloc failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    }
    lw.d_arena = arena;
    lw.arena_bytes = total;

    uint8_t *base = static_cast<uint8_t *>(arena);
    size_t off = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto &j = jobs[i];
        if (!j.t) continue;
        off = fastload::AlignUp(off, fastload::kArenaAlign);
        void *view = base + off;
        *j.dst = view;
        if (j.bytes_cap) *j.bytes_cap = static_cast<int64_t>(sz[i]);
        if ((j.rows_cap || j.cols_cap) && j.t->rank == 2) {
            if (j.rows_cap) *j.rows_cap = static_cast<int>(j.t->shape[0]);
            if (j.cols_cap) *j.cols_cap = static_cast<int>(j.t->shape[1]);
        }
        cudaError_t e = j.to_fp16
            ? ring.UploadBf16AsFp16(j.t->data,
                  static_cast<int64_t>(sz[i] / 2),
                  static_cast<__half *>(view), stream, bytes_acc)
            : ring.UploadRaw(j.t->data, view, sz[i], stream,
                             bytes_acc);
        if (e != cudaSuccess) {
            log::error("[load_pinned] upload failed for '" +
                       std::string(j.t->name)
                       + "': " + cudaGetErrorString(e));
            return e;
        }
        off += sz[i];
    }
    return cudaSuccess;
}

arrow::Result<TensorBatchViews> load_arrow_file(const std::string &path)
{
    return LoadTensorBatchViews(path);
}

} // anonymous namespace

ModelTopology load_topology(const std::string &model_path)
{
    ModelTopology topo;
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(ipath);
    if (!f) {
        topo.error_msg = "Cannot open " + ipath.string();
        return topo;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        topo.error_msg = "JSON parse error in " + ipath.string();
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "Expected JSON object in " + ipath.string();
        return topo;
    }

    std::string verr;
    if (!lsxcommon::ParseVramBlock(json, topo.vram, verr)) {
        ModelTopology failed;   // num_layers == 0 -> pipeline guard fires
        failed.error_msg = verr + " in " + ipath.string();
        return failed;
    }
    std::string_view fmt;
    if (root["format"].get_string().get(fmt) == simdjson::SUCCESS)
        topo.format.assign(fmt);
    int64_t nl = 0, epl = 0, te = 0;
    if (root["num_layers"].get_int64().get(nl) != simdjson::SUCCESS ||
        root["experts_per_layer"].get_int64().get(epl) != simdjson::SUCCESS ||
        root["total_experts"].get_int64().get(te) != simdjson::SUCCESS) {
        topo.error_msg = "missing routing counts in " + ipath.string();
        return topo;
    }
    topo.num_layers = static_cast<int>(nl);
    topo.experts_per_layer = static_cast<int>(epl);
    topo.total_experts = te;
    return topo;
}

std::string detect_model_type(const std::string &model_path)
{
    std::string result;
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
    const auto moe_ipath =
        std::filesystem::path(model_path) / "logestix-qwen35moe" /
        "model.index.json";
    if (!std::filesystem::exists(ipath) && std::filesystem::exists(moe_ipath))
        ipath = moe_ipath;
    std::ifstream f(ipath);
    if (!f) return result;
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return result;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return result;

    std::string_view name;
    if (root["model"].get_string().get(name) != simdjson::SUCCESS) return result;
    std::string raw(name);
    // Normalize: "DeepSeek-V4-Flash" -> "deepseek-v4-flash"
    for (char &c : raw) {
        if (c == ' ') c = '-';
        else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return raw;
}

// No per-model parser needed: the shared ExpertStore<TEntry>::DecodeEntry()
// extracts the six ExpertSchema bins + three global scales into TEntry.

ExpertDb load_expert_index(const std::string &model_path,
                            int num_layers, int experts_per_layer)
{
    ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    // Full-resident mmap table: the 7000 cap (Windows CRT limit) forced
    // constant churn past layer 27 and munmaps under in-flight prefetch DMA.
    db.max_open = 65536;
    db.log_prefix = "[load_expert_index] ";
    db.mmap_fail_verb = "mmap fail ";
    db.emit_diag = true;             // DeepSeek diag block
    db.log_batch_fails = false;      // DeepSeek logs no batch-failure lines
    db.Load(model_path, /*fallback=*/false);
    // ── Pin the routed-expert store into locked host memory ──────────────
    // File-backed mmap sources carry a driver VMA-lookup penalty on every
    // H2D copy (measured 5.7 tok/s with a 11008-VMA table) and vanish under
    // munmap while an async copy can be in flight. One big cudaHostAlloc
    // pool, laid out per expert as packed [w1|w3|w2] then scales [s1|s3|s2]
    // — the exact order the GEMV phase streams. Disable with
    // LSX_EXPERT_PIN=0 (falls back to mmap views).
    if (std::getenv("LSX_EXPERT_PIN") == nullptr
        || std::getenv("LSX_EXPERT_PIN")[0] != '0') {
        int64_t pool_bytes = 0;
        bool sizes_ok = true;
        for (const auto &en : db.entries) {
            if (!en.host_w1 || !en.host_w2 || !en.host_w3
                || !en.host_w1_scales || !en.host_w2_scales
                || !en.host_w3_scales
                || en.w1_bytes <= 0 || en.w2_bytes <= 0 || en.w3_bytes <= 0
                || en.w1_scales_bytes <= 0 || en.w2_scales_bytes <= 0
                || en.w3_scales_bytes <= 0) {
                sizes_ok = false;
                break;
            }
            pool_bytes += ((en.w1_bytes + en.w3_bytes + en.w2_bytes + 255)
                           / 256) * 256;
            pool_bytes += ((en.w1_scales_bytes + en.w3_scales_bytes
                            + en.w2_scales_bytes + 255) / 256) * 256;
        }
        void *pool = nullptr;
        if (sizes_ok && pool_bytes > 0
            && cudaHostAlloc(&pool, static_cast<size_t>(pool_bytes),
                             cudaHostAllocDefault) == cudaSuccess) {
            db.pinned_pool = pool;
            db.pinned_bytes = static_cast<std::size_t>(pool_bytes);
            uintptr_t cur = reinterpret_cast<uintptr_t>(pool);
            auto aln = [](uintptr_t v) { return (v + 255) / 256 * 256; };
            int pinned_cnt = 0;
            for (auto &en : db.entries) {
                en.pinned_blob = reinterpret_cast<const uint8_t *>(cur);
                en.blob_pack_bytes = en.w1_bytes + en.w3_bytes + en.w2_bytes;
                en.blob_scale_bytes = en.w1_scales_bytes + en.w3_scales_bytes
                                    + en.w2_scales_bytes;
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w1,
                            static_cast<std::size_t>(en.w1_bytes));
                en.host_w1 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w1_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w3,
                            static_cast<std::size_t>(en.w3_bytes));
                en.host_w3 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w3_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w2,
                            static_cast<std::size_t>(en.w2_bytes));
                en.host_w2 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w2_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w1_scales,
                            static_cast<std::size_t>(en.w1_scales_bytes));
                en.host_w1_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w1_scales_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w3_scales,
                            static_cast<std::size_t>(en.w3_scales_bytes));
                en.host_w3_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w3_scales_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w2_scales,
                            static_cast<std::size_t>(en.w2_scales_bytes));
                en.host_w2_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w2_scales_bytes);
                ++pinned_cnt;
            }
            log::info("[load_expert_index] pinned routed-expert store: "
                      + std::to_string(pinned_cnt) + " experts, "
                      + std::to_string(pool_bytes / (1024 * 1024))
                      + " MiB locked host memory");
        } else {
            log::warn("[load_expert_index] expert pinning unavailable — "
                      "keeping file-backed mmap sources");
        }
    }
    return db;
}

void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream,
                            cudaEvent_t evict_done)
{
    db.EnsureResident(layer, expert, stream, evict_done);
}

void free_expert_index(ExpertDb &db) { db.Free(); }

void LayerWeights::Free()
{
    if (d_arena) {
        if (h_tid2eid)   cudaFreeHost(h_tid2eid);
        if (h_gate_bias) cudaFreeHost(h_gate_bias);
        cudaFree(d_arena);
        *this = LayerWeights{};
        return;
    }
    if (d_ffn_norm)     cudaFree(d_ffn_norm);
    if (d_gate_weight)  cudaFree(d_gate_weight);
    if (d_shared_w1)    cudaFree(d_shared_w1);
    if (d_shared_w2)    cudaFree(d_shared_w2);
    if (d_shared_w3)    cudaFree(d_shared_w3);
    if (d_shared_w1_scales) cudaFree(d_shared_w1_scales);
    if (d_shared_w2_scales) cudaFree(d_shared_w2_scales);
    if (d_shared_w3_scales) cudaFree(d_shared_w3_scales);
    if (d_shared_w1_h)  cudaFree(d_shared_w1_h);
    if (d_shared_w3_h)  cudaFree(d_shared_w3_h);
    if (d_shared_w2_f32) cudaFree(d_shared_w2_f32);
    if (d_wq_a)         cudaFree(const_cast<uint8_t*>(d_wq_a));
    if (d_wq_a_scales)  cudaFree(const_cast<uint8_t*>(d_wq_a_scales));
    if (d_wq_b)         cudaFree(const_cast<uint8_t*>(d_wq_b));
    if (d_wq_b_scales)  cudaFree(const_cast<uint8_t*>(d_wq_b_scales));
    if (d_wkv)          cudaFree(const_cast<uint8_t*>(d_wkv));
    if (d_wkv_scales)   cudaFree(const_cast<uint8_t*>(d_wkv_scales));
    if (d_wo_a)         cudaFree(const_cast<uint8_t*>(d_wo_a));
    if (d_wo_a_scales)  cudaFree(const_cast<uint8_t*>(d_wo_a_scales));
    if (d_wo_b)         cudaFree(const_cast<uint8_t*>(d_wo_b));
    if (d_wo_b_scales)  cudaFree(const_cast<uint8_t*>(d_wo_b_scales));
    if (d_q_norm)       cudaFree(d_q_norm);
    if (d_kv_norm)      cudaFree(d_kv_norm);
    if (d_attn_norm)    cudaFree(d_attn_norm);
    if (d_attn_sink)    cudaFree(d_attn_sink);
    if (d_hc_attn_fn)   cudaFree(d_hc_attn_fn);
    if (d_hc_attn_base) cudaFree(d_hc_attn_base);
    if (d_hc_attn_scale)cudaFree(d_hc_attn_scale);
    if (d_hc_ffn_fn)    cudaFree(d_hc_ffn_fn);
    if (d_hc_ffn_base)  cudaFree(d_hc_ffn_base);
    if (d_hc_ffn_scale) cudaFree(d_hc_ffn_scale);
    if (h_tid2eid)      cudaFreeHost(h_tid2eid);
    if (h_gate_bias)    cudaFreeHost(h_gate_bias);
    // CSA compressor
    if (d_comp_wkv)     cudaFree(d_comp_wkv);
    if (d_comp_wgate)   cudaFree(d_comp_wgate);
    if (d_comp_ape)     cudaFree(d_comp_ape);
    if (d_comp_norm)    cudaFree(d_comp_norm);
    // CSA indexer
    if (d_idx_wq_b)     cudaFree(d_idx_wq_b);
    if (d_idx_wq_b_scales) cudaFree(d_idx_wq_b_scales);
    if (d_idx_weights_proj) cudaFree(d_idx_weights_proj);
    if (d_idx_comp_wkv)     cudaFree(d_idx_comp_wkv);
    if (d_idx_comp_wgate)   cudaFree(d_idx_comp_wgate);
    if (d_idx_comp_ape)     cudaFree(d_idx_comp_ape);
    if (d_idx_comp_norm)    cudaFree(d_idx_comp_norm);
    // Null every field so a second Free() is a no-op (load_pinned_weights
    // releases its partial allocation on error paths, then the caller frees).
    *this = LayerWeights{};
}

void PinnedWeights::Free()
{
    if (d_embed)      cudaFree(d_embed);
    if (d_head)       cudaFree(d_head);
    if (d_final_norm) cudaFree(d_final_norm);
    if (d_hc_head_fn)   cudaFree(d_hc_head_fn);
    if (d_hc_head_base) cudaFree(d_hc_head_base);
    if (d_hc_head_scale)cudaFree(d_hc_head_scale);
    // Idempotent: idempotent reload (Free at load_pinned_weights entry) and
    // caller cleanup after an error-path release must not double free.
    d_embed = nullptr;
    d_head = nullptr;
    d_final_norm = nullptr;
    d_hc_head_fn = nullptr;
    d_hc_head_base = nullptr;
    d_hc_head_scale = nullptr;
    for (auto &l : layers) l.Free();
    layers.clear();
    vocab_size = 0;
    resident_bytes = 0;
}

cudaError_t load_pinned_weights(
    const std::string &model_path,
    PinnedWeights &pw,
    cudaStream_t stream)
{
    namespace fs = std::filesystem;

    // Idempotent re-load: drop any previously uploaded set first so calling
    // Load twice cannot leak the old device allocations.
    pw.Free();

    const auto t0 = std::chrono::steady_clock::now();

    auto pinned_dir = fs::path(model_path) / "logestix" / "pinned";

    // Drains the stream on every error exit below so pinned staging never
    // dies while a DMA transfer is still in flight.
    fastload::StreamDrain drain{stream};

    lsxcommon::fastload::PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // ── Topology: layer count comes from model.index.json (was hardcoded) ──
    auto topo = load_topology(model_path);
    if (!topo.error_msg.empty() || topo.num_layers <= 0) {
        log::error("[load_pinned] topology error: "
                   + (topo.error_msg.empty()
                          ? std::string("num_layers <= 0")
                          : topo.error_msg));
        return cudaErrorInvalidValue;
    }

    // Per-layer compress_ratio table (decoder grid of DeepSeek-V4-Flash:
    // pure MLA=0, CSA>0; ratio==4 uses coff=2 overlap).
    static const int compress_ratios[] = {
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4
    };
    constexpr int kRatioTableSize =
        static_cast<int>(sizeof(compress_ratios) / sizeof(compress_ratios[0]));
    const int num_layers = topo.num_layers;
    if (num_layers > kRatioTableSize) {
        log::error("[load_pinned] topology num_layers="
                   + std::to_string(num_layers)
                   + " exceeds supported decoder grid (" +
                   std::to_string(kRatioTableSize) + ")");
        return cudaErrorInvalidValue;
    }
    pw.layers.resize(static_cast<size_t>(num_layers));

    std::uint64_t rb = 0;   // device-resident byte total (weights only)

    // ── global.arrow ──────────────────────────────────────────────────────
    auto global_path = pinned_dir / "global.arrow";
    auto global_tensors_res = load_arrow_file(global_path.string());
    if (!global_tensors_res.ok()) {
        log::error("[load_pinned] ERROR loading global.arrow: "
                   + global_tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    auto &gt = *global_tensors_res;

    // Strict global set: embed/head/norm are mandatory (the model cannot
    // run without them).
    auto *embed_t = find_tensor_suffix(gt, "embed.weight");
    auto *head_t  = find_tensor_suffix(gt, "head.weight");
    auto *norm_t  = find_tensor_suffix(gt, "norm.weight");
    if (!embed_t || embed_t->rank != 2 ||
        embed_t->bytes <
            static_cast<size_t>(embed_t->shape[0]) *
                static_cast<size_t>(embed_t->shape[1]) * 2) {
        log::error("[load_pinned] embed.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }
    if (!head_t || head_t == embed_t || head_t->rank != 2 ||
        head_t->bytes <
            static_cast<size_t>(head_t->shape[0]) *
                static_cast<size_t>(head_t->shape[1]) * 2) {
        log::error("[load_pinned] head.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }
    if (!norm_t || norm_t->rank != 1 ||
        norm_t->bytes < static_cast<size_t>(norm_t->shape[0]) * 2) {
        log::error("[load_pinned] norm.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }

    pw.vocab_size = static_cast<int>(embed_t->shape[0]);
    {
        int64_t numel =
            static_cast<int64_t>(embed_t->shape[0]) * embed_t->shape[1];
        cudaError_t ge = cudaMalloc(&pw.d_embed,
                                    static_cast<size_t>(numel) * 2);
        if (ge != cudaSuccess) return ge;
        rb += static_cast<std::uint64_t>(numel) * 2;
        ge = ring.UploadBf16AsFp16(embed_t->data, numel,
                                   pw.d_embed, stream, rb);
        if (ge != cudaSuccess) {
            log::error(std::string("[load_pinned] embed upload failed: ")
                       + cudaGetErrorString(ge));
            return ge;
        }
    }
    {
        int64_t numel =
            static_cast<int64_t>(head_t->shape[0]) * head_t->shape[1];
        cudaError_t ge = cudaMalloc(&pw.d_head,
                                    static_cast<size_t>(numel) * 2);
        if (ge != cudaSuccess) return ge;
        rb += static_cast<std::uint64_t>(numel) * 2;
        ge = ring.UploadBf16AsFp16(head_t->data, numel,
                                   pw.d_head, stream, rb);
        if (ge != cudaSuccess) {
            log::error(std::string("[load_pinned] head upload failed: ")
                       + cudaGetErrorString(ge));
            return ge;
        }
    }
    {
        int64_t numel = norm_t->shape[0];
        cudaError_t ge = cudaMalloc(&pw.d_final_norm,
                                    static_cast<size_t>(numel) * 2);
        if (ge != cudaSuccess) return ge;
        rb += static_cast<std::uint64_t>(numel) * 2;
        ge = ring.UploadBf16AsFp16(norm_t->data, numel,
                                   pw.d_final_norm, stream, rb);
        if (ge != cudaSuccess) {
            log::error(std::string("[load_pinned] final_norm upload failed: ")
                       + cudaGetErrorString(ge));
            return ge;
        }
    }

    // HC head (FP32) — strict presence and status checks.
    for (auto [suffix, dst] : std::initializer_list<
             std::pair<const char *, float **>>{
             {"hc_head_fn", &pw.d_hc_head_fn},
             {"hc_head_base", &pw.d_hc_head_base},
             {"hc_head_scale", &pw.d_hc_head_scale}}) {
        auto *t = find_tensor_suffix(gt, suffix);
        if (!t || t->bytes == 0) {
            log::error(std::string("[load_pinned] hc head tensor missing: ")
                       + suffix);
            return cudaErrorInvalidValue;
        }
        cudaError_t he = cudaMalloc(dst, t->bytes);
        if (he != cudaSuccess) return he;
        rb += t->bytes;
        he = ring.UploadRaw(t->data, *dst, t->bytes, stream, rb);
        if (he != cudaSuccess) {
            log::error(std::string("[load_pinned] upload failed for ")
                       + suffix + ": " + cudaGetErrorString(he));
            return he;
        }
    }

    // ── Per-layer files (strict: every layer file MUST exist) ─────────────
    for (int l = 0; l < num_layers; ++l) {
        auto fn = "layer_" + std::to_string(l) + ".arrow";
        auto layer_path = pinned_dir / fn;
        if (!fs::exists(layer_path)) {
            log::error("[load_pinned] required layer file missing: " + fn);
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto tensors_res = load_arrow_file(layer_path.string());
        if (!tensors_res.ok()) {
            log::error("[load_pinned] ERROR loading " + fn + ": "
                       + tensors_res.status().ToString());
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto &lt = *tensors_res;
        auto &lw = pw.layers[l];

        lw.compress_ratio = compress_ratios[l];
        lw.rope_theta = (lw.compress_ratio > 0) ? 160000.0f : 10000.0f;
        // CSA overlap: ratio==4 uses coff=2 (reference Compressor:
        // coff = 1 + overlap).
        lw.comp_coff = (lw.compress_ratio == 4) ? 2 : 1;

        std::vector<ArenaJob> jobs = {
            {find_tensor_substr(lt, "ffn_norm.weight"), &lw.d_ffn_norm},
            {find_tensor_substr(lt, "ffn.gate.weight"), &lw.d_gate_weight},
            {find_tensor_suffix(lt, "attn.q_norm.weight"), &lw.d_q_norm},
            {find_tensor_suffix(lt, "attn.kv_norm.weight"), &lw.d_kv_norm},
            {find_tensor_suffix(lt, "attn_norm.weight"), &lw.d_attn_norm},
        };

        // CSA Compressor (ratio>0): wkv/wgate BF16, ape F32, norm BF16.
        if (lw.compress_ratio > 0) {
            jobs.push_back({find_tensor_suffix(lt,
                 "attn.compressor.wkv.weight"), &lw.d_comp_wkv});
            jobs.push_back({find_tensor_suffix(lt,
                 "attn.compressor.wgate.weight"), &lw.d_comp_wgate});
            jobs.push_back({find_tensor_suffix(lt, "attn.compressor.ape"),
                            &lw.d_comp_ape});
            jobs.push_back({find_tensor_suffix(lt,
                 "attn.compressor.norm.weight"), &lw.d_comp_norm});

            // CSA Indexer (ratio==4 only).
            if (lw.compress_ratio == 4) {
                auto *idx_w = find_tensor_suffix(lt,
                    "attn.indexer.wq_b.weight");
                auto *idx_s = find_tensor_suffix(lt,
                    "attn.indexer.wq_b.scale");
                if (idx_w) {
                    if (!idx_s) {
                        log::error("[load_pinned] indexer wq_b scale "
                                   "missing in " + fn);
                        pw.Free();
                        return cudaErrorInvalidValue;
                    }
                    lw.idx_wq_b_rows = 8192;   // canonical geometry
                    lw.idx_wq_b_cols = 1024;
                    jobs.push_back({idx_w, &lw.d_idx_wq_b,
                                    nullptr, &lw.idx_wq_b_rows,
                                    &lw.idx_wq_b_cols});
                    jobs.push_back({idx_s, &lw.d_idx_wq_b_scales});
                }
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.weights_proj.weight"),
                     &lw.d_idx_weights_proj});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.wkv.weight"),
                     &lw.d_idx_comp_wkv});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.wgate.weight"),
                     &lw.d_idx_comp_wgate});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.ape"),
                     &lw.d_idx_comp_ape});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.norm.weight"),
                     &lw.d_idx_comp_norm});
            }
        }

        // MLA attention weights (F8_E4M3 + F8_E8M0 scales, strict pairing:
        // an attention slot present without its scale is fatal).
        const std::pair<const char *, const char *> attn_pairs[] = {
            {"attn.wq_a.weight", "attn.wq_a.scale"},
            {"attn.wq_b.weight", "attn.wq_b.scale"},
            {"attn.wkv.weight",  "attn.wkv.scale"},
            {"attn.wo_a.weight", "attn.wo_a.scale"},
            {"attn.wo_b.weight", "attn.wo_b.scale"},
        };
        const std::tuple<const uint8_t **, const uint8_t **,
                         int64_t *, int64_t *, int *, int *> attn_fields[] = {
            {&lw.d_wq_a, &lw.d_wq_a_scales, &lw.wq_a_n, &lw.wq_a_s_n,
             &lw.wq_a_rows, &lw.wq_a_cols},
            {&lw.d_wq_b, &lw.d_wq_b_scales, &lw.wq_b_n, &lw.wq_b_s_n,
             &lw.wq_b_rows, &lw.wq_b_cols},
            {&lw.d_wkv,  &lw.d_wkv_scales,  &lw.wkv_n,  &lw.wkv_s_n,
             &lw.wkv_rows, &lw.wkv_cols},
            {&lw.d_wo_a, &lw.d_wo_a_scales, &lw.wo_a_n, &lw.wo_a_s_n,
             &lw.wo_a_rows, &lw.wo_a_cols},
            {&lw.d_wo_b, &lw.d_wo_b_scales, &lw.wo_b_n, &lw.wo_b_s_n,
             &lw.wo_b_rows, &lw.wo_b_cols},
        };
        for (int p = 0; p < 5; ++p) {
            auto *wt = find_tensor_suffix(lt, attn_pairs[p].first);
            if (!wt) continue;   // absent attention slot stays legal
            auto *st = find_tensor_suffix(lt, attn_pairs[p].second);
            if (!st) {
                log::error("[load_pinned] scale missing for '"
                           + std::string(attn_pairs[p].first) + "' in "
                           + fn);
                pw.Free();
                return cudaErrorInvalidValue;
            }
            auto [dw, dsc, wn, sn, rw, cw] = attn_fields[p];
            jobs.push_back({wt, dw, wn, rw, cw});
            jobs.push_back({st, dsc, sn});
        }

        // Attention sink + Hyper-Connections (FP32).
        jobs.push_back({find_tensor_suffix(lt, "attn.attn_sink"),
                        &lw.d_attn_sink});
        for (auto [suffix, dst] :
             std::initializer_list<std::pair<const char *, float **>>{
                 {"hc_attn_fn", &lw.d_hc_attn_fn},
                 {"hc_attn_base", &lw.d_hc_attn_base},
                 {"hc_attn_scale", &lw.d_hc_attn_scale},
                 {"hc_ffn_fn", &lw.d_hc_ffn_fn},
                 {"hc_ffn_base", &lw.d_hc_ffn_base},
                 {"hc_ffn_scale", &lw.d_hc_ffn_scale}}) {
            jobs.push_back({find_tensor_suffix(lt, suffix), dst});
        }

        cudaError_t le = UploadLayerJobs(std::move(jobs), lw, stream, ring,
                                         rb);
        if (le != cudaSuccess) {
            pw.Free();
            return le;
        }

        // Router selection bias (F32) + hash table (I64) — host pinned.
        auto *gate_bias = find_tensor_substr(lt, "ffn.gate.bias");
        if (gate_bias && gate_bias->rank == 1 &&
            gate_bias->dtype == "F32") {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias,
                               static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data,
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }
        auto *tid2eid = find_tensor_substr(lt, "ffn.gate.tid2eid");
        if (tid2eid && tid2eid->rank == 2 &&
            tid2eid->dtype == "I64") {
            int64_t numel = tid2eid->shape[0] * tid2eid->shape[1];
            if (cudaMallocHost(&lw.h_tid2eid,
                               static_cast<size_t>(numel) * sizeof(int64_t))
                == cudaSuccess) {
                std::memcpy(lw.h_tid2eid, tid2eid->data,
                            static_cast<size_t>(numel) * sizeof(int64_t));
                lw.tid2eid_vocab = static_cast<int>(tid2eid->shape[0]);
                lw.tid2eid_topk  = static_cast<int>(tid2eid->shape[1]);
            }
        }
    }

    // Shared experts: NVFP4-packed metadata from pinned/shared_experts.arrow.
    // Loaded AFTER the layer loop so the descriptor registry lives
    // independently of the per-layer pinned files. Absent file stays legal;
    // present-but-broken file is fatal.
    {
        auto shared_path = pinned_dir / "shared_experts.arrow";
        if (fs::exists(shared_path)) {
            auto mmap_res = arrow::io::MemoryMappedFile::Open(
                shared_path.string(), arrow::io::FileMode::READ);
            if (!mmap_res.ok()) {
                log::error("[load_pinned] shared_experts open failed: "
                           + mmap_res.status().ToString());
                pw.Free();
                return cudaErrorInvalidValue;
            }
            SharedExpertDb db;
            auto st = load_shared_experts_metadata(*mmap_res, db);
            if (!st.ok()) {
                log::error("[load_pinned] shared_experts metadata: "
                           + st.ToString());
                pw.Free();
                return cudaErrorInvalidValue;
            }
            log::info("[load_pinned] shared_experts: "
                      + std::to_string(db.num_layers()) + " layers");
            for (int l = 0; l < db.num_layers() &&
                            l < static_cast<int>(pw.layers.size()); ++l) {
                auto &lw = pw.layers[l];
                auto upload = [&](const SharedExpertDescriptor &desc,
                                  uint8_t *&d_buf, uint8_t *&d_scales,
                                  float &gs, int64_t &nbytes) -> cudaError_t {
                    nbytes = desc.packed_bytes;
                    if (nbytes <= 0) return cudaSuccess;
                    cudaError_t ue = cudaMalloc(
                        &d_buf, static_cast<size_t>(desc.packed_bytes));
                    if (ue != cudaSuccess) return ue;
                    rb += static_cast<std::uint64_t>(desc.packed_bytes);
                    ue = ring.UploadRaw(desc.host_ptr, d_buf,
                                        static_cast<size_t>(desc.packed_bytes),
                                        stream, rb);
                    if (ue != cudaSuccess) return ue;
                    if (desc.scale_bytes > 0) {
                        ue = cudaMalloc(&d_scales,
                                        static_cast<size_t>(desc.scale_bytes));
                        if (ue != cudaSuccess) return ue;
                        rb += static_cast<std::uint64_t>(desc.scale_bytes);
                        ue = ring.UploadRaw(desc.scale_host_ptr, d_scales,
                                            static_cast<size_t>(desc.scale_bytes),
                                            stream, rb);
                        if (ue != cudaSuccess) return ue;
                    }
                    gs = desc.global_scale;
                    return cudaSuccess;
                };
                cudaError_t ue = upload(db.w1[l], lw.d_shared_w1,
                                        lw.d_shared_w1_scales,
                                        lw.shared_w1_global_scale,
                                        lw.shared_w1_bytes);
                if (ue == cudaSuccess)
                    ue = upload(db.w2[l], lw.d_shared_w2,
                                lw.d_shared_w2_scales,
                                lw.shared_w2_global_scale,
                                lw.shared_w2_bytes);
                if (ue == cudaSuccess)
                    ue = upload(db.w3[l], lw.d_shared_w3,
                                lw.d_shared_w3_scales,
                                lw.shared_w3_global_scale,
                                lw.shared_w3_bytes);
                if (ue != cudaSuccess) {
                    log::error("[load_pinned] shared expert upload failed "
                               "(layer " + std::to_string(l) + "): "
                               + cudaGetErrorString(ue));
                    pw.Free();
                    return ue;
                }
                // Load-time FP16/FP32 dequant caches (F-12 Task 4): the
                // same kernels the per-call path uses, executed ONCE at
                // load; bit-exact by construction, works with fast_math
                // either way. ~67 MB per layer across 43 layers.
                if (lw.shared_w1_bytes > 0 && lw.shared_w3_bytes > 0 &&
                    lw.shared_w2_bytes > 0) {
                    const int64_t w1el = lw.shared_w1_bytes;
                    const int64_t w2el = lw.shared_w2_bytes;
                    ue = cudaMalloc(&lw.d_shared_w1_h,
                                    static_cast<size_t>(w1el) * sizeof(__half));
                    if (ue == cudaSuccess)
                        ue = cudaMalloc(&lw.d_shared_w3_h,
                                        static_cast<size_t>(w1el) * sizeof(__half));
                    if (ue == cudaSuccess)
                        ue = cudaMalloc(&lw.d_shared_w2_f32,
                                        static_cast<size_t>(w2el) * sizeof(float));
                    if (ue == cudaSuccess)
                        ue = dequant_shared_f8_half(
                            stream, lw.d_shared_w1, lw.shared_w1_bytes,
                            lw.d_shared_w1_scales, 0, lw.d_shared_w1_h,
                            w1el, kHiddenDim);
                    if (ue == cudaSuccess)
                        ue = dequant_shared_f8_half(
                            stream, lw.d_shared_w3, lw.shared_w3_bytes,
                            lw.d_shared_w3_scales, 0, lw.d_shared_w3_h,
                            w1el, kHiddenDim);
                    if (ue == cudaSuccess)
                        ue = dequant_shared_f8(
                            stream, lw.d_shared_w2, lw.shared_w2_bytes,
                            lw.d_shared_w2_scales, 0, lw.d_shared_w2_f32,
                            w2el, kIntermediateDim);
                    if (ue == cudaSuccess) {
                        rb += (static_cast<std::uint64_t>(w1el) * 2 +
                               static_cast<std::uint64_t>(w2el) * 4);
                    } else {
                        log::error("[load_pinned] shared dequant cache "
                                   "failed (layer " + std::to_string(l) +
                                   "): " + cudaGetErrorString(ue));
                        cudaFree(lw.d_shared_w1_h);
                        cudaFree(lw.d_shared_w3_h);
                        cudaFree(lw.d_shared_w2_f32);
                        lw.d_shared_w1_h = nullptr;
                        lw.d_shared_w3_h = nullptr;
                        lw.d_shared_w2_f32 = nullptr;
                        ue = cudaSuccess;   // fall back to per-call dequant
                    }
                }
            }
        }
    }

    pw.resident_bytes = rb;

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double gib =
        static_cast<double>(rb) / (1024.0 * 1024.0 * 1024.0);
    const double gibs = (ms > 0.0) ? (gib / (ms / 1000.0)) : 0.0;
    log::info("[load_pinned] pinned weights: " + std::to_string(rb >> 20)
              + " MiB in " + std::to_string(static_cast<int>(ms + 0.5))
              + " ms (" + std::to_string(gibs) + " GiB/s)");

    return cudaSuccess;
}

// ── DeepSeekInferArena: single-allocation per-run footprint ────────────────
// Mirrors GlmMoEArena::AllocateGeneration's page() lambda: one device block +
// one pinned host block, 16-byte-aligned slices, sized exactly as the old
// malloc-per-buffer pipeline (deepseek_moe_inference_pipeline.cu) and
// alloc_csa_state. The CSA score buffers must start at -inf (the overlap
// pooling masks rows [0, ratio) before the first shift), everything else zero.

namespace {

__global__ void ds_fill_neg_inf_kernel(float* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = -INFINITY;
}

} // namespace

cudaError_t DeepSeekInferArena::Allocate(cudaStream_t stream, int num_layers,
                                         int max_seq_len,
                                         const int* compress_ratios, int vocab)
{
    Free();
    constexpr int64_t kIdxHeadDim = 128;

    // Sizing replicated verbatim from the old pipeline (lines 252-268) and
    // alloc_csa_state so the arena is a pure allocation refactor.
    const int64_t dequant_sz = static_cast<int64_t>(kHiddenDim) * kHiddenDim
                             * sizeof(float);
    const int64_t shared_int_sz = static_cast<int64_t>(kHiddenDim)
                                * sizeof(__half);
    const int64_t max_attn_elems = static_cast<int64_t>(kAttnHeads) * kHeadDim
                                 * kQloraRank;
    const int64_t max_attn_elems_wo = 8192LL * 4096;
    const int64_t attn_dequant_max = std::max(max_attn_elems, max_attn_elems_wo);
    // Extra FP32 scratch past the dequant block for the largest single attention
    // GEMM: f8_gemm_1d needs cols for the FP32 activation copy plus rows for the
    // FP32 output (clamped to FP16 afterwards). wq_b [32768,1024] needs
    // kQloraRank + kAttnHeads*kHeadDim; wo_b [4096,8192] needs 8192 + 4096. The
    // wq_b case (33,792) dominates. An undersized scratch here overflows into
    // the adjacent kv cache (corrupting layer-0 window slots on the first step).
    const int64_t act_plus_out = static_cast<int64_t>(kQloraRank)
                               + static_cast<int64_t>(kAttnHeads) * kHeadDim;
    const int64_t attn_dequant_sz = (attn_dequant_max + act_plus_out)
                                  * sizeof(float);
    const int64_t max_compressed = max_seq_len / 4 + 1;
    const int64_t kv_cache_sz = static_cast<int64_t>(kNumLayers)
                              * (kAttnWindow + max_compressed) * kHeadDim
                              * sizeof(__half);
    const int64_t pack_total = 3 * (static_cast<int64_t>(kIntermediateDim)
                                    * kHiddenDim / 2);
    // v1-nvfp4: per-16 scale arrays (2x the legacy per-32 size).
    const int64_t scale_total = 3 * (static_cast<int64_t>(kIntermediateDim)
                                     * kHiddenDim / 16);

    int64_t off = 0;
    auto page = [&](int64_t bytes_, int64_t align) -> int64_t {
        off = (off + align - 1) / align * align;
        const int64_t at = off;
        off += bytes_;
        return at;
    };
    const int64_t o_single       = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_single_f32   = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_normed       = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_ffn_out      = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_hidden       = page(static_cast<int64_t>(kHcDim) * sizeof(float), 16);
    const int64_t o_hidden_next  = page(static_cast<int64_t>(kHcDim) * sizeof(float), 16);
    const int64_t o_shared_int   = page(shared_int_sz, 16);
    const int64_t o_shared_gate  = page(shared_int_sz, 16);
    const int64_t o_routed_buf   = page(shared_int_sz, 16);
    const int64_t o_dequant      = page(dequant_sz, 16);
    const int64_t o_act_quant    = page(std::max(kHiddenDim, kIntermediateDim) * sizeof(float), 16);
    const int64_t o_ffn_accum    = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_swiglu       = page(static_cast<int64_t>(kIntermediateDim) * sizeof(float), 16);
    const int64_t o_act_scales   = page(((std::max(kHiddenDim, kIntermediateDim) + 127) / 128) * sizeof(float), 16);
    const int64_t o_gate_f32     = page(static_cast<int64_t>(kIntermediateDim) * sizeof(float), 16);
    const int64_t o_up_f32       = page(static_cast<int64_t>(kIntermediateDim) * sizeof(float), 16);
    const int64_t o_attn_dequant = page(attn_dequant_sz, 16);
    const int64_t o_kv_cache     = page(kv_cache_sz, 16);
    const int64_t o_q            = page(static_cast<int64_t>(kAttnHeads) * kHeadDim * sizeof(__half), 16);
    const int64_t o_qr           = page(static_cast<int64_t>(kQloraRank) * sizeof(__half), 16);
    const int64_t o_qr2          = page(static_cast<int64_t>(kQloraRank) * sizeof(__half), 16);
    const int64_t o_kv           = page(static_cast<int64_t>(kHeadDim) * sizeof(__half), 16);
    const int64_t o_kv2          = page(static_cast<int64_t>(kHeadDim) * sizeof(__half), 16);
    const int64_t o_o            = page(static_cast<int64_t>(kAttnHeads) * kHeadDim * sizeof(__half), 16);
    const int64_t o_olora        = page(static_cast<int64_t>(kOGroups) * kOloraRank * sizeof(__half), 16);
    const int64_t o_attn_out_h   = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_attn_out     = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_hc_post      = page(static_cast<int64_t>(kHcMult) * sizeof(float), 16);
    const int64_t o_hc_comb      = page(static_cast<int64_t>(kHcMult) * kHcMult * sizeof(float), 16);
    const int64_t o_hc_pre       = page(static_cast<int64_t>(kHcMult) * sizeof(float), 16);
    const int64_t o_logits       = page(static_cast<int64_t>(vocab) * sizeof(float), 16);
    const int64_t o_norm_scratch = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_norm_gamma   = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_next_token   = page(sizeof(int32_t), 16);
    const int64_t o_next_maxv    = page(sizeof(float), 16);
    const int64_t o_pack_stage   = page(pack_total, 16);
    const int64_t o_scale_stage  = page(scale_total, 16);
    const int64_t o_scale_stage_tc = page(scale_total, 16);
    const int64_t o_routing      = page(static_cast<int64_t>(kNumRouted) * sizeof(__half), 16);

    // Per-layer CSA region (exact alloc_csa_state sizes; ratio<=0 skipped).
    const int64_t o_csa = off;
    struct CsaOffsets {
        int64_t kv_state, score_state, compressed_kv;
        int64_t idx_kv_state, idx_score_state, idx_compressed_kv;
    };
    std::vector<CsaOffsets> csa_off(num_layers > 0 ? num_layers : 0);
    for (int l = 0; l < num_layers; ++l) {
        const int ratio = compress_ratios ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        const int coff = (ratio == 4) ? 2 : 1;
        const int state_sz = coff * ratio * coff * kHeadDim;
        const int max_comp = max_seq_len / ratio + 1;
        const int idx_state_sz = 2 * 4 * 2 * static_cast<int>(kIdxHeadDim);
        csa_off[static_cast<size_t>(l)].kv_state =
            page(static_cast<int64_t>(state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].score_state =
            page(static_cast<int64_t>(state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].compressed_kv =
            page(static_cast<int64_t>(max_comp) * kHeadDim * sizeof(__half), 16);
        csa_off[static_cast<size_t>(l)].idx_kv_state =
            page(static_cast<int64_t>(idx_state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].idx_score_state =
            page(static_cast<int64_t>(idx_state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].idx_compressed_kv =
            page(static_cast<int64_t>(max_comp) * kIdxHeadDim * sizeof(__half), 16);
    }
    bytes = static_cast<size_t>(off);

    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_base), bytes);
    if (e != cudaSuccess) { bytes = 0; Free(); return e; }

    // Pinned host block: top-k result slots + route_token logits staging.
    int64_t hoff = 0;
    auto hpage = [&](int64_t bytes_, int64_t align) -> int64_t {
        hoff = (hoff + align - 1) / align * align;
        const int64_t at = hoff;
        hoff += bytes_;
        return at;
    };
    const int64_t ho_topk    = hpage(static_cast<int64_t>(kTopK) * sizeof(int), 16);
    const int64_t ho_topw    = hpage(static_cast<int64_t>(kTopK) * sizeof(float), 16);
    const int64_t ho_routing = hpage(static_cast<int64_t>(kNumRouted) * sizeof(float), 16);
    host_bytes = static_cast<size_t>(hoff);

    e = cudaMallocHost(reinterpret_cast<void**>(&h_base), host_bytes);
    if (e != cudaSuccess) { host_bytes = 0; Free(); return e; }

    InferenceAllocCount().fetch_add(2);

    // Bind device slices.
    d_single       = reinterpret_cast<__half*>(d_base + o_single);
    d_single_f32   = reinterpret_cast<float*>(d_base + o_single_f32);
    d_normed       = reinterpret_cast<__half*>(d_base + o_normed);
    d_ffn_out      = reinterpret_cast<__half*>(d_base + o_ffn_out);
    d_hidden       = reinterpret_cast<float*>(d_base + o_hidden);
    d_hidden_next  = reinterpret_cast<float*>(d_base + o_hidden_next);
    d_shared_int   = reinterpret_cast<__half*>(d_base + o_shared_int);
    d_shared_gate  = reinterpret_cast<__half*>(d_base + o_shared_gate);
    d_routed_buf   = reinterpret_cast<__half*>(d_base + o_routed_buf);
    d_dequant_scratch = reinterpret_cast<float*>(d_base + o_dequant);
    d_act_quant    = reinterpret_cast<float*>(d_base + o_act_quant);
    d_ffn_accum    = reinterpret_cast<float*>(d_base + o_ffn_accum);
    d_swiglu_out   = reinterpret_cast<float*>(d_base + o_swiglu);
    d_act_scales   = reinterpret_cast<float*>(d_base + o_act_scales);
    d_gate_f32     = reinterpret_cast<float*>(d_base + o_gate_f32);
    d_up_f32       = reinterpret_cast<float*>(d_base + o_up_f32);
    d_attn_dequant = reinterpret_cast<float*>(d_base + o_attn_dequant);
    d_kv_cache     = reinterpret_cast<__half*>(d_base + o_kv_cache);
    d_q            = reinterpret_cast<__half*>(d_base + o_q);
    d_qr           = reinterpret_cast<__half*>(d_base + o_qr);
    d_qr2          = reinterpret_cast<__half*>(d_base + o_qr2);
    d_kv           = reinterpret_cast<__half*>(d_base + o_kv);
    d_kv2          = reinterpret_cast<__half*>(d_base + o_kv2);
    d_o            = reinterpret_cast<__half*>(d_base + o_o);
    d_olora        = reinterpret_cast<__half*>(d_base + o_olora);
    d_attn_out_h   = reinterpret_cast<__half*>(d_base + o_attn_out_h);
    d_attn_out     = reinterpret_cast<float*>(d_base + o_attn_out);
    d_hc_post      = reinterpret_cast<float*>(d_base + o_hc_post);
    d_hc_comb      = reinterpret_cast<float*>(d_base + o_hc_comb);
    d_hc_pre       = reinterpret_cast<float*>(d_base + o_hc_pre);
    d_logits       = reinterpret_cast<float*>(d_base + o_logits);
    d_norm_scratch = reinterpret_cast<float*>(d_base + o_norm_scratch);
    d_norm_gamma_scratch = reinterpret_cast<float*>(d_base + o_norm_gamma);
    d_next_token   = reinterpret_cast<int32_t*>(d_base + o_next_token);
    d_next_maxv    = reinterpret_cast<float*>(d_base + o_next_maxv);
    d_pack_stage   = d_base + o_pack_stage;
    d_scale_stage  = d_base + o_scale_stage;
    d_scale_stage_tc = d_base + o_scale_stage_tc;
    d_routing_scratch = reinterpret_cast<__half*>(d_base + o_routing);

    // Bind pinned slices.
    h_topk          = reinterpret_cast<int*>(h_base + ho_topk);
    h_top_weights   = reinterpret_cast<float*>(h_base + ho_topw);
    h_routing_scratch = reinterpret_cast<float*>(h_base + ho_routing);

    // Bind + init per-layer CSA state (same zero / -inf init as alloc_csa_state).
    csa_states = new CSAState[num_layers];
    for (int l = 0; l < num_layers; ++l) {
        const int ratio = compress_ratios ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        CSAState& s = csa_states[l];
        const CsaOffsets& co = csa_off[static_cast<size_t>(l)];
        s.d_comp_kv_state    = reinterpret_cast<float*>(d_base + co.kv_state);
        s.d_comp_score_state = reinterpret_cast<float*>(d_base + co.score_state);
        s.d_compressed_kv    = reinterpret_cast<__half*>(d_base + co.compressed_kv);
        s.d_idx_kv_state     = reinterpret_cast<float*>(d_base + co.idx_kv_state);
        s.d_idx_score_state  = reinterpret_cast<float*>(d_base + co.idx_score_state);
        s.d_idx_compressed_kv = reinterpret_cast<__half*>(d_base + co.idx_compressed_kv);
    }

    // CSA init (mirror alloc_csa_state): zero KV/compressed, -inf score states.
    for (int l = 0; l < num_layers; ++l) {
        const int ratio = compress_ratios ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        CSAState& s = csa_states[l];
        const int coff = (ratio == 4) ? 2 : 1;
        const int state_sz = coff * ratio * coff * kHeadDim;
        const int max_comp = max_seq_len / ratio + 1;
        const int idx_state_sz = 2 * 4 * 2 * static_cast<int>(kIdxHeadDim);

        if ((e = cudaMemsetAsync(s.d_comp_kv_state, 0,
                 static_cast<size_t>(state_sz) * sizeof(float), stream)) != cudaSuccess) goto fail;
        ds_fill_neg_inf_kernel<<<(state_sz + 255) / 256, 256, 0, stream>>>(
            s.d_comp_score_state, state_sz);
        if ((e = cudaGetLastError()) != cudaSuccess) goto fail;
        if ((e = cudaMemsetAsync(s.d_compressed_kv, 0,
                 static_cast<size_t>(max_comp) * kHeadDim * sizeof(__half), stream)) != cudaSuccess) goto fail;
        if ((e = cudaMemsetAsync(s.d_idx_kv_state, 0,
                 static_cast<size_t>(idx_state_sz) * sizeof(float), stream)) != cudaSuccess) goto fail;
        ds_fill_neg_inf_kernel<<<(idx_state_sz + 255) / 256, 256, 0, stream>>>(
            s.d_idx_score_state, idx_state_sz);
        if ((e = cudaGetLastError()) != cudaSuccess) goto fail;
        if ((e = cudaMemsetAsync(s.d_idx_compressed_kv, 0,
                 static_cast<size_t>(max_comp) * kIdxHeadDim * sizeof(__half), stream)) != cudaSuccess) goto fail;
    }
    return cudaSuccess;

fail:
    Free();
    return e;
}

void DeepSeekInferArena::Free()
{
    if (d_base) cudaFree(d_base);
    if (h_base) cudaFreeHost(h_base);
    delete[] csa_states;
    d_base = nullptr; h_base = nullptr; csa_states = nullptr;
    d_single = d_normed = d_ffn_out = nullptr;
    d_single_f32 = nullptr;
    d_hidden = d_hidden_next = nullptr;
    d_shared_int = d_shared_gate = d_routed_buf = nullptr;
    d_dequant_scratch = d_act_quant = d_ffn_accum = d_swiglu_out = nullptr;
    d_act_scales = d_gate_f32 = d_up_f32 = d_attn_dequant = nullptr;
    d_kv_cache = nullptr;
    d_q = d_qr = d_qr2 = d_kv = d_kv2 = d_o = d_olora = d_attn_out_h = nullptr;
    d_attn_out = nullptr;
    d_hc_post = d_hc_comb = d_hc_pre = nullptr;
    d_logits = d_norm_scratch = d_norm_gamma_scratch = nullptr;
    d_next_token = nullptr;
    d_next_maxv = nullptr;
    d_pack_stage = d_scale_stage = d_scale_stage_tc = nullptr;
    d_routing_scratch = nullptr;
    h_topk = nullptr;
    h_top_weights = nullptr;
    h_routing_scratch = nullptr;
    bytes = 0; host_bytes = 0;
}

} // namespace gpu
} // namespace lsxcommon
