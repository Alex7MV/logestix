#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/glm_inference.h"
#include "lsxcommon/glm_convert.h"
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

#include "glm_load_internal.h"

namespace lsxcommon {
namespace glm {

namespace {

cudaError_t UploadRaw(const std::vector<uint8_t> &data, float **d_out,
                      cudaStream_t stream, std::uint64_t &bytes_acc)
{
    if (data.empty()) return cudaSuccess;
    float *d = nullptr;
    cudaError_t e = cudaMalloc(&d, data.size());
    if (e != cudaSuccess) return e;
    bytes_acc += data.size();
    e = cudaMemcpyAsync(d, data.data(), data.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaFree(d);
        return e;
    }
    *d_out = d;
    return cudaSuccess;
}

// Uploads one NVFP4-packed expert tensor (packed E2M1 nibbles + per-block E4M3
// scales) into a GlmExpertDeviceBlock slot. On any failure the slot stays
// inactive (null payload pointers) — the deterministic zero-contribution skip.
cudaError_t UploadExpertTensor(const std::vector<uint8_t> &packed,
                               const std::vector<uint8_t> &scales,
                               float global_scale,
                               const uint8_t **d_packed,
                               const uint8_t **d_scales,
                               float *out_global_scale,
                               int64_t *out_packed_bytes,
                               int64_t *out_scale_bytes,
                               cudaStream_t stream)
{
    void *d1 = nullptr;
    void *d2 = nullptr;
    cudaError_t e = cudaMalloc(&d1, packed.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d1, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d1); return e; }
    e = cudaMalloc(&d2, scales.size());
    if (e != cudaSuccess) { cudaFree(d1); return e; }
    e = cudaMemcpyAsync(d2, scales.data(), scales.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d1); cudaFree(d2); return e; }
    *d_packed         = static_cast<const uint8_t *>(d1);
    *d_scales         = static_cast<const uint8_t *>(d2);
    *out_global_scale = global_scale;
    *out_packed_bytes = static_cast<int64_t>(packed.size());
    *out_scale_bytes  = static_cast<int64_t>(scales.size());
    return cudaSuccess;
}

// Frees the six payload pointers of one NVFP4 expert descriptor.
void FreeExpertBlock(GlmExpertDeviceBlock &b)
{
    cudaFree(const_cast<uint8_t *>(b.d_w1_packed));
    cudaFree(const_cast<uint8_t *>(b.d_w1_scales));
    cudaFree(const_cast<uint8_t *>(b.d_w2_packed));
    cudaFree(const_cast<uint8_t *>(b.d_w2_scales));
    cudaFree(const_cast<uint8_t *>(b.d_w3_packed));
    cudaFree(const_cast<uint8_t *>(b.d_w3_scales));
    b = GlmExpertDeviceBlock{};
}

const TensorRecord *FindTensor(const std::vector<TensorRecord> &tensors,
                               const std::string &suffix)
{
    for (const auto &t : tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(),
                           suffix.size(), suffix) == 0)
            return &t;
    }
    return nullptr;
}

arrow::Result<std::vector<TensorRecord>> LoadArrowFile(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok()) return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    return PinnedSchema::UnpackBatch(*batch_res);
}

// Found-but-malformed => error; absent => ok (optional tensor in the
// heterogeneous layer grid). Validation lives in ValidateJob/UploadJobs.

cudaError_t LoadGlobal(const std::filesystem::path &pinned_dir,
                       GlmPinnedWeights &pw, cudaStream_t stream,
                       std::uint64_t &bytes_acc)
{
    auto tensors_res = LoadArrowFile((pinned_dir / "global.arrow").string());
    if (!tensors_res.ok()) {
        log::error("[glm_load] ERROR loading global.arrow: "
                   + tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    auto &tensors = *tensors_res;

    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    auto *embed = FindTensor(tensors, "embed_tokens.weight");
    if (embed && embed->shape.size() >= 2 &&
        embed->data.size() >=
            static_cast<size_t>(embed->shape[0]) *
                static_cast<size_t>(embed->shape[1]) * 2) {
        pw.vocab_size = static_cast<int>(embed->shape[0]);
        pw.hidden = static_cast<int>(embed->shape[1]);
        cudaError_t e = cudaMalloc(&pw.d_embed,
            static_cast<size_t>(pw.vocab_size) * pw.hidden * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(embed->data.data(),
                                  static_cast<int64_t>(pw.vocab_size) *
                                      pw.hidden,
                                  pw.d_embed, stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[glm_load] embed upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    } else {
        log::error("[glm_load] embed_tokens.weight missing/malformed in "
                   "global.arrow — model cannot run without it");
        return cudaErrorInvalidValue;
    }
    auto *head = FindTensor(tensors, "lm_head.weight");
    if (head && head->shape.size() >= 2 && head != embed &&
        head->data.size() >=
            static_cast<size_t>(head->shape[0]) *
                static_cast<size_t>(head->shape[1]) * 2) {
        int64_t numel = head->shape[0] * head->shape[1];
        cudaError_t e = cudaMalloc(&pw.d_head, static_cast<size_t>(numel) * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(head->data.data(), numel, pw.d_head,
                                  stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[glm_load] head upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    }
    auto *norm = FindTensor(tensors, "model.norm.weight");
    if (norm && norm->shape.size() == 1 && norm->shape[0] > 0 &&
        norm->data.size() >= static_cast<size_t>(norm->shape[0]) * 2) {
        cudaError_t e = cudaMalloc(&pw.d_final_norm,
                                   static_cast<size_t>(norm->shape[0]) * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(norm->data.data(), norm->shape[0],
                                  pw.d_final_norm, stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[glm_load] model.norm upload failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    } else {
        log::error("[glm_load] model.norm.weight missing/malformed in "
                   "global.arrow — final RMSNorm cannot run without it");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}



constexpr size_t kArenaAlign = 256;

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};

struct Fp16Job {
    const TensorRecord *t;
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
    const auto &s = j.t->shape;
    if (s.size() == 1 && s[0] > 0) {
        sh.is_2d = false;
        sh.numel = static_cast<size_t>(s[0]);
    } else if (s.size() == 2 && s[0] > 0 && s[1] > 0) {
        sh.is_2d = true;
        sh.numel = static_cast<size_t>(s[0]) * static_cast<size_t>(s[1]);
    } else {
        log::error("[glm_load] malformed tensor '" + j.t->name + "'");
        return cudaErrorInvalidValue;
    }
    if (j.t->data.size() < sh.numel * 2) {
        log::error("[glm_load] malformed tensor '" + j.t->name + "'");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// Validates every job, issues ONE arena allocation sized to the layer's FP16
// footprint and uploads each tensor through the pinned ring into its
// 256-byte-aligned arena view. On success lw.d_arena/arena_bytes own the
// block; every *j.dst points inside it.
cudaError_t UploadJobs(std::vector<Fp16Job> jobs, GlmLayerWeights &lw,
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
            log::error("[glm_load] arena alloc failed: " +
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
        cudaError_t e = ring.UploadBf16AsFp16(j.t->data.data(),
                            static_cast<int64_t>(shapes[i].numel),
                            view, stream, bytes_acc);
        if (e != cudaSuccess) return e;
        off += shapes[i].numel * 2;
    }
    return cudaSuccess;
}


} // anonymous namespace

cudaError_t LoadSingleLayer(const std::vector<TensorRecord> &tensors,
                            GlmLayerWeights &lw, cudaStream_t stream,
                            std::uint64_t &bytes_acc)
{
    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // Required-for-type validation (heterogeneous-grid rules):
    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[glm_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }
    const bool moe = FindTensor(tensors, "mlp.gate.weight") != nullptr;
    if (!moe && !FindTensor(tensors, "mlp.gate_proj.weight")) {
        log::error("[glm_load] dense layer missing mlp.gate_proj");
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

    jobs.push_back({FindTensor(tensors, "self_attn.indexer.wq_b.weight"),
                    &lw.d_idx_wq_b, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "self_attn.indexer.wk.weight"),
                    &lw.d_idx_wk, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "self_attn.indexer.weights_proj.weight"),
                    &lw.d_idx_weights_proj, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "self_attn.indexer.k_norm.weight"),
                    &lw.d_idx_k_norm, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "eh_proj.weight"),
                    &lw.d_eh_proj, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "enorm.weight"),
                    &lw.d_enorm, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "hnorm.weight"),
                    &lw.d_hnorm, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "shared_head.norm.weight"),
                    &lw.d_shared_head_norm, nullptr, nullptr});

    cudaError_t e = UploadJobs(std::move(jobs), lw, stream, ring, bytes_acc);
    if (e != cudaSuccess) return e;

    auto *router = FindTensor(tensors, "mlp.gate.weight");
    if (router) {
        auto *gate_bias =
            FindTensor(tensors, "mlp.gate.e_score_correction_bias");
        if (gate_bias && gate_bias->shape.size() == 1 &&
            gate_bias->data.size() >=
                static_cast<size_t>(gate_bias->shape[0]) * 4) {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias,
                               static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data.data(),
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }
    }

    auto *k_bias = FindTensor(tensors, "self_attn.indexer.k_norm.bias");
    if (k_bias && !k_bias->data.empty()) {
        void *d = nullptr;
        cudaError_t ke = cudaMalloc(&d, k_bias->data.size());
        if (ke != cudaSuccess) return ke;
        lw.d_idx_k_norm_bias = static_cast<float *>(d);
        ke = ring.UploadRaw(k_bias->data.data(), d, k_bias->data.size(),
                            stream, bytes_acc);
        if (ke != cudaSuccess) return ke;
    }
    return cudaSuccess;
}


void GlmLayerWeights::Free()
{
    if (d_arena) {
        if (h_gate_bias) cudaFreeHost(h_gate_bias);
        cudaFree(d_arena);
        *this = GlmLayerWeights{};
        return;
    }
    if (d_input_norm)          cudaFree(d_input_norm);
    if (d_post_attn_norm)      cudaFree(d_post_attn_norm);
    if (d_q_a_proj)            cudaFree(d_q_a_proj);
    if (d_q_a_norm)            cudaFree(d_q_a_norm);
    if (d_q_b_proj)            cudaFree(d_q_b_proj);
    if (d_kv_a_proj)           cudaFree(d_kv_a_proj);
    if (d_kv_a_norm)           cudaFree(d_kv_a_norm);
    if (d_kv_b_proj)           cudaFree(d_kv_b_proj);
    if (d_o_proj)              cudaFree(d_o_proj);
    if (d_router_gate)         cudaFree(d_router_gate);
    if (h_gate_bias)           cudaFreeHost(h_gate_bias);
    if (d_mlp_gate)            cudaFree(d_mlp_gate);
    if (d_mlp_down)            cudaFree(d_mlp_down);
    if (d_mlp_up)              cudaFree(d_mlp_up);
    if (d_idx_wq_b)            cudaFree(d_idx_wq_b);
    if (d_idx_wk)              cudaFree(d_idx_wk);
    if (d_idx_weights_proj)    cudaFree(d_idx_weights_proj);
    if (d_idx_k_norm)          cudaFree(d_idx_k_norm);
    if (d_idx_k_norm_bias)     cudaFree(d_idx_k_norm_bias);
    if (d_eh_proj)             cudaFree(d_eh_proj);
    if (d_enorm)               cudaFree(d_enorm);
    if (d_hnorm)               cudaFree(d_hnorm);
    if (d_shared_head_norm)    cudaFree(d_shared_head_norm);

    d_input_norm = d_post_attn_norm = d_q_a_proj = d_q_a_norm = nullptr;
    d_q_b_proj = d_kv_a_proj = d_kv_a_norm = d_kv_b_proj = d_o_proj = nullptr;
    d_router_gate = nullptr; h_gate_bias = nullptr; gate_bias_n = 0;
    d_mlp_gate = d_mlp_down = d_mlp_up = nullptr;
    d_idx_wq_b = d_idx_wk = d_idx_weights_proj = nullptr;
    d_idx_k_norm = nullptr; d_idx_k_norm_bias = nullptr;
    d_eh_proj = d_enorm = d_hnorm = d_shared_head_norm = nullptr;
}

void GlmPinnedWeights::Free()
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
}

cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream,
                                 const std::vector<int> *layer_filter)
{
    namespace fs = std::filesystem;

    // Idempotent re-load: drop any previously uploaded set first (Free() is
    // safe on a nulled/empty set), so calling Load twice without an explicit
    // Free() in between cannot leak the old device allocations.
    pw.Free();

    const auto t0 = std::chrono::steady_clock::now();

    // Accept both the logestix root itself and its container directory:
    //   <model>/logestix/pinned/...   (model dir, like the DeepSeek loader)
    //   <logestix>/pinned/...         (converter output_root)
    fs::path pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    fs::path index_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(pinned_dir) && fs::exists(fs::path(model_path) / "pinned")) {
        pinned_dir = fs::path(model_path) / "pinned";
        index_path = fs::path(model_path) / "model.index.json";
    }

    // Topology: number of layers sizes the layers vector (heterogeneous grid).
    auto topo = load_topology(index_path.string());
    if (!topo.error_msg.empty()) {
        log::error("[glm_load] topology error: " + topo.error_msg);
        return cudaErrorInvalidValue;
    }
    pw.layers.resize(static_cast<size_t>(topo.num_layers));

    // Drains the stream on every error exit below so pinned staging never
    // dies while a DMA transfer is still in flight.
    StreamDrain drain{stream};

    auto err = LoadGlobal(pinned_dir, pw, stream, pw.resident_bytes);
    if (err != cudaSuccess) return err;

    for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
        if (layer_filter &&
            std::find(layer_filter->begin(), layer_filter->end(), l) ==
                layer_filter->end())
            continue;
        auto fn = absl::StrCat("layer_", l, ".arrow");
        auto layer_path = pinned_dir / fn;
        if (!fs::exists(layer_path)) {
            log::error("[glm_load] required layer file missing: " + fn);
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto tensors_res = LoadArrowFile(layer_path.string());
        if (!tensors_res.ok()) {
            log::error("[glm_load] ERROR loading " + fn + ": "
                       + tensors_res.status().ToString());
            pw.Free();
            return cudaErrorInvalidValue;
        }
        cudaError_t lerr =
            LoadSingleLayer(*tensors_res, pw.layers[static_cast<size_t>(l)],
                            stream, pw.resident_bytes);
        if (lerr != cudaSuccess) {
            pw.Free();
            return lerr;
        }
    }

    // Shared experts: one NVFP4-packed PackedExpert per layer in
    // pinned/shared_experts.arrow (w1=gate, w2=down, w3=up). Kept host-side in
    // pw.shared_host (index = layer); the faithful generation pass stages each
    // layer's shared expert into the reusable per-expert device staging
    // buffers on demand, so the ~1.4 GiB shared set costs no VRAM.
    pw.shared_host.assign(pw.layers.size(), PackedExpert{});
    auto se_path = pinned_dir / "shared_experts.arrow";
    if (fs::exists(se_path)) {
        auto se_batch_res = ReadRecordBatchFromFile(se_path);
        if (se_batch_res.ok()) {
            auto sh_res = SharedExpertSchema::UnpackBatch(*se_batch_res);
            if (sh_res.ok()) {
                auto &sh = *sh_res;
                for (size_t l = 0;
                     l < sh.size() && l < pw.layers.size(); ++l) {
                    pw.shared_host[l] = std::move(sh[l]);
                }
            } else {
                log::warn("[glm_load] shared_experts.arrow unpack failed: " +
                          sh_res.status().ToString());
            }
        } else {
            log::warn("[glm_load] shared_experts.arrow read failed: " +
                      se_batch_res.status().ToString());
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double gib =
        static_cast<double>(pw.resident_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double gibs = (ms > 0.0) ? (gib / (ms / 1000.0)) : 0.0;
    log::info(absl::StrCat("[glm_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms (", gibs,
                           " GiB/s)"));

    return cudaSuccess;
}

cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream)
{
    return LoadGlmPinnedWeights(model_path, pw, stream, nullptr);
}

void GlmExpertTables::Free()
{
    for (auto &tab : tables)
        for (auto &b : tab) FreeExpertBlock(b);
    tables.clear();
}

// ── On-demand routed-expert index (layers 3-78) ─────────────────────────────
namespace {

// No per-model parser needed: the shared ExpertStore<TEntry>::DecodeEntry()
// extracts the six ExpertSchema bins + three global scales into TEntry.

} // anonymous namespace

GlmExpertDb load_glm_expert_index(const std::string &model_path,
                                  int num_layers, int experts_per_layer)
{
    GlmExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.log_prefix = "[glm_load] ";
    db.mmap_fail_verb = "expert mmap fail ";
    db.emit_diag = false;            // GLM emits no [diag-expert] block
    db.log_batch_fails = true;       // GLM logs batch-failure lines
    db.Load(model_path, /*fallback=*/true);
    return db;
}

void ensure_glm_expert_resident(GlmExpertDb &db, int layer, int expert,
                                cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_glm_expert_index(GlmExpertDb &db) { db.Free(); }

cudaError_t LoadGlmExpertTables(const std::string &model_path,
                                GlmExpertTables &experts,
                                cudaStream_t stream,
                                const std::vector<int> *layer_filter,
                                int max_experts)
{
    namespace fs = std::filesystem;

    // Idempotent re-load: drop any previously loaded expert tables first so a
    // second Load without an explicit Free() cannot leak device allocations.
    experts.Free();

    // Accept both the logestix root and its container directory, mirroring
    // the pinned loader:
    //   <model>/logestix/experts/layer_<l>/exp_<e>.arrow
    //   <model>/experts/layer_<l>/exp_<e>.arrow
    fs::path exp_dir = fs::path(model_path) / "logestix" / "experts";
    if (!fs::exists(exp_dir) && fs::exists(fs::path(model_path) / "experts"))
        exp_dir = fs::path(model_path) / "experts";

    fs::path index_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(index_path) &&
        fs::exists(fs::path(model_path) / "model.index.json"))
        index_path = fs::path(model_path) / "model.index.json";

    auto topo = load_topology(index_path.string());
    if (!topo.error_msg.empty()) {
        log::error("[glm_load] expert topology error: " + topo.error_msg);
        return cudaErrorInvalidValue;
    }

    experts.tables.resize(static_cast<size_t>(topo.num_layers));
    for (auto &tab : experts.tables)
        tab.resize(static_cast<size_t>(kGlmNumRouted));  // all-inactive default

    for (int l = 0; l < topo.num_layers; ++l) {
        if (layer_filter &&
            std::find(layer_filter->begin(), layer_filter->end(), l) ==
                layer_filter->end())
            continue;
        auto layer_dir = exp_dir / absl::StrCat("layer_", l);
        if (!fs::is_directory(layer_dir)) continue;
        auto &tab = experts.tables[static_cast<size_t>(l)];
        for (int e = 0; e < max_experts; ++e) {
            auto p = layer_dir / absl::StrCat("exp_", e, ".arrow");
            if (!fs::exists(p)) continue;
            auto batch_res = ReadRecordBatchFromFile(p);
            if (!batch_res.ok()) continue;
            auto pe_res = ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) continue;
            const PackedExpert &pe = *pe_res;
            if (pe.w1_packed.empty() || pe.w2_packed.empty() ||
                pe.w3_packed.empty() || pe.w1_scales.empty() ||
                pe.w2_scales.empty() || pe.w3_scales.empty())
                continue;

            GlmExpertDeviceBlock b;
            cudaError_t err = UploadExpertTensor(
                pe.w1_packed, pe.w1_scales, pe.w1_global_scale,
                &b.d_w1_packed, &b.d_w1_scales, &b.w1_global_scale,
                &b.w1_packed_bytes, &b.w1_scale_bytes, stream);
            if (err == cudaSuccess)
                err = UploadExpertTensor(
                    pe.w2_packed, pe.w2_scales, pe.w2_global_scale,
                    &b.d_w2_packed, &b.d_w2_scales, &b.w2_global_scale,
                    &b.w2_packed_bytes, &b.w2_scale_bytes, stream);
            if (err == cudaSuccess)
                err = UploadExpertTensor(
                    pe.w3_packed, pe.w3_scales, pe.w3_global_scale,
                    &b.d_w3_packed, &b.d_w3_scales, &b.w3_global_scale,
                    &b.w3_packed_bytes, &b.w3_scale_bytes, stream);
            if (err != cudaSuccess) {
                log::error(absl::StrCat("[glm_load] expert upload failed: layer ", l,
                            " exp ", e));
                FreeExpertBlock(b);  // slot stays inactive
                continue;
            }
            tab[static_cast<size_t>(e)] = b;
        }
    }

    return cudaSuccess;
}

} // namespace glm
} // namespace lsxcommon
