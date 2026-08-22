#include "lsxcommon/gemma4_inference_load.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>

#include "glm_load_internal.h"

namespace lsxcommon {
namespace gemma {

namespace fs = std::filesystem;

namespace {

using glm::PinnedStagingRing;

const TensorRecord *FindTensor(const std::vector<TensorRecord> &tensors,
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

constexpr size_t kArenaAlign = 256;

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct Fp16Job {
    const TensorRecord *t;
    __half **dst;
    int64_t *rows; // optional dim0 capture (2D only)
    int64_t *cols; // optional dim1 capture (2D only)
};

cudaError_t ValidateJob(const Fp16Job &j, size_t &numel, bool &is_2d)
{
    const auto &s = j.t->shape;
    if (s.size() == 1 && s[0] > 0) {
        is_2d = false;
        numel = static_cast<size_t>(s[0]);
    } else if (s.size() == 2 && s[0] > 0 && s[1] > 0) {
        is_2d = true;
        numel = static_cast<size_t>(s[0]) * static_cast<size_t>(s[1]);
    } else {
        log::error("[gemma_load] malformed tensor '" + j.t->name + "'");
        return cudaErrorInvalidValue;
    }
    if (j.t->data.size() < numel * 2) {
        log::error("[gemma_load] malformed tensor '" + j.t->name + "'");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t UploadJobs(std::vector<Fp16Job> jobs, GemmaLayerWeights &lw,
                       cudaStream_t stream, PinnedStagingRing &ring,
                       std::uint64_t &bytes_acc)
{
    struct Shape {
        size_t numel = 0;
        bool is_2d = false;
    };
    std::vector<Shape> shapes(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        cudaError_t e = ValidateJob(jobs[i], shapes[i].numel, shapes[i].is_2d);
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
            log::error("[gemma_load] arena alloc failed: " +
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
        if (shapes[i].is_2d && j.rows) {
            *j.rows = j.t->shape[0];
            if (j.cols) *j.cols = j.t->shape[1];
        }
        cudaError_t e = ring.UploadBf16AsFp16(j.t->data.data(),
                                              static_cast<int64_t>(shapes[i].numel),
                                              view, stream, bytes_acc);
        if (e != cudaSuccess) return e;
        off += shapes[i].numel * 2;
    }
    return cudaSuccess;
}

// BF16 host -> F32 host (exact: BF16 is a truncated F32).
std::vector<float> Bf16BytesToF32(const std::vector<uint8_t> &data,
                                  int64_t numel)
{
    std::vector<float> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        uint32_t wide = static_cast<uint32_t>(
                            data[static_cast<size_t>(i) * 2]) |
                        (static_cast<uint32_t>(
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
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
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

cudaError_t LoadGlobal(const fs::path &pinned_dir, GemmaPinnedWeights &pw,
                       cudaStream_t stream, std::uint64_t &bytes_acc)
{
    auto tensors_res = LoadArrowFile((pinned_dir / "global.arrow").string());
    if (!tensors_res.ok()) {
        log::error("[gemma_load] ERROR loading global.arrow: " +
                   tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    auto &tensors = *tensors_res;

    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    auto *embed = FindTensor(tensors, "embed_tokens.weight");
    if (embed && embed->shape.size() == 2 &&
        embed->data.size() >=
            static_cast<size_t>(embed->shape[0]) *
                static_cast<size_t>(embed->shape[1]) * 2) {
        pw.vocab_size = static_cast<int>(embed->shape[0]);
        pw.hidden = static_cast<int>(embed->shape[1]);
        cudaError_t e = cudaMalloc(
            &pw.d_embed,
            static_cast<size_t>(pw.vocab_size) * pw.hidden * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(embed->data.data(),
                                  static_cast<int64_t>(pw.vocab_size) *
                                      pw.hidden,
                                  pw.d_embed, stream, bytes_acc);
        if (e != cudaSuccess) {
            log::error(std::string("[gemma_load] embed upload failed: ") +
                       cudaGetErrorString(e));
            return e;
        }
    } else {
        log::error("[gemma_load] embed_tokens.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
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
            log::error(std::string("[gemma_load] final norm upload failed: ") +
                       cudaGetErrorString(e));
            return e;
        }
    } else {
        log::error("[gemma_load] model.norm.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t LoadLayer(const std::vector<TensorRecord> &tensors,
                      GemmaLayerWeights &lw, cudaStream_t stream,
                      std::uint64_t &bytes_acc)
{
    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[gemma_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }

    std::vector<Fp16Job> jobs = {
        {in_norm, &lw.d_input_norm, nullptr, nullptr},
        {post_norm, &lw.d_post_norm, nullptr, nullptr},
        {FindTensor(tensors, "pre_feedforward_layernorm_2.weight"),
         &lw.d_pre_ffn2, nullptr, nullptr},
        {FindTensor(tensors, "post_feedforward_layernorm_2.weight"),
         &lw.d_post_ffn2, nullptr, nullptr},
        {FindTensor(tensors, "pre_feedforward_layernorm.weight"),
         &lw.d_pre_ffn, nullptr, nullptr},
        {FindTensor(tensors, "post_feedforward_layernorm.weight"),
         &lw.d_post_ffn, nullptr, nullptr},
        {FindTensor(tensors, "self_attn.q_proj.weight"), &lw.d_q,
         &lw.q_rows, nullptr},
        {FindTensor(tensors, "self_attn.k_proj.weight"), &lw.d_k,
         &lw.kv_rows, nullptr},
        {FindTensor(tensors, "self_attn.v_proj.weight"), &lw.d_v,
         nullptr, nullptr},
        {FindTensor(tensors, "self_attn.o_proj.weight"), &lw.d_o,
         nullptr, &lw.o_cols},
        {FindTensor(tensors, "self_attn.q_norm.weight"), &lw.d_q_norm,
         nullptr, nullptr},
        {FindTensor(tensors, "self_attn.k_norm.weight"), &lw.d_k_norm,
         nullptr, nullptr},
        {FindTensor(tensors, "mlp.gate_proj.weight"), &lw.d_mlp_gate,
         nullptr, nullptr},
        {FindTensor(tensors, "mlp.up_proj.weight"), &lw.d_mlp_up,
         nullptr, nullptr},
        {FindTensor(tensors, "mlp.down_proj.weight"), &lw.d_mlp_down,
         nullptr, nullptr},
        {FindTensor(tensors, "router.proj.weight"), &lw.d_router_proj,
         nullptr, nullptr},
    };

    cudaError_t e = UploadJobs(std::move(jobs), lw, stream, ring, bytes_acc);
    if (e != cudaSuccess) return e;

    // F32 / BF16-scalar side uploads.
    if (auto *sc = FindTensor(tensors, "layer_scalar");
        sc && sc->dtype == "F32" && sc->shape.size() == 1 &&
        sc->data.size() >= sizeof(float)) {
        e = UploadF32(reinterpret_cast<const float *>(sc->data.data()), 1,
                      &lw.d_layer_scalar, stream, ring, bytes_acc);
        if (e != cudaSuccess) return e;
    }
    if (auto *pes = FindTensor(tensors, "router.per_expert_scale");
        pes && pes->dtype == "F32" && pes->shape.size() == 1 &&
        pes->data.size() >= static_cast<size_t>(pes->shape[0]) * 4) {
        e = UploadF32(reinterpret_cast<const float *>(pes->data.data()),
                      pes->shape[0], &lw.d_per_expert_scale, stream, ring,
                      bytes_acc);
        if (e != cudaSuccess) return e;
    }
    if (auto *rs = FindTensor(tensors, "router.scale");
        rs && rs->dtype == "BF16" && rs->shape.size() == 1 &&
        rs->data.size() >= static_cast<size_t>(rs->shape[0]) * 2) {
        auto f32 = Bf16BytesToF32(rs->data, rs->shape[0]);
        e = UploadF32(f32.data(), rs->shape[0], &lw.d_router_scale, stream,
                      ring, bytes_acc);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace

void GemmaLayerWeights::Free()
{
    if (d_arena) {
        cudaFree(d_arena);
    }
    if (d_layer_scalar) cudaFree(d_layer_scalar);
    if (d_router_scale) cudaFree(d_router_scale);
    if (d_per_expert_scale) cudaFree(d_per_expert_scale);
    *this = GemmaLayerWeights{};
}

void GemmaPinnedWeights::Free()
{
    if (d_embed) cudaFree(d_embed);
    if (d_final_norm) cudaFree(d_final_norm);
    d_embed = d_final_norm = nullptr;
    vocab_size = 0;
    hidden = 0;
    resident_bytes = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
}

cudaError_t LoadGemmaPinnedWeights(const std::string &model_path,
                                   GemmaPinnedWeights &pw,
                                   cudaStream_t stream)
{
    namespace fs = std::filesystem;
    pw.Free();

    fs::path pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir) &&
        fs::exists(fs::path(model_path) / "pinned"))
        pinned_dir = fs::path(model_path) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[gemma_load] pinned dir not found under " + model_path);
        return cudaErrorInvalidValue;
    }

    std::uint64_t bytes_acc = 0;
    cudaError_t e = LoadGlobal(pinned_dir, pw, stream, bytes_acc);
    if (e != cudaSuccess) return e;

    // Enumerate layer files (heterogeneous grid tolerated: missing layer
    // files are skipped, matching GLM).
    int max_layer = -1;
    for (const auto &entry : fs::directory_iterator(pinned_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("layer_", 0) != 0) continue;
        if (name.size() < 7 || name.substr(name.size() - 6) != ".arrow")
            continue;
        int l = std::atoi(name.c_str() + 6);
        max_layer = std::max(max_layer, l);
    }
    if (max_layer < 0) {
        log::error("[gemma_load] no layer_*.arrow files in " +
                   pinned_dir.string());
        return cudaErrorInvalidValue;
    }
    pw.layers.assign(static_cast<size_t>(max_layer + 1), GemmaLayerWeights{});

    for (int l = 0; l <= max_layer; ++l) {
        auto path = pinned_dir / absl::StrCat("layer_", l, ".arrow");
        if (!fs::exists(path)) continue; // heterogeneous grid: skip
        auto tensors_res = LoadArrowFile(path.string());
        if (!tensors_res.ok()) {
            log::error("[gemma_load] ERROR loading " + path.string() + ": " +
                       tensors_res.status().ToString());
            return cudaErrorInvalidValue;
        }
        e = LoadLayer(*tensors_res, pw.layers[static_cast<size_t>(l)], stream,
                      bytes_acc);
        if (e != cudaSuccess) return e;
    }

    pw.resident_bytes = bytes_acc;
    return cudaSuccess;
}

// ── Expert store wiring ─────────────────────────────────────────────────────

GemmaExpertDb load_gemma_expert_index(const std::string &model_path,
                                      int num_layers, int experts_per_layer)
{
    GemmaExpertDb db;
    db.log_prefix = "[gemma_load] ";
    db.mmap_fail_verb = "gemma expert mmap fail ";
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.Load(model_path, /*fallback=*/true);
    return db;
}

void ensure_gemma_expert_resident(GemmaExpertDb &db, int layer, int expert,
                                  cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_gemma_expert_index(GemmaExpertDb &db) { db.Free(); }

} // namespace gemma
} // namespace lsxcommon
