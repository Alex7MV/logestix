#include "lsxcommon/gemma_moe_load.h"
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

#include "glm_moe_load_internal.h"

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
        e = ring.UploadRaw(embed->data.data(), pw.d_embed,
                           static_cast<size_t>(pw.vocab_size) * pw.hidden * 2,
                           stream, bytes_acc);
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
        e = ring.UploadRaw(norm->data.data(), pw.d_final_norm,
                           static_cast<size_t>(norm->shape[0]) * 2, stream,
                           bytes_acc);
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
                      GemmaLayerWeights &lw, float *h_layer_scalar,
                      cudaStream_t stream,
                      std::uint64_t &bytes_acc)
{
    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // NVFP4 dense-MLP triple detection: "<name>.packed" + ".scales" +
    // ".global_scale" records (converted schema). When all three projections
    // are present the BF16 MLP upload jobs are dropped.
    struct Nvfp4Recs {
        const TensorRecord *packed = nullptr;
        const TensorRecord *scales = nullptr;
        const TensorRecord *gs = nullptr;
    };
    auto find_rec = [&](std::string_view suffix) -> const TensorRecord * {
        for (const auto &t : tensors)
            if (t.name.size() > suffix.size() &&
                t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                               suffix) == 0)
                return &t;
        return nullptr;
    };
    auto gather = [&](std::string_view base) {
        Nvfp4Recs r;
        r.packed = find_rec(std::string(base) + ".packed");
        r.scales = find_rec(std::string(base) + ".scales");
        r.gs = find_rec(std::string(base) + ".global_scale");
        return r;
    };
    const Nvfp4Recs gate_rec =
        gather("mlp.gate_proj.weight");
    const Nvfp4Recs up_rec = gather("mlp.up_proj.weight");
    const Nvfp4Recs down_rec = gather("mlp.down_proj.weight");
    const bool mlp_nv = gate_rec.packed && gate_rec.scales && gate_rec.gs &&
                        up_rec.packed && up_rec.scales && up_rec.gs &&
                        down_rec.packed && down_rec.scales && down_rec.gs;
    auto upload_nv = [&](const Nvfp4Recs &r,
                         lsxcommon::lsx_nvfp4::LsxNvfp4Weight &w)
        -> cudaError_t {
        if (!r.packed || !r.scales || !r.gs) return cudaErrorInvalidValue;
        w.rows = r.packed->shape.size() >= 2
                     ? static_cast<int>(r.packed->shape[0])
                     : 0;
        w.cols = r.packed->shape.size() >= 2
                     ? static_cast<int>(r.packed->shape[1])
                     : 0;
        if (w.rows <= 0 || w.cols <= 0) return cudaErrorInvalidValue;
        std::memcpy(&w.global_scale, r.gs->data.data(), sizeof(float));
        w.packed_bytes = static_cast<int64_t>(r.packed->data.size());
        w.scale_bytes = static_cast<int64_t>(r.scales->data.size());
        void *d1 = nullptr, *d2 = nullptr;
        cudaError_t e = cudaMalloc(&d1, r.packed->data.size());
        if (e != cudaSuccess) { w.Free(); return e; }
        e = cudaMalloc(&d2, r.scales->data.size());
        if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
        e = ring.UploadRaw(r.packed->data.data(), d1, r.packed->data.size(),
                           stream, bytes_acc);
        if (e == cudaSuccess)
            e = ring.UploadRaw(r.scales->data.data(), d2,
                               r.scales->data.size(), stream, bytes_acc);
        if (e != cudaSuccess) { cudaFree(d1); cudaFree(d2); w.Free(); return e; }
        w.d_packed = static_cast<const uint8_t *>(d1);
        w.d_scales = static_cast<const uint8_t *>(d2);
        return cudaSuccess;
    };
    if (mlp_nv) {
        cudaError_t e = upload_nv(gate_rec, lw.mlp_gate);
        if (e == cudaSuccess) e = upload_nv(up_rec, lw.mlp_up);
        if (e == cudaSuccess) e = upload_nv(down_rec, lw.mlp_down);
        if (e != cudaSuccess) return e;
        lw.mlp_nvfp4 = true;
    }

    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[gemma_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }

    // Adapter: the shared fastload engine consumes TensorView; the Gemma4
    // pinned schema keeps owning TensorRecords.
    auto view_of = [](const TensorRecord &r) {
        lsxcommon::TensorView v{};
        v.name = r.name;
        v.rank = static_cast<int64_t>(r.shape.size());
        v.shape = r.shape.data();
        v.bytes = r.data.size();
        v.data = r.data.data();
        return v;
    };
    // Keep TensorView adapters in a local vector so shape/data pointers stay
    // stable for the duration of the RunUploadJobs call.
    std::vector<lsxcommon::TensorView> views;
    views.reserve(17);  // job count below; prevents pointer-invalidating growth
    auto make_job = [&](const TensorRecord *t, void **dst,
                        int64_t *rows, int64_t *cols) {
        if (t) views.push_back(view_of(*t));
        return fastload::UploadJob{t ? &views.back() : nullptr, dst,
                                   fastload::CvKind::kBf16Raw, rows, cols};
    };
    std::vector<fastload::UploadJob> jobs = {
        make_job(in_norm, reinterpret_cast<void **>(&lw.d_input_norm),
                 nullptr, nullptr),
        make_job(post_norm, reinterpret_cast<void **>(&lw.d_post_norm),
                 nullptr, nullptr),
        make_job(FindTensor(tensors, "pre_feedforward_layernorm_2.weight"),
                 reinterpret_cast<void **>(&lw.d_pre_ffn2), nullptr, nullptr),
        make_job(FindTensor(tensors, "post_feedforward_layernorm_2.weight"),
                 reinterpret_cast<void **>(&lw.d_post_ffn2), nullptr, nullptr),
        make_job(FindTensor(tensors, "pre_feedforward_layernorm.weight"),
                 reinterpret_cast<void **>(&lw.d_pre_ffn), nullptr, nullptr),
        make_job(FindTensor(tensors, "post_feedforward_layernorm.weight"),
                 reinterpret_cast<void **>(&lw.d_post_ffn), nullptr, nullptr),
        make_job(FindTensor(tensors, "post_feedforward_layernorm_1.weight"),
                 reinterpret_cast<void **>(&lw.d_post_ffn1), nullptr, nullptr),
        make_job(FindTensor(tensors, "self_attn.q_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_q), &lw.q_rows, nullptr),
        make_job(FindTensor(tensors, "self_attn.k_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_k), &lw.kv_rows, nullptr),
        make_job(FindTensor(tensors, "self_attn.v_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_v), nullptr, nullptr),
        make_job(FindTensor(tensors, "self_attn.o_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_o), nullptr, &lw.o_cols),
        make_job(FindTensor(tensors, "self_attn.q_norm.weight"),
                 reinterpret_cast<void **>(&lw.d_q_norm), nullptr, nullptr),
        make_job(FindTensor(tensors, "self_attn.k_norm.weight"),
                 reinterpret_cast<void **>(&lw.d_k_norm), nullptr, nullptr),
        // NVFP4 MLP: the BF16 d_mlp_* jobs are skipped (raw records absent).
        make_job(mlp_nv ? nullptr
                        : FindTensor(tensors, "mlp.gate_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_mlp_gate), nullptr, nullptr),
        make_job(mlp_nv ? nullptr
                        : FindTensor(tensors, "mlp.up_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_mlp_up), nullptr, nullptr),
        make_job(mlp_nv ? nullptr
                        : FindTensor(tensors, "mlp.down_proj.weight"),
                 reinterpret_cast<void **>(&lw.d_mlp_down), nullptr, nullptr),
        make_job(FindTensor(tensors, "router.proj.weight"),
                 reinterpret_cast<void **>(&lw.d_router_proj), nullptr,
                 nullptr),
    };

    void *gemma_arena = nullptr;
    size_t gemma_arena_bytes = 0;
    cudaError_t e = fastload::RunUploadJobs(
        std::move(jobs), &gemma_arena, &gemma_arena_bytes, stream, ring,
        bytes_acc, nullptr);
    lw.d_arena = gemma_arena;
    lw.arena_bytes = gemma_arena_bytes;
    if (e != cudaSuccess) return e;

    // F32 / BF16-scalar side uploads. Real checkpoints store layer_scalar
    // and router.per_expert_scale as BF16 (2 bytes/value) — the size guards
    // below must follow the dtype, not sizeof(float).
    if (auto *sc = FindTensor(tensors, "layer_scalar");
        sc && sc->shape.size() == 1 &&
        ((sc->dtype == "F32" && sc->data.size() >= sizeof(float)) ||
         (sc->dtype == "BF16" && sc->data.size() >= 2))) {
        if (sc->dtype == "F32") {
            e = UploadF32(reinterpret_cast<const float *>(sc->data.data()), 1,
                          &lw.d_layer_scalar, stream, ring, bytes_acc);
            if (h_layer_scalar)
                *h_layer_scalar =
                    *reinterpret_cast<const float *>(sc->data.data());
        } else if (sc->dtype == "BF16") {
            auto f32 = Bf16BytesToF32(sc->data, 1);
            e = UploadF32(f32.data(), 1, &lw.d_layer_scalar, stream, ring,
                          bytes_acc);
            if (h_layer_scalar) *h_layer_scalar = f32[0];
        } else {
            e = cudaErrorInvalidValue;
        }
        if (e != cudaSuccess) return e;
    }
    if (auto *pes = FindTensor(tensors, "router.per_expert_scale");
        pes && pes->shape.size() == 1 &&
        ((pes->dtype == "F32" &&
          pes->data.size() >=
              static_cast<size_t>(pes->shape[0]) * sizeof(float)) ||
         (pes->dtype == "BF16" &&
          pes->data.size() >= static_cast<size_t>(pes->shape[0]) * 2))) {
        const size_t n = static_cast<size_t>(pes->shape[0]);
        if (pes->dtype == "F32") {
            e = UploadF32(reinterpret_cast<const float *>(pes->data.data()),
                          n, &lw.d_per_expert_scale, stream, ring, bytes_acc);
        } else {
            auto f32 = Bf16BytesToF32(pes->data, n);
            e = UploadF32(f32.data(), n, &lw.d_per_expert_scale, stream, ring,
                          bytes_acc);
        }
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
    mlp_gate.Free();
    mlp_up.Free();
    mlp_down.Free();
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
    h_layer_scalar.clear();
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

    pw.h_layer_scalar.resize(pw.layers.size(), 1.0f);
    for (int l = 0; l <= max_layer; ++l) {
        auto path = pinned_dir / absl::StrCat("layer_", l, ".arrow");
        if (!fs::exists(path)) continue; // heterogeneous grid: skip
        auto tensors_res = LoadArrowFile(path.string());
        if (!tensors_res.ok()) {
            log::error("[gemma_load] ERROR loading " + path.string() + ": " +
                       tensors_res.status().ToString());
            return cudaErrorInvalidValue;
        }
        e = LoadLayer(*tensors_res, pw.layers[static_cast<size_t>(l)],
                      pw.h_layer_scalar.empty()
                          ? nullptr
                          : &pw.h_layer_scalar[static_cast<size_t>(l)],
                      stream, bytes_acc);
        if (e != cudaSuccess) return e;
    }

    pw.resident_bytes = bytes_acc;

    // Topology from the index (mandatory for generation; the loader still
    // succeeds without it so weight-only tooling keeps working).
    auto idx_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(idx_path))
        idx_path = fs::path(model_path) / "model.index.json";
    if (fs::exists(idx_path)) {
        pw.top = load_gemma_topology(idx_path.string());
        if (!pw.top.error_msg.empty())
            log::error("[gemma_load] index topology parse failed: " +
                       pw.top.error_msg);
    }

    return cudaSuccess;
}

// ── Expert store wiring ─────────────────────────────────────────────────────

cudaError_t GemmaKvCache::Allocate(const GemmaPinnedWeights &pw, int max_total)
{
    Free();
    if (!pw.top.error_msg.empty() || pw.layers.empty() || max_total <= 0)
        return cudaErrorInvalidValue;
    dk.resize(pw.layers.size(), nullptr);
    dv.resize(pw.layers.size(), nullptr);
    std::uint64_t bytes = 0;
    for (size_t l = 0; l < pw.layers.size(); ++l) {
        const int64_t stride = pw.layers[l].kv_rows; // kv_heads * head_dim
        if (stride <= 0) {
            Free();
            return cudaErrorInvalidValue;
        }
        size_t sz = static_cast<size_t>(stride) * max_total * 2;
        if (cudaMalloc(&dk[l], sz) != cudaSuccess ||
            cudaMalloc(&dv[l], sz) != cudaSuccess) {
            Free();
            return cudaErrorMemoryAllocation;
        }
        bytes += 2 * sz;
    }
    this->max_total = max_total;
    cache_bytes = bytes;
    return cudaSuccess;
}

void GemmaKvCache::Free()
{
    for (auto p : dk) if (p) cudaFree(p);
    for (auto p : dv) if (p) cudaFree(p);
    dk.clear();
    dv.clear();
    max_total = 0;
    cache_bytes = 0;
}

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


