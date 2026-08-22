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

constexpr size_t kArenaAlign = 256;

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Drains the stream on scope exit so pinned staging never dies while a DMA
// transfer is still in flight (error paths of LoadQwenPinnedWeights).
struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};

enum class CvKind {
    kBf16ToHalf,  // raw BF16 bits -> FP16 on device (in-place kernel)
    kF32Raw,      // raw F32 passthrough
    kF32ToHalf,   // host scalar F32 -> FP16 (tiny tensors only)
    kBf16ToF32,   // host scalar BF16 bits -> F32 (tiny tensors only)
};

size_t OutElemSize(CvKind k)
{
    return (k == CvKind::kF32Raw || k == CvKind::kBf16ToF32) ? 4 : 2;
}

struct UploadJob {
    const TensorRecord *t;   // null = absent tensor, skipped
    void **dst;
    CvKind kind;
};

cudaError_t ValidateJob(const UploadJob &j, int64_t &numel)
{
    const auto &s = j.t->shape;
    if (!s.empty()) {
        // Any-rank contiguous tensor (1D norms, [C, K] / [C, 1, K] conv1d,
        // [H, 1] biases): the squeezed layout is contiguous either way.
        numel = 1;
        for (int64_t d : s) {
            if (d <= 0) {
                log::error("[qwen_load] malformed tensor '" + j.t->name + "'");
                return cudaErrorInvalidValue;
            }
            numel *= d;
        }
    } else {
        std::string sh;
        for (int64_t d : s) absl::StrAppend(&sh, sh.empty() ? "" : "x", d);
        log::error("[qwen_load] malformed tensor '" + j.t->name + "' dtype=" +
                   j.t->dtype + " shape=[" + sh + "] bytes=" +
                   absl::StrCat(j.t->data.size()));
        return cudaErrorInvalidValue;
    }
    size_t in_bytes = (j.kind == CvKind::kF32Raw || j.kind == CvKind::kF32ToHalf)
                          ? 4 : 2;
    if (j.t->data.size() < static_cast<size_t>(numel) * in_bytes) {
        log::error("[qwen_load] malformed tensor '" + j.t->name +
                   "' dtype=" + j.t->dtype + ": data " +
                   absl::StrCat(j.t->data.size(), " < needed ",
                                static_cast<uint64_t>(numel) * in_bytes));
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// Validates every job, issues ONE arena allocation sized to the layer's
// converted footprint and uploads each tensor through the pinned ring into
// its 256-byte-aligned arena view. On success *arena_out/*arena_bytes_out own
// the block; every *j.dst points inside it.
cudaError_t RunUploadJobs(std::vector<UploadJob> jobs,
                          void **arena_out, size_t *arena_bytes_out,
                          cudaStream_t stream,
                          fastload::PinnedStagingRing &ring,
                          std::uint64_t &bytes_acc)
{
    std::vector<int64_t> numels(jobs.size(), 0);
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        cudaError_t e = ValidateJob(jobs[i], numels[i]);
        if (e != cudaSuccess) return e;
    }

    size_t total = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        total = AlignUp(total, kArenaAlign) +
                static_cast<size_t>(numels[i]) * OutElemSize(jobs[i].kind);
    }

    void *arena = nullptr;
    if (total > 0) {
        cudaError_t e = cudaMalloc(&arena, total);
        if (e != cudaSuccess) {
            log::error("[qwen_load] arena alloc failed: " +
                       std::string(cudaGetErrorString(e)));
            return e;
        }
    }
    *arena_out = arena;
    *arena_bytes_out = total;

    size_t off = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto &j = jobs[i];
        if (!j.t) continue;
        off = AlignUp(off, kArenaAlign);
        uint8_t *view = static_cast<uint8_t *>(arena) + off;
        *j.dst = view;
        const int64_t n = numels[i];
        cudaError_t e = cudaSuccess;
        switch (j.kind) {
        case CvKind::kBf16ToHalf:
            e = ring.UploadBf16AsFp16(j.t->data.data(), n,
                                      reinterpret_cast<__half *>(view),
                                      stream, bytes_acc);
            break;
        case CvKind::kF32Raw:
            e = ring.UploadRaw(j.t->data.data(), view,
                               static_cast<size_t>(n) * 4, stream, bytes_acc);
            break;
        default: {
            // Tiny host-converted tensors (F32->FP16 norm / BF16->F32 bias).
            size_t bytes = static_cast<size_t>(n) * OutElemSize(j.kind);
            std::vector<uint8_t> h(bytes);
            if (j.kind == CvKind::kF32ToHalf) {
                const float *src =
                    reinterpret_cast<const float *>(j.t->data.data());
                __half *dst = reinterpret_cast<__half *>(h.data());
                for (int64_t k = 0; k < n; ++k)
                    dst[k] = __float2half(src[k]);
            } else {
                const uint16_t *src =
                    reinterpret_cast<const uint16_t *>(j.t->data.data());
                float *dst = reinterpret_cast<float *>(h.data());
                for (int64_t k = 0; k < n; ++k) {
                    uint32_t bits = static_cast<uint32_t>(src[k]) << 16;
                    std::memcpy(&dst[k], &bits, sizeof(bits));
                }
            }
            e = cudaMemcpyAsync(view, h.data(), bytes,
                                cudaMemcpyHostToDevice, stream);
            if (e == cudaSuccess) bytes_acc += bytes;
            break;
        }
        }
        if (e != cudaSuccess) return e;
        off += static_cast<size_t>(numels[i]) * OutElemSize(j.kind);
    }
    return cudaSuccess;
}

// Uploads one NVFP4 weight (packed E2M1 nibbles + E4M3 scales) as two raw
// device blocks staged through the pinned ring.
cudaError_t UploadNvfp4Weight(const DenseNvfp4Record *r, QwenNvfp4Weight &w,
                              cudaStream_t stream,
                              fastload::PinnedStagingRing &ring,
                              std::uint64_t &bytes_acc)
{
    if (!r || r->packed.empty()) return cudaSuccess;
    w.rows = r->shape.size() >= 2 ? static_cast<int>(r->shape[0]) : 0;
    w.cols = r->shape.size() >= 2 ? static_cast<int>(r->shape[1]) : 0;
    w.global_scale = r->global_scale;
    w.packed_bytes = static_cast<int64_t>(r->packed.size());
    w.scale_bytes = static_cast<int64_t>(r->scales.size());
    void *d1 = nullptr;
    cudaError_t e = cudaMalloc(&d1, r->packed.size());
    if (e != cudaSuccess) { w.Free(); return e; }
    e = ring.UploadRaw(r->packed.data(), d1, r->packed.size(),
                       stream, bytes_acc);
    if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
    void *d2 = nullptr;
    e = cudaMalloc(&d2, r->scales.size());
    if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
    e = ring.UploadRaw(r->scales.data(), d2, r->scales.size(),
                       stream, bytes_acc);
    if (e != cudaSuccess) { cudaFree(d1); cudaFree(d2); w.Free(); return e; }
    w.d_packed = static_cast<const uint8_t*>(d1);
    w.d_scales = static_cast<const uint8_t*>(d2);
    return cudaSuccess;
}

// Matches a tensor record whose name ENDS with `suffix`.
const TensorRecord *FindRaw(const std::vector<TensorRecord> &tensors,
                            const std::string &suffix)
{
    for (const auto &t : tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return &t;
    }
    return nullptr;
}

const DenseNvfp4Record *FindNvfp4(const std::vector<DenseNvfp4Record> &recs,
                                  const std::string &suffix)
{
    for (const auto &r : recs) {
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return &r;
    }
    return nullptr;
}

arrow::Result<std::vector<TensorRecord>> LoadRawFile(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok()) return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    return PinnedSchema::UnpackBatch(*batch_res);
}

arrow::Result<std::vector<DenseNvfp4Record>> LoadNvfp4File(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok()) return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    return DenseNvfp4Schema::UnpackBatch(*batch_res);
}

// Picks the upload kind for a 1D FP16-destined tensor by its on-disk dtype.
CvKind NormKind(const TensorRecord *t)
{
    return (t && t->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
}

// Dtype-aware kind for F32-destined tensors: honor BF16-on-disk records
// (e.g. A_log in real Qwen checkpoints) instead of reinterpreting bits.
CvKind CvKindFromDtype(const TensorRecord *t, CvKind f32_kind)
{
    return (t && t->dtype == "BF16") ? CvKind::kBf16ToF32 : f32_kind;
}

cudaError_t LoadLayer(const std::vector<TensorRecord> &raw,
                      const std::vector<DenseNvfp4Record> &nv,
                      QwenLayerWeights &lw, cudaStream_t stream,
                      fastload::PinnedStagingRing &ring,
                      std::uint64_t &bytes_acc, bool is_linear)
{
    lw.is_linear = is_linear;

    auto *in_norm = FindRaw(raw, "input_layernorm.weight");
    if (in_norm && !in_norm->shape.empty())
        lw.norm_size = static_cast<int>(in_norm->shape[0]);
    auto *post_norm = FindRaw(raw, "post_attention_layernorm.weight");

    std::vector<UploadJob> jobs = {
        {in_norm,   reinterpret_cast<void **>(&lw.d_input_norm),
                    NormKind(in_norm)},
        {post_norm, reinterpret_cast<void **>(&lw.d_post_norm),
                    NormKind(post_norm)},
    };

    if (is_linear) {
        auto *conv = FindRaw(raw, "linear_attn.conv1d.weight");
        if (conv && !conv->shape.empty()) {
            // nn.Conv1d weight is [C, 1, K] on disk; legacy fixtures store
            // [C, K]. The squeezed [C, K] layout is contiguous either way.
            lw.conv_dim = static_cast<int>(conv->shape[0]);
            lw.conv_kernel = conv->shape.size() >= 3
                ? static_cast<int>(conv->shape[2])
                : static_cast<int>(conv->shape[conv->shape.size() - 1]);
        }
        jobs.push_back({conv,
                        reinterpret_cast<void **>(&lw.d_conv1d),
                        CvKind::kBf16ToHalf});
        jobs.push_back({FindRaw(raw, "linear_attn.A_log"),
                        reinterpret_cast<void **>(&lw.d_A_log),
                        CvKindFromDtype(FindRaw(raw, "linear_attn.A_log"),
                                        CvKind::kF32Raw)});
        auto *dt = FindRaw(raw, "linear_attn.dt_bias");
        // dt_bias is BF16 on disk; kernels consume it as FP32.
        CvKind dt_kind = (!dt || dt->dtype == "BF16" || dt->dtype.empty())
                             ? CvKind::kBf16ToF32 : CvKind::kF32Raw;
        jobs.push_back({dt, reinterpret_cast<void **>(&lw.d_dt_bias), dt_kind});
        auto *an = FindRaw(raw, "linear_attn.norm.weight");
        if (an && !an->shape.empty())
            lw.attn_norm_size = static_cast<int>(an->shape[0]);
        jobs.push_back({an, reinterpret_cast<void **>(&lw.d_attn_norm),
                        NormKind(an)});
    } else {
        auto *qn = FindRaw(raw, "self_attn.q_norm.weight");
        if (qn && !qn->shape.empty())
            lw.q_norm_size = static_cast<int>(qn->shape[0]);
        auto *kn = FindRaw(raw, "self_attn.k_norm.weight");
        if (kn && !kn->shape.empty())
            lw.k_norm_size = static_cast<int>(kn->shape[0]);
        jobs.push_back({qn, reinterpret_cast<void **>(&lw.d_q_norm),
                        NormKind(qn)});
        jobs.push_back({kn, reinterpret_cast<void **>(&lw.d_k_norm),
                        NormKind(kn)});
    }

    cudaError_t e = RunUploadJobs(std::move(jobs), &lw.d_arena,
                                  &lw.arena_bytes, stream, ring, bytes_acc);
    if (e != cudaSuccess) return e;

    if (is_linear) {
        e = UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_qkv.weight"), lw.qkv, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_z.weight"), lw.z, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_b.weight"), lw.b, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_a.weight"), lw.a, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.out_proj.weight"), lw.out, stream, ring, bytes_acc);
    } else {
        e = UploadNvfp4Weight(FindNvfp4(nv, "self_attn.q_proj.weight"), lw.q, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "self_attn.k_proj.weight"), lw.k, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "self_attn.v_proj.weight"), lw.v, stream, ring, bytes_acc);
        if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "self_attn.o_proj.weight"), lw.o, stream, ring, bytes_acc);
    }
    if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "mlp.gate_proj.weight"), lw.mlp_gate, stream, ring, bytes_acc);
    if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "mlp.up_proj.weight"),   lw.mlp_up,   stream, ring, bytes_acc);
    if (e == cudaSuccess) e = UploadNvfp4Weight(FindNvfp4(nv, "mlp.down_proj.weight"), lw.mlp_down, stream, ring, bytes_acc);
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
        auto raw_res = LoadRawFile((pinned_dir / "global.arrow").string());
        if (!raw_res.ok()) {
            log::error("[qwen_load] global.arrow read failed: " +
                       raw_res.status().ToString());
            pw.Free();
            return false;
        }
        auto *fn = FindRaw(*raw_res, "norm.weight");
        if (fn && !fn->shape.empty()) {
            int64_t n = fn->shape[0];
            pw.final_norm_size = static_cast<int>(n);
            cudaError_t e = cudaMalloc(&pw.d_final_norm,
                                       static_cast<size_t>(n) * sizeof(__half));
            if (e != cudaSuccess) { pw.Free(); return false; }
            if (fn->dtype == "F32") {
                std::vector<__half> h(static_cast<size_t>(n));
                const float *src =
                    reinterpret_cast<const float *>(fn->data.data());
                for (int64_t i = 0; i < n; ++i)
                    h[static_cast<size_t>(i)] = __float2half(src[i]);
                e = cudaMemcpyAsync(pw.d_final_norm, h.data(),
                                    static_cast<size_t>(n) * sizeof(__half),
                                    cudaMemcpyHostToDevice, stream);
            } else {
                e = ring.UploadBf16AsFp16(fn->data.data(), n, pw.d_final_norm,
                                          stream, bytes_acc);
            }
            if (e != cudaSuccess) { pw.Free(); return false; }
        }
    }
    // Global NVFP4 (embed + head).
    if (fs::exists(pinned_dir / "global_nvfp4.arrow")) {
        auto nv_res = LoadNvfp4File((pinned_dir / "global_nvfp4.arrow").string());
        if (!nv_res.ok()) {
            log::error("[qwen_load] global_nvfp4.arrow read failed: " +
                       nv_res.status().ToString());
            pw.Free();
            return false;
        }
        cudaError_t e = UploadNvfp4Weight(FindNvfp4(*nv_res, "embed_tokens.weight"),
                                          pw.embed, stream, ring, bytes_acc);
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(FindNvfp4(*nv_res, "lm_head.weight"),
                                  pw.head, stream, ring, bytes_acc);
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

        auto raw = LoadRawFile((pinned_dir / raw_name).string());
        if (!raw.ok()) {
            log::error("[qwen_load] " + raw_name + " read failed");
            continue;
        }
        std::vector<DenseNvfp4Record> nv;
        if (fs::exists(pinned_dir / nv_name)) {
            auto nv_res = LoadNvfp4File((pinned_dir / nv_name).string());
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