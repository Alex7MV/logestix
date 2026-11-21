#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>
#include <string_view>
#include <fstream>

#include "lsxcommon/fast_load.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace fastload {

// ── Shared upload/conversion helpers (dedup phase 5) ────────────────────
// Byte-identical per-family copies moved here: UploadBytes (mistral/mistral-
// large4/xing load.cpp), Bf16ToHalf (alice/kimi-k3 load.cpp).

// Allocates device memory, copies @p n bytes H2D synchronously, accounts the
// bytes; on copy failure the allocation is freed. n == 0 is a no-op success.
inline cudaError_t UploadBytes(const uint8_t *src, size_t n,
                               const uint8_t **dst, std::size_t &accounted)
{
    if (n == 0) return cudaSuccess;
    void *p = nullptr;
    const cudaError_t err = cudaMalloc(&p, n);
    if (err != cudaSuccess) return err;
    const cudaError_t cp = cudaMemcpy(p, src, n, cudaMemcpyHostToDevice);
    if (cp != cudaSuccess)
    {
        cudaFree(p);
        return cp;
    }
    *dst = static_cast<const uint8_t *>(p);
    accounted += n;
    return cudaSuccess;
}

// Host BF16 (little-endian byte pairs) → FP16 vector, bit-exact widening.
inline std::vector<__half> Bf16ToHalf(const std::vector<uint8_t> &data)
{
    const size_t n = data.size() / 2;
    std::vector<__half> out(n);
    for (size_t i = 0; i < n; ++i) {
        const uint16_t b =
            static_cast<uint16_t>(data[2 * i] | (data[2 * i + 1] << 8));
        const uint32_t f = static_cast<uint32_t>(b) << 16;
        float v;
        std::memcpy(&v, &f, 4);
        out[i] = __float2half(v);
    }
    return out;
}

// Double-buffered pinned staging for async H2D uploads on one stream.
// Buffer parity is guarded by a recorded event: before the host refills a
// buffer, the previous copy enqueued from it has been event-synchronized, so
// an in-flight DMA read can never race the next host memcpy. Same-stream
// ordering guarantees every conversion kernel runs strictly after its copy.
class PinnedStagingRing {
public:
    static constexpr size_t kDefaultChunkBytes = size_t{8} << 20;

    explicit PinnedStagingRing(size_t chunk_bytes = kDefaultChunkBytes)
        : chunk_bytes_(chunk_bytes < (size_t{1} << 20) ? (size_t{1} << 20)
                                                       : chunk_bytes)
    {
        for (int i = 0; i < 2; ++i) {
            bufs_[i] = nullptr;
            ev_[i] = nullptr;
        }
        for (int i = 0; i < 2; ++i) {
            if (cudaMallocHost(&bufs_[i], chunk_bytes_) != cudaSuccess)
                bufs_[i] = nullptr;
        }
        for (int i = 0; i < 2; ++i)
            cudaEventCreateWithFlags(&ev_[i], cudaEventDisableTiming);
    }

    ~PinnedStagingRing()
    {
        // Both events were recorded right after each buffer's latest copy:
        // synchronizing them proves no DMA read from bufs_[i] is still in
        // flight before cudaFreeHost (a freed-while-in-flight pinned buffer
        // poisons the CUDA context for every later call).
        for (int i = 0; i < 2; ++i)
            if (ev_[i]) cudaEventSynchronize(ev_[i]);
        for (int i = 0; i < 2; ++i) {
            if (bufs_[i]) cudaFreeHost(bufs_[i]);
            if (ev_[i]) cudaEventDestroy(ev_[i]);
        }
    }

    PinnedStagingRing(const PinnedStagingRing &) = delete;
    PinnedStagingRing &operator=(const PinnedStagingRing &) = delete;

    bool ok() const { return bufs_[0] && bufs_[1] && ev_[0] && ev_[1]; }

    // Uploads numel BF16 values to d_dst, converting them to FP16 on device
    // (bit-exact legacy path). d_dst must be 16-byte aligned.
    cudaError_t UploadBf16AsFp16(const uint8_t *host_bf16, int64_t numel,
                                 __half *d_dst, cudaStream_t stream,
                                 std::uint64_t &bytes_acc)
    {
        if (!ok()) return cudaErrorMemoryAllocation;
        const int64_t per_chunk =
            static_cast<int64_t>(chunk_bytes_ / 2);
        if (per_chunk <= 0) return cudaErrorInvalidValue;
        int bi = 0;
        for (int64_t off = 0; off < numel; ) {
            const int64_t cnt =
                (numel - off < per_chunk) ? (numel - off) : per_chunk;
            cudaError_t e = cudaEventSynchronize(ev_[bi]);
            if (e != cudaSuccess) return e;
            std::memcpy(bufs_[bi], host_bf16 + off * 2,
                        static_cast<size_t>(cnt) * 2);
            uint16_t *dst16 = reinterpret_cast<uint16_t *>(d_dst) + off;
            e = cudaMemcpyAsync(dst16, bufs_[bi],
                                static_cast<size_t>(cnt) * 2,
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaEventRecord(ev_[bi], stream);
            if (e != cudaSuccess) return e;
            e = Bf16ToHalfInPlace(dst16, cnt, stream);
            if (e != cudaSuccess) return e;
            off += cnt;
            bi ^= 1;
        }
        bytes_acc += static_cast<std::uint64_t>(numel) * 2;
        return cudaSuccess;
    }

    // Plain chunked byte upload (F32/F8 tensors etc.), same event-guarded ring.
    cudaError_t UploadRaw(const void *src, void *d_dst, size_t bytes,
                          cudaStream_t stream, std::uint64_t &bytes_acc)
    {
        if (!ok()) return cudaErrorMemoryAllocation;
        int bi = 0;
        size_t off = 0;
        while (off < bytes) {
            const size_t cnt =
                (bytes - off < chunk_bytes_) ? (bytes - off) : chunk_bytes_;
            cudaError_t e = cudaEventSynchronize(ev_[bi]);
            if (e != cudaSuccess) return e;
            std::memcpy(bufs_[bi], static_cast<const uint8_t *>(src) + off,
                        cnt);
            e = cudaMemcpyAsync(static_cast<uint8_t *>(d_dst) + off,
                                bufs_[bi], cnt, cudaMemcpyHostToDevice,
                                stream);
            if (e != cudaSuccess) return e;
            e = cudaEventRecord(ev_[bi], stream);
            if (e != cudaSuccess) return e;
            off += cnt;
            bi ^= 1;
        }
        bytes_acc += bytes;
        return cudaSuccess;
    }

private:
    size_t chunk_bytes_;
    uint8_t *bufs_[2];
    cudaEvent_t ev_[2];
};

// ── Shared loader helpers (deduplicated 2026-08: were copied per loader) ────

// Device-arena view alignment for coalesced access.
inline constexpr size_t kArenaAlign = 256;

inline size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Drains the stream on scope exit so pinned staging never dies while a DMA
// transfer is still in flight (error paths of pinned-weight loaders).
struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};

// Returns the tensor whose name ends with `suffix`, or nullptr.
inline const TensorView *FindTensorSuffix(const TensorBatchViews &tensors,
                                          std::string_view suffix)
{
    for (const auto &t : tensors.tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &t;
    }
    return nullptr;
}

// Same suffix lookup over NVFP4 record batches.
inline const DenseNvfp4View *FindNvfp4Suffix(const DenseNvfp4BatchViews &recs,
                                             std::string_view suffix)
{
    for (const auto &r : recs.records) {
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &r;
    }
    return nullptr;
}

// ── Shared arena upload-job engine ──────────────────────────────────────────
// One engine for all pinned-weight loaders (was: six per-loader copies of the
// same validate -> single arena cudaMalloc -> 256B-aligned views -> ring
// upload pipeline; see docs/superpowers/specs/2026-08-24-model-unification).

#include <string>
#include <vector>

enum class CvKind {
    kBf16ToHalf,  // raw BF16 bits -> FP16 on device (in-place kernel)
    kBf16Raw,     // raw BF16 passthrough (Kimi keeps BF16 weights resident)
    kF32Raw,      // raw F32 passthrough
    kF32ToHalf,   // host scalar F32 -> FP16 (tiny tensors only)
    kBf16ToF32,   // host scalar BF16 bits -> F32 (tiny tensors only)
};

inline size_t OutElemSize(CvKind k)
{
    return (k == CvKind::kF32Raw || k == CvKind::kBf16ToF32) ? 4 : 2;
}

struct UploadJob {
    const TensorView *t = nullptr;  // null = absent tensor, skipped
    void **dst = nullptr;
    CvKind kind = CvKind::kBf16ToHalf;
    int64_t *rows = nullptr;  // optional 2-D shape capture (GLM parity)
    int64_t *cols = nullptr;
};

inline cudaError_t ValidateUploadJob(const UploadJob &j, int64_t &numel,
                                     std::string *error)
{
    if (!j.t || !j.dst) {
        if (error) *error = "null tensor or dst in upload job";
        return cudaErrorInvalidValue;
    }
    if (j.t->rank <= 0) {
        if (error)
            *error = "malformed tensor '" + std::string(j.t->name) +
                     "' rank=0";
        return cudaErrorInvalidValue;
    }
    numel = 1;
    for (int64_t d = 0; d < j.t->rank; ++d) {
        if (j.t->shape[d] <= 0) {
            if (error)
                *error = "malformed tensor '" + std::string(j.t->name) + "'";
            return cudaErrorInvalidValue;
        }
        numel *= j.t->shape[d];
    }
    const size_t in_bytes =
        (j.kind == CvKind::kF32Raw || j.kind == CvKind::kF32ToHalf) ? 4 : 2;
    if (j.t->bytes < static_cast<size_t>(numel) * in_bytes) {
        if (error)
            *error = "malformed tensor '" + std::string(j.t->name) +
                     "': insufficient data bytes";
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// Validates every job, issues ONE arena allocation sized to the converted
// footprint and uploads each tensor through the pinned ring into its
// 256-byte-aligned arena view. `error` may be null (messages are then
// dropped); it is cleared on success and filled on failure.
cudaError_t RunUploadJobs(std::vector<UploadJob> jobs,
                          void **arena_out, size_t *arena_bytes_out,
                          cudaStream_t stream, PinnedStagingRing &ring,
                          std::uint64_t &bytes_acc, std::string *error);

// Uploads one NVFP4 weight as two raw device blocks staged through the ring.
// WeightT must expose rows/cols/global_scale/packed_bytes/scale_bytes/
// d_packed/d_scales/Free() (lsx_nvfp4::LsxNvfp4Weight satisfies this).
template <typename WeightT>
cudaError_t UploadNvfp4Weight(const DenseNvfp4View *r, WeightT &w,
                              cudaStream_t stream, PinnedStagingRing &ring,
                              std::uint64_t &bytes_acc, std::string *error)
{
    if (!r || r->packed_bytes == 0) return cudaSuccess;
    w.rows = r->rank >= 2 ? static_cast<int>(r->shape[0]) : 0;
    w.cols = r->rank >= 2 ? static_cast<int>(r->shape[1]) : 0;
    w.global_scale = r->global_scale;
    w.packed_bytes = r->packed_bytes;
    w.scale_bytes = r->scale_bytes;
    void *d1 = nullptr;
    cudaError_t e = cudaMalloc(&d1, static_cast<size_t>(r->packed_bytes));
    if (e != cudaSuccess) { w.Free(); return e; }
    e = ring.UploadRaw(r->packed, d1, static_cast<size_t>(r->packed_bytes),
                       stream, bytes_acc);
    if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
    void *d2 = nullptr;
    e = cudaMalloc(&d2, static_cast<size_t>(r->scale_bytes));
    if (e != cudaSuccess) { cudaFree(d1); w.Free(); return e; }
    e = ring.UploadRaw(r->scales, d2, static_cast<size_t>(r->scale_bytes),
                       stream, bytes_acc);
    if (e != cudaSuccess) { cudaFree(d1); cudaFree(d2); w.Free(); return e; }
    w.d_packed = static_cast<const uint8_t *>(d1);
    w.d_scales = static_cast<const uint8_t *>(d2);
    if (error) error->clear();
    return cudaSuccess;
}

inline cudaError_t RunUploadJobs(std::vector<UploadJob> jobs,
                                 void **arena_out, size_t *arena_bytes_out,
                                 cudaStream_t stream, PinnedStagingRing &ring,
                                 std::uint64_t &bytes_acc, std::string *error)
{
    auto set_err = [&](const std::string &msg) {
        if (error) *error = msg;
    };
    std::string err_sink;
    std::string *errp = error ? error : &err_sink;

    std::vector<int64_t> numels(jobs.size(), 0);
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        cudaError_t e = ValidateUploadJob(jobs[i], numels[i], errp);
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
            set_err(std::string("arena alloc failed: ") +
                    cudaGetErrorString(e));
            return e;
        }
    }
    *arena_out = arena;
    *arena_bytes_out = total;

    size_t off = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const UploadJob &j = jobs[i];
        if (!j.t) continue;
        off = AlignUp(off, kArenaAlign);
        uint8_t *view = static_cast<uint8_t *>(arena) + off;
        *j.dst = view;
        const int64_t n = numels[i];
        if (j.t->rank == 2) {
            if (j.rows) *j.rows = j.t->shape[0];
            if (j.cols) *j.cols = j.t->shape[1];
        }
        cudaError_t e = cudaSuccess;
        switch (j.kind) {
        case CvKind::kBf16ToHalf:
            e = ring.UploadBf16AsFp16(j.t->data, n,
                                      reinterpret_cast<__half *>(view),
                                      stream, bytes_acc);
            break;
        case CvKind::kBf16Raw:
            e = ring.UploadRaw(j.t->data, view,
                               static_cast<size_t>(n) * 2, stream, bytes_acc);
            break;
        case CvKind::kF32Raw:
            e = ring.UploadRaw(j.t->data, view,
                               static_cast<size_t>(n) * 4, stream, bytes_acc);
            break;
        default: {
            size_t bytes = static_cast<size_t>(n) * OutElemSize(j.kind);
            std::vector<uint8_t> h(bytes);
            if (j.kind == CvKind::kF32ToHalf) {
                const float *src =
                    reinterpret_cast<const float *>(j.t->data);
                __half *dst = reinterpret_cast<__half *>(h.data());
                for (int64_t k = 0; k < n; ++k)
                    dst[k] = __float2half(src[k]);
            } else {
                const uint16_t *src =
                    reinterpret_cast<const uint16_t *>(j.t->data);
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
        if (e != cudaSuccess) {
            set_err("upload failed for '" + std::string(j.t->name) +
                    "': " + cudaGetErrorString(e));
            return e;
        }
        off += static_cast<size_t>(numels[i]) * OutElemSize(j.kind);
    }
    if (error) error->clear();
    return cudaSuccess;
}

// Single-tensor upload through the shared job engine (the nemotron/nemotron-
// ultra UploadOne copy; dedup phase 5). Missing tensor / upload failure fills
// @p error with the tensor's role (@p what).
inline bool UploadOne(const TensorView *t, CvKind kind, void **dst,
                      const std::string &what, cudaStream_t stream,
                      PinnedStagingRing &ring, std::uint64_t &acc,
                      std::string &error)
{
    if (!t) {
        error = "missing tensor: " + what;
        return false;
    }
    std::vector<UploadJob> jobs = {{t, dst, kind, nullptr, nullptr}};
    void *arena = nullptr;
    size_t arena_bytes = 0;
    std::string e2;
    cudaError_t e = RunUploadJobs(std::move(jobs), &arena, &arena_bytes, stream,
                                  ring, acc, &e2);
    if (e != cudaSuccess) {
        error = "upload failed: " + what + ": " + e2;
        return false;
    }
    return true;
}

} // namespace fastload
} // namespace lsxcommon
