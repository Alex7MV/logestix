#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "lsxcommon/fast_load.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace fastload {

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

} // namespace fastload
} // namespace lsxcommon
