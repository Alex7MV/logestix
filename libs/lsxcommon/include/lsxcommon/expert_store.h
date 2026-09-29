// expert_store.h — shared MoE mmap-expert LRU store (kernel-unification §3.2).
//
// Collapses the two byte-identical expert loaders (DeepSeek `gpu::ExpertDb`,
// GLM `glm::GlmExpertDb`) into one template, keyed on the per-model entry
// type. Both entries share the same 9-column ExpertSchema and identical member
// names in the same order, so a single DecodeEntry() serves both.
//
// Host-side only: no __device__ kernels, no absl. Safe under both the MSVC
// host compiler and NVCC .cu TUs. LRU is NOT internally synchronized (same
// contract as the loaders it replaces: one consumer per instance).
#pragma once

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/record_batch.h>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#include <algorithm>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {

// Pending-copy drain hook (implemented by the MoE inference TU that issues
// prefetch copies from these mmaps). Called immediately before an mmap is
// unmapped so any in-flight copy_stream DMA staged from this mapping can be
// awaited (prefetch-async copies intentionally outlive their API call).
void lsx_pending_copy_drain(const void* host_ptr);

namespace gpu {
namespace expert_store {

template <typename TEntry>
struct ExpertStore {
    std::vector<TEntry> entries;
    std::vector<std::shared_ptr<arrow::io::MemoryMappedFile>> mmaps;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::vector<std::string> paths;      // per flat index (for lazy open)
    std::vector<size_t> lru;             // flat indices, back = least recently used
    // 11008 DeepSeek experts = 11008 fds. The historical 7000 cap (Windows
    // CRT 8192 limit) forced constant mmap churn past layer 27 — and every
    // eviction munmap'ed entries whose async prefetch DMA could still be
    // in flight. Linux fd limits are effectively unbounded here.
    size_t max_open = 65536;
    int num_layers = 0;
    int experts_per_layer = 0;
    std::string error_msg;

    // Diagnostics: ParseFile invocation / success counters (tests, loaders).
    int64_t parse_calls = 0;
    int64_t parse_ok = 0;
    // Optional locked-host pool backing the entries (DeepSeek NVFP4 pin).
    // When set, entry host pointers point INTO this allocation and Free()
    // releases it with cudaFreeHost.
    void *pinned_pool = nullptr;
    std::size_t pinned_bytes = 0;
    // Chunked variant (Kimi 1T store): the driver fails a single >~300 GiB
    // cudaHostAlloc, so large stores pin as a vector of 16 GiB chunks.
    std::vector<void *> pinned_pools;
    // true = chunk is anonymous mmap + cudaHostRegister (fast path, no
    // driver zerofill); false = cudaHostAlloc (DeepSeek path). Free() picks
    // the matching release primitive per chunk.
    std::vector<bool> pinned_registered;
    std::vector<size_t> pinned_chunk_sizes;

    // Model toggles (see top-of-file).
    std::string log_prefix = "";
    std::string mmap_fail_verb = "mmap fail ";   // DeepSeek; GLM = "expert mmap fail "
    bool emit_diag = false;       // DeepSeek-only [diag-expert] block
    bool log_batch_fails = false; // GLM-only per-batch failure debug lines

    bool Load(const std::string &model_path, bool fallback)
    {
        namespace fs = std::filesystem;
        int total = num_layers * experts_per_layer;
        if (total <= 0) return false;
        entries.assign(static_cast<size_t>(total), TEntry());
        mmaps.assign(static_cast<size_t>(total), nullptr);
        batches.assign(static_cast<size_t>(total), nullptr);
        paths.resize(static_cast<size_t>(total));
        lru.reserve(max_open);

        fs::path exp_dir = fs::path(model_path) / "logestix" / "experts";
        if (fallback && !fs::exists(exp_dir) &&
            fs::exists(fs::path(model_path) / "experts"))
            exp_dir = fs::path(model_path) / "experts";

        for (int layer = 0; layer < num_layers; ++layer) {
            for (int exp = 0; exp < experts_per_layer; ++exp) {
                size_t idx = static_cast<size_t>(layer) *
                                 static_cast<size_t>(experts_per_layer) +
                             static_cast<size_t>(exp);
                paths[idx] = (exp_dir / ("layer_" + std::to_string(layer)) /
                              ("exp_" + std::to_string(exp) + ".arrow")).string();
            }
        }

        // Warm the cache up to max_open (matches the per-model loop order).
        for (size_t idx = 0; idx < paths.size() && lru.size() < max_open; ++idx) {
            if (ParseFile(idx)) lru.push_back(idx);
        }
        return true;
    }

    // Parallel bulk-parse (Linux): same work as the Load() warm loop but
    // spread over worker threads. NVMe reads at 23040-file scale are
    // latency-bound single-threaded (~2.5 GB/s); parallel mmap+parse
    // saturates the device (~7 GB/s). Thread safety: each worker touches
    // only its own entries[idx]/batches[idx]/mmaps[idx] slot; lru is
    // filled afterwards on the calling thread. Used by the pinned-store
    // path where every entry is resident anyway (max_open cap not applied
    // — no eviction bookkeeping needed).
    bool LoadParallelParse(int n_threads)
    {
#if defined(__unix__) || defined(__APPLE__)
        namespace fs = std::filesystem;
        const size_t total = entries.size();
        if (total == 0 || paths.size() != total) return false;
        if (n_threads <= 0)
            n_threads = std::clamp<unsigned>(
                std::thread::hardware_concurrency() / 4u, 8u, 32u);
        std::atomic<size_t> next{0};
        std::atomic<size_t> ok_cnt{0};
        auto worker = [&]() {
            for (;;) {
                const size_t idx = next.fetch_add(1);
                if (idx >= total) return;
                if (ParseFile(idx)) ok_cnt.fetch_add(1);
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 1; t < n_threads; ++t) pool.emplace_back(worker);
        worker();
        for (auto &th : pool) th.join();
        for (size_t idx = 0; idx < total; ++idx)
            if (mmaps[idx]) lru.push_back(idx);
        log::info(log_prefix + "parallel parse: " +
                  std::to_string(ok_cnt.load()) + "/" +
                  std::to_string(total) + " experts in " +
                  std::to_string(n_threads) + " threads, " +
                  std::to_string(std::chrono::duration_cast<
                                     std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count()) + " ms");
        return ok_cnt.load() > 0;
#else
        (void)n_threads;
        return false;
#endif
    }

    // Decodes the shared 6-bin + 3-float ExpertSchema fields into @p out.
    static void DecodeEntry(const std::shared_ptr<arrow::RecordBatch> &batch,
                            TEntry &out)
    {
        auto get_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
            auto arr = batch->column(col);
            auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(arr);
            if (!bin) return {nullptr, 0};
            auto view = bin->GetView(0);
            return {reinterpret_cast<const uint8_t *>(view.data()),
                    static_cast<int64_t>(view.size())};
        };
        auto get_scale_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
            auto lst = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(col));
            if (!lst) return {nullptr, 0};
            auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
            if (!u8) return {nullptr, 0};
            auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
            if (!offs) return {nullptr, 0};
            int64_t begin = offs->Value(0);
            int64_t end   = offs->Value(1);
            return {u8->raw_values() + begin, end - begin};
        };
        auto get_gs = [&](int col) -> float {
            auto arr = batch->column(col);
            auto f = std::dynamic_pointer_cast<arrow::FloatArray>(arr);
            return f ? f->Value(0) : 1.0f;
        };

        auto [w1p, w1s] = get_bin(lsxcommon::ExpertSchema::kIdxW1Weight);
        auto [w2p, w2s] = get_bin(lsxcommon::ExpertSchema::kIdxW2Weight);
        auto [w3p, w3s] = get_bin(lsxcommon::ExpertSchema::kIdxW3Weight);
        auto [w1sp, w1ss] = get_scale_bin(lsxcommon::ExpertSchema::kIdxW1Scale);
        auto [w2sp, w2ss] = get_scale_bin(lsxcommon::ExpertSchema::kIdxW2Scale);
        auto [w3sp, w3ss] = get_scale_bin(lsxcommon::ExpertSchema::kIdxW3Scale);

        out = {w1p, w2p, w3p, w1sp, w2sp, w3sp,
               get_gs(lsxcommon::ExpertSchema::kIdxW1GlobalScale),
               get_gs(lsxcommon::ExpertSchema::kIdxW2GlobalScale),
               get_gs(lsxcommon::ExpertSchema::kIdxW3GlobalScale),
               w1s, w2s, w3s, w1ss, w2ss, w3ss};
    }

    bool ParseFile(size_t idx)
    {
        ++parse_calls;
        auto &apath = paths[idx];
        auto mmap_res = arrow::io::MemoryMappedFile::Open(
            apath, arrow::io::FileMode::READ);
        if (!mmap_res.ok()) {
            if (idx < 5 || idx % 1000 == 0)
                log::debug(log_prefix + mmap_fail_verb + apath + ": "
                           + mmap_res.status().ToString());
            return false;
        }
        auto mmap = *mmap_res;

        auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
        if (!reader_res.ok() || (*reader_res)->num_record_batches() == 0) {
            if (log_batch_fails)
                log::debug(log_prefix + "expert file has no record batches: " + apath);
            return false;
        }
        auto batch_res = (*reader_res)->ReadRecordBatch(0);
        if (!batch_res.ok()) {
            if (log_batch_fails)
                log::debug(log_prefix + "expert batch read failed for " + apath +
                           ": " + batch_res.status().ToString());
            return false;
        }
        auto batch = *batch_res;

        DecodeEntry(batch, entries[idx]);

        if (emit_diag && (idx == 0 || idx == 1 ||
                          idx == static_cast<size_t>(experts_per_layer))) {
            const TEntry &e = entries[idx];
            auto fmt_scale_bytes = [](const uint8_t *p, int64_t n) -> std::string {
                std::ostringstream os; os << "[";
                for (int64_t i = 0; i < n && i < 8; ++i) {
                    if (i) os << ",";
                    char b[4]; std::snprintf(b, sizeof(b), "%02x", p ? p[i] : 0);
                    os << b;
                }
                return os.str() + "]";
            };
            auto implied_blk = [](int64_t packed, int64_t scl) -> std::string {
                if (scl <= 0) return "n/a (0 scales)";
                return std::to_string((packed * 2) / scl) + " elems/scale (numel=" +
                       std::to_string(packed * 2) + " scales=" + std::to_string(scl) + ")";
            };
            std::ostringstream os;
            os << "[diag-expert] idx=" << idx
               << " w1: packed=" << e.w1_bytes << "B scales=" << e.w1_scales_bytes
               << "B gs=" << e.w1_global_scale
               << " blk=" << implied_blk(e.w1_bytes, e.w1_scales_bytes)
               << " s1[0..7]=" << fmt_scale_bytes(e.host_w1_scales, e.w1_scales_bytes)
               << " | w2: packed=" << e.w2_bytes << "B scales=" << e.w2_scales_bytes
               << "B gs=" << e.w2_global_scale
               << " blk=" << implied_blk(e.w2_bytes, e.w2_scales_bytes)
               << " | w3: packed=" << e.w3_bytes << "B scales=" << e.w3_scales_bytes
               << "B gs=" << e.w3_global_scale
               << " blk=" << implied_blk(e.w3_bytes, e.w3_scales_bytes);
            log::info(os.str());
        }

        ++parse_ok;
        mmaps[idx] = std::move(mmap);
        batches[idx] = std::move(batch);
        return true;
    }

    // Touch a resident entry or parse it, evicting the LRU tail under max_open.
    // The caller's stream is synchronized ONCE, only before the first eviction
    // (pre-munmap barrier for in-flight cudaMemcpyAsync reading host mmap).
    // @p evict_done, when non-null, releases that barrier via
    // cudaEventRecord + cudaStreamWaitEvent instead of cudaStreamSynchronize:
    // an in-flight async H2D reading an about-to-be-munmap'd host mmap is
    // awaited without draining unrelated stream work. nullptr preserves the
    // prior behavior exactly (existing DeepSeek call sites pass none).
    void EnsureResident(int layer, int expert, cudaStream_t stream,
                        cudaEvent_t evict_done = nullptr)
    {
        if (expert < 0 || expert >= experts_per_layer) return;
        if (paths.empty()) return;
        size_t idx = static_cast<size_t>(layer) *
                         static_cast<size_t>(experts_per_layer) +
                     static_cast<size_t>(expert);
        if (idx >= entries.size()) return;

        if (entries[idx].host_w1) {
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                if (*it == idx) { lru.erase(it); lru.insert(lru.begin(), idx); break; }
            }
            return;
        }

        if (lru.size() >= max_open) {
            if (evict_done) {
                cudaEventRecord(evict_done, stream);
                cudaStreamWaitEvent(stream, evict_done, 0);
            } else {
                cudaStreamSynchronize(stream);
            }
        }
        while (lru.size() >= max_open) {
            size_t evict = lru.back();
            lru.pop_back();
            // Retire any in-flight prefetch DMA reading this mapping before
            // it disappears (munmap under an async copy = silent garbage).
            lsx_pending_copy_drain(entries[evict].host_w1);
            mmaps[evict].reset();
            batches[evict].reset();
            entries[evict] = TEntry();
        }
        if (ParseFile(idx))
            lru.insert(lru.begin(), idx);
    }

    void Free()
    {
        if (pinned_pool) cudaFreeHost(pinned_pool);
        pinned_pool = nullptr;
        pinned_bytes = 0;
        for (size_t ci = 0; ci < pinned_pools.size(); ++ci) {
            void *p = pinned_pools[ci];
            if (!p) continue;
            const bool reg = ci < pinned_registered.size()
                                 ? pinned_registered[ci] : false;
            const size_t bytes = ci < pinned_chunk_sizes.size()
                                     ? pinned_chunk_sizes[ci] : 0;
            // PinChunked chunks are anonymous mmap + cudaHostRegister;
            // registered or not, release is munmap. PinChunked is compiled
            // only on Unix, so on Windows pinned_pools stays empty and this
            // loop body never runs -- but munmap is undeclared there, so the
            // call has to be guarded or the header fails to compile under
            // nvcc/MSVC.
            if (reg) cudaHostUnregister(p);
#if defined(__unix__) || defined(__APPLE__)
            if (bytes > 0) munmap(p, bytes);
#else
            (void)bytes;
#endif
        }
        pinned_pools.clear();
        pinned_registered.clear();
        pinned_chunk_sizes.clear();
        for (auto &m : mmaps) m.reset();
        for (auto &b : batches) b.reset();
        mmaps.clear(); batches.clear(); lru.clear(); entries.clear(); paths.clear();
        error_msg.clear();
        num_layers = 0; experts_per_layer = 0;
    }

    // ── Chunked locked-host pin (shared by Kimi / GLM / Gemma / any TEntry
    // with the six host_w* / *_bytes fields) ─────────────────────────────
    //
    // Pins the whole routed-expert store into locked host memory as a
    // vector of kPinChunkBytes chunks. Phases (timings for the 542 GiB
    // Kimi store, 2026-09-12):
    //   plan    — span sizes + chunk assignment (no IO)
    //   chunks  — parallel anonymous mmap (NO cudaHostAlloc: its driver-
    //             serialized zerofill costs 332 s at 542 GiB and evicts
    //             the page cache, turning the copy into a disk re-read)
    //   copy    — parallel fault-at-write span copies from the mmap views
    //             (reads the hot page cache; faults pinned pages with data)
    //   register— cudaHostRegister last, pinning already-populated pages
    //             (~1.6 s/chunk vs 5.7 s on unfaulted pages)
    //
    // kSplitFence layout packs per expert [w1|w3][s1|s3][w2][s2] and fills
    // the entry's blob_a..d/pinned_blob fields (requires TEntry to have
    // them — KimiExpertEntry); the stream layout packs [w1|w3|w2][s1|s3|s2]
    // (DeepSeek round-8 convention) without touching blob fields.
    //
    // Returns false when nothing was pinned (alloc failure) — the caller
    // falls back to mmap views; entries keep their mmap pointers in that
    // case. Entries without weights (dense holes) are skipped.
    enum class PinLayout { kStream, kSplitFence };
    using element_type = TEntry;

    template <PinLayout kLayout>
    bool PinChunked(int n_threads = 0)
    {
#if defined(__unix__) || defined(__APPLE__)
        if (entries.empty()) return false;
        constexpr int64_t kPinChunkBytes = int64_t{16} << 30;
        auto aln = [](uintptr_t v) { return (v + 255) / 256 * 256; };
        auto t0 = std::chrono::steady_clock::now();

        // Phase 1 — plan.
        struct PinSpan {
            TEntry *en;
            uint32_t chunk;
            int64_t off[6];   // w1,w3,w2,s1,s3,s2 span starts
        };
        std::vector<PinSpan> plan;
        std::vector<const uint8_t *> src[6];
        {
            int64_t cur = 0;
            uint32_t chunk_idx = 0;
            for (auto &en : entries) {
                if (!en.host_w1 || !en.host_w2 || !en.host_w3
                    || !en.host_w1_scales || !en.host_w2_scales
                    || !en.host_w3_scales)
                    continue;
                int64_t block;
                if constexpr (kLayout == PinLayout::kSplitFence) {
                    en.blob_a_bytes = aln(en.w1_bytes) + en.w3_bytes;
                    en.blob_b_bytes = aln(en.w1_scales_bytes)
                                    + en.w3_scales_bytes;
                    en.blob_c_bytes = en.w2_bytes;
                    en.blob_d_bytes = en.w2_scales_bytes;
                    block = aln(en.blob_a_bytes) + aln(en.blob_b_bytes)
                          + aln(en.blob_c_bytes) + aln(en.blob_d_bytes);
                } else {
                    block = aln(en.w1_bytes) + aln(en.w3_bytes)
                          + aln(en.w2_bytes) + aln(en.w1_scales_bytes)
                          + aln(en.w3_scales_bytes) + aln(en.w2_scales_bytes);
                }
                if (cur != 0 && cur + block > kPinChunkBytes) {
                    ++chunk_idx;
                    cur = 0;
                }
                PinSpan ps{};
                ps.en = &en;
                ps.chunk = chunk_idx;
                if constexpr (kLayout == PinLayout::kSplitFence) {
                    ps.off[0] = cur;                       // w1
                    ps.off[1] = ps.off[0] + aln(en.w1_bytes);  // w3
                    ps.off[3] = ps.off[1] + aln(en.w3_bytes);  // s1
                    ps.off[4] = ps.off[3] + aln(en.w1_scales_bytes);  // s3
                    ps.off[2] = ps.off[4] + aln(en.w3_scales_bytes);  // w2
                    ps.off[5] = ps.off[2] + aln(en.w2_bytes);         // s2
                    cur = ps.off[5] + aln(en.w2_scales_bytes);
                } else {
                    ps.off[0] = cur;
                    ps.off[1] = ps.off[0] + aln(en.w1_bytes);
                    ps.off[2] = ps.off[1] + aln(en.w3_bytes);
                    ps.off[3] = ps.off[2] + aln(en.w2_bytes);
                    ps.off[4] = ps.off[3] + aln(en.w1_scales_bytes);
                    ps.off[5] = ps.off[4] + aln(en.w3_scales_bytes);
                    cur = ps.off[5] + aln(en.w2_scales_bytes);
                }
                src[0].push_back(en.host_w1);
                src[1].push_back(en.host_w3);
                src[2].push_back(en.host_w2);
                src[3].push_back(en.host_w1_scales);
                src[4].push_back(en.host_w3_scales);
                src[5].push_back(en.host_w2_scales);
                plan.push_back(ps);
            }
        }
        if (plan.empty()) return false;

        // Phase 2 — parallel chunk mmap.
        const int n_chunks = static_cast<int>(plan.back().chunk + 1);
        pinned_pools.assign(static_cast<size_t>(n_chunks), nullptr);
        pinned_registered.assign(static_cast<size_t>(n_chunks), false);
        pinned_chunk_sizes.assign(static_cast<size_t>(n_chunks), 0);
        {
            std::atomic<int> next{0};
            auto alloc_worker = [&]() {
                for (;;) {
                    const int i = next.fetch_add(1);
                    if (i >= n_chunks) return;
                    void *chunk = mmap(nullptr,
                                       static_cast<size_t>(kPinChunkBytes),
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS
                                           | MAP_NORESERVE,
                                       -1, 0);
                    if (chunk == MAP_FAILED) continue;
                    pinned_pools[static_cast<size_t>(i)] = chunk;
                    pinned_chunk_sizes[static_cast<size_t>(i)] =
                        static_cast<size_t>(kPinChunkBytes);
                }
            };
            if (n_threads <= 0)
                n_threads = static_cast<int>(std::clamp<unsigned>(
                    std::thread::hardware_concurrency() / 16u, 4u, 8u));
            std::vector<std::thread> pool;
            for (int t = 1; t < n_threads; ++t) pool.emplace_back(alloc_worker);
            alloc_worker();
            for (auto &th : pool) th.join();
        }
        bool all_ok = true;
        for (void *c : pinned_pools)
            if (!c) all_ok = false;
        if (!all_ok) {
            for (void *c : pinned_pools)
                if (c) munmap(c, static_cast<size_t>(kPinChunkBytes));
            pinned_pools.clear();
            pinned_registered.clear();
            pinned_chunk_sizes.clear();
            return false;
        }
        pinned_bytes = static_cast<std::size_t>(n_chunks)
                     * static_cast<std::size_t>(kPinChunkBytes);

        // Phase 3 — parallel fault-at-write span copies.
        {
            std::atomic<size_t> next{0};
            auto copy_worker = [&]() {
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= plan.size()) return;
                    const PinSpan &ps = plan[i];
                    auto *base = static_cast<uint8_t *>(
                        pinned_pools[ps.chunk]);
                    TEntry &en = *ps.en;
                    const int64_t lens[6] = {en.w1_bytes, en.w3_bytes,
                                             en.w2_bytes, en.w1_scales_bytes,
                                             en.w3_scales_bytes,
                                             en.w2_scales_bytes};
                    for (int s = 0; s < 6; ++s)
                        std::memcpy(base + ps.off[s], src[s][i],
                                    static_cast<std::size_t>(lens[s]));
                }
            };
            const int nt = std::clamp<unsigned>(
                std::thread::hardware_concurrency() / 8u, 8u, 24u);
            std::vector<std::thread> pool;
            for (int t = 1; t < nt; ++t) pool.emplace_back(copy_worker);
            copy_worker();
            for (auto &th : pool) th.join();
        }

        // Phase 4 — register populated pages; entry pointers re-pointed.
        for (size_t ci = 0; ci < pinned_pools.size(); ++ci) {
            void *chunk = pinned_pools[ci];
            if (!chunk) continue;
            const cudaError_t ce = cudaHostRegister(
                chunk, pinned_chunk_sizes[ci], cudaHostRegisterDefault);
            if (ce != cudaSuccess) {
                cudaGetLastError();
                log::warn(log_prefix + "cudaHostRegister failed for chunk "
                          + std::to_string(ci) + " — chunk stays pageable");
            } else {
                pinned_registered[ci] = true;
            }
        }
        for (size_t i = 0; i < plan.size(); ++i) {
            PinSpan &ps = plan[i];
            auto *base = static_cast<uint8_t *>(pinned_pools[ps.chunk]);
            TEntry &en = *ps.en;
            en.host_w1 = base + ps.off[0];
            en.host_w3 = base + ps.off[1];
            en.host_w2 = base + ps.off[2];
            en.host_w1_scales = base + ps.off[3];
            en.host_w3_scales = base + ps.off[4];
            en.host_w2_scales = base + ps.off[5];
            if constexpr (kLayout == PinLayout::kSplitFence) {
                en.pinned_blob = base + ps.off[0];
            }
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        log::info(log_prefix + "pinned routed-expert store: "
                  + std::to_string(plan.size()) + " experts, "
                  + std::to_string(n_chunks) + " x 16 GiB chunks in "
                  + std::to_string(ms) + " ms");
        return true;
#else
        (void)n_threads;
        return false;
#endif
    }
};

} // namespace expert_store
} // namespace gpu
} // namespace lsxcommon