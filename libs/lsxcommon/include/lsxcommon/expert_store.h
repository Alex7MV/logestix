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
#include <cstdlib>
#include <cuda_runtime.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lsxcommon/arrow_validate.h"
#include "lsxcommon/log.h"
#include "lsxcommon/pread_pool.h"
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
    // ── Where each plane lives in its file ────────────────────────────────
    // The staging ring fills pinned slots with pread, which needs file offsets,
    // and the store is the only place that knows how a file maps onto the nine
    // ExpertSchema columns. Six spans per flat index, in ExpertSchema order
    // (w1, w2, w3, w1_scales, w2_scales, w3_scales); all zeros for a non-resident
    // entry. Written only by that entry's own ParseFile, which is the same
    // thread-ownership rule entries[]/mmaps[]/batches[] already follow.
    struct PlaneLocator {
        int64_t offset = 0;  // bytes from the start of the file
        int64_t bytes = 0;
    };
    std::vector<std::array<PlaneLocator, 6>> planes;
    // Per-store geometry, filled in the single-threaded phases of Load() /
    // LoadParallelParse() after workers have joined. Kept as plain scalars
    // because ExpertStore is moved by value (a mutex or an atomic member would
    // delete the move constructor several families rely on).
    int64_t plane_bytes[6] = {0, 0, 0, 0, 0, 0};
    int64_t file_bytes = 0;
    // 11008 DeepSeek experts = 11008 fds. The historical 7000 cap (Windows
    // CRT 8192 limit) forced constant mmap churn past layer 27 — and every
    // eviction munmap'ed entries whose async prefetch DMA could still be
    // in flight. Linux fd limits are effectively unbounded here.
    size_t max_open = 65536;
    int num_layers = 0;
    int experts_per_layer = 0;
    // Layers [0, first_moe_layer) are dense (no routed experts, no files on
    // disk) and are skipped when enumerating/opening the store. The flat index
    // math stays full-layer (layer*experts_per_layer+eid) so callers need no
    // remap; the dense slots just stay empty.
    int first_moe_layer = 0;
    // Expert file naming: most converters write exp_<e>.arrow; glm-5.3-flash
    // writes exp_<e>.arrow zero-padded to 3 digits. Only the file NAME differs
    // — the payload schema is ExpertSchema for every family.
    bool zero_pad_expert_names = false;
    // PinChunked chunk size (0 = the 16 GiB default). A window pinned from the
    // LRU warm set is a few GiB, and allocating whole 16 GiB chunks for it
    // would lock 4x the memory it is meant to save.
    int64_t pin_chunk_bytes = 0;
    std::string error_msg;
    // Converted trees are uniformly per-16 (measured over the whole vault);
    // a pre-repack source store sets 0 to disable the check.
    int32_t scale_block_size = 16;
    // Load() pre-parses up to max_open experts into the LRU so the first token
    // does not crawl. That is the right default for a small window, but it is a
    // pure startup tax once the window is large: it walks experts in PATH order
    // (layer 1..N, expert 0..895), not in the order decode will ask for them, at
    // ~2.4 ms per expert. A KimiLinear tree with a 65 536 window paid 156 s of
    // load for capacity the LRU would have filled on its own within two tokens,
    // so that family opts out and lets the window fill from real accesses.
    bool warm_on_load = true;

    // Diagnostics: ParseFile invocation / success counters (tests, loaders).
    int64_t parse_calls = 0;
    int64_t parse_ok = 0;
    // ── Deferred unmap of evicted slots ───────────────────────────────────
    // The decode loop issues an async H2D copy from an expert's mmap and may
    // evict that expert several residency calls later, so the mapping cannot be
    // released on the spot. The historical answer was cudaStreamSynchronize —
    // which stalls the host on EVERY eviction and waits for unrelated work
    // (measured: a 20 ms queued memset made a 2-slot eviction take 20.3 ms).
    // Instead the mapping is RETIRED with an event recorded on the stream that
    // issued the copies, and released once cudaEventQuery reports completion.
    // Nothing is unmapped while a DMA that could still read it is in flight,
    // and the host never blocks.
    struct RetiredMapping {
        std::shared_ptr<arrow::io::MemoryMappedFile> mmap;
        std::shared_ptr<arrow::RecordBatch> batch;
        cudaEvent_t ev = nullptr;  // null = nothing in flight, release at will
    };
    std::vector<RetiredMapping> retired;
    std::vector<cudaEvent_t> retired_event_pool;
    // Upper bound on mappings kept alive past their eviction; on overflow the
    // oldest event is synchronized (a rare, amortized host wait) so memory
    // stays bounded even if the GPU falls many layers behind.
    size_t max_retired = 64;
    // Optional per-store drain hook for the pending-prefetch-copy registry.
    // A family whose prefetch copies live in its own registry sets this;
    // when null the shared lsx_pending_copy_drain hook is called.
    void (*pending_drain)(const void *) = nullptr;
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

    // ── Pinned staging ring ───────────────────────────────────────────────
    // Fills pinned buffers from the expert files with a parallel pread pool and
    // DMAs them to the caller's device layout, so the streaming path stops
    // paying the pageable-copy cost (measured 5.2 GB/s pageable vs 10.5 GB/s
    // through this ring, 39.8 GB/s array ceiling).
    //
    // Off by default: `slots = 0` makes EnqueueStage refuse, and every family
    // keeps its own copy loop until it opts in. That is what makes this a
    // change with no blast radius on the 15 store aliases.
    struct StageRingConfig {
        size_t slots = 0;          // 0 = disabled
        int pread_threads = 4;     // <= 0 => 4
        size_t chunk_experts = 1;  // staging granularity (see InitStageRing)
    };
    StageRingConfig stage_ring;

    // Engagement counters. Without these, "the ring is engaged for this family"
    // is an assumption; a family silently falling back would look like a change
    // that simply did not help.
    struct StageStats {
        int64_t staged_experts = 0;
        int64_t staged_bytes = 0;
        int64_t legacy_fallbacks = 0;
        int64_t ring_waits = 0;
        int64_t short_reads = 0;
    };
    StageStats stage_stats;

    // One pinned buffer plus the event that says its DMA has executed. The
    // event is the whole point: a slot may not be refilled while a copy reading
    // it is still in flight.
    struct StageSlot {
        uint8_t *host = nullptr;
        cudaEvent_t ev = nullptr;
        int64_t last_idx = -1;
    };
    std::vector<StageSlot> stage_slots;
    // flat index -> slot currently holding it (or -1).
    std::vector<int> stage_slot_of;
    std::unique_ptr<io::PreadPool> stage_pool;
    size_t stage_cursor = 0;
    int64_t max_stage_bytes_ = 0;   // one slot's size, for PrefetchStage's bound
    std::vector<int> open_fds_;     // fds held across one PrefetchStage batch

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
        planes.assign(static_cast<size_t>(total), {});
        mmaps.assign(static_cast<size_t>(total), nullptr);
        batches.assign(static_cast<size_t>(total), nullptr);
        paths.resize(static_cast<size_t>(total));
        lru.reserve(max_open);

        fs::path exp_dir = fs::path(model_path) / "logestix" / "experts";
        if (fallback && !fs::exists(exp_dir) &&
            fs::exists(fs::path(model_path) / "experts"))
            exp_dir = fs::path(model_path) / "experts";

        for (int layer = 0; layer < num_layers; ++layer) {
            if (layer < first_moe_layer) continue;  // dense prefix, no files
            for (int exp = 0; exp < experts_per_layer; ++exp) {
                size_t idx = static_cast<size_t>(layer) *
                                 static_cast<size_t>(experts_per_layer) +
                             static_cast<size_t>(exp);
                char num[16];
                if (zero_pad_expert_names)
                    std::snprintf(num, sizeof(num), "%03d", exp);
                else
                    std::snprintf(num, sizeof(num), "%d", exp);
                paths[idx] = (exp_dir / ("layer_" + std::to_string(layer)) /
                              (std::string("exp_") + num + ".arrow")).string();
            }
        }

        // Warm the cache up to max_open (matches the per-model loop order).
        if (warm_on_load) {
            for (size_t idx = 0; idx < paths.size() && lru.size() < max_open;
                 ++idx) {
                if (paths[idx].empty()) continue;  // dense-prefix slot
                if (ParseFile(idx)) lru.push_back(idx);
            }
        }
        FillStoreGeometry();
        return true;
    }

    // Reads the uniform geometry of the tree from the first resident entry.
    // Called only from the single-threaded phases (after Load()'s warm loop, or
    // after LoadParallelParse's workers have joined), so it needs no locking and
    // ExpertStore stays movable by value.
    void FillStoreGeometry()
    {
        for (size_t idx = 0; idx < entries.size(); ++idx) {
            if (!entries[idx].host_w1 || idx >= planes.size()) continue;
            bool any = false;
            for (int k = 0; k < 6; ++k) {
                plane_bytes[k] = planes[idx][static_cast<size_t>(k)].bytes;
                any = any || plane_bytes[k] > 0;
            }
            if (!any) continue;
            const int64_t end = planes[idx][5].offset + planes[idx][5].bytes;
            file_bytes = end > 0 ? end : 0;
            if (idx < mmaps.size() && mmaps[idx]) {
                auto sz = mmaps[idx]->GetSize();
                if (sz.ok()) file_bytes = *sz;
            }
            return;
        }
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
                if (paths[idx].empty()) continue;  // dense-prefix slot
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
        // Workers have joined: the shared geometry is ours to fill now.
        FillStoreGeometry();
        return ok_cnt.load() > 0;
#else
        (void)n_threads;
        return false;
#endif
    }

    // Decodes the shared 6-bin + 3-float ExpertSchema fields into @p out.
    // Returns a status on a malformed/foreign batch so the caller never keeps a
    // partially decoded entry (null payload pointers or bogus byte counts).
    // `plane_ptrs`/`plane_sizes`, when given, receive the six host pointers the
    // decoder resolved, in ExpertSchema order. They exist so the file locators
    // (ParseFile) are derived from the SAME column mapping the decode uses —
    // a second copy of that mapping in ParseFile would be free to drift, and a
    // drifted mapping stages a neighbouring plane (wrong model, no error).
    static arrow::Status DecodeEntry(const std::shared_ptr<arrow::RecordBatch> &batch,
                                     TEntry &out,
                                     int32_t scale_block = 16,
                                     const uint8_t *plane_ptrs[6] = nullptr,
                                     int64_t plane_sizes[6] = nullptr)
    {
        if (!batch ||
            batch->num_columns() != lsxcommon::ExpertSchema::kNumFields)
            return arrow::Status::Invalid(
                "expert batch has ", batch ? batch->num_columns() : 0,
                " columns, expected ", lsxcommon::ExpertSchema::kNumFields);
        if (batch->num_rows() != 1)
            return arrow::Status::Invalid(
                "expert batch must have one row, got ", batch->num_rows());

        auto get_bin = [&](int col, const char *what)
            -> arrow::Result<std::pair<const uint8_t *, int64_t>> {
            auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
                batch->column(col));
            if (!bin)
                return arrow::Status::TypeError(what, " is not large_binary");
            auto view = bin->GetView(0);
            ARROW_RETURN_NOT_OK(lsxcommon::arrow_check::CheckViewRange(
                view.data(), static_cast<int64_t>(view.size()),
                bin->value_data(), what, "expert"));
            return std::make_pair(
                reinterpret_cast<const uint8_t *>(view.data()),
                static_cast<int64_t>(view.size()));
        };
        auto get_scale_bin = [&](int col, const char *what)
            -> arrow::Result<std::pair<const uint8_t *, int64_t>> {
            auto lst = std::dynamic_pointer_cast<arrow::ListArray>(
                batch->column(col));
            if (!lst)
                return arrow::Status::TypeError(what, " is not list");
            auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(
                lst->values());
            auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(
                lst->offsets());
            if (!u8 || !offs)
                return arrow::Status::TypeError(what, " is not list<uint8>");
            ARROW_RETURN_NOT_OK(lsxcommon::arrow_check::CheckListSlice(
                *offs, 0, u8->length(), what, "expert"));
            const int64_t begin = offs->Value(0);
            const int64_t end = offs->Value(1);
            return std::make_pair(u8->raw_values() + begin, end - begin);
        };
        auto get_gs = [&](int col, const char *what) -> arrow::Result<float> {
            auto f = std::dynamic_pointer_cast<arrow::FloatArray>(
                batch->column(col));
            if (!f)
                return arrow::Status::TypeError(what, " is not float32");
            return f->Value(0);
        };

        ARROW_ASSIGN_OR_RAISE(auto w1, get_bin(
            lsxcommon::ExpertSchema::kIdxW1Weight, "w1_weight"));
        ARROW_ASSIGN_OR_RAISE(auto w2, get_bin(
            lsxcommon::ExpertSchema::kIdxW2Weight, "w2_weight"));
        ARROW_ASSIGN_OR_RAISE(auto w3, get_bin(
            lsxcommon::ExpertSchema::kIdxW3Weight, "w3_weight"));
        ARROW_ASSIGN_OR_RAISE(auto s1, get_scale_bin(
            lsxcommon::ExpertSchema::kIdxW1Scale, "w1_scale"));
        ARROW_ASSIGN_OR_RAISE(auto s2, get_scale_bin(
            lsxcommon::ExpertSchema::kIdxW2Scale, "w2_scale"));
        ARROW_ASSIGN_OR_RAISE(auto s3, get_scale_bin(
            lsxcommon::ExpertSchema::kIdxW3Scale, "w3_scale"));
        ARROW_ASSIGN_OR_RAISE(auto g1, get_gs(
            lsxcommon::ExpertSchema::kIdxW1GlobalScale, "w1_global_scale"));
        ARROW_ASSIGN_OR_RAISE(auto g2, get_gs(
            lsxcommon::ExpertSchema::kIdxW2GlobalScale, "w2_global_scale"));
        ARROW_ASSIGN_OR_RAISE(auto g3, get_gs(
            lsxcommon::ExpertSchema::kIdxW3GlobalScale, "w3_global_scale"));

        if (scale_block > 0) {
            ARROW_RETURN_NOT_OK(lsxcommon::arrow_check::CheckExpertScaleCount(
                w1.second, s1.second, scale_block, "w1"));
            ARROW_RETURN_NOT_OK(lsxcommon::arrow_check::CheckExpertScaleCount(
                w2.second, s2.second, scale_block, "w2"));
            ARROW_RETURN_NOT_OK(lsxcommon::arrow_check::CheckExpertScaleCount(
                w3.second, s3.second, scale_block, "w3"));
        }
        out = {w1.first, w2.first, w3.first,
               s1.first, s2.first, s3.first, g1, g2, g3,
               w1.second, w2.second, w3.second,
               s1.second, s2.second, s3.second};
        if (plane_ptrs != nullptr) {
            plane_ptrs[0] = w1.first;
            plane_ptrs[1] = w2.first;
            plane_ptrs[2] = w3.first;
            plane_ptrs[3] = s1.first;
            plane_ptrs[4] = s2.first;
            plane_ptrs[5] = s3.first;
        }
        if (plane_sizes != nullptr) {
            plane_sizes[0] = w1.second;
            plane_sizes[1] = w2.second;
            plane_sizes[2] = w3.second;
            plane_sizes[3] = s1.second;
            plane_sizes[4] = s2.second;
            plane_sizes[5] = s3.second;
        }
        return arrow::Status::OK();
    }

    // Converts the decoder's six host pointers into file offsets.
    //
    // The mapping base comes from a zero-copy ReadAt(0, 8): Arrow's
    // MemoryMappedFile::ReadAt(position, n) returns a Slice of the mapping, so
    // at position 0 the returned buffer's address IS the mapping base. Asking
    // for 8 bytes rather than the whole file is deliberate — ReadAt calls
    // MemoryAdviseWillNeed on the requested range, and advising the full
    // 18.6 MB of every expert is exactly the page-cache prefetch that was
    // measured as an 18% loss (§R17 P5) and must not come back by accident.
    //
    // The gate is three checks at once, and it is the reason a wrong base is
    // loud rather than silent: offsets must be ascending and non-overlapping in
    // schema order, every span must sit inside the file, and the spans must
    // partition as packed planes followed by scale planes. A base off by the
    // Arrow header (16 bytes) or by a page breaks all three.
    static arrow::Result<std::array<PlaneLocator, 6>> RecordPlaneLocators(
        arrow::io::MemoryMappedFile *mmap, const uint8_t *const plane_ptrs[6],
        const int64_t plane_sizes[6])
    {
        if (mmap == nullptr)
            return arrow::Status::Invalid("no mapping for expert file");
        auto head = mmap->ReadAt(0, 8);
        if (!head.ok()) return head.status();
        const uint8_t *base =
            reinterpret_cast<const uint8_t *>((*head)->address());
        if (base == nullptr)
            return arrow::Status::Invalid("empty mapping for expert file");
        auto size_res = mmap->GetSize();
        if (!size_res.ok()) return size_res.status();
        const int64_t size = *size_res;

        std::array<PlaneLocator, 6> out{};
        int64_t prev_end = 0;
        for (int k = 0; k < 6; ++k) {
            // An EMPTY plane is legal and expected: NemotronH experts are
            // non-gated relu^2 MLPs whose converter writes w1 with shape {0,0},
            // so the file genuinely has no w1. Rejecting it made ParseFile fail
            // and left the entry pointing into a mapping that was about to be
            // destroyed — the nemotron model test segfaulted on it. An empty
            // plane is recorded as a zero span and skipped by the staging code;
            // the ordering and bounds checks still cover every plane that has
            // bytes, which is where a wrong mapping base shows up.
            if (plane_sizes[k] <= 0) {
                out[static_cast<size_t>(k)] = {0, 0};
                continue;
            }
            if (plane_ptrs[k] == nullptr)
                return arrow::Status::Invalid("plane ", k, " has ", plane_sizes[k],
                                              " bytes but no pointer");
            const int64_t off = static_cast<int64_t>(plane_ptrs[k] - base);
            if (off < prev_end)
                return arrow::Status::Invalid(
                    "plane ", k, " starts at ", off,
                    " inside the previous plane (ends ", prev_end,
                    ") — the mapping base is wrong");
            if (off + plane_sizes[k] > size)
                return arrow::Status::Invalid(
                    "plane ", k, " spans [", off, ",", off + plane_sizes[k],
                    ") outside the ", size, "-byte file");
            out[static_cast<size_t>(k)] = {off, plane_sizes[k]};
            prev_end = off + plane_sizes[k];
        }
        return out;
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

        const uint8_t *plane_ptrs[6] = {};
        int64_t plane_sizes[6] = {};
        const auto dec = DecodeEntry(batch, entries[idx], scale_block_size,
                                     plane_ptrs, plane_sizes);
        if (!dec.ok()) {
            if (log_batch_fails)
                log::debug(log_prefix + "expert batch decode failed for " +
                           apath + ": " + dec.ToString());
            return false;
        }
        // The ring stages with pread, so it needs file offsets, not pointers.
        // Deriving them here (rather than in the ring) keeps one owner for the
        // mapping and lets the sanity gate below run where the failure can still
        // be reported against a file name.
        auto loc = RecordPlaneLocators(mmap.get(), plane_ptrs, plane_sizes);
        if (!loc.ok()) {
            log::error(log_prefix + "expert plane locators rejected for " +
                       apath + ": " + loc.status().ToString());
            // DecodeEntry has already written into entries[idx], and those
            // pointers address THIS mapping, which is released when this
            // function returns. Leaving them behind is a dangling pointer the
            // caller cannot distinguish from a resident expert — that is how a
            // rejected locator became a SIGSEGV instead of a clean "not
            // resident". Clear it.
            entries[idx] = TEntry();
            return false;
        }
        planes[idx] = *loc;

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

    // ── pinned resident payload ─────────────────────────────────────────
    // cudaHostRegister on the six payload spans of one mapped expert. The
    // spans are disjoint sub-ranges of the file mapping, so registering them
    // individually keeps the lock to payload rather than file padding.
    // cudaHostRegister rejects a range whose start is not page-aligned, and an
    // expert payload starts at an arbitrary offset inside the file mapping — so
    // each span is rounded DOWN to the page boundary and its length extended to
    // cover the tail. Adjacent spans of one file therefore overlap by at most a
    // page, which the driver accepts.
    // Page-aligned, COALESCED spans of one expert's payload.
    //
    // Two constraints, both learned the hard way:
    //  - cudaHostRegister rejects a start that is not page-aligned, and an
    //    expert payload begins at an arbitrary offset inside the file mapping;
    //  - the six spans of one expert are ADJACENT in that file, so rounding
    //    each one down to a page makes neighbouring spans overlap. Registering
    //    an already-registered page range returns
    //    cudaErrorHostMemoryAlreadyRegistered, which is sticky and takes down
    //    every later CUDA call in the process — it presented as "generation
    //    fails right after a successful load", with no hint about pinning.
    // So: align down, sort, merge, and register the disjoint result. Pin and
    // unpin must use the SAME helper or the bookkeeping drifts.
    // Releases every retired mapping whose copies have completed. Called after
    // each eviction (bounded per call) and at teardown.
    //
    // Every retiree's event is recorded on the SAME stream, so they complete in
    // FIFO order: once the oldest is not ready, none of the younger ones can be.
    // Checking only the front makes this O(completed) — amortized one
    // cudaEventQuery per eviction — instead of O(retired), which was up to 64
    // driver calls per eviction and cost 6.6% of a 2048-token mistral-small-4
    // run (42.06 vs 45.04 t/s) before it was fixed.
    void ReapRetired()
    {
        while (!retired.empty()) {
            RetiredMapping &r = retired.front();
            if (r.ev) {
                const cudaError_t q = cudaEventQuery(r.ev);
                if (q == cudaErrorNotReady) return;
                if (q != cudaSuccess) {
                    // The stream or context is gone: no copy can still be
                    // reading this mapping. Clear the sticky error.
                    (void)cudaGetLastError();
                }
                retired_event_pool.push_back(r.ev);
            }
            r.mmap.reset();
            r.batch.reset();
            retired.erase(retired.begin());
        }
    }

    // Moves an evicted slot's mapping out of the live tables and keeps it alive
    // until `stream`'s work issued so far has executed. Falls back to a full
    // sync only if no event can be obtained (then the release is immediate and
    // therefore must be ordered).
    void RetireSlot(size_t idx, cudaStream_t stream)
    {
        RetiredMapping r;
        r.mmap = std::move(mmaps[idx]);
        r.batch = std::move(batches[idx]);
        entries[idx] = TEntry();
        if (!r.mmap) return;  // never mapped (or pinned store): nothing to wait on

        cudaEvent_t ev = nullptr;
        if (!retired_event_pool.empty()) {
            ev = retired_event_pool.back();
            retired_event_pool.pop_back();
        } else if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) !=
                   cudaSuccess) {
            (void)cudaGetLastError();
            ev = nullptr;
        }
        if (ev) {
            if (cudaEventRecord(ev, stream) != cudaSuccess) {
                (void)cudaGetLastError();
                retired_event_pool.push_back(ev);
                ev = nullptr;
            }
        }
        if (!ev) {
            // No event: order the release the slow way rather than unmap under
            // a possible in-flight copy.
            if (stream) (void)cudaStreamSynchronize(stream);
            else (void)cudaDeviceSynchronize();
        }
        r.ev = ev;
        retired.push_back(std::move(r));
    }

    // Touch a resident entry or parse it, evicting the LRU tail under max_open.
    // Eviction NEVER drains the host on the stream: an evicted expert's mapping
    // is retired with an event (RetireSlot) and released once the stream has
    // executed the copies issued so far — see the RetiredMapping comment for
    // the measurement and the bound. (@p evict_done, when non-null, still
    // records an event and makes the stream wait on it, so a family that wants
    // that ordering point can keep it; the DeepSeek call sites use it, Kimi
    // passes nullptr and relies on the retire/event path.)
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
            }
            // Deliberately NO host drain here: an eviction retires the mapping
            // with an event (RetireSlot) instead of waiting for the whole
            // stream. See the RetiredMapping comment.
        }
        while (lru.size() >= max_open) {
            size_t evict = lru.back();
            lru.pop_back();
            // Retire any in-flight prefetch DMA reading this mapping before
            // it disappears (munmap under an async copy = silent garbage).
            // Families with their own copy registry drain it here; the shared
            // hook covers the rest. RetireSlot keeps the mapping itself alive
            // until the stream has executed the copies issued so far.
            if (entries[evict].host_w1) {
                if (pending_drain) pending_drain(entries[evict].host_w1);
                else lsx_pending_copy_drain(entries[evict].host_w1);
            }
            RetireSlot(evict, stream);
        }
        ReapRetired();
        if (retired.size() > max_retired) {
            // Bounded memory: wait for the oldest retiree only, then reap what
            // has completed. Amortized and rare — it needs the GPU to be
            // max_retired experts behind the host.
            if (retired.front().ev) (void)cudaEventSynchronize(retired.front().ev);
            ReapRetired();
        }
        if (ParseFile(idx))
            lru.insert(lru.begin(), idx);
    }

    // ── Staging ring ─────────────────────────────────────────────────────
    // Reads the four knobs into stage_ring. Defaults matter more than parsing
    // here: `slots = 0` (off) is what every family gets until it opts in, and a
    // threads value of 0 means the measured default (4), never a zero-worker
    // pool that would complete nothing while looking configured.
    //
    // LSX_EXPERT_STAGE_RING gates the rest: a SLOTS left in someone's shell is
    // not consent, so a ring flag of 0 keeps slots at 0.
    void ResolveStageRingFromEnv()
    {
        auto env_int = [](const char *name, long dflt) -> long {
            const char *v = std::getenv(name);
            if (v == nullptr || v[0] == '\0') return dflt;
            char *end = nullptr;
            const long parsed = std::strtol(v, &end, 10);
            return (end == v) ? dflt : parsed;
        };
        const long on = env_int("LSX_EXPERT_STAGE_RING", 0);
        if (on == 0) {
            stage_ring.slots = 0;
            return;
        }
        const long slots = env_int("LSX_EXPERT_STAGE_SLOTS", 64);
        stage_ring.slots = slots > 0 ? static_cast<size_t>(slots) : 64;
        const long threads = env_int("LSX_EXPERT_STAGE_THREADS", 0);
        stage_ring.pread_threads = threads > 0 ? static_cast<int>(threads) : 4;
        const long chunk = env_int("LSX_EXPERT_STAGE_CHUNK", 0);
        stage_ring.chunk_experts = chunk > 0 ? static_cast<size_t>(chunk) : 1;
    }

    // One line, and only when the ring actually staged something: a family that
    // fell back must not print a line that looks like engagement.
    void LogStageStats() const
    {
        if (stage_stats.staged_experts == 0) return;
        log::info(log_prefix + "staging ring: " +
                  std::to_string(stage_stats.staged_experts) +
                  " experts, " + std::to_string(stage_stats.staged_bytes) +
                  " B staged, " + std::to_string(stage_stats.legacy_fallbacks) +
                  " legacy fallbacks, " +
                  std::to_string(stage_stats.ring_waits) + " slot waits, " +
                  std::to_string(stage_stats.short_reads) + " short reads");
    }

    // Allocates the pinned slots, their events and the pread pool.
    // False (with a log line) when slots == 0 or pinned memory is unavailable.
    bool InitStageRing()
    {
        FreeStageRing();
        if (stage_ring.slots == 0) return false;
        if (stage_ring.pread_threads <= 0) stage_ring.pread_threads = 4;
        if (stage_ring.chunk_experts == 0) stage_ring.chunk_experts = 1;

        // Plane geometry comes from a parsed expert, and a store with
        // warm_on_load = false has none at this point — that is the KimiLinear
        // case, where the eager pre-parse cost 156 s for nothing (§R17 P5) and
        // was disabled. Parse ONE expert here instead: the tree is uniform, so
        // one file gives the geometry, and it costs ~2.4 ms rather than a load
        // pass. Found on the real model: without this the ring logged "no plane
        // geometry" and every expert silently took the legacy path.
        int64_t max_bytes = 0;
        for (int k = 0; k < 6; ++k) max_bytes += plane_bytes[k];
        if (max_bytes <= 0) {
            for (size_t idx = 0; idx < paths.size(); ++idx) {
                if (paths[idx].empty()) continue;  // dense-prefix slot
                if (!ParseFile(idx)) continue;
                lru.insert(lru.begin(), idx);
                break;
            }
            FillStoreGeometry();
            max_bytes = 0;
            for (int k = 0; k < 6; ++k) max_bytes += plane_bytes[k];
        }
        if (max_bytes <= 0) {
            log::error(log_prefix +
                       "staging ring: no plane geometry (no expert could be "
                       "parsed)");
            return false;
        }
        stage_slot_of.assign(entries.size(), -1);
        stage_slots.reserve(stage_ring.slots);
        for (size_t i = 0; i < stage_ring.slots; ++i) {
            StageSlot slot;
            // The locked-memory limit is the failure that presents as "every
            // later CUDA call fails", so name it here.
            const cudaError_t alloc = cudaHostAlloc(
                reinterpret_cast<void **>(&slot.host),
                static_cast<size_t>(max_bytes), cudaHostAllocDefault);
            if (alloc != cudaSuccess) {
                (void)cudaGetLastError();
                log::error(log_prefix + "staging ring: cudaHostAlloc of " +
                           std::to_string(max_bytes) + " B failed (" +
                           cudaGetErrorString(alloc) +
                           "); raise `ulimit -l` or lower "
                           "LSX_EXPERT_STAGE_SLOTS — pinned staging is off");
                FreeStageRing();
                return false;
            }
            if (cudaEventCreateWithFlags(&slot.ev, cudaEventDisableTiming) !=
                cudaSuccess) {
                (void)cudaGetLastError();
                cudaFreeHost(slot.host);
                FreeStageRing();
                log::error(log_prefix + "staging ring: event creation failed");
                return false;
            }
            // Nothing has been recorded on this event yet; mark it completed so
            // the first recycle attempt does not wait on it.
            (void)cudaEventRecord(slot.ev, nullptr);
            (void)cudaEventSynchronize(slot.ev);
            stage_slots.push_back(slot);
        }
        max_stage_bytes_ = max_bytes;
        stage_pool = std::make_unique<io::PreadPool>(stage_ring.pread_threads);
        stage_cursor = 0;
        return true;
    }

    // Stages one resident expert into @p dst_dev.
    //
    // @p dst_off is indexed in ExpertSchema order (w1, w2, w3, w1_scales,
    // w2_scales, w3_scales), so a family keeps whatever device layout it
    // already has. Returns false when the ring is disabled, the entry is not
    // resident, or no slot can be recycled without stalling — the caller then
    // uses its legacy copy for that expert.
    bool EnqueueStage(size_t idx, uint8_t *dst_dev, const int64_t (&dst_off)[6],
                      cudaStream_t stream)
    {
        if (stage_ring.slots == 0 || stage_slots.empty() || !stage_pool) {
            ++stage_stats.legacy_fallbacks;
            return false;
        }
        if (idx >= entries.size() || idx >= planes.size() || !entries[idx].host_w1) {
            ++stage_stats.legacy_fallbacks;
            return false;
        }
        if (dst_dev == nullptr) {
            ++stage_stats.legacy_fallbacks;
            return false;
        }

        // Recycle: prefer a slot whose DMA has EXECUTED, else wait for the
        // oldest one to have executed.
        //
        // This wait MUST be host-side (cudaEventSynchronize), and the reason is
        // easy to get wrong: the refill below writes the pinned buffer from the
        // host with pread, so ordering the stream is not enough — a stream-side
        // wait orders device work, while the host is already writing the buffer
        // that the not-yet-executed DMA is about to read. (First implementation
        // used cudaStreamWaitEvent here and its hazard test passed against the
        // broken code; the doctored-slot experiment is what exposed it.)
        //
        // The wait is rare by construction: with slots >= one layer's top-k
        // every expert of a layer owns a slot, so recycling happens a layer
        // later, long after that layer's DMAs retired, and the query succeeds
        // without blocking. slots < top-k turns this into a per-expert stall,
        // which is why LSX_EXPERT_STAGE_SLOTS must be at least top-k.
        size_t pick = stage_slots.size();
        for (size_t i = 0; i < stage_slots.size(); ++i) {
            const size_t cand = (stage_cursor + i) % stage_slots.size();
            const cudaError_t q = cudaEventQuery(stage_slots[cand].ev);
            if (q == cudaSuccess) { pick = cand; break; }
            if (q != cudaErrorNotReady) (void)cudaGetLastError();
        }
        if (pick == stage_slots.size()) {
            pick = stage_cursor % stage_slots.size();
            const cudaError_t w = cudaEventSynchronize(stage_slots[pick].ev);
            if (w != cudaSuccess) (void)cudaGetLastError();
            ++stage_stats.ring_waits;
        }
        // The previous occupant is gone from this slot.
        if (stage_slots[pick].last_idx >= 0 &&
            static_cast<size_t>(stage_slots[pick].last_idx) < stage_slot_of.size())
            stage_slot_of[static_cast<size_t>(stage_slots[pick].last_idx)] = -1;

        StageSlot &slot = stage_slots[pick];
        std::vector<io::PreadJob> jobs;
        jobs.reserve(6);
        int64_t staged = 0;
        const auto &path = paths[idx];
        // Open per staging call: the tree is 1.4 TB against 65 536 descriptors in
        // this store, so holding one fd per expert is the churn the window was
        // raised to avoid. The pool then reads the six planes in parallel.
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            log::error(log_prefix + "staging ring: cannot open " + path);
            ++stage_stats.legacy_fallbacks;
            return false;
        }
        int64_t cursor = 0;
        for (int k = 0; k < 6; ++k) {
            const auto &p = planes[idx][static_cast<size_t>(k)];
            // Explicit construction, not brace-init: this header is compiled by
            // NVCC too, and aggregate init with a uint8_t* into a void* member
            // does not resolve there.
            io::PreadJob job;
            job.fd = fd;
            job.offset = p.offset;
            job.dst = static_cast<void *>(slot.host + cursor);
            job.len = static_cast<size_t>(p.bytes);
            jobs.push_back(job);
            cursor += p.bytes;
            staged += p.bytes;
        }
        const size_t ok_jobs = stage_pool->Run(jobs);
        ::close(fd);
        if (ok_jobs != jobs.size()) {
            // Never DMA a half-filled slot.
            ++stage_stats.short_reads;
            ++stage_stats.legacy_fallbacks;
            return false;
        }

        // The pread jobs packed the six planes back to back in the slot, so the
        // DMA source for plane k is the running sum of the planes before it.
        int64_t slot_off = 0;
        for (int k = 0; k < 6; ++k) {
            const auto &p = planes[idx][static_cast<size_t>(k)];
            void *dst = dst_dev + dst_off[k];
            const cudaError_t cp = cudaMemcpyAsync(
                dst, slot.host + slot_off, static_cast<size_t>(p.bytes),
                cudaMemcpyHostToDevice, stream);
            slot_off += p.bytes;
            if (cp != cudaSuccess) {
                (void)cudaGetLastError();
                log::error(log_prefix + "staging ring: DMA failed for " + path);
                ++stage_stats.legacy_fallbacks;
                return false;
            }
        }
        if (cudaEventRecord(slot.ev, stream) != cudaSuccess) {
            (void)cudaGetLastError();
            log::error(log_prefix + "staging ring: event record failed");
            ++stage_stats.legacy_fallbacks;
            return false;
        }
        slot.last_idx = static_cast<int64_t>(idx);
        stage_slot_of[idx] = static_cast<int>(pick);
        stage_cursor = (pick + 1) % stage_slots.size();
        ++stage_stats.staged_experts;
        stage_stats.staged_bytes += staged;
        return true;
    }

    // Fills pinned slots for a whole layer's expert set in ONE pool pass.
    //
    // Why a batch and not N calls to EnqueueStage: the measured lever is
    // parallelism across the layer's planes, not per-expert. The probe staged
    // one layer (16 experts x 6 planes = 96 preads over 4 threads) at
    // 26.9 ms cold against 54.8 ms for the pageable path, while a per-expert
    // refill can only ever offer 6 jobs per barrier — measured on the real
    // model at 32 tokens, the per-expert form was neutral-to-worse.
    //
    // Pure host work: no CUDA call, so the caller can prefetch a whole layer
    // while the device is still busy with the previous one. Returns the number
    // of experts whose six planes are ready in a slot.
    size_t PrefetchStage(const size_t *idx, size_t count)
    {
        if (stage_ring.slots == 0 || stage_slots.empty() || !stage_pool)
            return 0;
        if (count > stage_slots.size()) count = stage_slots.size();
        std::vector<io::PreadJob> jobs;
        std::vector<std::pair<size_t, size_t>> owners;  // (idx, slot)
        jobs.reserve(count * 6);
        owners.reserve(count);
        for (size_t n = 0; n < count; ++n) {
            const size_t flat = idx[n];
            if (flat >= entries.size() || flat >= planes.size() ||
                !entries[flat].host_w1 ||
                paths.empty() || paths[flat].empty())
                continue;
            const size_t slot_i = n;  // one slot per expert of the batch
            StageSlot &slot = stage_slots[slot_i];
            const cudaError_t q = cudaEventQuery(slot.ev);
            if (q != cudaSuccess) {
                if (q != cudaErrorNotReady) (void)cudaGetLastError();
                (void)cudaEventSynchronize(slot.ev);
                ++stage_stats.ring_waits;
            }
            // Size check BEFORE anything is queued or opened. A tree whose
            // experts are not uniform (a repack, a mixed variant) would
            // otherwise overflow the slot sized from the first parsed expert,
            // and the first implementation leaked the descriptor while doing it:
            // it pushed the jobs, noticed the overflow and continued without
            // recording the fd for closing.
            int64_t need = 0;
            for (int k = 0; k < 6; ++k)
                need += planes[flat][static_cast<size_t>(k)].bytes;
            if (need > max_stage_bytes_) {
                log::error(log_prefix + "staging ring: " + paths[flat] + " needs " +
                           std::to_string(need) + " B but a slot holds " +
                           std::to_string(max_stage_bytes_) +
                           " B (geometry from the first parsed expert); this "
                           "tree is not uniform — the ring declines");
                ++stage_stats.short_reads;
                continue;
            }
            const int fd = ::open(paths[flat].c_str(), O_RDONLY);
            if (fd < 0) continue;
            int64_t cursor = 0;
            for (int k = 0; k < 6; ++k) {
                const auto &pl = planes[flat][static_cast<size_t>(k)];
                io::PreadJob job;
                job.fd = fd;
                job.offset = pl.offset;
                job.dst = static_cast<void *>(slot.host + cursor);
                job.len = static_cast<size_t>(pl.bytes);
                jobs.push_back(job);
                cursor += pl.bytes;
            }
            // The fd must stay open until the pool has read from it; Run() below
            // is synchronous per batch, so closing after it is safe.
            open_fds_.push_back(fd);
            slot.last_idx = static_cast<int64_t>(flat);
            owners.push_back({flat, slot_i});
        }
        if (jobs.empty()) return 0;
        const size_t done = stage_pool->Run(jobs);
        for (int fd : open_fds_) ::close(fd);
        open_fds_.clear();
        if (done != jobs.size()) {
            stage_stats.short_reads += static_cast<int64_t>(jobs.size() - done);
            return 0;  // never DMA a partially filled batch
        }
        for (const auto &o : owners) {
            if (o.first < stage_slot_of.size())
                stage_slot_of[o.first] = static_cast<int>(o.second);
            ++stage_stats.staged_experts;
        }
        return owners.size();
    }

    // DMAs one expert's six planes out of the slot PrefetchStage filled.
    bool StageFromSlot(size_t idx, uint8_t *dst_dev,
                       const int64_t (&dst_off)[6], cudaStream_t stream)
    {
        if (idx >= stage_slot_of.size()) return false;
        const int slot_i = stage_slot_of[idx];
        if (slot_i < 0 || static_cast<size_t>(slot_i) >= stage_slots.size())
            return false;
        StageSlot &slot = stage_slots[static_cast<size_t>(slot_i)];
        int64_t slot_off = 0;
        for (int k = 0; k < 6; ++k) {
            const auto &pl = planes[idx][static_cast<size_t>(k)];
            const cudaError_t cp = cudaMemcpyAsync(
                dst_dev + dst_off[k], slot.host + slot_off,
                static_cast<size_t>(pl.bytes), cudaMemcpyHostToDevice, stream);
            slot_off += pl.bytes;
            if (cp != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
        }
        stage_stats.staged_bytes += slot_off;
        return cudaEventRecord(slot.ev, stream) == cudaSuccess;
    }

    // Makes @p stream wait for the DMA queued for @p idx. No host sync.
    void AwaitStage(size_t idx, cudaStream_t stream)
    {
        if (idx >= stage_slot_of.size()) return;
        const int slot = stage_slot_of[idx];
        if (slot < 0 || static_cast<size_t>(slot) >= stage_slots.size()) return;
        (void)cudaStreamWaitEvent(stream, stage_slots[static_cast<size_t>(slot)].ev,
                                  0);
    }

    void FreeStageRing()
    {
        for (auto &slot : stage_slots) {
            if (slot.host) (void)cudaFreeHost(slot.host);
            if (slot.ev) (void)cudaEventDestroy(slot.ev);
            slot = StageSlot{};
        }
        stage_slots.clear();
        stage_slot_of.clear();
        stage_pool.reset();
    }

    void Free()
    {
        FreeStageRing();
        // Retired mappings first: an evicted expert's mmap may still have a
        // copy reading it, so wait for its event before releasing anything.
        for (auto &r : retired)
            if (r.ev) (void)cudaEventSynchronize(r.ev);
        retired.clear();
        for (auto ev : retired_event_pool) (void)cudaEventDestroy(ev);
        retired_event_pool.clear();
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
        const int64_t chunk_bytes =
            pin_chunk_bytes > 0 ? pin_chunk_bytes : kPinChunkBytes;
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
                if (cur != 0 && cur + block > chunk_bytes) {
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
                                       static_cast<size_t>(chunk_bytes),
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS
                                           | MAP_NORESERVE,
                                       -1, 0);
                    if (chunk == MAP_FAILED) continue;
                    pinned_pools[static_cast<size_t>(i)] = chunk;
                    pinned_chunk_sizes[static_cast<size_t>(i)] =
                        static_cast<size_t>(chunk_bytes);
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
                if (c) munmap(c, static_cast<size_t>(chunk_bytes));
            pinned_pools.clear();
            pinned_registered.clear();
            pinned_chunk_sizes.clear();
            return false;
        }
        pinned_bytes = static_cast<std::size_t>(n_chunks)
                     * static_cast<std::size_t>(chunk_bytes);

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