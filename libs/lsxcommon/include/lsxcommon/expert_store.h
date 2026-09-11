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
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
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
        for (auto &m : mmaps) m.reset();
        for (auto &b : batches) b.reset();
        mmaps.clear(); batches.clear(); lru.clear(); entries.clear(); paths.clear();
        error_msg.clear();
        num_layers = 0; experts_per_layer = 0;
    }
};

} // namespace expert_store
} // namespace gpu
} // namespace lsxcommon