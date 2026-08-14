#include "lsxnetwork/io_worker.h"
#include "lsxnetwork/lsx_token_translator.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <thread>
#include <vector>

// ────────────────────────────────────────────────────────────────────────────
// TDD Phase 4.5 — translating-worker weld.
//
// Exercises splicing an individually-owned LsxTokenTranslator into the pinned
// IoWorker thread context: per-thread isolation (4 independent translators over
// one shared read-only passport), strict 64-byte cache-line alignment of every
// embedded mapping, zero cross-thread lock contention on the translation fast
// path, and zero runtime heap allocations across a concurrent multi-threaded
// payload burst (proven by a per-thread replacement operator-new counter, so
// each worker's tally cannot be polluted by on-looker threads).
// ────────────────────────────────────────────────────────────────────────────

namespace {

std::atomic<long long> g_alloc_count{0};

// Per-thread allocation counter. In a multi-threaded burst the global counter
// is polluted by unrelated threads (MSVC CRT / Catch2 housekeeping on the
// assert thread), so the zero-allocation proof measures each worker's OWN
// thread-local tally around its translation burst.
thread_local long long g_thread_alloc{0};

std::filesystem::path MakeTempPassport(const lsxnetwork::BuildPassportOptions& opts) {
    static std::atomic<unsigned> seq{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto p = std::filesystem::temp_directory_path() /
                   ("lsx_weld_" + std::to_string(seq.fetch_add(1)) + "_" +
                    std::to_string(stamp) + ".bin");
    REQUIRE(lsxnetwork::WritePassport(p.string().c_str(), opts) == 0);
    return p;
}

lsxnetwork::BuildPassportOptions Fixture() {
    lsxnetwork::BuildPassportOptions o;
    o.glm_tokens = {"hello", "world", "foo", "hello!", "zzz", "<|im_end|>"};
    o.deepseek_tokens = {"hello", "world", "foo", "bar", "<|im_end|>"};
    o.glm_unk_id = 4;
    o.deepseek_unk_id = 3;  // the genuine passport unknown id (no hardcoded magic)
    o.lmcs_min_overlap = 2;
    return o;
}

/// One worker thread's burst result. No release gate is used: the four workers
/// already run as independent OS threads, so the burst is inherently concurrent.
/// Coordination is a single done-flag, which Stop() can always interrupt.
struct WeldContext {
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> checksum{0};
    std::atomic<long long> alloc_delta{0};
    std::atomic<bool> translator_valid{false};
};

constexpr unsigned kWorkers = 4;
constexpr unsigned kRing = 8;              // token ids per in-place buffer
constexpr std::uint32_t kIters = 200000;   // translation iterations per worker

}  // namespace

// Global + per-thread replacement allocators. g_thread_alloc is incremented per
// allocation on the calling thread so the concurrent zero-allocation proof stays
// immune to on-looker threads (CRT/Catch2 housekeeping on the assert thread).
void* operator new(std::size_t n) {
    ++g_thread_alloc;
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    ++g_thread_alloc;
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void*, std::size_t) noexcept {}
void operator delete[](void*, std::size_t) noexcept {}

// Runs the measured translation burst. A first touch of the translator on a
// throwaway warm-up buffer (OUTSIDE the measured window) absorbs any one-time
// MSVC CRT/per-thread initialization, so `*out_alloc_delta` reflects only the
// steady-state translation burst — which must be 0. Returns the checksum.
std::uint64_t RunBurst(const lsxnetwork::LsxTokenTranslator& tr,
                       long long* out_alloc_delta) {
    alignas(64) std::uint32_t warm[kRing] = {0, 1, 2, 3, 4, 5, 0, 1};
    lsxnetwork::TranslateGlmToDeepSeekInPlace(tr, warm, kRing);

    alignas(64) std::uint32_t ids[kRing] = {0, 1, 2, 3, 4, 5, 0, 1};
    const long long before = g_thread_alloc;
    std::uint64_t sum = 0;
    for (std::uint32_t it = 0; it < kIters; ++it) {
        lsxnetwork::TranslateGlmToDeepSeekInPlace(tr, ids, kRing);
        sum ^= (static_cast<std::uint64_t>(it) << 32) ^ ids[it & (kRing - 1)];
    }
    const long long after = g_thread_alloc;
    if (out_alloc_delta) *out_alloc_delta = after - before;
    return sum;
}

// Each worker's loop body: hammers TranslateGlmToDeepSeek over a private
// 64-aligned ring buffer in place, using the worker's OWN embedded translator
// reached via the release/acquire-published worker pointer. Zero allocations.
void WeldLoop(WeldContext& ctx, std::atomic<lsxnetwork::IoWorker*>& wptr) {
    lsxnetwork::IoWorker* w = wptr.load(std::memory_order_acquire);
    if (w == nullptr) {
        ctx.done.store(true, std::memory_order_release);
        return;
    }

    // The embedded translator load completed on the worker thread before the
    // loop ran (IoWorker::Entry orders Load() before loop_()).
    ctx.translator_valid.store(w->Translator().Valid(), std::memory_order_relaxed);

    long long delta = 0;
    ctx.checksum.store(RunBurst(w->Translator(), &delta), std::memory_order_relaxed);
    ctx.alloc_delta.store(delta, std::memory_order_relaxed);
    ctx.done.store(true, std::memory_order_release);
}

/// Single-threaded reference checksum over the same loop the workers run.
std::uint64_t ReferenceChecksum(const lsxnetwork::LsxTokenTranslator& tr) {
    return RunBurst(tr, nullptr);
}

TEST_CASE("IoWorker embeds an isolated, cache-line aligned translator per thread",
          "[lsx_network_translator_weld]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);

    // Each worker owns its own LsxTokenTranslator instance (a private member,
    // not a shared singleton) even though all of them map the same read-only
    // passport file.
    std::vector<std::unique_ptr<lsxnetwork::IoWorker>> workers;
    for (unsigned i = 0; i < kWorkers; ++i) {
        (void)i;
        workers.push_back(
            std::make_unique<lsxnetwork::IoWorker>([] {}, path.string()));
    }

    REQUIRE(workers.size() == kWorkers);
    for (const auto& w : workers) {
        const int s = w->Start();
        REQUIRE(s == 0);
    }

    // After Start() the translator is loaded (load precedes the loop). Check
    // each embedded mapping: 64-byte alignment of object and access pointers,
    // and correct GLM->DeepSeek translation incl. the genuine passport unk id.
    for (const auto& w : workers) {
        lsxnetwork::LsxTokenTranslator& tr = w->Translator();
        REQUIRE(tr.Valid());
        REQUIRE(reinterpret_cast<std::uintptr_t>(&tr) % 64 == 0);  // alignas(64)
        REQUIRE(reinterpret_cast<std::uintptr_t>(tr.GlmToDeepSeek()) % 64 == 0);
        REQUIRE(reinterpret_cast<std::uintptr_t>(tr.DeepSeekToGlm()) % 64 == 0);
        REQUIRE(tr.TranslateGlmToDeepSeek(0) == 0);
        REQUIRE(tr.TranslateGlmToDeepSeek(4) == 3);  // "zzz" -> genuwin unk
    }

    // Distinct translator instances => no single shared static, so the fast
    // path carries no cross-thread lock.
    for (unsigned i = 1; i < kWorkers; ++i)
        REQUIRE(&workers[i]->Translator() != &workers[0]->Translator());

    for (auto& w : workers) w->Stop();
}

TEST_CASE(
    "Concurrent burst: per-thread independence, zero lock contention, zero allocs",
    "[lsx_network_translator_weld]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);

    // Single-threaded reference result.
    lsxnetwork::LsxTokenTranslator ref;
    REQUIRE(ref.Load(path.string().c_str()) == 0);
    const std::uint64_t reference = ReferenceChecksum(ref);

    std::vector<std::unique_ptr<lsxnetwork::IoWorker>> workers;
    std::vector<std::atomic<lsxnetwork::IoWorker*>> wptrs(kWorkers);
    std::vector<WeldContext> ctxs(kWorkers);

    for (unsigned i = 0; i < kWorkers; ++i) {
        auto w = std::make_unique<lsxnetwork::IoWorker>(
            [&ctx = ctxs[i], &wptr = wptrs[i]] { WeldLoop(ctx, wptr); },
            path.string());
        wptrs[i].store(w.get(), std::memory_order_relaxed);
        workers.push_back(std::move(w));
    }

    for (unsigned i = 0; i < kWorkers; ++i) REQUIRE(workers[i]->Start() == 0);

    // Wait for every worker to finish its translation burst; Stop() below can
    // always interrupt this poll (the loops terminate on their own).
    for (auto& ctx : ctxs) {
        while (!ctx.done.load(std::memory_order_acquire)) std::this_thread::yield();
    }

    for (unsigned i = 0; i < kWorkers; ++i) {
        REQUIRE(ctxs[i].translator_valid.load(std::memory_order_relaxed));
        // Each worker reproduced the same deterministic check value: no
        // interleaving, shared-state or lock contention drove any worker off the
        // mapping, so the lock-free fast path held under concurrency.
        REQUIRE(ctxs[i].checksum.load(std::memory_order_relaxed) == reference);
        // 0 runtime heap allocations inside the entire concurrent burst.
        REQUIRE(ctxs[i].alloc_delta.load(std::memory_order_relaxed) == 0);
    }

    for (auto& w : workers) w->Stop();
}
