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
#include <string>
#include <vector>

// ────────────────────────────────────────────────────────────────────────────
// Counting heap allocator: overrides the global (replacement) operator
// new/delete and tallies call sites into a process-wide atomic. The translator
// fast path must not touch it — the tests diff the counter across a translation
// burst to prove 0 runtime allocations.
// ────────────────────────────────────────────────────────────────────────────
namespace {
std::atomic<long long> g_alloc_count{0};
std::filesystem::path MakeTempPassport(const lsxnetwork::BuildPassportOptions& opts) {
    static std::atomic<unsigned> seq{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto p = std::filesystem::temp_directory_path() /
                   ("lsx_translator_" + std::to_string(seq.fetch_add(1)) + "_" +
                    std::to_string(stamp) + ".bin");
    REQUIRE(lsxnetwork::WritePassport(p.string().c_str(), opts) == 0);
    return p;
}

lsxnetwork::BuildPassportOptions Fixture() {
    lsxnetwork::BuildPassportOptions o;
    // glm tokens (id == vector index):
    o.glm_tokens = {"hello", "world", "foo", "hello!", "zzz", "<|im_end|>"};
    o.deepseek_tokens = {"hello", "world", "foo", "bar", "<|im_end|>"};
    o.glm_unk_id = 4;
    o.deepseek_unk_id = 3;
    o.lmcs_min_overlap = 2;
    return o;
}

const std::uint32_t kG3 = 3;  // "hello!" -> exact-stem overlap "hello" == ds0
}  // namespace

void* operator new(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void*, std::size_t) noexcept {}
void operator delete[](void*, std::size_t) noexcept {}

TEST_CASE("passport header occupies exactly one aligned cache line",
          "[lsx_token_translator]") {
    STATIC_REQUIRE(sizeof(lsxnetwork::token_passport::PassportHeader) == 64);
    STATIC_REQUIRE(alignof(lsxnetwork::token_passport::PassportHeader) == 64);
}

TEST_CASE("BuiltPassport produces 64-byte blocks with correct header",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto bytes = lsxnetwork::BuildPassport(opts);
    // header(64) + glm block(64) + ds block(64) == 192
    REQUIRE(bytes.size() == 64 + 64 + 64);

    const auto* hdr =
        reinterpret_cast<const lsxnetwork::token_passport::PassportHeader*>(bytes.data());
    REQUIRE(hdr->magic == lsxnetwork::token_passport::kMagic);
    REQUIRE(hdr->version == lsxnetwork::token_passport::kVersion);
    REQUIRE(hdr->glm_vocab_size == 6);
    REQUIRE(hdr->deepseek_vocab_size == 5);
    REQUIRE(hdr->glm_unk_id == 4);
    REQUIRE(hdr->deepseek_unk_id == 3);
}

TEST_CASE("Mapping via CreateFileMapping exposes aligned zero-alloc pointers",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);

    lsxnetwork::LsxTokenTranslator tr;
    REQUIRE(tr.Load(path.string().c_str()) == 0);
    REQUIRE(tr.Valid());

    REQUIRE(tr.GlmVocabSize() == 6);
    REQUIRE(tr.DeepSeekVocabSize() == 5);
    REQUIRE(tr.GlmUnkId() == 4);
    REQUIRE(tr.DeepSeekUnkId() == 3);

    const std::uintptr_t fwd = reinterpret_cast<std::uintptr_t>(tr.GlmToDeepSeek());
    const std::uintptr_t rev = reinterpret_cast<std::uintptr_t>(tr.DeepSeekToGlm());
    REQUIRE(fwd % 64 == 0);
    REQUIRE(rev % 64 == 0);
    REQUIRE((rev - fwd) % 64 == 0);
}

TEST_CASE("Forward GLM->DeepSeek translation is exact for all mapped tokens",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);
    lsxnetwork::LsxTokenTranslator tr;
    REQUIRE(tr.Load(path.string().c_str()) == 0);

    REQUIRE(tr.TranslateGlmToDeepSeek(0) == 0);  // "hello" -> "hello"
    REQUIRE(tr.TranslateGlmToDeepSeek(1) == 1);  // "world" -> "world"
    REQUIRE(tr.TranslateGlmToDeepSeek(2) == 2);  // "foo"   -> "foo"
    REQUIRE(tr.TranslateGlmToDeepSeek(kG3) == 0);      // "hello!" stem -> "hello"
    REQUIRE(tr.TranslateGlmToDeepSeek(4) == 3);        // "zzz" unmatched -> ds unk
    REQUIRE(tr.TranslateGlmToDeepSeek(5) == 4);        // "<|im_end|>" -> "<|im_end|>"
    REQUIRE(tr.TranslateGlmToDeepSeek(1000) == 3);     // out of range -> ds unk
}

TEST_CASE("Reverse DeepSeek->GLM translation matches the deterministic table",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);
    lsxnetwork::LsxTokenTranslator tr;
    REQUIRE(tr.Load(path.string().c_str()) == 0);

    REQUIRE(tr.TranslateDeepSeekToGlm(0) == 0);  // first glm id that hits ds0
    REQUIRE(tr.TranslateDeepSeekToGlm(1) == 1);
    REQUIRE(tr.TranslateDeepSeekToGlm(2) == 2);
    REQUIRE(tr.TranslateDeepSeekToGlm(3) == 4);  // "bar" never targeted -> glm unk
    REQUIRE(tr.TranslateDeepSeekToGlm(4) == 5);
    REQUIRE(tr.TranslateDeepSeekToGlm(999) == 4);  // out of range -> glm unk
}

TEST_CASE("Multi-token stream round-trips both directions with 100% stability",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);
    lsxnetwork::LsxTokenTranslator tr;
    REQUIRE(tr.Load(path.string().c_str()) == 0);

    const std::uint32_t in[] = {0, 1, 2, kG3, 5, 0, 2, 1};
    std::uint32_t fwd[8];
    std::uint32_t back[8];
    for (std::size_t i = 0; i < 8; ++i) fwd[i] = tr.TranslateGlmToDeepSeek(in[i]);
    for (std::size_t i = 0; i < 8; ++i) back[i] = tr.TranslateDeepSeekToGlm(fwd[i]);

    const std::uint32_t expect_fwd[] = {0, 1, 2, 0, 4, 0, 2, 1};
    const std::uint32_t expect_back[] = {0, 1, 2, 0, 5, 0, 2, 1};
    REQUIRE(std::equal(std::begin(expect_fwd), std::end(expect_fwd), std::begin(fwd)));
    REQUIRE(std::equal(std::begin(expect_back), std::end(expect_back), std::begin(back)));

    // Bijective exact tokens are stable through a full round-trip.
    for (std::size_t i = 0; i < 8; ++i)
        if (in[i] != kG3) REQUIRE(back[i] == in[i]);
}

TEST_CASE("Translation burst performs zero runtime heap allocations",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);
    lsxnetwork::LsxTokenTranslator tr;
    REQUIRE(tr.Load(path.string().c_str()) == 0);

    const long long before = g_alloc_count.load(std::memory_order_relaxed);
    volatile std::uint32_t sink_a = 0, sink_b = 0;
    for (std::uint32_t it = 0; it < 1000000; ++it) {
        sink_a ^= tr.TranslateGlmToDeepSeek(it & 15u);
        sink_b ^= tr.TranslateDeepSeekToGlm(it & 7u);
    }
    const long long after = g_alloc_count.load(std::memory_order_relaxed);
    (void)sink_a;
    (void)sink_b;
    REQUIRE(after - before == 0);
}

TEST_CASE("Garbage and truncated passports are rejected without Valid state",
          "[lsx_token_translator]") {
    const auto garbage = std::filesystem::temp_directory_path() / "lsx_garbage.bin";
    {
        std::ofstream out(garbage.string(), std::ios::binary | std::ios::trunc);
        const unsigned char junk[128] = {0xde, 0xad, 0xbe, 0xef};
        out.write(reinterpret_cast<const char*>(junk), sizeof(junk));
    }
    lsxnetwork::LsxTokenTranslator tr;
    REQUIRE(tr.Load(garbage.string().c_str()) == -1);
    REQUIRE_FALSE(tr.Valid());
    std::filesystem::remove(garbage);
}

TEST_CASE("Move transfers the mapping and invalidates the source",
          "[lsx_token_translator]") {
    const auto opts = Fixture();
    const auto path = MakeTempPassport(opts);
    lsxnetwork::LsxTokenTranslator src;
    REQUIRE(src.Load(path.string().c_str()) == 0);

    lsxnetwork::LsxTokenTranslator dst(std::move(src));
    REQUIRE_FALSE(src.Valid());
    REQUIRE(dst.Valid());
    REQUIRE(dst.TranslateGlmToDeepSeek(0) == 0);
    REQUIRE(dst.TranslateDeepSeekToGlm(0) == 0);
}