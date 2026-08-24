// TDD — fastload shared helpers (host-only: AlignUp + suffix lookup; no CUDA
// allocations, no model weights).
#include <catch2/catch_test_macros.hpp>

#include "fast_load_internal.h"
#include "lsxcommon/schema.h"

TEST_CASE("AlignUp rounds up to alignment", "[fastload]")
{
    using lsxcommon::fastload::AlignUp;
    STATIC_REQUIRE(lsxcommon::fastload::kArenaAlign == 256);
    CHECK(AlignUp(0, 256) == 0);
    CHECK(AlignUp(1, 256) == 256);
    CHECK(AlignUp(255, 256) == 256);
    CHECK(AlignUp(256, 256) == 256);
    CHECK(AlignUp(257, 256) == 512);
    CHECK(AlignUp(4096, 256) == 4096);
}

TEST_CASE("FindTensorSuffix matches by name suffix", "[fastload]")
{
    using lsxcommon::fastload::FindTensorSuffix;
    lsxcommon::TensorBatchViews batch{};
    const int64_t shape_a[2] = {3, 4};
    lsxcommon::TensorView a{};
    a.name = "model.layers.3.input_layernorm.weight";
    a.rank = 2;
    a.shape = shape_a;
    const int64_t shape_b[2] = {8, 4};
    lsxcommon::TensorView b{};
    b.name = "model.layers.3.mlp.gate.weight";
    b.rank = 2;
    b.shape = shape_b;
    batch.tensors = {a, b};

    const auto *hit = FindTensorSuffix(batch, "input_layernorm.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "model.layers.3.input_layernorm.weight");

    hit = FindTensorSuffix(batch, "mlp.gate.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "model.layers.3.mlp.gate.weight");

    CHECK(FindTensorSuffix(batch, "missing.weight") == nullptr);
    // A "suffix" longer than the name must not match.
    CHECK(FindTensorSuffix(batch, "x.model.layers.3.mlp.gate.weight") ==
          nullptr);
}

TEST_CASE("FindNvfp4Suffix matches by record name suffix", "[fastload]")
{
    using lsxcommon::fastload::FindNvfp4Suffix;
    lsxcommon::DenseNvfp4BatchViews batch{};
    lsxcommon::DenseNvfp4View r{};
    r.name = "blk.0.attn_q.weight";
    r.rank = 2;
    batch.records = {r};

    const auto *hit = FindNvfp4Suffix(batch, "attn_q.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "blk.0.attn_q.weight");
    CHECK(FindNvfp4Suffix(batch, "attn_k.weight") == nullptr);
}

// ── Arena upload-job engine (GPU: uses device allocs + default stream) ──────

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace {

// Minimal weight receiver satisfying the UploadNvfp4Weight contract
// (rows/cols/global_scale/packed_bytes/scale_bytes/d_packed/d_scales/Free).
struct FakeNvfp4Weight {
    int rows = 0;
    int cols = 0;
    float global_scale = 0.f;
    int64_t packed_bytes = 0;
    int64_t scale_bytes = 0;
    const uint8_t *d_packed = nullptr;
    const uint8_t *d_scales = nullptr;
    void Free()
    {
        if (d_packed) cudaFree(const_cast<uint8_t *>(d_packed));
        if (d_scales) cudaFree(const_cast<uint8_t *>(d_scales));
        d_packed = nullptr;
        d_scales = nullptr;
    }
};

lsxcommon::TensorView MakeBf16View(std::vector<uint8_t> &storage,
                                   const int64_t shape[2], const char *name)
{
    lsxcommon::TensorView tv{};
    tv.name = name;
    tv.rank = 2;
    tv.shape = shape;
    tv.bytes = storage.size();
    tv.data = storage.data();
    return tv;
}

} // namespace

TEST_CASE("RunUploadJobs allocates one arena and uploads both jobs",
          "[fastload][gpu]")
{
    using namespace lsxcommon::fastload;
    PinnedStagingRing ring;
    REQUIRE(ring.ok());

    std::vector<uint8_t> bf16_a(12, 0xAB);  // 2x3 BF16 elements
    std::vector<uint8_t> bf16_b(8, 0xCD);   // 2x2 BF16 elements
    const int64_t shape_a[2] = {2, 3};
    const int64_t shape_b[2] = {2, 2};
    lsxcommon::TensorView a = MakeBf16View(bf16_a, shape_a, "a");
    lsxcommon::TensorView b = MakeBf16View(bf16_b, shape_b, "b");

    void *dst_a = nullptr;
    void *dst_b = nullptr;
    void *arena = nullptr;
    size_t arena_bytes = 0;
    std::uint64_t acc = 0;
    std::string err;

    cudaError_t e =
        RunUploadJobs({{&a, &dst_a, CvKind::kBf16ToHalf},
                       {&b, &dst_b, CvKind::kBf16ToHalf}},
                      &arena, &arena_bytes, /*stream=*/0, ring, acc, &err);
    REQUIRE(e == cudaSuccess);
    CHECK(err.empty());
    // 12 bytes then align-up to 256 plus 8 payload bytes.
    CHECK(arena_bytes == 264);
    REQUIRE(arena != nullptr);
    CHECK(dst_a == arena);  // first view starts at offset 0
    auto off_b = static_cast<const uint8_t *>(dst_b) -
                 static_cast<const uint8_t *>(arena);
    CHECK(off_b % kArenaAlign == 0);  // second view is 256-aligned
    CHECK(acc == 20);

    cudaFree(arena);
}

TEST_CASE("RunUploadJobs rejects malformed tensors", "[fastload][gpu]")
{
    using namespace lsxcommon::fastload;
    PinnedStagingRing ring;
    REQUIRE(ring.ok());

    const int64_t bad_shape[1] = {0};
    lsxcommon::TensorView bad{};
    bad.name = "broken";
    bad.rank = 1;
    bad.shape = bad_shape;
    bad.bytes = 4;
    static uint8_t dummy[4] = {};
    bad.data = dummy;

    void *dst = nullptr;
    void *arena = nullptr;
    size_t arena_bytes = 99;
    std::uint64_t acc = 0;
    std::string err = "sentinel";
    cudaError_t e = RunUploadJobs({{&bad, &dst, CvKind::kBf16ToHalf}}, &arena,
                                  &arena_bytes, 0, ring, acc, &err);
    CHECK(e == cudaErrorInvalidValue);
    CHECK(err.find("malformed tensor 'broken'") != std::string::npos);
    CHECK(arena == nullptr);
}

TEST_CASE("RunUploadJobs captures optional 2-D shapes", "[fastload][gpu]")
{
    using namespace lsxcommon::fastload;
    PinnedStagingRing ring;
    REQUIRE(ring.ok());

    std::vector<uint8_t> bf16(16, 0);
    const int64_t shape[2] = {4, 2};
    lsxcommon::TensorView tv = MakeBf16View(bf16, shape, "w");

    void *dst = nullptr;
    int64_t rows = -1;
    int64_t cols = -1;
    void *arena = nullptr;
    size_t arena_bytes = 0;
    std::uint64_t acc = 0;
    std::string err;
    cudaError_t e = RunUploadJobs(
        {{&tv, &dst, CvKind::kBf16ToHalf, &rows, &cols}}, &arena,
        &arena_bytes, 0, ring, acc, &err);
    REQUIRE(e == cudaSuccess);
    CHECK(rows == 4);
    CHECK(cols == 2);
    cudaFree(arena);
}

TEST_CASE("RunUploadJobs kBf16Raw keeps bits untouched", "[fastload][gpu]")
{
    using namespace lsxcommon::fastload;
    PinnedStagingRing ring;
    REQUIRE(ring.ok());

    // Raw BF16 payload must be uploaded byte-exact (no FP16 conversion).
    std::vector<uint8_t> bf16(8, 0x9C);
    const int64_t shape[1] = {4};
    lsxcommon::TensorView tv{};
    tv.name = "bf16_raw";
    tv.rank = 1;
    tv.shape = shape;
    tv.bytes = bf16.size();
    tv.data = bf16.data();

    void *dst = nullptr;
    void *arena = nullptr;
    size_t arena_bytes = 0;
    std::uint64_t acc = 0;
    std::string err;
    cudaError_t e = RunUploadJobs({{&tv, &dst, CvKind::kBf16Raw}}, &arena,
                                  &arena_bytes, 0, ring, acc, &err);
    REQUIRE(e == cudaSuccess);
    CHECK(arena_bytes == 8);
    CHECK(acc == 8);

    std::vector<uint8_t> back(8);
    cudaMemcpy(back.data(), dst, 8, cudaMemcpyDeviceToHost);
    CHECK(back == bf16);
    cudaFree(arena);
}

TEST_CASE("UploadNvfp4Weight uploads packed+scales and fills metadata",
          "[fastload][gpu]")
{
    using namespace lsxcommon::fastload;
    PinnedStagingRing ring;
    REQUIRE(ring.ok());

    std::vector<uint8_t> packed(64, 0x5A);
    std::vector<uint8_t> scales(8, 0x7E);
    lsxcommon::DenseNvfp4View rec{};
    rec.name = "expert.w1";
    rec.rank = 2;
    const int64_t shape[2] = {32, 32};
    rec.shape = shape;
    rec.packed = packed.data();
    rec.packed_bytes = static_cast<int64_t>(packed.size());
    rec.scales = scales.data();
    rec.scale_bytes = static_cast<int64_t>(scales.size());
    rec.global_scale = 2.5f;

    FakeNvfp4Weight w;
    std::uint64_t acc = 0;
    std::string err;
    cudaError_t e =
        UploadNvfp4Weight(&rec, w, /*stream=*/0, ring, acc, &err);
    REQUIRE(e == cudaSuccess);
    CHECK(w.rows == 32);
    CHECK(w.cols == 32);
    CHECK(w.global_scale == 2.5f);
    CHECK(w.packed_bytes == 64);
    CHECK(w.scale_bytes == 8);
    REQUIRE(w.d_packed != nullptr);
    REQUIRE(w.d_scales != nullptr);
    CHECK(acc == 72);

    // Device contents match the host buffers.
    std::vector<uint8_t> back(64);
    cudaMemcpy(back.data(), w.d_packed, 64, cudaMemcpyDeviceToHost);
    CHECK(back == packed);

    w.Free();
    CHECK(w.d_packed == nullptr);

    // Null record is a no-op success.
    FakeNvfp4Weight w2;
    CHECK(UploadNvfp4Weight(nullptr, w2, 0, ring, acc, &err) == cudaSuccess);
}

