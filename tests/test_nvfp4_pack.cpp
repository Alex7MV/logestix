// PackExpertNVFP4 characterization tests: already-quantized passthrough
// contract, empty-input rejection and the GPU NVFP4 packing output contract.
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"

namespace {

bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

lsxcommon::ExpertData MakeQuantizedExpert(const std::string &dtype)
{
    lsxcommon::ExpertData raw;
    raw.w1_dtype = raw.w2_dtype = raw.w3_dtype = dtype;
    raw.w1_shape = {2, 16};
    raw.w2_shape = {2, 16};
    raw.w3_shape = {2, 16};
    raw.w1_raw.assign(32, 0x2Au);
    raw.w2_raw.assign(32, 0x5Au);
    raw.w3_raw.assign(32, 0xA5u);
    // 2 blocks of 16 elements -> 2 E8M0 scale bytes per tensor.
    raw.w1_scale_raw.assign(2, 0x7Eu);
    raw.w2_scale_raw.assign(2, 0x7Fu);
    raw.w3_scale_raw.assign(2, 0x80u);
    raw.w1_scale_shape = {2, 1};
    raw.w2_scale_shape = {2, 1};
    raw.w3_scale_shape = {2, 1};
    return raw;
}

}  // namespace

TEST_CASE("PackExpertNVFP4 passes through already-quantized F8_E4M3 weights",
          "[nvfp4][converter]")
{
    auto raw = MakeQuantizedExpert("F8_E4M3");
    auto res = lsxcommon::PackExpertNVFP4(raw);
    REQUIRE(res.ok());
    const auto &pe = *res;

    CHECK(pe.w1_packed == raw.w1_raw);
    CHECK(pe.w2_packed == raw.w2_raw);
    CHECK(pe.w3_packed == raw.w3_raw);
    CHECK(pe.w1_scales == raw.w1_scale_raw);
    CHECK(pe.w2_scales == raw.w2_scale_raw);
    CHECK(pe.w3_scales == raw.w3_scale_raw);
    CHECK(pe.w1_global_scale == 1.0f);
    CHECK(pe.w2_global_scale == 1.0f);
    CHECK(pe.w3_global_scale == 1.0f);
    CHECK(pe.w1_shape == raw.w1_shape);
    CHECK(pe.w2_shape == raw.w2_shape);
    CHECK(pe.w3_shape == raw.w3_shape);
}

TEST_CASE("PackExpertNVFP4 passes through U8 weights paired with E8M0 scales",
          "[nvfp4][converter]")
{
    auto raw = MakeQuantizedExpert("U8");
    auto res = lsxcommon::PackExpertNVFP4(raw);
    REQUIRE(res.ok());
    const auto &pe = *res;

    CHECK(pe.w1_packed == raw.w1_raw);
    CHECK(pe.w1_scales == raw.w1_scale_raw);
    CHECK(pe.w1_global_scale == 1.0f);
}

TEST_CASE("PackExpertNVFP4 rejects empty expert data", "[nvfp4][converter]")
{
    lsxcommon::ExpertData raw;
    auto res = lsxcommon::PackExpertNVFP4(raw);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().IsExecutionError());
    REQUIRE(res.status().ToString().find("w1") != std::string::npos);
}

TEST_CASE("PackExpertNVFP4 GPU-quantizes BF16 into nibbles plus per-block scales",
          "[nvfp4][converter]")
{
    if (!GpuAvailable())
        std::exit(4);

    lsxcommon::ExpertData raw;
    raw.w1_dtype = raw.w2_dtype = raw.w3_dtype = "BF16";
    raw.w1_shape = {2, 16};
    raw.w2_shape = {2, 16};
    raw.w3_shape = {2, 16};

    // BF16 1.0 little-endian payload for every element (32 elems = 64 bytes).
    constexpr uint8_t kOne[2] = {0x00, 0x3F};
    for (auto *dst : {&raw.w1_raw, &raw.w2_raw, &raw.w3_raw}) {
        dst->reserve(64);
        for (int i = 0; i < 32; ++i) {
            dst->push_back(kOne[0]);
            dst->push_back(kOne[1]);
        }
    }

    auto res = lsxcommon::PackExpertNVFP4(raw);
    REQUIRE(res.ok());
    const auto &pe = *res;

    // 32 elements -> 16 nibble-packed bytes; 2 NVFP4 blocks -> 2 E4M3 scales.
    CHECK(pe.w1_packed.size() == 16);
    CHECK(pe.w2_packed.size() == 16);
    CHECK(pe.w3_packed.size() == 16);
    CHECK(pe.w1_scales.size() == 2);
    CHECK(pe.w2_scales.size() == 2);
    CHECK(pe.w3_scales.size() == 2);
    CHECK(std::isfinite(pe.w1_global_scale));
    CHECK(pe.w1_global_scale > 0.0f);
    CHECK(std::isfinite(pe.w2_global_scale));
    CHECK(pe.w2_global_scale > 0.0f);
    CHECK(pe.w3_shape == raw.w3_shape);
}
