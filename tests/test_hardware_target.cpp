// HardwareTarget + SupportedCodecs capability table (host-only, no GPU).
//
// This table IS the "hardware native" contract: a codec appears in a target's
// mask only if that target executes it in silicon. NVFP4 and FP8 are absent
// from every Ampere row because GA10x has no E2M1/E4M3 tensor-core
// instruction — not because the scalar kernel could not decode them.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hardware_target.h"

using namespace lsxcommon::hw;

namespace {

constexpr CodecMask kAllSeven =
    Mask(PlaneCodec::kF32) | Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kF16) |
    Mask(PlaneCodec::kI8Sym) | Mask(PlaneCodec::kMxFp8) |
    Mask(PlaneCodec::kNvFp4) | Mask(PlaneCodec::kMxFp4);

constexpr CodecMask kAmpere =
    Mask(PlaneCodec::kF32) | Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kF16) |
    Mask(PlaneCodec::kI8Sym);

} // namespace

TEST_CASE("Blackwell executes all seven plane codecs", "[hardware_target]") {
    for (int minor : {0, 1}) {
        const auto t = TargetFromComputeCapability(12, minor);
        INFO(t.Id());
        REQUIRE(t.Id() == (minor == 0 ? "nvidia:sm_120" : "nvidia:sm_121"));
        REQUIRE(SupportedCodecs(t) == kAllSeven);
    }
}

TEST_CASE("Ampere executes F32/BF16/F16/INT8 and never FP8 or FP4",
          "[hardware_target]") {
    for (int minor : {6, 7}) {
        const auto t = TargetFromComputeCapability(8, minor);
        INFO(t.Id());
        REQUIRE(t.Id() == (minor == 6 ? "nvidia:sm_86" : "nvidia:sm_87"));
        REQUIRE(SupportedCodecs(t) == kAmpere);
        REQUIRE((SupportedCodecs(t) & Mask(PlaneCodec::kMxFp8)) == 0);
        REQUIRE((SupportedCodecs(t) & Mask(PlaneCodec::kNvFp4)) == 0);
        REQUIRE((SupportedCodecs(t) & Mask(PlaneCodec::kMxFp4)) == 0);
    }
}

TEST_CASE("sm_86 and sm_87 are distinct targets", "[hardware_target]") {
    CHECK(TargetFromComputeCapability(8, 6).Id() !=
          TargetFromComputeCapability(8, 7).Id());
}

TEST_CASE("Ada is not a capability row", "[hardware_target]") {
    CHECK(SupportedCodecs(TargetFromComputeCapability(8, 9)) == 0);
    CHECK(TargetFromComputeCapability(8, 9).vendor == Vendor::kUnknown);
}

TEST_CASE("unknown compute capability yields kUnknown, not a default",
          "[hardware_target]") {
    CHECK(TargetFromComputeCapability(9, 0).vendor == Vendor::kUnknown);
    CHECK(TargetFromComputeCapability(8, 0).vendor == Vendor::kUnknown);
    CHECK(SupportedCodecs(TargetFromComputeCapability(9, 0)) == 0);
}

TEST_CASE("cpu target is feature-derived, never assumed from the family name",
          "[hardware_target]") {
    const auto bare = TargetFromCpuFeatures(true, false, false, false, false, false);
    REQUIRE(bare.Id() == "cpu:avx512");
    REQUIRE(SupportedCodecs(bare) == Mask(PlaneCodec::kF32));

    const auto full = TargetFromCpuFeatures(true, true, true, true, false, false);
    REQUIRE(full.Id() == "cpu:avx512+bf16+fp16+vnni");
    REQUIRE(SupportedCodecs(full) == kAmpere);
}

TEST_CASE("Jetson host side is neon, not avx512", "[hardware_target]") {
    const auto t = TargetFromCpuFeatures(false, false, false, false, true, true);
    REQUIRE(t.Id() == "cpu:neon+dotprod");
    REQUIRE((SupportedCodecs(t) & Mask(PlaneCodec::kI8Sym)) != 0);
    REQUIRE((SupportedCodecs(t) & Mask(PlaneCodec::kNvFp4)) == 0);
}

TEST_CASE("amd is reserved and supports nothing yet", "[hardware_target]") {
    bool ok = false;
    const auto t = TargetFromId("amd:gfx942", &ok);
    REQUIRE(ok);
    REQUIRE(t.vendor == Vendor::kAmd);
    CHECK(SupportedCodecs(t) == 0);
}

TEST_CASE("TargetFromId round-trips every id the tree can produce",
          "[hardware_target]") {
    for (const char *id : {"nvidia:sm_120", "nvidia:sm_121", "nvidia:sm_86",
                           "nvidia:sm_87", "cpu:avx512",
                           "cpu:avx512+bf16+fp16+vnni", "cpu:neon+dotprod",
                           "amd:gfx942"}) {
        bool ok = false;
        const auto t = TargetFromId(id, &ok);
        INFO(id);
        REQUIRE(ok);
        REQUIRE(t.Id() == id);
    }
}

TEST_CASE("TargetFromId rejects malformed ids", "[hardware_target]") {
    bool ok = true;
    (void)TargetFromId("sm_120", &ok);            // no vendor prefix
    CHECK_FALSE(ok);
    (void)TargetFromId("nvidia:", &ok);           // no arch
    CHECK_FALSE(ok);
    (void)TargetFromId("", &ok);
    CHECK_FALSE(ok);
    (void)TargetFromId("nvidia:sm_120", nullptr); // null out is tolerated
}

TEST_CASE("an unknown vendor is never treated as a supported target",
          "[hardware_target]") {
    bool ok = true;
    const auto t = TargetFromId("tpu:v5", &ok);
    INFO(t.Id());
    CHECK_FALSE(ok);
    CHECK(SupportedCodecs(t) == 0);
}

// ── I3 as a pure function ──────────────────────────────────────────────────

TEST_CASE("gate allows a subset and reports the exact missing bits",
          "[hardware_target]") {
    const auto sm86 = TargetFromComputeCapability(8, 6);
    CHECK(CheckCodecs(Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kF32), sm86).allowed);

    const auto d = CheckCodecs(Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kNvFp4), sm86);
    REQUIRE_FALSE(d.allowed);
    REQUIRE(d.missing == Mask(PlaneCodec::kNvFp4));
    // The message has to name the codec AND the target: an operator reading a
    // log needs both to know whether to reconvert or to change machine.
    CHECK(d.message.find("nvfp4") != std::string::npos);
    CHECK(d.message.find("nvidia:sm_86") != std::string::npos);
}

TEST_CASE("gate on an unimplemented target reports the target, not a codec",
          "[hardware_target]") {
    bool ok = false;
    const auto amd = TargetFromId("amd:gfx942", &ok);
    REQUIRE(ok);
    const auto d = CheckCodecs(Mask(PlaneCodec::kBF16), amd);
    REQUIRE_FALSE(d.allowed);
    CHECK(d.message.find("not implemented") != std::string::npos);
}

TEST_CASE("gate refuses when the target is unknown", "[hardware_target]") {
    REQUIRE_FALSE(CheckCodecs(Mask(PlaneCodec::kBF16), HardwareTarget{}).allowed);
    CHECK_FALSE(CheckCodecs(Mask(PlaneCodec::kBF16),
                            TargetFromComputeCapability(8, 9)).allowed);
}

TEST_CASE("gate on an empty mask is allowed for a real target",
          "[hardware_target]") {
    // A model with no quantized planes (pure bf16/f32) is always portable.
    CHECK(CheckCodecs(0, TargetFromComputeCapability(8, 6)).allowed);
}

TEST_CASE("ActiveTarget override is scoped to the test process",
          "[hardware_target]") {
    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    REQUIRE(ActiveTarget().Id() == "nvidia:sm_86");
    SetActiveTargetForTesting(TargetFromComputeCapability(12, 0));
    REQUIRE(ActiveTarget().Id() == "nvidia:sm_120");
    ResetActiveTargetForTesting();
}

TEST_CASE("the probed ActiveTarget on this host is a supported target",
          "[hardware_target]") {
    ResetActiveTargetForTesting();
    const auto t = ActiveTarget();
    INFO(t.Id());
    // Either a supported GPU or a cpu: row; never unknown on a machine that can
    // run this suite.
    CHECK(t.vendor != Vendor::kUnknown);
    CHECK(SupportedCodecs(t) != 0);
}
