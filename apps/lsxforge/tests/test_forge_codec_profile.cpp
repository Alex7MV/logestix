// --codec-profile parsing and the declared-plane table (host-only).
//
// The declared planes are one half of the contract the emitter verifies: a
// profile that declares a plane the converter did not write fails the
// conversion, and a profile that declares nothing is refused before any work
// starts.
#include <catch2/catch_test_macros.hpp>

#include "codec_profiles.h"
#include "lsxcommon/hardware_target.h"

#include <algorithm>
#include <string>

using namespace lsxforge;
using lsx::storage::PlaneDecl;
using namespace lsxcommon::hw;

namespace {

bool AnyCodec(const std::vector<PlaneDecl> &v, uint64_t bit)
{
    return std::any_of(v.begin(), v.end(),
                       [bit](const PlaneDecl &p) { return p.codec_bit == bit; });
}

bool HasExpertsPlane(const std::vector<PlaneDecl> &v)
{
    return std::any_of(v.begin(), v.end(), [](const PlaneDecl &p) {
        return p.path.rfind("experts", 0) == 0;
    });
}

} // namespace

TEST_CASE("the default profile is the nvfp4 profile, not a no-op profile", "[forge_profile]") {
    // `kDefault` is what a flagless `lsxforge --phi --input <src>` carries. It
    // used to short-circuit the emitter, so that run exited 0 and published a
    // tree with no artifact.fb — a tree lsxforge --verify rejects and BeginLoad
    // refuses forever. There is one profile language now, and it is nvfp4.
    CHECK(ResolveCodecProfile(CodecProfile::kDefault) == CodecProfile::kNvFp4);
    CHECK(std::string(CodecProfileName(ResolveCodecProfile(CodecProfile::kDefault))) ==
          "nvfp4");
}

TEST_CASE("the default profile declares the same planes as nvfp4", "[forge_profile]") {
    const auto mask_of = [](CodecProfile p, PipelineKind k) {
        uint64_t m = 0;
        for (const auto &plane : DeclaredPlanes(p, k, 4)) m |= plane.codec_bit;
        return m;
    };
    CHECK(mask_of(CodecProfile::kDefault, PipelineKind::kDense) ==
          mask_of(CodecProfile::kNvFp4, PipelineKind::kDense));
    CHECK(mask_of(CodecProfile::kDefault, PipelineKind::kMoe) ==
          mask_of(CodecProfile::kNvFp4, PipelineKind::kMoe));
}

TEST_CASE("unknown codec profile is rejected", "[forge_profile]") {
    CodecProfile p{};
    CHECK_FALSE(ParseCodecProfile("fp8", &p));
    CHECK_FALSE(ParseCodecProfile("", &p));      // empty means default, not a name
    CHECK_FALSE(ParseCodecProfile("BF16", &p));  // names are lowercase
    CHECK_FALSE(ParseCodecProfile("bf16", nullptr));
}

TEST_CASE("the canonical profile names parse", "[forge_profile]") {
    CodecProfile p{};
    REQUIRE(ParseCodecProfile("bf16", &p));
    CHECK(p == CodecProfile::kBf16);
    REQUIRE(ParseCodecProfile("nvfp4", &p));
    CHECK(p == CodecProfile::kNvFp4);
    REQUIRE(ParseCodecProfile("int8-w8a16", &p));
    CHECK(p == CodecProfile::kInt8W8A16);
}

TEST_CASE("the nvfp4 profile declares nvfp4 planes, the bf16 one does not",
          "[forge_profile]") {
    const auto nv = DeclaredPlanes(CodecProfile::kNvFp4, PipelineKind::kDense, 40);
    const auto bf = DeclaredPlanes(CodecProfile::kBf16, PipelineKind::kDense, 40);

    CHECK(AnyCodec(nv, static_cast<uint64_t>(PlaneCodec::kNvFp4)));
    CHECK(AnyCodec(bf, static_cast<uint64_t>(PlaneCodec::kBF16)));
    CHECK_FALSE(AnyCodec(bf, static_cast<uint64_t>(PlaneCodec::kNvFp4)));
}

TEST_CASE("int8-w8a16 is not offered while it has no quantizer",
          "[forge_profile]") {
    CHECK(DeclaredPlanes(CodecProfile::kInt8W8A16, PipelineKind::kDense, 40).empty());
    CHECK(DeclaredPlanes(CodecProfile::kInt8W8A16, PipelineKind::kMoe, 40).empty());
}

TEST_CASE("no profile declares a ballast plane, for any pipeline",
          "[forge_profile]") {
    // v1 ships no ballast: a plane the load path cannot reach is not written by
    // ANY profile, so `used_codecs` needs no stored-vs-executed distinction.
    for (auto p : {CodecProfile::kDefault, CodecProfile::kNvFp4,
                   CodecProfile::kBf16, CodecProfile::kInt8W8A16}) {
        for (auto k : {PipelineKind::kDense, PipelineKind::kMoe}) {
            for (const auto &plane : DeclaredPlanes(p, k, 40)) {
                INFO(CodecProfileName(p) << " " << plane.path);
                CHECK(plane.path.rfind("mtp", 0) != 0);
            }
        }
    }
}

TEST_CASE("the profile table's codec bits match the enum", "[forge_profile]") {
    const auto nv = DeclaredPlanes(CodecProfile::kNvFp4, PipelineKind::kMoe, 40);
    CHECK(AnyCodec(nv, Mask(PlaneCodec::kNvFp4)));
    CHECK(AnyCodec(nv, Mask(PlaneCodec::kBF16)));
    CHECK(AnyCodec(nv, Mask(PlaneCodec::kF32)));
    const auto bf = DeclaredPlanes(CodecProfile::kBf16, PipelineKind::kMoe, 40);
    CHECK(AnyCodec(bf, Mask(PlaneCodec::kBF16)));
    CHECK(AnyCodec(bf, Mask(PlaneCodec::kF32)));
}

TEST_CASE("a moe profile declares the routed-expert plane and a dense one does not",
          "[forge_profile]") {
    const auto moe = DeclaredPlanes(CodecProfile::kBf16, PipelineKind::kMoe, 40);
    const auto dense = DeclaredPlanes(CodecProfile::kBf16, PipelineKind::kDense, 40);
    CHECK(AnyCodec(moe, Mask(PlaneCodec::kBF16)));
    CHECK(HasExpertsPlane(moe));
    CHECK_FALSE(HasExpertsPlane(dense));
}

TEST_CASE("zero layers declares nothing", "[forge_profile]") {
    CHECK(DeclaredPlanes(CodecProfile::kBf16, PipelineKind::kDense, 0).empty());
    CHECK(DeclaredPlanes(CodecProfile::kBf16, PipelineKind::kDense, -1).empty());
}
