// --codec-profile: parsing and the per-pipeline plane table.
//
// The declared-plane table is deliberately small and boring: it states what
// each profile WILL write, and the emitter then verifies that against what
// actually landed on disk. Keeping the declaration in one file makes the
// declared side of that comparison auditable at a glance.
#include "codec_profiles.h"

#include <array>

namespace lsxforge {

namespace {

// Bit values mirrored from lsxcommon::hw::PlaneCodec. The shell cannot take a
// dependency on the codec header without dragging CUDA in, and the values are an
// on-disk contract that tests/test_forge_codec_profile.cpp pins against the enum
// (every DeclaredPlanes bit it compares comes from hw::Mask(), so a drift here
// fails that test rather than silently mislabelling an artifact).
constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kNvFp4 = 1ull << 5;

} // namespace

CodecProfile ResolveCodecProfile(CodecProfile p)
{
    // One profile language. `kDefault` is "the flag was not passed", and the
    // layout a flagless conversion produces has always been nvfp4; resolving it
    // here is what lets the emitter run on EVERY conversion instead of being
    // skipped for the one flag value most runbooks pass.
    return p == CodecProfile::kDefault ? CodecProfile::kNvFp4 : p;
}

std::vector<lsx::storage::PlaneDecl> DeclaredPlanes(CodecProfile p,
                                                   PipelineKind kind,
                                                   int layers)
{
    using lsx::storage::PlaneDecl;
    if (layers <= 0) return {};

    switch (p) {
    case CodecProfile::kDefault:
    case CodecProfile::kNvFp4: {
        // Quantized linears as NVFP4 nibbles, everything else BF16/F32.
        std::vector<PlaneDecl> planes;
        planes.push_back({"dense.mlp.nvfp4", kNvFp4, "pinned/nvfp4.planes", 0});
        planes.push_back({"dense.residual", kBF16, "pinned/bf16.planes", 0});
        planes.push_back({"norm", kF32, "pinned/f32.planes", 0});
        if (kind == PipelineKind::kMoe)
            planes.push_back({"expert.routed", kNvFp4, "experts", 0});
        return planes;
    }
    case CodecProfile::kBf16: {
        std::vector<PlaneDecl> planes;
        planes.push_back({"dense.mlp", kBF16, "pinned/bf16.planes", 0});
        planes.push_back({"norm", kF32, "pinned/f32.planes", 0});
        if (kind == PipelineKind::kMoe)
            planes.push_back({"expert.routed", kBF16, "experts", 0});
        return planes;
    }
    case CodecProfile::kInt8W8A16:
        // Not implemented: needs a per-channel quantizer and a W8A16 kernel
        // (Branch 3). Empty means the CLI refuses before converting.
        return {};
    }
    return {};
}

} // namespace lsxforge
