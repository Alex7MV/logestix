#pragma once

// ── Codec profiles: what `lsxforge --codec-profile` selects ────────────────
//
// A profile is a named set of plane encodings plus the hardware class it
// targets. It is the unit invariant 4 multiplies: one source model, several
// artifacts, and lsxfabric picks whichever the current device executes natively.
//
// v1 ships no ballast in ANY profile (design §1.4): a plane the load path cannot
// reach is simply not written, so `used_codecs` has exactly one meaning — the
// codecs of the planes in this artifact.
//
//   nvfp4        NVFP4 weights, BF16/F32 rest        -> sm_120/121
//   bf16         BF16 weights, F32 norms/scales      -> every target
//   int8-w8a16   int8 weights + f32 per-channel      -> sm_86/87, sm_120
//
// `int8-w8a16` needs a quantizer and a W8A16 kernel that do not exist yet, so
// DeclaredPlanes returns nothing for it and the flag is rejected before any
// conversion work starts.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/codec_profile.h"
#include "lsxstorage/artifact_writer.h"

namespace lsxforge {

// The enum, its spellings and its parser are lsxcommon's
// (libs/lsxcommon/include/lsxcommon/codec_profile.h): a CONVERTER has to branch
// on them, and libs/ cannot include apps/. Forge keeps what it owns — the
// declared-plane table, the CLI, and ResolveCodecProfile below — and re-exports
// the vocabulary so the app's own code reads unchanged.
using lsxcommon::CodecProfile;
using lsxcommon::CodecProfileName;
using lsxcommon::ParseCodecProfile;

// Which conversion pipeline a profile is being applied to. Pipelines differ in
// the subtrees they write (experts, shared, ple), not in their codec rules.
enum class PipelineKind {
    kDense,   // phi, qwen dense, granite dense, ...
    kMoe,     // qwen/ornith/glm/nemotron/... expert families
};

// The profile an EMITTER should build for a given parse result.
//
// kDefault exists only to say "the flag was absent". A conversion must always
// produce an artifact.fb (design §0.2), and the flagless conversion historically
// produced the nvfp4 layout, so kDefault resolves to kNvFp4 — there is one
// profile language and no code path that publishes a tree with no declaration.
CodecProfile ResolveCodecProfile(CodecProfile p);

// The planes this profile intends to write for `layers` layers. Empty when the
// profile is not implemented for that pipeline yet, which is how an
// unimplemented profile is refused before any work happens rather than after.
std::vector<lsx::storage::PlaneDecl> DeclaredPlanes(CodecProfile p,
                                                   PipelineKind kind,
                                                   int layers);

} // namespace lsxforge
