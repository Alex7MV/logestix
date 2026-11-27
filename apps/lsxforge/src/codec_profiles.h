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

#include "lsxstorage/artifact_writer.h"

namespace lsxforge {

enum class CodecProfile {
    kDefault,     // the family's historical layout (nvfp4); "" on the CLI
    kBf16,
    kNvFp4,
    kInt8W8A16,
};

// Which conversion pipeline a profile is being applied to. Pipelines differ in
// the subtrees they write (experts, shared, ple), not in their codec rules.
enum class PipelineKind {
    kDense,   // phi, qwen dense, granite dense, ...
    kMoe,     // qwen/ornith/glm/nemotron/... expert families
};

// Parses the flag value. Accepts exactly the canonical names; "" is the default
// profile and is NOT a name (a CLI user who types nothing gets kDefault from
// their own code, not from here).
bool ParseCodecProfile(std::string_view name, CodecProfile *out);

// Canonical name for logs and for artifact.fb's profile/profile_arg fields.
const char *CodecProfileName(CodecProfile p);

// The planes this profile intends to write for `layers` layers. Empty when the
// profile is not implemented for that pipeline yet, which is how an
// unimplemented profile is refused before any work happens rather than after.
std::vector<lsx::storage::PlaneDecl> DeclaredPlanes(CodecProfile p,
                                                   PipelineKind kind,
                                                   int layers);

} // namespace lsxforge
