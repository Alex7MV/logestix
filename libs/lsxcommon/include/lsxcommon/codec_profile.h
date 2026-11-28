#pragma once

// ── Codec profiles: the storage encodings one conversion may produce ───────
//
// A profile is a named set of plane encodings. It is the unit invariant I4
// multiplies: one source model, several artifacts, and the loader/gate picks
// whichever the current device executes natively.
//
// WHY THE ENUM LIVES *HERE* AND NOT IN apps/lsxforge
//
// The enum started in `apps/lsxforge/src/codec_profiles.h`, which was fine while
// nothing below the app needed it. Then a CONVERTER had to branch on it — "write
// BF16 planes instead of quantizing" — and `libs/` cannot include `apps/`. The
// wrong repair is obvious and was nearly taken: give each family its own copy of
// the profile table. That is the per-family codec knowledge the design calls out
// as the `validator.cpp` mistake (§0.5). So the enum moves down, and forge keeps
// the POLICY it owns: parsing the flag, the declared-plane table, the CLI.
//
// This header is reachable from `.cu`/`.cuh` TUs (a converter signature is, and
// so is anything that includes a converter header from a device TU), so it
// carries `<cstdint>` and `<string_view>` only: no absl, no Arrow, no logging.

#include <cstdint>
#include <string_view>

namespace lsxcommon {

enum class CodecProfile {
    // The family's historical layout (nvfp4). `kDefault` is what a flagless run
    // carries; it is NOT a distinct encoding, and every consumer that branches
    // must treat it as nvfp4 — see ProfileUsesNvfp4Weights below.
    kDefault,
    kBf16,
    kNvFp4,
    kInt8W8A16,
};

// Canonical spelling, shared by `--codec-profile`, artifact.fb's `profile` field
// and every log message. kDefault renders as its actual encoding, because an
// artifact must not be labelled with an absence.
inline constexpr const char *CodecProfileName(CodecProfile p)
{
    switch (p) {
    case CodecProfile::kBf16: return "bf16";
    case CodecProfile::kNvFp4: return "nvfp4";
    case CodecProfile::kInt8W8A16: return "int8-w8a16";
    case CodecProfile::kDefault: break;
    }
    return "nvfp4";
}

// Parses the canonical spelling. `""` is rejected: an empty flag means "the
// operator did not pass one", which is the caller's default to apply, not a name.
// "fp8" is rejected too — it is ambiguous between E4M3 and E5M2, and the only
// FP8 variant this tree stores must be spelled mxfp8.
inline constexpr bool ParseCodecProfile(std::string_view name, CodecProfile *out)
{
    if (out == nullptr || name.empty()) return false;
    if (name == "nvfp4") { *out = CodecProfile::kNvFp4; return true; }
    if (name == "bf16") { *out = CodecProfile::kBf16; return true; }
    if (name == "int8-w8a16") { *out = CodecProfile::kInt8W8A16; return true; }
    return false;
}

// ── The one branch a CONVERTER takes ──────────────────────────────────────
//
// True when this profile packs weights as NVFP4 nibbles (plus the per-tensor
// E4M3 scales and, where a family writes one, the residual plane). False means
// the weights are written as the high-precision planes they arrived as.
//
// This predicate exists so the conversion branch is one call, not a `switch` per
// family: a family that adds a `case` of its own is a family that will one day
// disagree with the others about what "bf16" means. It also fixes `kDefault`'s
// meaning in exactly one place — a flagless run is nvfp4, because every artifact
// measured before profiles existed was built that way.
inline constexpr bool ProfileUsesNvfp4Weights(CodecProfile p)
{
    return p != CodecProfile::kBf16 && p != CodecProfile::kInt8W8A16;
}

} // namespace lsxcommon
