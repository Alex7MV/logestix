#pragma once

// ── Hardware-native codec contract: storage encodings ──────────────────────
//
// `PlaneCodec` names HOW a plane's bytes are encoded on disk. It is orthogonal
// to which kernel consumes them: the same NVFP4 nibbles are decoded by a
// tensor-core MMA on Blackwell and (in principle) by a scalar kernel anywhere
// else. A bitmask over PlaneCodec therefore answers exactly one question —
// "which of these encodings does a given target execute IN HARDWARE?" — and
// that question is what the load gate in ModelBase::BeginLoad asks.
//
// This header is reachable from `.cu`/`.cuh` TUs (ModelBase includes it), so it
// carries POD types, <cstdint>, <string> and <string_view> only: no absl
// containers (NVCC fails in raw_hash_map), no Arrow, no FlatBuffers, no
// exceptions, no logging.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {
namespace hw {

// Bits 0..6 are codecs shipped today. Bits 7..31 are reserved for future plane
// encodings; bits 32..63 are vendor extensions and must never appear in
// another vendor's SupportedCodecs() row.
//
// The numeric values are an on-disk contract: `artifact.fb` stores the mask as
// an integer, so renumbering silently reinterprets every existing artifact.
// tests/test_hardware_target.cpp pins them.
enum class PlaneCodec : uint64_t {
    kNone  = 0,
    kF32   = 1ull << 0,   // IEEE binary32
    kBF16  = 1ull << 1,   // bfloat16
    kF16   = 1ull << 2,   // IEEE binary16
    kI8Sym = 1ull << 3,   // signed int8, symmetric per-channel, f32 scale companion
    kMxFp8 = 1ull << 4,   // E4M3, block-32, E8M0 scale
    kNvFp4 = 1ull << 5,   // E2M1, block-16, UE4M3 scale (+ per-tensor f32 global)
    kMxFp4 = 1ull << 6,   // E2M1, block-32, E8M0 scale
};

using CodecMask = uint64_t;

constexpr CodecMask Mask(PlaneCodec c) { return static_cast<CodecMask>(c); }

// ── Targets ────────────────────────────────────────────────────────────────

enum class Vendor : uint8_t { kNvidia, kCpu, kAmd, kUnknown };

// One accelerator (or CPU) identified by vendor + architecture. `arch` is the
// vendor's own spelling: "sm_120", "sm_86", "avx512+bf16+fp16+vnni", "gfx942".
// Never carries a codec: capability is asked of SupportedCodecs(), so a target
// cannot claim support it does not have.
struct HardwareTarget {
    Vendor vendor = Vendor::kUnknown;
    std::string arch;

    // Normalized identity, also the filename segment of an artifact profile:
    // "nvidia:sm_120", "cpu:avx512", "amd:gfx942". "unknown:" when unset.
    std::string Id() const;
};

bool operator==(const HardwareTarget &a, const HardwareTarget &b);
bool operator!=(const HardwareTarget &a, const HardwareTarget &b);

// THE capability table: which plane encodings the target executes in hardware.
//
// Policy, not capability archaeology: a row exists only for silicon this tree
// targets. Ada (sm_89) has FP8 tensor cores but is not a supported target, so
// it gets 0 — the table is not a hardware database.
//
//   nvidia:sm_120/121  all seven (mma ... kind::mxf4nvf4.block_scale for both
//                      block-scale FP4 flavours; kind::f8f6f4 for E4M3)
//   nvidia:sm_86/87    F32,BF16,F16,I8 — mma.m16n8k16 f16/bf16 and
//                      mma.m16n8k32.s8.s8.s32 (IMMA) are native on GA10x.
//                      NO FP8 and NO FP4: GA10x has no such tensor-core op.
//   cpu:avx512*        F32 always; BF16 only with avx512_bf16, F16 only with
//                      avx512_fp16, I8 only with avx512_vnni
//   cpu:neon*          F32,BF16,F16,I8 (aarch64 NEON + optional dotprod);
//                      AVX512 does not exist on Jetson, so cpu:avx512 can never
//                      be the only CPU row.
//   amd:*              0 — reserved, reported as "target not implemented".
CodecMask SupportedCodecs(const HardwareTarget &t);

// Builds a target from CUDA's major/minor. Returns Vendor::kUnknown for
// anything outside {12.0, 12.1, 8.6, 8.7} — including sm_80, which AGENTS.md
// forbids. Unknown targets support nothing, so a mis-detected device fails the
// load gate closed instead of open.
HardwareTarget TargetFromComputeCapability(int major, int minor);

// Builds a cpu: target from CPUID-derived feature flags. Feature order in the
// id is fixed (avx512, then +bf16, +fp16, +vnni; or neon, +dotprod) so the id
// is stable for profile selection.
HardwareTarget TargetFromCpuFeatures(bool avx512, bool bf16, bool fp16,
                                     bool vnni, bool neon, bool dotprod);

// Parses "<vendor>:<arch>". Returns Vendor::kUnknown and sets *ok=false (when
// *ok is non-null) for a malformed id or an unknown vendor. A well-formed but
// unsupported vendor (e.g. "amd:gfx942") parses ok and supports nothing.
HardwareTarget TargetFromId(std::string_view id, bool *ok);

// ── Invariant I3: the admission decision as a pure function ────────────────
//
// `used` is the artifact's declared mask (artifact.fb `used_codecs`), `t` the
// device the process runs on. No hardware call happens here, so the whole
// matrix — seven codecs x every target — is testable on a machine with no
// Ampere card attached. Empty `message` on success.
struct GateDecision {
    bool allowed = false;
    CodecMask missing = 0;   // used & ~supported; 0 when allowed
    std::string message;     // human-readable reason, empty when allowed
};

GateDecision CheckCodecs(CodecMask used, const HardwareTarget &t);

// The active target of this process, resolved once on first call: CUDA device 0
// when a device is present, else a CPUID-derived cpu:* target. Cached, so
// repeated calls are free and the answer cannot drift mid-run.
const HardwareTarget &ActiveTarget();

// Test-only override. Deliberately not an env var and not a CLI flag: a
// production binary must not be able to pretend it runs on another target.
// ResetActiveTargetForTesting() drops the override AND the cached probe.
void SetActiveTargetForTesting(const HardwareTarget &t);
void ResetActiveTargetForTesting();

// Canonical lowercase spelling used by `--codec-profile`, artifact.fb and log
// messages. kNone renders as "none".
const char *PlaneCodecName(PlaneCodec c);

// Parses the canonical spelling. Returns false for anything it cannot name —
// including "fp8", which is ambiguous between E4M3 and E5M2 and must be spelled
// "mxfp8" (the only FP8 variant this tree stores).
bool ParsePlaneCodec(std::string_view name, PlaneCodec *out);

// Comma-separated set bits in ascending bit order, e.g. "bf16,nvfp4"; "none"
// for an empty mask. This exact text appears in refusal messages, so it is
// pinned by tests.
std::string FormatMask(CodecMask mask);

// ── Profile preference (design §2.7, invariant I4) ─────────────────────────
//
// Which of a model's compatible profiles a device should load, most desirable
// first. The rationale is bytes per element: the most compressed profile the
// target can EXECUTE wins, which is also the throughput order — and a profile
// with no kernel on that target is excluded by the capability mask, not by this
// list.
//
// `int8-w8a16` leads because on sm_120 it is native AND half the bytes of
// nvfp4's neighbours, but the list is a policy, not a measurement: spec §8.2
// says the order between nvfp4 and int8-w8a16 on Blackwell must be corrected
// from the Branch-3 perf rows, and this is the one place to correct it.
//
// Declared here (host-only, no allocation in the hot path) so that a caller
// cannot invent its own order: storage takes the list as a parameter and has no
// opinion, lsxcommon has exactly one.
inline const std::vector<std::string> &CodecProfilePreference()
{
    static const std::vector<std::string> v = {"int8-w8a16", "nvfp4", "mxfp8",
                                               "bf16"};
    return v;
}

} // namespace hw
} // namespace lsxcommon