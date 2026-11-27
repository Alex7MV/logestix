#include "lsxcommon/hardware_target.h"

#include <cuda_runtime.h>

#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace lsxcommon {
namespace hw {

namespace {

struct CodecNameRow {
    PlaneCodec codec;
    const char *name;
};

// One row per accepted spelling, in bit order. ParsePlaneCodec and
// PlaneCodecName read the same table, so a name can never parse to one codec
// and render as another.
constexpr CodecNameRow kCodecNames[] = {
    {PlaneCodec::kF32, "f32"},
    {PlaneCodec::kBF16, "bf16"},
    {PlaneCodec::kF16, "f16"},
    {PlaneCodec::kI8Sym, "i8"},
    {PlaneCodec::kMxFp8, "mxfp8"},
    {PlaneCodec::kNvFp4, "nvfp4"},
    {PlaneCodec::kMxFp4, "mxfp4"},
};

constexpr const char *kVendorNvidia = "nvidia";
constexpr const char *kVendorCpu = "cpu";
constexpr const char *kVendorAmd = "amd";

constexpr CodecMask kF32Mask = Mask(PlaneCodec::kF32);

// GA10x: f16/bf16 MMA (m16n8k16) and integer MMA (m16n8k32.s8) are native;
// there is no E4M3/E2M1 tensor-core instruction, so FP8 and FP4 are absent.
constexpr CodecMask kAmpereMask = kF32Mask | Mask(PlaneCodec::kBF16) |
                                  Mask(PlaneCodec::kF16) |
                                  Mask(PlaneCodec::kI8Sym);

// Blackwell consumer: every plane encoding this tree produces.
constexpr CodecMask kBlackwellMask =
    kAmpereMask | Mask(PlaneCodec::kMxFp8) | Mask(PlaneCodec::kNvFp4) |
    Mask(PlaneCodec::kMxFp4);

// Splits "<vendor>:<arch>". Returns false when either half is empty.
bool SplitId(std::string_view id, std::string_view *vendor, std::string_view *arch)
{
    const auto colon = id.find(':');
    if (colon == std::string_view::npos) return false;
    *vendor = id.substr(0, colon);
    *arch = id.substr(colon + 1);
    return !vendor->empty() && !arch->empty();
}

// cpu: feature suffix -> mask. The suffix is derived from the arch string so
// TargetFromId and TargetFromCpuFeatures agree by construction.
CodecMask CpuMaskForArch(std::string_view arch)
{
    CodecMask m = kF32Mask;
    if (arch.find("+bf16") != std::string_view::npos)
        m |= Mask(PlaneCodec::kBF16);
    if (arch.find("+fp16") != std::string_view::npos)
        m |= Mask(PlaneCodec::kF16);
    // avx512_vnni and ARM dotprod both give a native int8 dot product.
    if (arch.find("+vnni") != std::string_view::npos ||
        arch.find("+dotprod") != std::string_view::npos)
        m |= Mask(PlaneCodec::kI8Sym);
    return m;
}

} // namespace

std::string HardwareTarget::Id() const
{
    std::string out;
    switch (vendor) {
    case Vendor::kNvidia: out = kVendorNvidia; break;
    case Vendor::kCpu: out = kVendorCpu; break;
    case Vendor::kAmd: out = kVendorAmd; break;
    default: return "unknown:";
    }
    out.push_back(':');
    out.append(arch);
    return out;
}

bool operator==(const HardwareTarget &a, const HardwareTarget &b)
{
    return a.vendor == b.vendor && a.arch == b.arch;
}

bool operator!=(const HardwareTarget &a, const HardwareTarget &b)
{
    return !(a == b);
}

CodecMask SupportedCodecs(const HardwareTarget &t)
{
    switch (t.vendor) {
    case Vendor::kNvidia:
        if (t.arch == "sm_120" || t.arch == "sm_121") return kBlackwellMask;
        if (t.arch == "sm_86" || t.arch == "sm_87") return kAmpereMask;
        // sm_89 has FP8 tensor cores but is not a supported target; sm_80 is
        // forbidden by AGENTS.md. Both support nothing here.
        return 0;
    case Vendor::kCpu:
        if (t.arch.rfind("avx512", 0) == 0 || t.arch.rfind("neon", 0) == 0)
            return CpuMaskForArch(t.arch);
        return 0;
    case Vendor::kAmd:
        return 0; // reserved: no kernels, so nothing is executed natively
    default:
        return 0;
    }
}

HardwareTarget TargetFromComputeCapability(int major, int minor)
{
    HardwareTarget t;
    if (major != 8 && major != 12) return t;
    if (major == 12 && minor != 0 && minor != 1) return t;
    if (major == 8 && minor != 6 && minor != 7) return t;
    t.vendor = Vendor::kNvidia;
    t.arch = "sm_" + std::to_string(major) + std::to_string(minor);
    return t;
}

HardwareTarget TargetFromCpuFeatures(bool avx512, bool bf16, bool fp16,
                                     bool vnni, bool neon, bool dotprod)
{
    HardwareTarget t;
    t.vendor = Vendor::kCpu;
    if (neon) {
        t.arch = "neon";
        if (dotprod) t.arch += "+dotprod";
        return t;
    }
    if (avx512) {
        t.arch = "avx512";
        if (bf16) t.arch += "+bf16";
        if (fp16) t.arch += "+fp16";
        if (vnni) t.arch += "+vnni";
        return t;
    }
    // No vector ISA we can name: leave vendor unset so the gate fails closed.
    t.vendor = Vendor::kUnknown;
    t.arch.clear();
    return t;
}

HardwareTarget TargetFromId(std::string_view id, bool *ok)
{
    const auto fail = [ok]() {
        if (ok != nullptr) *ok = false;
        return HardwareTarget{};
    };
    if (id.empty()) return fail();

    std::string_view vendor, arch;
    if (!SplitId(id, &vendor, &arch)) return fail();

    HardwareTarget t;
    if (vendor == kVendorNvidia) {
        if (arch != "sm_120" && arch != "sm_121" && arch != "sm_86" &&
            arch != "sm_87")
            return fail();
        t.vendor = Vendor::kNvidia;
    } else if (vendor == kVendorCpu) {
        // Feature suffixes are open-ended, but the ISA root must be one we know.
        if (arch.rfind("avx512", 0) != 0 && arch.rfind("neon", 0) != 0)
            return fail();
        t.vendor = Vendor::kCpu;
    } else if (vendor == kVendorAmd) {
        t.vendor = Vendor::kAmd;
    } else {
        return fail(); // unknown vendor: never a supported target
    }
    t.arch = std::string(arch);
    if (ok != nullptr) *ok = true;
    return t;
}

const char *PlaneCodecName(PlaneCodec c)
{
    for (const auto &row : kCodecNames)
        if (row.codec == c) return row.name;
    return "none";
}

bool ParsePlaneCodec(std::string_view name, PlaneCodec *out)
{
    if (out == nullptr || name.empty()) return false;
    for (const auto &row : kCodecNames) {
        if (name == row.name) {
            *out = row.codec;
            return true;
        }
    }
    return false;
}

std::string FormatMask(CodecMask mask)
{
    if (mask == 0) return "none";
    std::string out;
    for (const auto &row : kCodecNames) {
        if ((mask & Mask(row.codec)) == 0) continue;
        if (!out.empty()) out.push_back(',');
        out.append(row.name);
    }
    return out;
}

// ── The gate ───────────────────────────────────────────────────────────────

GateDecision CheckCodecs(CodecMask used, const HardwareTarget &t)
{
    GateDecision d;
    const CodecMask supported = SupportedCodecs(t);

    if (t.vendor == Vendor::kUnknown) {
        d.missing = used;
        d.message = "no supported accelerator detected (target '" + t.Id() +
                    "'): refusing to guess which codecs are hardware-native";
        return d;
    }
    if (supported == 0) {
        // A named target with no row: say so, because "missing codecs" would
        // send the operator looking for a reconversion that cannot help.
        d.missing = used;
        d.message = "target '" + t.Id() +
                    "' is not implemented in this build (no hardware-native "
                    "codec profile exists for it)";
        return d;
    }

    d.missing = used & ~supported;
    if (d.missing == 0) {
        d.allowed = true;
        return d;
    }
    d.message = "artifact uses [" + FormatMask(d.missing) +
                "] which '" + t.Id() +
                "' does not execute natively (it supports [" +
                FormatMask(supported) + "])";
    return d;
}

// ── Active target ──────────────────────────────────────────────────────────

namespace {

std::mutex g_target_mu;
HardwareTarget g_override;
bool g_has_override = false;
HardwareTarget g_probed;
bool g_probed_valid = false;

HardwareTarget ProbeCpuTarget()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, nullptr) < 7)
        return TargetFromCpuFeatures(false, false, false, false, false, false);
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    const bool avx512f = (ebx & (1u << 16)) != 0;
    const bool avx512bf16 = (eax & (1u << 5)) != 0;
    const bool avx512fp16 = (edx & (1u << 23)) != 0;
    const bool avx512vnni = (ecx & (1u << 11)) != 0;
    return TargetFromCpuFeatures(avx512f, avx512bf16, avx512fp16, avx512vnni,
                                 false, false);
#elif defined(__aarch64__) || defined(_M_ARM64)
    bool dotprod = false;
#if defined(__ARM_FEATURE_DOTPROD)
    dotprod = true;
#endif
    return TargetFromCpuFeatures(false, true, true, false, true, dotprod);
#else
    return TargetFromCpuFeatures(false, false, false, false, false, false);
#endif
}

HardwareTarget ProbeTarget()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            auto t = TargetFromComputeCapability(prop.major, prop.minor);
            if (t.vendor != Vendor::kUnknown) return t;
            // A present device we do not target (sm_89, sm_80, ...) must not
            // silently fall through to the CPU row: the gate has to refuse.
        }
    }
    return ProbeCpuTarget();
}

} // namespace

const HardwareTarget &ActiveTarget()
{
    std::lock_guard<std::mutex> lock(g_target_mu);
    if (g_has_override) return g_override;
    if (!g_probed_valid) {
        g_probed = ProbeTarget();
        g_probed_valid = true;
    }
    return g_probed;
}

void SetActiveTargetForTesting(const HardwareTarget &t)
{
    std::lock_guard<std::mutex> lock(g_target_mu);
    g_override = t;
    g_has_override = true;
}

void ResetActiveTargetForTesting()
{
    std::lock_guard<std::mutex> lock(g_target_mu);
    g_has_override = false;
    g_override = HardwareTarget{};
    g_probed_valid = false;
    g_probed = HardwareTarget{};
}

} // namespace hw
} // namespace lsxcommon
