// Arch policy as a fact the tree can be held to (plan Task 14).
//
// Two things live here, and they fail in different ways if the policy reverts:
//
//   * `SupportedCodecs(nvidia:sm_86|sm_87)` must stay non-empty. AGENTS.md once
//    said "never add 80-real/86-real/87-real"; it now says Ampere is a target.
//    A prose-only policy is re-forgotten the next time someone reads the old
//    sentence, so it is pinned by a test: a revert of the table makes this red
//    instead of making the next Ampere build a mystery.
//   * every preset's `LSX_CUDA_ARCH` must be accepted by the validator. Three
//    hand-written arch lists (presets, the module, the guard) drifted the moment
//    `86` was added; this case asks the REAL validator (cmake/LsxArch.cmake via
//    `cmake -P`) about every value a preset carries, so a preset can never ship
//    an arch the configure would refuse.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hardware_target.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif
#ifndef LSX_CMAKE_COMMAND
#define LSX_CMAKE_COMMAND "cmake"
#endif

namespace fs = std::filesystem;
using namespace lsxcommon::hw;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// '# ...' to end of line out. A CMake comment that PROHIBITS a flag must not read
// as that flag: this file's subject matter is full of "never set -march here".
std::string StripHashComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool comment = false;
    for (const char c : text) {
        if (comment) {
            if (c == '\n') { comment = false; out.push_back(c); }
            continue;
        }
        if (c == '#') { comment = true; continue; }
        out.push_back(c);
    }
    return out;
}

// Every "<preset name>" -> "LSX_CUDA_ARCH value" pair in CMakePresets.json,
// without a JSON dependency: the file is small, uniform, and a test that pulls a
// parser in would be harder to trust than a scanner this narrow.
struct PresetArch {
    std::string name;
    std::string arch;   // empty when the preset does not set one
};

std::vector<PresetArch> PresetArches()
{
    const std::string text = ReadFile(fs::path(LSX_SOURCE_DIR) / "CMakePresets.json");
    std::vector<PresetArch> out;
    // Split on preset objects: every configurePreset carries "name" then, later,
    // its cacheVariables. Scanning linearly keeps the pairing obvious.
    static const std::regex name_re(R"re("name"\s*:\s*"([^"]+)")re");
    static const std::regex arch_re(R"re("LSX_CUDA_ARCH"\s*:\s*"([^"]+)")re");
    // Only the configurePresets block: the build/test presets repeat the same
    // names and carry no arch.
    const auto cfg_begin = text.find("\"configurePresets\"");
    const auto cfg_end = text.find("\"buildPresets\"");
    if (cfg_begin == std::string::npos || cfg_end == std::string::npos) return out;
    const std::string cfg = text.substr(cfg_begin, cfg_end - cfg_begin);

    // Walk object starts and pick up the arch between this name and the next.
    std::vector<std::pair<std::string, size_t>> named;
    for (auto it = std::sregex_iterator(cfg.begin(), cfg.end(), name_re);
         it != std::sregex_iterator(); ++it)
        named.emplace_back((*it)[1].str(), static_cast<size_t>(it->position()));
    for (size_t i = 0; i < named.size(); ++i) {
        const size_t from = named[i].second;
        const size_t to = i + 1 < named.size() ? named[i + 1].second : cfg.size();
        PresetArch p;
        p.name = named[i].first;
        const std::string slice = cfg.substr(from, to - from);
        std::smatch m;
        if (std::regex_search(slice, m, arch_re)) p.arch = m[1].str();
        out.push_back(std::move(p));
    }
    return out;
}

// Runs the real validator on `arches`; returns its exit code and output.
struct ProbeResult {
    int rc = -1;
    std::string out;
};

ProbeResult ProbeArch(const std::string &arches)
{
    ProbeResult r;
    const fs::path script =
        fs::path(LSX_SOURCE_DIR) / "tests" / "cmake" / "probe_arch.cmake";
    const fs::path module = fs::path(LSX_SOURCE_DIR) / "cmake" / "LsxArch.cmake";
    const std::string cmd = std::string("\"") + LSX_CMAKE_COMMAND +
                            "\" -D PROBE_ARCH=\"" + arches + "\" -D LsxArch=\"" +
                            module.string() + "\" -P \"" + script.string() +
                            "\" 2>&1";
#if defined(_WIN32)
    FILE *pipe = _popen(cmd.c_str(), "r");
#else
    FILE *pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) return r;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) r.out += buf;
#if defined(_WIN32)
    r.rc = _pclose(pipe);
#else
    r.rc = (pclose(pipe) >> 8) & 0xff;
#endif
    return r;
}

} // namespace

// ── The policy: Ampere is a target, and it executes four codecs ────────────

TEST_CASE("arch support: sm_86 and sm_87 are supported targets",
          "[guard][arch_support]") {
    // The pin against a policy revert: if someone restores "Blackwell only",
    // SupportedCodecs returns 0 here and every Ampere build becomes a refusal at
    // load time with no explanation. This is the test that says otherwise.
    for (const char *id : {"nvidia:sm_86", "nvidia:sm_87"}) {
        bool ok = false;
        const auto t = TargetFromId(id, &ok);
        INFO("target " << id);
        REQUIRE(ok);
        REQUIRE(SupportedCodecs(t) != 0);
        // Exactly the four the hardware has: no FP8, no FP4 anywhere on GA10x.
        CHECK(SupportedCodecs(t) ==
              (Mask(PlaneCodec::kF32) | Mask(PlaneCodec::kBF16) |
               Mask(PlaneCodec::kF16) | Mask(PlaneCodec::kI8Sym)));
        CHECK((SupportedCodecs(t) & Mask(PlaneCodec::kMxFp8)) == 0);
        CHECK((SupportedCodecs(t) & Mask(PlaneCodec::kNvFp4)) == 0);
    }
    // And a bf16/f32 artifact is admitted there, which is the whole point of
    // Branch 2: the gate must have somewhere to say yes.
    const auto ampere = TargetFromComputeCapability(8, 6);
    CHECK(CheckCodecs(Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kF32), ampere)
              .allowed);
    // While an nvfp4 one is refused, naming what is missing.
    const auto d = CheckCodecs(Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kNvFp4),
                              ampere);
    CHECK_FALSE(d.allowed);
    CHECK(d.missing == Mask(PlaneCodec::kNvFp4));
}

TEST_CASE("arch support: the build targets both supported families",
          "[guard][arch_support]") {
    // The validator's accepted set, asked through the real module. A guard that
    // only read the source text would pass while the regex rejected 86.
    const auto r = ProbeArch("120;121;86;87");
    INFO(r.out);
    REQUIRE(r.rc == 0);
    for (const char *arch : {"sm_86", "sm_87", "sm_120", "sm_121"})
        CHECK(r.out.find(arch) != std::string::npos);
    // Hopper stays out — the policy is "Blackwell consumer + Ampere GA10x", not
    // "anything modern".
    const auto hopper = ProbeArch("90");
    CHECK(hopper.rc != 0);
}

// ── Presets can never carry an arch the configure would refuse ─────────────

TEST_CASE("arch support: every preset's LSX_CUDA_ARCH is accepted",
          "[guard][arch_support]") {
    const auto presets = PresetArches();
    INFO("presets found: " << presets.size());
    REQUIRE(presets.size() >= 8);   // 6 original + the 2 Ampere ones

    int with_arch = 0;
    for (const auto &p : presets) {
        if (p.arch.empty()) continue;
        ++with_arch;
        INFO("preset '" << p.name << "' arch='" << p.arch << "'");
        const auto r = ProbeArch(p.arch);
        INFO(r.out);
        CHECK(r.rc == 0);
    }
    // The Ampere presets must actually pin their arch: a preset that inherits
    // the 120;121 default would build a 3090 tree that cannot run on the 3090.
    INFO("presets pinning an arch: " << with_arch);
    CHECK(with_arch >= 2);
}

TEST_CASE("arch support: the Ampere presets exist and pin the right arches",
          "[guard][arch_support]") {
    const auto presets = PresetArches();
    const auto find = [&presets](const std::string &name) -> std::string {
        for (const auto &p : presets)
            if (p.name == name) return p.arch;
        return "<absent>";
    };
    // linux-release-sm86 is the RTX 3090 host: sm_86 only, so the binary cannot
    // silently grow an sm_87 image it will never launch.
    CHECK(find("linux-release-sm86") == "86");
    // Jetson Orin NX/Nano is sm_87 — and it ships sm_86 SASS too, because the
    // toolchain is shared with the 3090 host and one fatbin serves both.
    CHECK(find("jetson-orin-release") == "86;87");
}

TEST_CASE("arch support: the Jetson toolchain file exists and pins CUDA 12.6",
          "[guard][arch_support]") {
    // JetPack 6.2.1's toolkit cannot share the x86 build tree, and its compiler
    // is the one thing a preset cannot express portably: the path is in the
    // toolchain file, which is why the plan asks for one.
    const fs::path tc = fs::path(LSX_SOURCE_DIR) / "cmake" / "toolchains" /
                        "jetson-orin.cmake";
    INFO("toolchain: " << tc.string());
    REQUIRE(fs::is_regular_file(tc));
    const std::string text = StripHashComments(ReadFile(tc));
    CHECK(text.find("cuda-12.6") != std::string::npos);
    // aarch64 target, and no x86 flags: a -march here would be a configure-time
    // error at best and a silently wrong binary at worst.
    CHECK(text.find("aarch64") != std::string::npos);
    CHECK(text.find("-march=native") == std::string::npos);
    // The x86 conveniences are turned OFF by the file itself, not left to the
    // caller: an inherited -march=znver4 in front of an ARM compiler is a
    // configure error, and LTO is a link-time budget this board does not have.
    CHECK(text.find("LSX_ARCH_ZNVER4 OFF") != std::string::npos);
    CHECK(text.find("LSX_LTO OFF") != std::string::npos);
    // And it must NOT pin the arch list: the preset owns that, and a second
    // source of truth for LSX_CUDA_ARCH is exactly what the module exists to
    // prevent.
    CHECK(text.find("set(LSX_CUDA_ARCH") == std::string::npos);
    // A toolchain file is re-included in every nested configure (try_compile for
    // the compiler-ABI probe gets a FRESH, EMPTY cache), so a hard FATAL on a
    // missing toolkit aborts the probe and reports "CMAKE_C_COMPILER not set"
    // instead of the real reason. Measured 2026-11-28. The file must therefore
    // resolve loudly and let find_package(CUDAToolkit REQUIRED) refuse.
    CHECK(text.find("message(FATAL_ERROR") == std::string::npos);
    CHECK(text.find("message(WARNING") != std::string::npos);
}

// ── The documents that state the policy must state this one ────────────────
//
// Three places told the world "Blackwell only, pre-Blackwell rejected": AGENTS.md,
// README's Prerequisites and docs/NVIDIA_DOCS_AMPERE.md (the last one has since
// been removed from the tree). A policy that lives in prose AND in a table drifts
// the moment one side moves — and the prose side is the one a newcomer reads
// first, so a stale sentence costs more than a stale comment. These cases pin the
// two files this task owns; AGENTS.md is operator-reviewed text and is not
// scanned here (Task 16 applies it by hand).

namespace {

std::string ReadDoc(const char *rel)
{
    return ReadFile(fs::path(LSX_SOURCE_DIR) / rel);
}

} // namespace

TEST_CASE("arch support: README states both supported target families",
          "[guard][arch_support][docs]") {
    const std::string readme = ReadDoc("README.md");
    REQUIRE_FALSE(readme.empty());
    const auto at = readme.find("## Prerequisites");
    REQUIRE(at != std::string::npos);
    const std::string prereq = readme.substr(at);

    // The Ampere rows, by capability — the thing a reader searches for.
    CHECK(prereq.find("sm_86") != std::string::npos);
    CHECK(prereq.find("sm_87") != std::string::npos);
    CHECK(prereq.find("3090") != std::string::npos);
    CHECK(prereq.find("JetPack") != std::string::npos);
    // And the sentence that used to be true and is now false: Ampere is NOT in
    // the rejected list. Scoped to Prerequisites so the changelog and history
    // sections stay readable.
    // The sentence that used to be true and is now false. Scoped to the SENTENCE,
    // not a character window: the Prerequisites legitimately discusses Ampere two
    // lines later (it is supported), and a fixed window cannot tell "Ampere is
    // rejected" from "on Ampere, an artifact is refused".
    const auto rejected = prereq.find("deliberately unsupported");
    if (rejected != std::string::npos) {
        const auto stop = prereq.find('.', rejected);
        const std::string sentence =
            prereq.substr(rejected, stop == std::string::npos ? 200 : stop - rejected);
        INFO("rejection sentence: " << sentence);
        CHECK(sentence.find("Ampere") == std::string::npos);
        CHECK(sentence.find("86") == std::string::npos);
        CHECK(sentence.find("87") == std::string::npos);
    }
    // The toolkit floor, and the two new presets.
    CHECK(prereq.find("12.6") != std::string::npos);
    CHECK(prereq.find("linux-release-sm86") != std::string::npos);
    CHECK(prereq.find("jetson-orin-release") != std::string::npos);
}

TEST_CASE("arch support: the numeric contract scopes R3 to a target group",
          "[guard][arch_support][docs]") {
    const std::string doc = ReadDoc("docs/numeric-contract.md");
    REQUIRE_FALSE(doc.empty());
    // §1.1 item 4 is the amendment: a capability may select a kernel FAMILY.
    // Without it the contract forbids exactly what the codec profiles do, and a
    // reader implementing Ampere support is told the design is illegal.
    const auto at = doc.find("### 1.1");
    REQUIRE(at != std::string::npos);
    const std::string section = doc.substr(at, 6000);
    CHECK(section.find("HardwareTarget") != std::string::npos);
    CHECK(section.find("target group") != std::string::npos);
    const bool family_wording = section.find("kernel *family*") != std::string::npos ||
                               section.find("kernel family") != std::string::npos;
    CHECK(family_wording);
    // The pair table that makes it operational: the two Ampere dies are ONE
    // group, and sm_86 vs sm_120 is not a relationship unless the profile is
    // shared.
    CHECK(section.find("sm_86") != std::string::npos);
    CHECK(section.find("sm_120") != std::string::npos);
}
