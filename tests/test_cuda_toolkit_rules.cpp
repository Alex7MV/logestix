// cuda_toolkit_guard's rules, testable where no toolkit is installed.
//
// The guard itself is a `.cu` file: it reads CUDART_VERSION, touches cuda_fp4.h
// and enumerates devices, so "does it accept CUDA 12.6 now" could only be proven
// on a JetPack machine — which is exactly the machine a change to the guard
// cannot assume. Splitting the arithmetic into test_cuda_toolkit_rules.h makes
// threshold checkable here (any host, any toolkit), and the two structural cases
// below pin the things that made the guard Blackwell-specific: a bare
// `#include <cuda_fp4.h>`, a major-equality clause, and a device-class check that
// re-derives the supported set instead of asking the capability table.
//
// The scan is the right tool for the structural half — "does this file name a
// constant" is a property of the text — while "is 12.6 above the floor" is a
// property of an evaluation, so it is a function call.
#include <catch2/catch_test_macros.hpp>

#include "test_cuda_toolkit_rules.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using cuda_toolkit::kFp4HeaderRuntime;
using cuda_toolkit::kMinCudaRuntime;
using cuda_toolkit::RuntimeFloorProblem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Line comments out, so a comment that EXPLAINS the old rule is not read as the
// rule. (This file's own subject matter is full of both.)
std::string StripLineComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool comment = false;
    for (size_t i = 0; i < text.size(); ++i) {
        if (comment) {
            if (text[i] == '\n') {
                comment = false;
                out.push_back('\n');
            }
            continue;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            comment = true;
            ++i;
            continue;
        }
        out.push_back(text[i]);
    }
    return out;
}

fs::path GuardFile()
{
    return fs::path(LSX_SOURCE_DIR) / "tests" / "test_cuda_toolkit_guard.cu";
}

} // namespace

// ── The threshold itself ──────────────────────────────────────────────────

TEST_CASE("toolkit rules: the floor is CUDA 12.6, not a vendor release",
          "[cuda_toolkit_rules]") {
    // 12.6 = JetPack 6.2.1 on Orin. Both supported toolkits are above it.
    CHECK(kMinCudaRuntime == 12060);
    CHECK(RuntimeFloorProblem(12060).empty());
    CHECK(RuntimeFloorProblem(12080).empty());
    CHECK(RuntimeFloorProblem(13050).empty());   // this x86 host
    CHECK(RuntimeFloorProblem(14000).empty());   // no upper bound: not this rule's job
    // Below the floor the message names both numbers so the operator knows what
    // to install, rather than reading "unsupported".
    const auto below = RuntimeFloorProblem(12050);
    CHECK_FALSE(below.empty());
    CHECK(below.find("12.5") != std::string::npos);
    CHECK(below.find("12.6") != std::string::npos);
    CHECK_FALSE(RuntimeFloorProblem(11080).empty());
    CHECK_FALSE(RuntimeFloorProblem(0).empty());
}

TEST_CASE("toolkit rules: the fp4 header threshold is a fact about the header",
          "[cuda_toolkit_rules]") {
    // cuda_fp4.h arrived in 12.8. The GATE is `__has_include` (see the header);
    // this constant exists so a SKIP can explain itself, and it must therefore
    // agree with what the guard says.
    CHECK(kFp4HeaderRuntime == 12080);
    CHECK_FALSE(cuda_toolkit::kFp4HeaderExpectedAt(12060));
    CHECK(cuda_toolkit::kFp4HeaderExpectedAt(12080));
    CHECK(cuda_toolkit::kFp4HeaderExpectedAt(13050));
}

// ── The structural half: no Blackwell/toolkit assumption in the guard ─────

TEST_CASE("toolkit rules: cuda_fp4.h is included through __has_include",
          "[cuda_toolkit_rules]") {
    const std::string text = StripLineComments(ReadFile(GuardFile()));
    REQUIRE_FALSE(text.empty());

    // A bare include is a compile error on JetPack 6.2.1 (CUDA 12.6) BEFORE any
    // SKIP can run — the case cannot protect itself from a missing header it
    // needs at preprocessing time.
    CHECK(text.find("__has_include") != std::string::npos);
    CHECK(text.find("#if __has_include(<cuda_fp4.h>)") != std::string::npos);

    // Every appearance of the include must be inside that guard: count them and
    // require the guard to be present at least as often.
    size_t includes = 0;
    for (size_t at = text.find("#include <cuda_fp4.h>"); at != std::string::npos;
         at = text.find("#include <cuda_fp4.h>", at + 1))
        ++includes;
    size_t guards = 0;
    for (size_t at = text.find("#if __has_include(<cuda_fp4.h>)");
         at != std::string::npos;
         at = text.find("#if __has_include(<cuda_fp4.h>)", at + 1))
        ++guards;
    INFO("includes=" << includes << " guards=" << guards);
    CHECK(includes == guards);
}

TEST_CASE("toolkit rules: the guard carries no major-equality or CC clause",
          "[cuda_toolkit_rules]") {
    const std::string text = StripLineComments(ReadFile(GuardFile()));
    REQUIRE_FALSE(text.empty());

    // "CUDART_VERSION / 1000 == 13" fails on JetPack. The floor above is the
    // check; a major equality is a second, contradictory gate.
    CHECK(text.find("CUDART_VERSION / 1000 == 13") == std::string::npos);
    CHECK(text.find("CUDART_VERSION/1000 == 13") == std::string::npos);

    // "prop.major == 12" re-derives the supported device set by hand. The
    // capability table is the authority (hardware_target.h), and the four
    // supported CCs are {12.0, 12.1, 8.6, 8.7} — a literal here fails on a 3090.
    CHECK(text.find("prop.major == 12") == std::string::npos);
    CHECK(text.find("major == 12") == std::string::npos);
    // And it does ask the table, through the shared predicate.
    CHECK(text.find("hw::TargetFromComputeCapability") != std::string::npos);
    CHECK(text.find("hw::Vendor::kUnknown") != std::string::npos);
}

TEST_CASE("toolkit rules: the guard uses the shared floor, not its own copy",
          "[cuda_toolkit_rules]") {
    const std::string text = StripLineComments(ReadFile(GuardFile()));
    // One constant in one place: a second `kMinCudaRuntime` in the .cu would
    // drift from the tested one the first time either moved.
    CHECK(text.find("test_cuda_toolkit_rules.h") != std::string::npos);
    CHECK(text.find("kMinCudaRuntime = ") == std::string::npos);
    CHECK(text.find("RuntimeFloorProblem") != std::string::npos);
}
