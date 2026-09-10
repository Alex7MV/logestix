#include <catch2/catch_test_macros.hpp>
#include "cli.h"

#include <string>
#include <vector>

namespace {

std::vector<char *> MakeArgv(const std::vector<std::string> &storage) {
    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (const std::string &s : storage)
        argv.push_back(const_cast<char *>(s.c_str()));
    return argv;
}

lsxbenchmark::BenchmarkCliArgs Parse(const std::vector<std::string> &args) {
    std::vector<char *> argv = MakeArgv(args);
    return lsxbenchmark::ParseBenchmarkCliArgs(static_cast<int>(argv.size()),
                                               argv.data());
}

} // namespace

TEST_CASE("lsxbenchmark cli: defaults to bench mode on the vault model",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse({"lsxbenchmark"});
    REQUIRE(a.bench);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE(a.max_gen == 2048);
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
}

TEST_CASE("lsxbenchmark cli: --selfcheck selects verification mode",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse({"lsxbenchmark", "--selfcheck"});
    REQUIRE(a.selfcheck);
    REQUIRE_FALSE(a.bench);
    REQUIRE_FALSE(a.bit_exact);
}

TEST_CASE("lsxbenchmark cli: --bit-exact pins the deterministic scalar mode",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a =
        Parse({"lsxbenchmark", "--selfcheck", "--bit-exact"});
    REQUIRE(a.selfcheck);
    REQUIRE(a.bit_exact);

    const lsxbenchmark::BenchmarkCliArgs b =
        Parse({"lsxbenchmark", "--bench", "--bit-exact"});
    REQUIRE(b.bench);
    REQUIRE(b.bit_exact);
}

TEST_CASE("lsxbenchmark cli: bench options parse", "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse(
        {"lsxbenchmark", "--model", "/tmp/model_x", "--bench", "--prompt", "hi",
         "--max-gen", "7"});
    REQUIRE(a.model_path == "/tmp/model_x");
    REQUIRE(a.bench);
    REQUIRE(a.prompt == "hi");
    REQUIRE(a.max_gen == 7);
}

TEST_CASE("lsxbenchmark cli: unknown flags are silently ignored",
          "[lsxbenchmark][cli]") {
    const lsxbenchmark::BenchmarkCliArgs a = Parse(
        {"lsxbenchmark", "--fast-math", "--calibrate", "--wat"});
    REQUIRE(a.bench);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE(a.max_gen == 2048);
}
