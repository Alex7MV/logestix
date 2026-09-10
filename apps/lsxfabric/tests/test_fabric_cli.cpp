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

lsxfabric::FabricCliArgs Parse(const std::vector<std::string> &args) {
    std::vector<char *> argv = MakeArgv(args);
    return lsxfabric::ParseFabricCliArgs(static_cast<int>(argv.size()),
                                         argv.data());
}

} // namespace

TEST_CASE("fabric cli: defaults are auto-policy with no modes", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric"});
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.benchmark);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE_FALSE(a.selfcheck);
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE(a.max_gen == 64);
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
}

TEST_CASE("fabric cli: --bit-exact is the single numeric-mode flag", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--bit-exact"});
    REQUIRE(a.bit_exact);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.benchmark);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE_FALSE(a.selfcheck);
}

TEST_CASE("fabric cli: removed speed/perf flags are not recognized", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--fast-math", "--no-fast-math", "--calibrate"});
    REQUIRE_FALSE(a.bit_exact);
    REQUIRE_FALSE(a.benchmark);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE_FALSE(a.selfcheck);
}

TEST_CASE("fabric cli: kept flags parse", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--localmodel", "/tmp/model_x", "--prompt", "hi",
         "--max-gen", "7", "--validate", "--benchmark", "--testprompt",
         "--selfcheck"});
    REQUIRE(a.model_path == "/tmp/model_x");
    REQUIRE(a.prompt == "hi");
    REQUIRE(a.max_gen == 7);
    REQUIRE(a.validate);
    REQUIRE(a.benchmark);
    REQUIRE(a.testprompt);
    REQUIRE(a.selfcheck);
}
