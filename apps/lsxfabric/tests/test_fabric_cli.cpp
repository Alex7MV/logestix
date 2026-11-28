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

TEST_CASE("fabric cli: defaults are maximum-speed generation", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric"});
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE(a.max_gen == 64);
    REQUIRE(a.prompt == "The capital of France is");
    // The default is a model IDENTITY, not a path: the store is org-nested
    // (`<root>/<org>/<model>`), so a path built here would drop the org — the
    // old default did and pointed at a directory that never exists. Resolve
    // turns this name into the artifact.
    REQUIRE(a.model_path == "deepseek-v4-flash");
    REQUIRE(a.model_path.find('/') == std::string::npos);
}

TEST_CASE("fabric cli: measurement flags moved to lsxbenchmark are not recognized",
          "[lsxfabric][cli]") {
    // Silently ignored, same policy as the removed --fast-math/--calibrate.
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--benchmark", "--selfcheck", "--bit-exact",
         "--fast-math", "--no-fast-math", "--calibrate"});
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
    REQUIRE(a.max_gen == 64);
    REQUIRE(a.prompt == "The capital of France is");
}

TEST_CASE("fabric cli: kept flags parse", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse(
        {"lsxfabric", "--localmodel", "/tmp/model_x", "--prompt", "hi",
         "--max-gen", "7", "--validate", "--testprompt"});
    REQUIRE(a.model_path == "/tmp/model_x");
    REQUIRE(a.prompt == "hi");
    REQUIRE(a.max_gen == 7);
    REQUIRE(a.validate);
    REQUIRE(a.testprompt);
}
