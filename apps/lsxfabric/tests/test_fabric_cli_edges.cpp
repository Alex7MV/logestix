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

// Current behaviour (green characterisation). See docs/LSXFABRIC_AUDIT.md.

TEST_CASE("fabric cli: flag without a value is silently ignored",
          "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--localmodel"}).model_path.find(
                "DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE(Parse({"lsxfabric", "--max-gen"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--prompt"}).prompt ==
            "The capital of France is");
}

TEST_CASE("fabric cli: --prompt consumes the following flag token",
          "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--prompt", "--validate"});
    REQUIRE(a.prompt == "--validate");
    REQUIRE_FALSE(a.validate);
}

TEST_CASE("fabric cli: non-numeric --max-gen becomes 0", "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "abc"}).max_gen == 0);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "12x"}).max_gen == 12);
}

TEST_CASE("fabric cli: negative --max-gen is accepted verbatim",
          "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "-5"}).max_gen == -5);
}

TEST_CASE("fabric cli: repeated flags, last one wins", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--max-gen", "5", "--max-gen", "9",
               "--localmodel", "/a", "--localmodel", "/b"});
    REQUIRE(a.max_gen == 9);
    REQUIRE(a.model_path == "/b");
}

TEST_CASE("fabric cli: a flag-looking value is taken literally",
          "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--localmodel", "--validate", "/tmp/x"});
    REQUIRE(a.model_path == "--validate");
    REQUIRE_FALSE(a.validate);
    REQUIRE(a.max_gen == 64);
}

TEST_CASE("fabric cli: help flags are unknown and ignored", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--help", "-h"});
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
}

// RED spec tests: the expected contract where current behaviour diverges.
// Hidden (a tag starting with '.'), so the default ctest run stays green.
// Run explicitly: ./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"

// FINDING F5: --max-gen must not turn a non-numeric value into 0.
TEST_CASE("fabric cli [spec]: --max-gen rejects a non-numeric value",
          "[.audit-spec]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "abc"}).max_gen == 64);
}

// FINDING F5: --max-gen must be positive.
TEST_CASE("fabric cli [spec]: --max-gen is bounded to a positive value",
          "[.audit-spec]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "-5"}).max_gen >= 1);
}

// FINDING F4: --prompt must not swallow a following known flag.
TEST_CASE("fabric cli [spec]: --prompt does not swallow a known flag",
          "[.audit-spec]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--prompt", "--validate"});
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.validate);
}
