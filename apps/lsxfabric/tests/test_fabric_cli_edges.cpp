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

TEST_CASE("fabric cli: flag without a value is ignored", "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--localmodel"}).model_path.find(
                "DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE(Parse({"lsxfabric", "--max-gen"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--prompt"}).prompt ==
            "The capital of France is");
}

TEST_CASE("fabric cli: a value flag does not swallow a known flag",
          "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs p = Parse({"lsxfabric", "--prompt", "--validate"});
    REQUIRE(p.prompt == "The capital of France is");
    REQUIRE(p.validate);

    const lsxfabric::FabricCliArgs m =
        Parse({"lsxfabric", "--localmodel", "--validate", "/tmp/x"});
    REQUIRE(m.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE(m.validate);
}

TEST_CASE("fabric cli: non-numeric or non-positive --max-gen keeps the default",
          "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "abc"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "0"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "-5"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "12x"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "12"}).max_gen == 12);
}

TEST_CASE("fabric cli: a non-flag value is still consumed", "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--prompt", "hello world"}).prompt == "hello world");
    REQUIRE(Parse({"lsxfabric", "--localmodel", "/tmp/model_x"}).model_path ==
            "/tmp/model_x");
}

TEST_CASE("fabric cli: repeated flags, last one wins", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--max-gen", "5", "--max-gen", "9",
               "--localmodel", "/a", "--localmodel", "/b"});
    REQUIRE(a.max_gen == 9);
    REQUIRE(a.model_path == "/b");
}

TEST_CASE("fabric cli: help flags are unknown and ignored", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--help", "-h"});
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
}
