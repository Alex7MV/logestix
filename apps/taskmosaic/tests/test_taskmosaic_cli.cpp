#include <catch2/catch_test_macros.hpp>

#include "cli.h"

#include <string>
#include <vector>

namespace {

std::vector<char*> MakeArgv(const std::vector<std::string>& storage) {
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (const std::string& s : storage)
        argv.push_back(const_cast<char*>(s.c_str()));
    return argv;
}

taskmosaic::TaskMosaicCliArgs Parse(const std::vector<std::string>& args) {
    std::vector<char*> argv = MakeArgv(args);
    return taskmosaic::ParseTaskMosaicCliArgs(static_cast<int>(argv.size()),
                                              argv.data());
}

}  // namespace

TEST_CASE("taskmosaic cli: parses server/model/prompt/max-gen",
          "[taskmosaic][cli]") {
    const taskmosaic::TaskMosaicCliArgs a =
        Parse({"taskmosaic", "--server", "tcp://127.0.0.1:5555", "--model",
               "/models/x", "--prompt", "hello", "--max-gen", "9"});
    REQUIRE(a.server == "tcp://127.0.0.1:5555");
    REQUIRE(a.model == "/models/x");
    REQUIRE(a.prompt == "hello");
    REQUIRE(a.max_gen == 9);
}

TEST_CASE("taskmosaic cli: prompt defaults empty (stdin)",
          "[taskmosaic][cli]") {
    const taskmosaic::TaskMosaicCliArgs a =
        Parse({"taskmosaic", "--server", "tcp://x:1", "--model", "/m"});
    REQUIRE(a.prompt.empty());
    REQUIRE(a.max_gen == 64);
}

TEST_CASE("taskmosaic cli: missing required args stay empty",
          "[taskmosaic][cli]") {
    const taskmosaic::TaskMosaicCliArgs a = Parse({"taskmosaic"});
    REQUIRE(a.server.empty());
    REQUIRE(a.model.empty());
}
