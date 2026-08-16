#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/generation_cli.h"

#include <string>
#include <vector>

TEST_CASE("generation_cli: default blueprint prompt", "[generation_cli]") {
    REQUIRE(std::string(lsxcommon::cli::kDefaultGenerationPrompt) ==
            "Explain the phrase: That's one small step for a man, one giant leap for mankind");
}

TEST_CASE("generation_cli: ResolveGenerationPrompt falls back to the default", "[generation_cli]") {
    REQUIRE(lsxcommon::cli::ResolveGenerationPrompt("") ==
            lsxcommon::cli::kDefaultGenerationPrompt);
    REQUIRE(lsxcommon::cli::ResolveGenerationPrompt("  ") ==
            lsxcommon::cli::kDefaultGenerationPrompt);
}

TEST_CASE("generation_cli: ResolveGenerationPrompt keeps a real prompt", "[generation_cli]") {
    REQUIRE(lsxcommon::cli::ResolveGenerationPrompt("hello world") == "hello world");
}

TEST_CASE("generation_cli: argv without generation flag passes through", "[generation_cli]") {
    std::vector<char> a0{'-'};
    std::vector<char> a1{'-'};
    std::vector<char> a2{'-'};
    char* argv[] = {a0.data(), a1.data(), a2.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(3, argv);
    REQUIRE(out.size() == 3);
}

TEST_CASE("generation_cli: -r short alias expands to --run", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "-r";
    std::string s2 = "hello";
    char* argv[] = {s0.data(), s1.data(), s2.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(3, argv);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == "prog");
    REQUIRE(out[1] == "--run");
    REQUIRE(out[2] == "hello");
}

TEST_CASE("generation_cli: -r with =value maps to --run=value", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "-r=hello";
    char* argv[] = {s0.data(), s1.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(2, argv);
    REQUIRE(out.size() == 2);
    REQUIRE(out[1] == "--run=hello");
}

TEST_CASE("generation_cli: bare -r as last token falls back to default prompt", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "-r";
    char* argv[] = {s0.data(), s1.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(2, argv);
    REQUIRE(out.size() == 2);
    REQUIRE(out[1] == std::string("--run=") + lsxcommon::cli::kDefaultGenerationPrompt);
}

TEST_CASE("generation_cli: bare --run as last token falls back to default prompt", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "--run";
    char* argv[] = {s0.data(), s1.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(2, argv);
    REQUIRE(out.size() == 2);
    REQUIRE(out[1] == std::string("--run=") + lsxcommon::cli::kDefaultGenerationPrompt);
}

TEST_CASE("generation_cli: --run with explicit =value passes through", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "--run=hello";
    char* argv[] = {s0.data(), s1.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(2, argv);
    REQUIRE(out.size() == 2);
    REQUIRE(out[1] == "--run=hello");
}

TEST_CASE("generation_cli: --run with empty =value falls back to default", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "--run=";
    char* argv[] = {s0.data(), s1.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(2, argv);
    REQUIRE(out.size() == 2);
    REQUIRE(out[1] == std::string("--run=") + lsxcommon::cli::kDefaultGenerationPrompt);
}

TEST_CASE("generation_cli: --run followed by another flag falls back to default", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "--run";
    std::string s2 = "--glm";
    char* argv[] = {s0.data(), s1.data(), s2.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(3, argv);
    REQUIRE(out.size() == 3);
    REQUIRE(out[1] == std::string("--run=") + lsxcommon::cli::kDefaultGenerationPrompt);
    REQUIRE(out[2] == "--glm");
}

TEST_CASE("generation_cli: --run with space-separated value passes through", "[generation_cli]") {
    std::string s0 = "prog";
    std::string s1 = "--run";
    std::string s2 = "blueprint";
    char* argv[] = {s0.data(), s1.data(), s2.data()};
    auto out = lsxcommon::cli::NormalizeGenerationArgv(3, argv);
    REQUIRE(out.size() == 3);
    REQUIRE(out[1] == "--run");
    REQUIRE(out[2] == "blueprint");
}
