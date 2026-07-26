#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/core.h"

#include <fstream>

TEST_CASE("lsxcli core initialization and lifecycle", "[lsxcli]") {
    const std::string log_path = "test_lsxcli.log";

    lsxcommon::Core::Initialize(log_path);
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();

    std::ifstream log_file(log_path);
    REQUIRE(log_file.good());
    log_file.close();
    std::remove(log_path.c_str());
}
