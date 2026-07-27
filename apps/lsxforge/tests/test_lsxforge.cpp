#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/core.h"

#include <fstream>

TEST_CASE("lsxforge core initialization and lifecycle", "[lsxforge]") {
    const std::string log_path = "test_lsxforge.log";

    lsxcommon::Core::Initialize(log_path);
    lsxcommon::Core::Shutdown();

    std::ifstream log_file(log_path);
    REQUIRE(log_file.good());
    log_file.close();
    std::remove(log_path.c_str());
}
