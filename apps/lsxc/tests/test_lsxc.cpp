#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/core.h"

#include <fstream>

TEST_CASE("lsxc core initialization and lifecycle", "[lsxc]") {
    const std::string log_path = "test_lsxc.log";

    lsxcommon::Core::Initialize(log_path);
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();

    std::ifstream log_file(log_path);
    REQUIRE(log_file.good());
    log_file.close();
    std::remove(log_path.c_str());
}
