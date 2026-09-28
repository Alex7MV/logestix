#include <catch2/catch_test_macros.hpp>

#include "cli.h"

TEST_CASE("taskmosaic cli: server flags", "[taskmosaic][cli]") {
    const char* argv[] = {"taskmosaic", "--config", "/tmp/c.json",
                          "--host", "0.0.0.0", "--port", "9000"};
    auto a = taskmosaic::ParseTaskMosaicCliArgs(7, const_cast<char**>(argv));
    CHECK(a.config == "/tmp/c.json");
    CHECK(a.host == "0.0.0.0");
    CHECK(a.port == 9000);
}

TEST_CASE("taskmosaic cli: defaults", "[taskmosaic][cli]") {
    const char* argv[] = {"taskmosaic"};
    auto a = taskmosaic::ParseTaskMosaicCliArgs(1, const_cast<char**>(argv));
    CHECK(a.config.empty());
    CHECK(a.host.empty());
    CHECK(a.port == 0);
}
