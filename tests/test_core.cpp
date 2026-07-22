#include "lsxfabric/core.h"
#include "lsxfabric/version.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("core_version returns non-empty string", "[core]")
{
    const char* v = core_version();
    REQUIRE(v != nullptr);
    REQUIRE(v[0] != '\0');
}

TEST_CASE("core_init succeeds", "[core]")
{
    REQUIRE(core_init() == 0);
}
