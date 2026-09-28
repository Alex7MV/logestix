#include <catch2/catch_test_macros.hpp>

#include "backend.h"

using taskmosaic::Backend;
using taskmosaic::BackendConfig;

TEST_CASE("backend: admission is bounded by queue_depth", "[taskmosaic][backend]") {
    BackendConfig cfg;
    cfg.endpoint = "tcp://127.0.0.1:1";
    Backend b(cfg, 2);
    CHECK(b.TryAdmit());
    CHECK(b.TryAdmit());
    CHECK_FALSE(b.TryAdmit());  // third exceeds depth
    b.Release();
    CHECK(b.TryAdmit());
}

TEST_CASE("backend: request ids are unique and increasing",
          "[taskmosaic][backend]") {
    BackendConfig cfg;
    Backend b(cfg, 1);
    const auto a = b.NextRequestId();
    const auto c = b.NextRequestId();
    CHECK(c > a);
}
