#include <catch2/catch_test_macros.hpp>
#include "lsxfabric/core.h"
#include <spdlog/spdlog.h>

TEST_CASE("Core lifecycle and logging initialization", "[core][logging]") {
    // Проверяем, что инициализация ядра проходит без генерации исключений (exceptions)
    REQUIRE_NOTHROW(lsxfabric::Core::Initialize("test_output.log"));
    
    // Проверяем, что глобальный логгер успешно зарегистрирован в рантайме
    auto current_logger = spdlog::default_logger();
    REQUIRE(current_logger != nullptr);
    REQUIRE(current_logger->name() == "lsx_logger");

    // Проверяем вызов логов
    REQUIRE_NOTHROW(lsxfabric::Core::ExecuteDummyInference());

    // Освобождаем ресурсы логгера
    REQUIRE_NOTHROW(lsxfabric::Core::Shutdown());
}
