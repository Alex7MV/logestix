# ── Catch2 — загрузка через FetchContent ─────────────────────────────────────
#
# Catch2 v3.x скачивается и собирается автоматически.
# Не требуется предустановленных пакетов ни на Ubuntu, ни на Windows.
#
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.0
)

FetchContent_MakeAvailable(Catch2)

# ── Исполняемый файл тестов ──────────────────────────────────────────────────
#
# Линкуется с объектной библиотекой lsxfabric_objects (core.cpp)
# и с Catch2::Catch2WithMain (предоставляет main()).
#
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE lsxfabric_objects Catch2::Catch2WithMain)

add_test(NAME test_core COMMAND test_core)
