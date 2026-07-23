# ── Объектная библиотека ядра ────────────────────────────────────────────────
#
# core.cpp компилируется один раз, а объектные файлы переиспользуются
# в основной программе lsxfabric и в исполняемом файле тестов test_core.
# Это исключает двойную компиляцию и гарантирует идентичность ABI.
#
add_library(lsxfabric_objects OBJECT core.cpp)

# Публичные инклюды наследуются всеми целями, которые линкуются
# с lsxfabric_objects (через target_link_libraries).
#
target_include_directories(lsxfabric_objects
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/thrust>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/cub>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include>"
)

# Линкуем Apache Arrow и spdlog с объектами ядра. Модификатор PUBLIC
# автоматически делает заголовочные файлы и библиотеки доступными
# для lsxfabric и test_core.
target_link_libraries(lsxfabric_objects PUBLIC
    arrow_shared
    spdlog::spdlog
)

# ── Исполняемый файл ─────────────────────────────────────────────────────────
#
# main.cpp — единственный файл, уникальный для релизной сборки.
# core.cpp уже входит в lsxfabric_objects.
#
add_executable(lsxfabric main.cpp)
target_link_libraries(lsxfabric PRIVATE lsxfabric_objects ${CMAKE_DL_LIBS})

install(TARGETS lsxfabric RUNTIME DESTINATION bin)
