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
target_compile_definitions(lsxfabric_objects PUBLIC
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

target_include_directories(lsxfabric_objects
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/thrust>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/cub>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/arrow-build/src>"
)

# Линкуем Apache Arrow и spdlog с объектами ядра. Модификатор PUBLIC
# автоматически делает заголовочные файлы и библиотеки доступными
# для lsxfabric и test_core.
target_link_libraries(lsxfabric_objects PUBLIC
    arrow_shared
    spdlog::spdlog
    flatbuffers
    libzmq-static
    simdjson::simdjson
)

# ── Исполняемый файл ─────────────────────────────────────────────────────────
#
# main.cpp — единственный файл, уникальный для релизной сборки.
# core.cpp уже входит в lsxfabric_objects.
#
add_executable(lsxfabric main.cpp)
target_link_libraries(lsxfabric PRIVATE lsxfabric_objects ${CMAKE_DL_LIBS})

# Копируем arrow.dll в директорию с исполняемым файлом
if(WIN32)
    add_custom_command(TARGET lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxfabric>"
        COMMAND_EXPAND_LISTS
    )
endif()

install(TARGETS lsxfabric RUNTIME DESTINATION bin)
