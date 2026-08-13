find_package(Threads REQUIRED)

add_library(lsxnetwork STATIC
    src/affinity.cpp
    src/frame_queue.cpp
    src/io_worker.cpp
    src/lsx_token_translator.cpp
    src/openai_endpoint.cpp
    src/zmq_context.cpp
)

target_include_directories(lsxnetwork
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

# simdjson is referenced by lsxnetwork's public header (openai_endpoint.h), so it
# must be PUBLIC. libzmq-static and Threads must be PUBLIC as well: lsxnetwork is
# a STATIC library, so its link dependencies are only re-exported to consumers
# through the interface (otherwise zmq_*/pthread symbols go unresolved in the
# final executable).
target_link_libraries(lsxnetwork PUBLIC
    simdjson::simdjson
    libzmq-static
    Threads::Threads
)

# spdlog is header-only (FetchContent default), so it cannot leak link symbols;
# PRIVATE keeps its include dirs out of the public interface.
target_link_libraries(lsxnetwork PRIVATE
    spdlog::spdlog
)

# Static libzmq on Windows depends on Winsock/IP helper libraries.
if(WIN32)
    target_link_libraries(lsxnetwork PUBLIC
        ws2_32
        iphlpapi
    )
endif()
