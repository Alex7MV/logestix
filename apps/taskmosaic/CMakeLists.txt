add_executable(taskmosaic
    src/main.cpp
    src/cli.cpp
    src/config.cpp
    src/backend.cpp
    src/chat_bridge.cpp
    src/lsx_framer.cpp)
target_link_libraries(taskmosaic PRIVATE lsxcommon lsxnetwork simdjson::simdjson arrow_shared arrow_cuda_shared ${CMAKE_DL_LIBS})

lsx_copy_third_party_notices(taskmosaic)

lsx_copy_arrow_dlls(taskmosaic)

install(TARGETS taskmosaic RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
