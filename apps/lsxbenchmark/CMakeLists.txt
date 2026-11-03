add_executable(lsxbenchmark src/main.cpp src/cli.cpp)
target_link_libraries(lsxbenchmark PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson ${CMAKE_DL_LIBS})

lsx_copy_third_party_notices(lsxbenchmark)

lsx_copy_arrow_dlls(lsxbenchmark)

install(TARGETS lsxbenchmark RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
