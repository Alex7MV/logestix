add_executable(lsxbenchmark src/main.cpp src/cli.cpp)
target_link_libraries(lsxbenchmark PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxbenchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxbenchmark>"
    )
    add_custom_command(TARGET lsxbenchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxbenchmark>"
    )
endif()

install(TARGETS lsxbenchmark RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
