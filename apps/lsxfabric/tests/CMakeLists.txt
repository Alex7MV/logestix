add_executable(test_lsxfabric test_lsxfabric.cpp)
target_link_libraries(test_lsxfabric PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxfabric>"
    )
    add_custom_command(TARGET test_lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxfabric>"
    )
endif()

add_test(NAME test_lsxfabric COMMAND test_lsxfabric)
