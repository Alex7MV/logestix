add_executable(test_lsxc test_lsxc.cpp)
target_link_libraries(test_lsxc PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxc>"
    )
    add_custom_command(TARGET test_lsxc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxc>"
    )
endif()

add_test(NAME test_lsxc COMMAND test_lsxc)
