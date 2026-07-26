add_executable(test_lsxforge test_lsxforge.cpp)
target_link_libraries(test_lsxforge PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxforge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxforge>"
    )
    add_custom_command(TARGET test_lsxforge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxforge>"
    )
endif()

add_test(NAME test_lsxforge COMMAND test_lsxforge)
