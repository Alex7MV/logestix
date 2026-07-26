add_executable(test_lsxcli test_lsxcli.cpp)
target_link_libraries(test_lsxcli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxcli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxcli>"
    )
    add_custom_command(TARGET test_lsxcli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxcli>"
    )
endif()

add_test(NAME test_lsxcli COMMAND test_lsxcli)
