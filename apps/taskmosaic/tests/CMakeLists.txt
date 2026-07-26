add_executable(test_taskmosaic test_taskmosaic.cpp)
target_link_libraries(test_taskmosaic PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic>"
    )
    add_custom_command(TARGET test_taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic>"
    )
endif()

add_test(NAME test_taskmosaic COMMAND test_taskmosaic)
