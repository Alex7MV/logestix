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
set_tests_properties(test_taskmosaic PROPERTIES LABELS fast)

add_executable(test_taskmosaic_cli test_taskmosaic_cli.cpp ../src/cli.cpp)
target_include_directories(test_taskmosaic_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_cli>"
    )
    add_custom_command(TARGET test_taskmosaic_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_cli>"
    )
endif()

add_test(NAME test_taskmosaic_cli COMMAND test_taskmosaic_cli)
set_tests_properties(test_taskmosaic_cli PROPERTIES LABELS fast)
