add_executable(test_lsxbenchmark_cli test_lsxbenchmark_cli.cpp ../src/cli.cpp)
target_include_directories(test_lsxbenchmark_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_lsxbenchmark_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxbenchmark_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxbenchmark_cli>"
    )
    add_custom_command(TARGET test_lsxbenchmark_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxbenchmark_cli>"
    )
endif()

add_test(NAME test_lsxbenchmark_cli COMMAND test_lsxbenchmark_cli)
set_tests_properties(test_lsxbenchmark_cli PROPERTIES LABELS fast)
set_tests_properties(test_lsxbenchmark_cli PROPERTIES TIMEOUT 120)
