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
set_tests_properties(test_lsxforge PROPERTIES LABELS fast)

# Pipeline dispatch (host-only): the flag-to-pipeline precedence that main()
# used to carry inline.
add_executable(test_lsxforge_cli test_lsxforge_cli.cpp ../src/cli.cpp)
target_include_directories(test_lsxforge_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_lsxforge_cli PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_lsxforge_cli COMMAND test_lsxforge_cli)
set_tests_properties(test_lsxforge_cli PROPERTIES LABELS fast)
