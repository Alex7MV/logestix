add_executable(test_lsxbenchmark_cli test_lsxbenchmark_cli.cpp ../src/cli.cpp)
target_include_directories(test_lsxbenchmark_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_lsxbenchmark_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_lsxbenchmark_cli)

add_test(NAME test_lsxbenchmark_cli COMMAND test_lsxbenchmark_cli)
set_tests_properties(test_lsxbenchmark_cli PROPERTIES LABELS fast)
set_tests_properties(test_lsxbenchmark_cli PROPERTIES TIMEOUT 120)
