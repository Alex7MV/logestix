add_executable(test_lsxcli test_lsxcli.cpp)
target_link_libraries(test_lsxcli PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_lsxcli)

add_test(NAME test_lsxcli COMMAND test_lsxcli)
set_tests_properties(test_lsxcli PROPERTIES LABELS fast)
