add_executable(test_lsxfabric test_lsxfabric.cpp ../src/validator.cpp)
target_sources(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_validator.cpp)
target_include_directories(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_lsxfabric PRIVATE lsxcommon simdjson::simdjson Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_lsxfabric)

add_test(NAME test_lsxfabric COMMAND test_lsxfabric)
set_tests_properties(test_lsxfabric PROPERTIES LABELS fast)

add_executable(test_fabric_cli test_fabric_cli.cpp ../src/cli.cpp)
target_sources(test_fabric_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_fabric_cli_edges.cpp)
target_include_directories(test_fabric_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_fabric_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_fabric_cli)

add_test(NAME test_fabric_cli COMMAND test_fabric_cli)
set_tests_properties(test_fabric_cli PROPERTIES LABELS fast)
set_tests_properties(test_fabric_cli PROPERTIES TIMEOUT 120)

add_executable(test_fabric_server test_fabric_server.cpp ../src/fabric_server.cpp)
target_include_directories(test_fabric_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_fabric_server PRIVATE lsxcommon lsxnetwork arrow_shared arrow_cuda_shared Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_fabric_server)

add_test(NAME test_fabric_server COMMAND test_fabric_server)
set_tests_properties(test_fabric_server PROPERTIES LABELS fast)
set_tests_properties(test_fabric_server PROPERTIES TIMEOUT 120)

# --explain-artifact (design §2.7): the selection report, formatted by the app
# and tested without a GPU or a model load.
add_executable(test_explain_artifact test_explain_artifact.cpp
    ../src/explain_artifact.cpp)
target_include_directories(test_explain_artifact PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_explain_artifact PRIVATE lsxcommon lsxstorage
    Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_explain_artifact)
add_test(NAME test_explain_artifact COMMAND test_explain_artifact)
set_tests_properties(test_explain_artifact PROPERTIES LABELS fast)
