add_executable(lsxfabric src/main.cpp src/cli.cpp src/validator.cpp src/fabric_server.cpp
    src/explain_artifact.cpp)
target_link_libraries(lsxfabric PRIVATE lsxcommon lsxnetwork arrow_shared arrow_cuda_shared simdjson::simdjson ${CMAKE_DL_LIBS})

lsx_copy_third_party_notices(lsxfabric)

lsx_copy_arrow_dlls(lsxfabric)

install(TARGETS lsxfabric RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
