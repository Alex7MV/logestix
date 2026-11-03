add_executable(lsxcli src/main.cpp)
target_link_libraries(lsxcli PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson ${CMAKE_DL_LIBS})

lsx_copy_third_party_notices(lsxcli)

lsx_copy_arrow_dlls(lsxcli)

install(TARGETS lsxcli RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
