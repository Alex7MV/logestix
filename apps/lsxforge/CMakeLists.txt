add_executable(lsxforge src/main.cpp src/cli.cpp src/codec_profiles.cpp)
target_link_libraries(lsxforge PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson absl::flags_parse absl::flags ${CMAKE_DL_LIBS})

lsx_copy_third_party_notices(lsxforge)

lsx_copy_arrow_dlls(lsxforge)

install(TARGETS lsxforge RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
