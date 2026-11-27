add_executable(lsxforge src/main.cpp src/cli.cpp src/codec_profiles.cpp src/model_identity.cpp)
# model_identity.cpp reads the generated manifest table (model_manifest.h) so a
# model's family/model_type come from the single source of truth rather than
# from a filename.
target_include_directories(lsxforge PRIVATE "${LSX_MANIFEST_GENERATED_DIR}")
target_link_libraries(lsxforge PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson absl::flags_parse absl::flags ${CMAKE_DL_LIBS})

lsx_copy_third_party_notices(lsxforge)

lsx_copy_arrow_dlls(lsxforge)

install(TARGETS lsxforge RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
