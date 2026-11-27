add_executable(test_lsxforge test_lsxforge.cpp)
target_link_libraries(test_lsxforge PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_lsxforge)

add_test(NAME test_lsxforge COMMAND test_lsxforge)
set_tests_properties(test_lsxforge PROPERTIES LABELS fast)

# Pipeline dispatch (host-only): the flag-to-pipeline precedence that main()
# used to carry inline.
add_executable(test_lsxforge_cli test_lsxforge_cli.cpp ../src/cli.cpp)
target_include_directories(test_lsxforge_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_lsxforge_cli PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_lsxforge_cli COMMAND test_lsxforge_cli)
set_tests_properties(test_lsxforge_cli PROPERTIES LABELS fast)

# Codec profiles (host-only): --codec-profile parsing and the declared-plane
# table that the emitter checks against what actually landed on disk.
add_executable(test_forge_codec_profile
    test_forge_codec_profile.cpp
    ../src/codec_profiles.cpp)
target_include_directories(test_forge_codec_profile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_forge_codec_profile PRIVATE
    lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_forge_codec_profile COMMAND test_forge_codec_profile)
set_tests_properties(test_forge_codec_profile PROPERTIES LABELS fast)

# Model identity: vault dir -> manifest (family, model_type) -> profile dir.
# Reads the generated manifest table, so it needs the same include dir as the app.
add_executable(test_model_identity test_model_identity.cpp ../src/model_identity.cpp)
target_include_directories(test_model_identity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    "${LSX_MANIFEST_GENERATED_DIR}")
target_link_libraries(test_model_identity PRIVATE lsxstorage Catch2::Catch2WithMain)
add_test(NAME test_model_identity COMMAND test_model_identity)
set_tests_properties(test_model_identity PROPERTIES LABELS fast)
