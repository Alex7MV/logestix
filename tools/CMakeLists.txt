# Offline cross-model translation passport builder. Standalone utility that
# links lsxnetwork for the shared passport format + writer (WritePassport) and
# spdlog for logging. lsxnetwork does not pull Arrow, so no DLL staging needed.
add_executable(lsx_token_packer lsx_token_packer.cpp)
target_link_libraries(lsx_token_packer PRIVATE lsxnetwork spdlog::spdlog)

install(TARGETS lsx_token_packer RUNTIME DESTINATION bin)