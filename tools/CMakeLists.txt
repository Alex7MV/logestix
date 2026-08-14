# Offline cross-model translation passport builder. Standalone utility that
# links lsxnetwork for the shared passport format + writer (WritePassport) and
# lsxcommon for the centralized model-metadata vault root (path_registry).
# spdlog for logging. Arrow objects go unreferenced by this tool (they are
# discarded by the static linker), but the DLLs are staged on Windows to keep
# the executable robust regardless of link-ordering details.
add_executable(lsx_token_packer lsx_token_packer.cpp)
target_link_libraries(lsx_token_packer PRIVATE lsxnetwork lsxcommon spdlog::spdlog)

if(WIN32)
    add_custom_command(TARGET lsx_token_packer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>" "$<TARGET_FILE_DIR:lsx_token_packer>"
    )
    add_custom_command(TARGET lsx_token_packer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>" "$<TARGET_FILE_DIR:lsx_token_packer>"
    )
endif()

install(TARGETS lsx_token_packer RUNTIME DESTINATION bin)