add_executable(lsxcli src/main.cpp)
target_link_libraries(lsxcli PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxcli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxcli>"
    )
    add_custom_command(TARGET lsxcli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxcli>"
    )
endif()

install(TARGETS lsxcli RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
