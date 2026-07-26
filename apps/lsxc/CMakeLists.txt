add_executable(lsxc src/main.cpp)
target_link_libraries(lsxc PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxc>"
    )
    add_custom_command(TARGET lsxc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxc>"
    )
endif()

install(TARGETS lsxc RUNTIME DESTINATION bin)
