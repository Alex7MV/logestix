add_executable(lsxforge src/main.cpp)
target_link_libraries(lsxforge PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxforge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxforge>"
    )
    add_custom_command(TARGET lsxforge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxforge>"
    )
endif()

install(TARGETS lsxforge RUNTIME DESTINATION bin)
