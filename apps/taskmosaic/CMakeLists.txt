add_executable(taskmosaic src/main.cpp)
target_link_libraries(taskmosaic PRIVATE lsxcommon arrow_shared arrow_cuda_shared ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:taskmosaic>"
    )
    add_custom_command(TARGET taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:taskmosaic>"
    )
endif()

install(TARGETS taskmosaic RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
