add_executable(lsxfabric src/main.cpp)
target_link_libraries(lsxfabric PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxfabric>"
    )
    add_custom_command(TARGET lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxfabric>"
    )
endif()

install(TARGETS lsxfabric RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
