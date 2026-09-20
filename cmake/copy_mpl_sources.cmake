# POST_BUILD helper: copy the libzmq MPL-2.0 Corresponding-Source archive next to
# a target's binary. The MPL-2.0 §3.2 disclosure obligation attaches to the
# distributed Executable Form, so the copy happens only for release-like
# configurations (Release / RelWithDebInfo / MinSizeRel); debug builds no-op.
#
# Invoked from add_custom_command(POST_BUILD) as:
#   cmake -DLSX_CONFIG=<cfg> -DLSX_ZMQ_ARCHIVE=<path> -DLSX_DEST=<dir>
#         -P cmake/copy_mpl_sources.cmake

if(NOT LSX_CONFIG MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
    return()
endif()

if(NOT EXISTS "${LSX_ZMQ_ARCHIVE}")
    message(WARNING "libzmq source archive not found: ${LSX_ZMQ_ARCHIVE}")
    return()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${LSX_ZMQ_ARCHIVE}" "${LSX_DEST}"
    RESULT_VARIABLE _lsx_copy_rc)

if(NOT _lsx_copy_rc EQUAL 0)
    message(WARNING "failed to copy ${LSX_ZMQ_ARCHIVE} to ${LSX_DEST}")
endif()
