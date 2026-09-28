#pragma once

#include <string>

namespace taskmosaic {

/// taskmosaic is the OpenAI-compatible HTTP bridge to a running lsxfabric.
struct TaskMosaicCliArgs {
    std::string config;  // required: bridge JSON config path
    std::string host;    // optional override of listen.host
    int port = 0;        // optional override of listen.port (0 = keep config)
};

TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv);

}  // namespace taskmosaic
