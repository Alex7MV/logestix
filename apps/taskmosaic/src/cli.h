#pragma once

#include <string>

namespace taskmosaic {

/// taskmosaic is a client-only CLI: it talks to a running lsxfabric server.
struct TaskMosaicCliArgs {
    std::string server;  // ZMQ endpoint, required
    std::string model;   // converted model path (tokenizer + framing), required
    std::string prompt;  // empty => read stdin
    int max_gen = 64;    // sent verbatim
};

TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv);

}  // namespace taskmosaic
