#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    lsxcommon::Core::Initialize("lsxcli_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();
    return 0;
}
