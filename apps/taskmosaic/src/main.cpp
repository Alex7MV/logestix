#include "lsxcommon/core.h"
#include "lsxcommon/common.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    lsxcommon::Core::Initialize("taskmosaic_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    lsxcommon::Core::Shutdown();
    return 0;
}
