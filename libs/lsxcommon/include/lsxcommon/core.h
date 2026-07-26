#ifndef LSXCOMMON_CORE_H
#define LSXCOMMON_CORE_H

#include <string>

namespace lsxcommon {

class Core {
public:
    static void Initialize(const std::string& log_file_path = "logestix.log");
    static void Shutdown();
    static void ExecuteDummyInference();
};

} // namespace lsxcommon

#endif // LSXCOMMON_CORE_H
