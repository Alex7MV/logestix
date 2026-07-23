#ifndef LSXFABRIC_CORE_H
#define LSXFABRIC_CORE_H

#include <string>

namespace lsxfabric {

class Core {
public:
    // Инициализация инфраструктуры (логирование, аллокаторы памяти)
    static void Initialize(const std::string& log_file_path = "logestix.log");
    
    // Корректное завершение работы (сброс буферов логов на диск)
    static void Shutdown();

    // Пример бизнес-логики инференса для проверки окружения
    static void ExecuteDummyInference();
};

} // namespace lsxfabric

#endif // LSXFABRIC_CORE_H
