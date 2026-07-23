#include "lsxfabric/core.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    // Запускаем подсистему логирования и ядра
    lsxfabric::Core::Initialize("logestix_runtime.log");

    // Вызываем тестовый инференс
    lsxfabric::Core::ExecuteDummyInference();

    // Корректно завершаем работу приложения
    lsxfabric::Core::Shutdown();
    return 0;
}
