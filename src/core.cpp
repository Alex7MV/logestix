#include "lsxfabric/core.h"

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <memory>
#include <vector>

namespace lsxfabric {

void Core::Initialize(const std::string& log_file_path) {
    try {
        // 1. Инициализируем глобальный пул потоков для асинхронного логирования.
        // Очередь на 8192 сообщения, 1 фоновый поток на запись.
        spdlog::init_thread_pool(8192, 1);

        // 2. Создаем sink (приемник) для вывода в консоль с цветовым выделением уровней
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        
        // 3. Создаем sink для записи в файл с автоматической ротацией.
        // Максимальный размер файла — 5 МБ, хранить до 3 старых архивов.
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file_path, 1024 * 1024 * 5, 3
        );

        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};

        // 4. Создаем асинхронный логгер, объединяющий оба источника вывода
        auto logger = std::make_shared<spdlog::async_logger>(
            "lsx_logger", sinks.begin(), sinks.end(), 
            spdlog::thread_pool(), spdlog::async_overflow_policy::block
        );

        // Настраиваем enterprise-формат: [Дата Время.Миллисекунды] [Имя логгера] [ID потока] [Уровень] Сообщение
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");
        
        // Устанавливаем минимальный уровень логирования (по умолчанию INFO, для дебага можно сменить на DEBUG)
        logger->set_level(spdlog::level::info);

        // Регистрируем логгер как глобальный по умолчанию
        spdlog::set_default_logger(logger);

        spdlog::info("Logestix Fabric core logging system initialized successfully.");
    } 
    catch (const spdlog::spdlog_ex& ex) {
        // Если логирование не поднялось (например, нет прав на запись файла), выводим ошибку в стандартный stderr
        fprintf(stderr, "Critical: Log initialization failed: %s\n", ex.what());
    }
}

void Core::Shutdown() {
    spdlog::info("Shutting down Logestix Fabric core engine...");
    // Принудительно сбрасываем все недозаписанные логи из очереди на диск перед выходом
    spdlog::shutdown();
}

void Core::ExecuteDummyInference() {
    spdlog::debug("Starting execution pipeline..."); // Не выведется при уровне INFO
    spdlog::info("Processing target inference batch...");
    
    // Имитация предупреждения, если что-то пошло не по оптимальному пути на GPU
    spdlog::warn("Hardware allocation high utilization detected, falling back to stream pooling.");
}

} // namespace lsxfabric
