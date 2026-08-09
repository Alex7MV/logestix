# LsxModelFactory — Design Spec

## 1. Overview

В проект будут добавляться новые модели. Сейчас весь движок — одна функция
`gpu::RunInference(cfg)` в `libs/lsxcommon/src/inference_pipeline.cu`, жёстко
заточенная под DeepSeek-V4-Flash (константы геометрии в `inference_load.cuh`:
`kHiddenDim=4096`, `kNumLayers=43`, `kNumRouted=256` и т.д.).

Этот спринт создаёт фабричный паттерн для моделей:

- `IModel` — абстрактный интерфейс-«продукт» (Load / Infer / Unload / Inspect / SelfCheck).
- `LsxModelFactory` — фабрика со статическим реестром реализаций.
- `DeepSeekV4FlashModel` — текущая единственная модель, перенесённая в интерфейс
  как обёртка над существующим движком (без параметризации констант — они остаются
  в `inference_load.cuh`).
- Все приложения (`lsxfabric`, `lsxcli`) и тесты переходят на работу через фабрику;
  `RunInference` / `RunSelfCheck` удаляются из публичного API.

## 2. Constraints

- **Без параметризации движка.** Константы геометрии (`kHiddenDim`, `kNumLayers`,
  MLA/HC/CSA) остаются в `inference_load.cuh` и общем для движка виде. Новая модель
  другой архитектуры в этом спринте не добавляется; проектируется только точка
  расширения.
- **Полная замена API.** `gpu::RunInference` / `gpu::RunSelfCheck` удаляются из
  публичного API. Всё переводится на `LsxModelFactory`.
- **Статический реестр.** Новые модели добавляются записью в центральный файл
  `model_registry.cpp` (2-3 модели — статический список обозрим).
- **Гибридный выбор типа:** явный `ModelInitConfig.model_type` либо autodetect из
  поля `model` файла `logestix/model.index.json`.
- **Вся работа с моделями — через фабрику.** Сканирование safetensors (`--info`)
  тоже доступно только через `IModel::Inspect()`.

## 3. Public API

Новый заголовок `libs/lsxcommon/include/lsxcommon/model.h`:

```cpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb
#include "lsxcommon/safetensors.h"     // SafetensorsModel

namespace lsxcommon {

struct ModelInitConfig {
    std::string model_path;   // обязательно
    std::string model_type;   // пусто = autodetect из model.index.json
    int max_gen_tokens = 64;
};

struct ModelRequest {
    std::vector<int32_t> input_ids;   // промпт, уже токенизированный
    int max_gen_tokens = 64;
};

class IModel {
public:
    virtual ~IModel() = default;
    virtual bool Load() = 0;
    virtual gpu::InferenceResult Infer(const ModelRequest &req) = 0;
    virtual void Unload() = 0;
    virtual SafetensorsModel Inspect() = 0;
    virtual gpu::InferenceResult SelfCheck(const ModelRequest &req) = 0;
    virtual const std::string &Type() const = 0;
    virtual ArrowTokenizerDb &Tokenizer() = 0;
};

} // namespace lsxcommon
```

Примечания:

- Токенизация выполняется **вне** модели: `ModelRequest.input_ids` приходит готовым.
  `IModel::Tokenizer()` отдаёт токенизатор, чтобы caller (приложение/тест) мог
  токенизировать промпт тем же токенизатором, что использует модель.
- `Inspect()` возвращает `SafetensorsModel` — существующий тип из `safetensors.h`
  (то, что возвращает `scan_directory`).
- `SelfCheck()` обязателен (модель должна уметь самодиагностику); для моделей без
  такой поддержки реализация возвращает `gpu::InferenceResult{ok=false, error_msg}`.

## 4. DeepSeekV4FlashModel

Новый файл `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` + приватный заголовок
`libs/lsxcommon/src/deepseek_v4_flash_model.h`:

```cpp
namespace lsxcommon {

class DeepSeekV4FlashModel final : public IModel {
public:
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    const std::string &Type() const override { return kType; }
    ArrowTokenizerDb &Tokenizer() override;

    static constexpr const char *kType = "deepseek-v4-flash";

private:
    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    bool loaded_ = false;
};

} // namespace lsxcommon
```

Реализация:

- `Load()` — готовит `ArrowTokenizerDb` и материалы, общие для Infer/SelfCheck
  (в текущем `RunInference` токенизатор создаётся на каждый вызов; `Load` делает
  это один раз).
- `Infer(req)` — вызывает внутренний движок:
  `gpu::InferenceResult RunInferenceModel(model_path, req.input_ids, max_gen_tokens, *tok_db_)`.
- `SelfCheck(req)` — аналогично через `gpu::RunSelfCheckModel(...)`.
- `Inspect()` — вызывает `lsxcommon::scan_directory(cfg_.model_path)`. Это
  единственное место, где вызывается утилита сканирования safetensors.
- `Unload()` — освобождает ресурсы (`cudaFree` pinned весов и т.д.), сбрасывает `loaded_`.

## 5. Рефакторинг движка (inference_pipeline.cu)

`RunInference(cfg)` / `RunSelfCheck(cfg)` переписываются во внутренние функции с
новой сигнатурой (остаются в том же файле, namespace `lsxcommon::gpu`):

```cpp
// Входной промпт приходит токенизированным; токенизатор — из IModel.
gpu::InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);

gpu::InferenceResult RunSelfCheckModel(
    const std::string &model_path,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
```

Изменения внутри тела `RunInference`:

- Убрать жёстко закодированный промпт и его токенизацию (строки ~277):
  `input_ids` теперь параметр.
- Принимать `ArrowTokenizerDb&` вместо загрузки его заново.
- Всё остальное (аллоцирование буферов, цикл по слоям, генерация, dump-хендуки,
  env-флаги `LSX_*`) не трогается.

Изменения внутри тела `RunSelfCheck`:

- Аналогично: без повторной загрузки токенизатора, получает `tok_db` извне.

`InferenceConfig` / `InferenceResult` остаются в `inference_gpu.h` как внутренние
типы движка (используются `DeepSeekV4FlashModel::Infer`).

## 6. LsxModelFactory + статический реестр

Новый заголовок `libs/lsxcommon/include/lsxcommon/model_factory.h`:

```cpp
#pragma once
#include <functional>
#include <memory>
#include <string>
#include "lsxcommon/model.h"

namespace lsxcommon {

using CreateModelFn = std::function<std::unique_ptr<IModel>(const ModelInitConfig &)>;

class LsxModelFactory {
public:
    static void Register(const std::string &type, CreateModelFn fn);
    static std::unique_ptr<IModel> Create(const ModelInitConfig &cfg);
    static std::string DetectType(const std::string &model_path);
};

// Файл model_registry.cpp: гарантированная инициализация реестра перед Create.
void EnsureModelRegistry();

// Помощник регистрации.
bool RegisterModel(const char *type, CreateModelFn fn);

} // namespace lsxcommon
```

Реализация `libs/lsxcommon/src/model_factory.cpp`:

- Реестр — `std::unordered_map<std::string, CreateModelFn>` (синглтон через
  static-local function).
- `DetectType(path)` — читает `logestix/model.index.json`, возвращает поле `model`
  (например `"DeepSeek-V4-Flash"`), либо пустую строку при ошибке. Парсинг
  переиспользует существующий simdjson-код; выделяется отдельная helper-функция
  `detect_model_type(path)` в `inference_load.cu`, вызываемая из `DetectType`.
- `Create(cfg)`:
  1. `type = cfg.model_type`; если пусто → `cfg.model_type = DetectType(cfg.model_path)`.
  2. `EnsureModelRegistry()` — гарантирует, что записи зарегистрированы (callback
     на front of `Create`; не полагаемся на порядок инициализации статиков при
     STATIC-линковке).
  3. Поиск типа в реестре; нет записи → `nullptr` + `spdlog::error`.
  4. Возвращает `fn(cfg)`.

Карта типа в имя модели (DetectType):

- `model.index.json` `"model": "DeepSeek-V4-Flash"` → тип `"deepseek-v4-flash"`
  (нижний регистр, дефис вместо пробела). Нормализация: lowercase + замена
  не-`[a-z0-9]` на `-`, trim возможных суффиксов при необходимости (уточняется при
  реализации — см. раздел «Open items»).

Файл `libs/lsxcommon/src/model_registry.cpp`:

```cpp
#include "lsxcommon/model_factory.h"
#include "deepseek_v4_flash_model.h"

namespace lsxcommon {
namespace {
const bool kRegDeepSeekV4Flash = RegisterModel(
    DeepSeekV4FlashModel::kType,
    [](const ModelInitConfig &cfg) { return std::make_unique<DeepSeekV4FlashModel>(cfg); });
} // namespace

void EnsureModelRegistry() {
    static bool init = [] {
        (void)kRegDeepSeekV4Flash;   // touch: держит запись при STATIC-линковке
        return true;
    }();
    (void)init;
}
} // namespace lsxcommon
```

## 7. Обновление приложений

### apps/lsxfabric/src/main.cpp

Замена прямых вызовов `gpu::RunInference(cfg)` / `gpu::RunSelfCheck(cfg)`:

```cpp
lsxcommon::ModelInitConfig cfg{model_path, /*model_type=*/"", /*max_gen_tokens=*/64};
auto model = lsxcommon::LsxModelFactory::Create(cfg);
if (!model || !model->Load()) {
    spdlog::error("Failed to create/load model: {}", model_path);
    return 1;
}
auto &tok = model->Tokenizer();
// токенизация промпта (как в текущем testprompt-коде, через tok),
// результат → std::vector<int32_t> input_ids
lsxcommon::ModelRequest req{std::move(input_ids), 64};
auto result = model->Infer(req);
spdlog::info(... result.output_text, result.tokens_per_sec ...);
model->Unload();
```

- `--localmodel`, `--testprompt`, `--selfcheck` флаги и их значение не меняются.
- Для `--selfcheck`:
  ```cpp
  lsxcommon::ModelRequest req{std::move(prompt_ids), 1};
  auto result = model->SelfCheck(req);
  ```
  (`RunSelfCheck` в текущем коде заворачивается в `RunInference` с `max_gen_tokens=1`
  — воспроизводим это внутри `SelfCheck`).

### apps/lsxcli/src/main.cpp

Переходит на фабрику полностью (`--info` / `--infofull`):

```cpp
lsxcommon::ModelInitConfig cfg{model_path, /*model_type=*/"", /*max_gen_tokens=*/64};
auto model = lsxcommon::LsxModelFactory::Create(cfg);
if (!model || !model->Load()) {
    spdlog::error("Failed to create/load model: {}", model_path);
    return 1;
}
if (info || infofull) {
    auto mi = model->Inspect();
    if (infofull)
        for (auto &shard : mi.shards)
            lsxcommon::print_shard(shard);
    lsxcommon::print_model(mi);
}
model->Unload();
```

- `scan_directory` / `print_shard` / `print_model` остаются публичными утилитами
  в `safetensors.h`, но вызываются отныне только через `IModel::Inspect()`.

## 8. Тесты

Новый файл `tests/test_model_factory.cpp` (Catch2, добавляется в CMake наравне с
`test_core.cpp`):

1. `DetectType(valid_path)` → `deepseek-v4-flash` (для существующего модельного пути).
2. `DetectType(invalid_path)` → пустая строка, без исключений.
3. `Create({model_type="unknown-type"})` → `nullptr`.
4. Registry round-trip: `Register("mock", ...)` + `Create` возвращает экземпляр.
5. `Create(valid_path)` → непустой `IModel`, `Type() == "deepseek-v4-flash"`,
   `Load()` ок. `Inspect()` возвращает ненулевое число shards. (Не требует GPU —
   сканирование файлов). Для скипа при отсутствии модели — guard на существование
   `logestix/model.index.json`.

Существующие тесты (`tests/test_core.cpp` и др.) не меняются, кроме необходимости
перекомпиляции: они не вызывают `RunInference`/`RunSelfCheck`.

## 9. CMake

- `libs/lsxcommon/CMakeLists.txt`: новые исходники в `lsxcommon`:
  `src/model_factory.cpp`, `src/model_registry.cpp`,
  `src/deepseek_v4_flash_model.cpp`.
- `tests/CMakeLists.txt`: новый тестовый таргет `test_model_factory` на `tests/test_model_factory.cpp`, линкует `lsxcommon` (как существующие тесты).
- Header-only API: `model.h`, `model_factory.h` — в `target_include_directories`
  PUBLIC include `libs/lsxcommon/include` (соответствует существующей структуре).

## 10. Error handling

- Без исключений в публичном контракте `IModel`.
  - `Load()` → `bool`; при ошибке пишет `spdlog::error` и возвращает `false`.
  - `Infer()` / `SelfCheck()` → `gpu::InferenceResult`; при ошибке
    `result.ok == false` + `result.error_msg`.
  - `Create()` → `nullptr` при неизвестном типе / отсутствии `model.index.json`.
  - `DetectType()` → пустая строка при ошибке (не бросает).
- Каскад в приложениях: проверять `Create()` на `nullptr`, `Load()` на `false`,
  `Infer()` на `result.ok` — на каждой стадии.

## 11. Non-goals

- Параметризация констант геометрии движка (вторая модель этой архитектуры с
  другими размерами сюда не входит).
- Новые модели (тестирование фабрики на реальной второй модели).
- Нетоки: `temperature`/`top_p`/семплинг поверх argmax, batching, серверная обвязка.
- Изменение формата `logestix/model.index.json`.

## 12. Open items (решаются при реализации)

- Точная нормализация `model` → `type`: пока `"DeepSeek-V4-Flash"` → `"deepseek-v4-flash"`
  (lowercase, ` `→`-`, `-`→`-`). Если в реальных моделях будут другие паттерны —
  уточнить таблицу преобразования.
- Токенизация промпта в `lsxfabric` testprompt: текущий код токенизирует hardcoded
  промпт; текст промпта можно оставить там же, но через `tok` из `IModel::Tokenizer()`.
- Домашний каталог моделей для тестов (переменная окружения `LSX_TEST_MODEL_PATH`,
  fallback на `D:\Programs\huggingface.co\DeepSeek-V4-Flash`).