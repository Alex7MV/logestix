# Fabric Validator Design

## Purpose

`lsxfabric` --validate запускает проверку целостности logestix-модели:
индекс, pinned тензоры, MoE эксперты (NVFP4 кодировка, E4M3 scales,
global scales).

## CLI

```
lsxfabric --localmodel PATH --validate
```

Новый флаг `--validate`. Exit code 0 при успехе, 1 при ошибках.

## Modules

Новый файл `apps/lsxfabric/src/validator.h` / `validator.cpp`.

### Structs

```cpp
struct ModelInfo {
    std::string format;
    std::string model_name;
    int num_layers;
    int experts_per_layer;
    int64_t total_pinned;
    int64_t total_experts;
};

struct ValidationResult {
    bool ok = true;
    int errors = 0;
    int warnings = 0;
};
```

### Functions

```cpp
// Парсинг model.index.json
Result<ModelInfo> LoadModelIndex(const std::string &index_path);

// Загрузка pinned .arrow, проверка схемы, логирование
ValidationResult ValidatePinnedTensors(const std::string &pinned_dir,
                                       int64_t expected_count);

// Загрузка одного эксперта, проверка всех полей
ValidationResult ValidateExpert(const std::string &arrow_path,
                                int layer, int expert);

// NVFP4 data validation: каждый nibble ∈ [0x00, 0x0F]
// Нет невалидных кодов
bool ValidateNVFP4Data(const uint8_t *data, int64_t nbytes);

// E4M3 scale validation: sign=0, exp4 ∈ [1, 14]
bool ValidateE4M3Scales(const uint8_t *scales, int64_t count);

// Global scale: finite, > 0
bool ValidateGlobalScale(float scale);

// Полная валидация
ValidationResult RunValidator(const std::string &model_path);
```

### Поток RunValidator

1. Читает `model.index.json` → `ModelInfo`
2. Проверяет формат `logestix-v1`
3. Валидирует `pinned/`:
   - Загружает `pinned/global.arrow`
   - Для каждого слоя загружает `pinned/layer_NNN.arrow`
   - Проверяет что RecordBatch читается без ошибок
4. Валидирует экспертов:
   - Для каждого слоя, для первого эксперта (exp_000.arrow):
     - Читает RecordBatch
     - Проверяет 9 полей в схеме
     - Валидирует NVFP4 данные w1/w2/w3
     - Валидирует E4M3 scale массивы w1/w2/w3
     - Валидирует global_scale (float) w1/w2/w3
5. Сводка: сколько экспертов проверено, сколько ошибок, сколько pinned

### main.cpp

```cpp
if (validate) {
    auto result = RunValidator(model_path);
    return result.ok ? 0 : 1;
}
```

### Testing

Новый `apps/lsxfabric/tests/test_validator.cpp` с Catch2:
- `LoadModelIndex` с синтетическим index.json
- `ValidateNVFP4Data` с корректными/некорректными данными
- `ValidateE4M3Scales` с корректными/некорректными байтами
- `ValidateGlobalScale` с 0, negative, inf, normal
