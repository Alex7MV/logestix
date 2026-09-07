# Qwen3.6-35B-A3B Ingestion Design

Date: 2026-09-07
Status: Approved
Goal: Адаптация Qwen3.6-35B-A3B по Model Ingestion Protocol (8 этапов), автономной
от других моделей, с переиспользованием qwen_moe family-инфраструктуры и без новых
CUDA-ядер. Реализация через TDD в ветке `feat/qwen36-35b-a3b`.

## Разведка (2026-09-07)

Источник: `/mnt/llm_models/models/Qwen3.6-35B-A3B` — 67 GB, BF16, 26 шардов
(`model-00001-of-00026.safetensors`), полный токенизатор (vocab.json, merges.txt,
tokenizer.json, tokenizer_config.json).

- `architectures = Qwen3_5MoeForConditionalGeneration`, `model_type = qwen3_5_moe` —
  архитектурно идентична Qwen3.5-35B-A3B (уже интегрирована как `qwen3.5-35b-a3b`).
- Геометрия text_config **байт-в-байт совпадает** с 3.5: hidden=2048, 40 слоёв
  (GDN-гибрид, full_attention каждый 4-й: i%4==3), heads=16, kv_heads=2, head_dim=256,
  linear 16/32/128/128, conv_kernel=4, vocab=248320, E=256, top-8, moe_i=512,
  shared_i=512, rms_eps=1e-6, partial_rotary=0.25, rope_theta=1e7.
- Грамматика тензоров основных слоёв идентична 3.5 (fused `mlp.experts.gate_up_proj`
  [256,1024,2048] / `mlp.experts.down_proj` [256,2048,512]).
- **Единственное структурное отличие — MTP-блок**: у 3.6 эксперты fused
  (`mtp.layers.0.mlp.experts.gate_up_proj` / `down_proj`), у 3.5 — per-expert
  (768 тензоров). Generic-ветка Mtp существующего конвертера покрывает оба варианта
  (MTP хранится, не применяется — HF parity), спецобработка не требуется.
- Конфиг-дельта 3.6: явный `text_config.bos_token_id = 248044` (совпадает с
  eos_token_id), `pad_token_id = null`, `partial_rotary_factor = 0.25`,
  `tie_word_embeddings = false`; отсутствует `mlp_only_layers` (эквивалентно пустому).
- Спецтокены tokenizer_config.json идентичны 3.5: bos_token отсутствует,
  eos_token = `<|im_end|>`, pad_token = `<|endoftext|>`.

## Решения оператора

| Вопрос | Решение |
|---|---|
| Конвертер | Общее ядро: тело `ConvertQwen35MoeModel` → `ConvertQwenMoeModelCommon(spec)`; 3.5 и 3.6 — тонкие обёртки. Вывод 3.5 байт-идентичен |
| Шаг 7 (passport.bin) | Пропустить (как Qwen3.5-35B-A3B); общий `passport/passport.bin` занят мостом GLM-5.3-Flash↔DeepSeek-V4-Flash. Отклонение зафиксировано здесь |
| Модельный класс | Свой класс `Qwen36MoeModel` + извлечение lifecycle-тел Load/Infer/Benchmark в family-хелперы, используемые обеими моделями |
| Новые CUDA-ядра | Не требуются (архитектура идентична 3.5); `qwen_moe_inference.cu` не изменяется |

## Инвариант №1: Автономность моделей

- Свой frozen-паспорт: `qwen_moe_36_constants.h`, namespace `lsxcommon::qwen_moe_36::frozen`
  (leaf-заголовок, никаких cross-includes паспортов — independence guard).
- Свой kType `qwen3.6-35b-a3b`, свой `ValidateGeometry`, свой `Infer` с
  `LsxFastMathScope(cfg_.fast_math, ...)` (consumer-список independence guard).
- Файлы 3.5 (`qwen_moe_*`) затрагиваются только механической экстракцией хелперов
  (поведение 3.5 не меняется, регрессионный щит — существующие
  `test_qwen_moe_{convert,loader,model,kernels,selfcheck}`) и однострочными
  легитимными касаниями: `model_registry.cpp`, `tests/test_code_naming_guard.cpp`,
  `tests/test_model_independence_guard.cpp`, CMakeLists, `apps/lsxforge/src/main.cpp`.
- Family-инфраструктура (`qwen_moe_load`, `qwen_moe_inference.cu`,
  генерационный драйвер `RunQwenMoeGeneration`) переиспользуется по ссылке —
  это разрешённый общий слой семейства (аналог `qwen_inference` для dense+MoE).

## Замороженная геометрия (compile-time, static_assert)

Источник истины: `/mnt/llm_models/models/Qwen3.6-35B-A3B/config.json`
(text_config, model_type `qwen3_5_moe`). Значения совпадают с 3.5, но паспорт
собственный — независимый (independence guard: уникальный namespace на модель).

```
hidden = 2048, vocab = 248320
Слои: 40, layer_types[i] = linear_attention (i%4!=3) / full_attention (i%4==3)
heads = 16, kv_heads = 2, head_dim = 256
linear: num_key_heads = 16, num_value_heads = 32, key_head_dim = 128,
        value_head_dim = 128, conv_kernel_dim = 4
MoE: num_experts = 256, num_experts_per_tok = 8, moe_intermediate = 512,
     shared_expert_intermediate = 512
rms_norm_eps = 1e-6, partial_rotary_factor = 0.25, rope_theta = 1e7
bos_token_id = 248044 (из конфига 3.6; eos_token_id = 248044, pad отсутствует)
MTP: store-not-apply (fused-эксперты MTP квантизуются whole-tensor)
```

## Компоненты (naming: family=qwen, type=moe, variant=36)

| Файл | Роль |
|---|---|
| `libs/lsxcommon/src/qwen_moe_36_constants.h` | frozen-паспорт 3.6 + static_assert |
| `libs/lsxcommon/src/qwen_moe_36_convert.cpp` + `include/lsxcommon/qwen_moe_36_convert.h` | `ConvertQwen36MoeModel` — frozen-гейт против `qwen_moe_36::frozen` + вызов общего ядра (format `logestix-qwen36moe`, model `Qwen3.6-35B-A3B`, record_bos) |
| `libs/lsxcommon/src/qwen_moe_36_model.cpp` + `.h` | `Qwen36MoeModel` (ModelBase), kType=`qwen3.6-35b-a3b` |

Модификации существующих файлов (легитимные касания):

| Файл | Изменение |
|---|---|
| `libs/lsxcommon/src/qwen_moe_convert.cpp` + `include/lsxcommon/qwen_moe_convert.h` | Тело → `ConvertQwenMoeModelCommon(input, output, const QwenMoeConvertSpec&)`; `ParseQwenMoeConfig` парсит `bos_token_id` (default -1); индекс пишет `bos_token_id` только при `spec.record_bos && top.bos_token_id >= 0` (вывод 3.5 байт-идентичен) |
| `libs/lsxcommon/src/qwen_moe_load.{h,cpp}` | Извлечение `LoadQwenMoeStandardGraph(...)` — тело `Qwen35MoeModel::Load` (dense load, expert store, governor, cache sizing, prestaging), параметризовано geometry-гейтом |
| `libs/lsxcommon/src/qwen_moe_inference.cu` + `include/lsxcommon/qwen_moe_inference.h` | Извлечение infer/benchmark-драйверов над `RunQwenMoeGeneration` (session policy, FillInferenceResult/FillBenchmarkResult) в family-хелперы |
| `libs/lsxcommon/src/qwen_moe_model.{h,cpp}` | Делегирование извлечённым хелперам (поведение идентично) |
| `libs/lsxcommon/src/model_registry.cpp` | `LsxModelFactory::Register<Qwen36MoeModel>()` |
| `apps/lsxforge/src/main.cpp` | Флаг `--qwen36moe` → `ConvertQwen36MoeModel` |
| `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt` | Новые исходники и тест-таргеты |
| `tests/test_code_naming_guard.cpp` | variant-regex `(?:27b|flash)` → `(?:27b|flash|36)` |
| `tests/test_model_independence_guard.cpp` | Строка `kPassports` (`qwen_moe_36_constants.h`, `lsxcommon::qwen_moe_36::frozen`), `owner`-map, `qwen_files`, consumer `qwen_moe_36_model.cpp` |

## Конвертер

- `QwenMoeConvertSpec` (в `qwen_moe_convert.h`):
  `format_tag`, `model_name`, `bool (*check_frozen)(const QwenMoeTopology&, std::string&)`,
  `record_bos`.
- Классификатор, fused-expert split (gate=rows[0,I), up=rows[I,2I)), MTP-ветка,
  токенизатор (`BuildAndVerifyTokenizer`, bos_required=false / eos_required=true),
  unified-раскладка `logestix/` — без изменений, общие для 3.5/3.6.
- Специфика 3.6: свой frozen-гейт; в индекс добавляется `"bos_token_id": 248044`
  (только для 3.6, управляется `record_bos`).
- MTP fused-эксперты 3.6: generic Mtp-ветка квантизует whole-tensor
  ([256,1024,2048] / [256,2048,512]) — stored-not-applied, HF parity.
- Нормы/biases/embed/lm_head/router/conv1d/A_log/dt_bias — без квантования
  (инвариант точности протокола); линейные матрицы — NVFP4.

## Модельный класс

- `Qwen36MoeModel final : ModelBase`:
  - `kType = "qwen3.6-35b-a3b"`;
  - ctor: `qwen::SetQwenTcFastMath(cfg_.fast_math)` (паритет с 3.5);
  - `Load()`: `LoadQwenMoeStandardGraph(*this, geometry-gate-3.6, члены)`;
  - `Infer()`: `LsxFastMathScope(cfg_.fast_math, cfg_.fast_math)` → family-хелпер
    (session policy, `RunQwenMoeGeneration`, FillInferenceResult);
  - `SelfCheck()`: `runtime::RunParisSelfCheck` (промпт "The capital of France is"
    → "Paris", топ-1);
  - `Benchmark()`: family-хелпер, fresh session;
  - `ValidateGeometry()`: против `qwen_moe_36::frozen` (+ `SyntheticGeometryAllowed()`
    escape hatch для synthetic fixtures).
- Инкрементальная session-политика, единый сквозной `cudaStream_t`, expert-cache
  sizing/prestaging — унаследованы от извлечённых хелперов (идентичны 3.5).

## Метаданные

Индекс и артефакты — unified-раскладка `<output_root>/logestix/` через
`lsxcommon::lsx_model_metadata(rel)`; никаких хардкодов путей в конвертере.
Формат-тэг в `model.index.json`: `logestix-qwen36moe`, `"model":
"Qwen3.6-35B-A3B"`, `bos_token_id: 248044`.

## Тесты (TDD — сначала тесты, потом реализация)

| Executable | Содержание |
|---|---|
| `tests/test_qwen_moe_36_convert.cpp` | Классификатор 3.6-грамматики (вкл. MTP fused-имена → Mtp), frozen-гейт (принять 3.6-конфиг / отклонить чужую геометрию), synthetic roundtrip на tiny-fixture (`qwen_moe_test_fixture.h` паттерн), содержимое индекса (format/model/bos), record_bos-условие |
| `tests/test_qwen_moe_36_model.cpp` | Registry содержит `qwen3.6-35b-a3b`; ValidateGeometry accept/reject; synthetic lifecycle (SKIP-паттерн без CUDA/весов) |
| `tests/test_qwen_moe_36_selfcheck.cpp` | Реальный Paris-селфчек, 3 бит-точных цикла (`LSX_TEST_QWEN36_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4 без конвертированной модели) — протокол, этап 8 |
| Регресс 3.5 | `test_qwen_moe_{convert,loader,model,kernels,selfcheck}` остаются зелёными; вывод конвертера 3.5 байт-идентичен (guard — существующие 119 assertions convert-теста) |
| Гарды | `test_code_naming_guard.cpp` (variant `36`), `test_model_independence_guard.cpp` (паспорт/consumer) — расширяются в том же коммите, что и файлы модели |

Порядок TDD: convert-тесты (red) → общее ядро + 3.6-обёртка (green) → model-тесты
(red) → класс + хелперы (green) → guards/CMake (green) → конвертация реальной
модели (`lsxforge --qwen36moe`, 67 GB) → selfcheck-тест (red без модели → green).

## Риски

- Экстракция хелперов затрагивает `qwen_moe_model.cpp` — чисто механическая,
  поведение фиксируется существующим набором тестов 3.5.
- `bos_token_id == eos_token_id == 248044` в конфиге 3.6: EOS генерации берётся из
  tokenizer db (`<|im_end|>`), как у 3.5 — коллизии нет.
- Windows-ограничение тестов (`ctest -R "qwen|gemma"`) уже покрывает новые тесты
  префиксом `qwen`.
- Шаг 7 пропущен — сознательное отклонение, зафиксировано в этой спеке; при
  появлении сетевого моста для Qwen-семейства паспорт добавляется отдельным
  файлом (не перезаписью общего моста).
