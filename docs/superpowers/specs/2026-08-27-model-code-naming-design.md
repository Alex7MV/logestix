# Спека: Унификация именования файлов-моделей кода (family + type + role)

Дата: 2026-08-27
Статус: утверждена оператором
Область: только **код** (`libs/lsxcommon/src`, `libs/lsxcommon/include/lsxcommon`, `tests/`,
CMakeLists, AGENTS.md). Файлы данных моделей (`.arrow`, `model.index.json`, каталоги
конвертации) НЕ затрагиваются.

## 1. Проблема

Имена файлов кода моделей неоднородны:

- маркер типа архитектуры (`moe`/`dense`) почти нигде не присутствует: `glm_convert.cpp`
  — фактически MoE, но этого не видно из имени; `qwen_convert.cpp` vs `qwen_moe_convert.cpp`;
- версии моделей смешаны с именами файлов в разных стилях: `glm_v5_model`,
  `deepseek_v4_flash_model`, `qwen_3_8_27b_model`, `gemma_4_26b_a4b_model`;
- префиксы одного семейства разные: `gemma4_*` vs `gemma_4_26b_a4b_*` vs `gemma_inference.cu`;
- суффиксы ролей разнобойные: `_inference_load`, `load_internal`, пустой (`glm_inference.cu`).

## 2. Целевая схема

```
<family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>
```

| Токен | Значение |
|---|---|
| `family` | строчное имя семейства: glm, qwen, gemma, kimi, deepseek |
| `type` | **обязателен**: moe или dense |
| `variant` | только при коллизии двух моделей одного type внутри family |
| `role` | короткий словарь: convert, load, model, inference, kernels, … (реестр расширяется в guard-тесте) |
| `subsystem` | необязательное разбиение крупного ядра: attn, csa, head, moe, pipeline |

Версии моделей из имён файлов убраны навсегда — живут в именах классов и `kType`
(`GlmV5Model::kType = "glm-5.2"`). Имена классов/неймспейсов НЕ меняются.

## 3. Карта переименований

Пары `.cpp`/`.h`/`.cu` переименовываются согласованно.

### GLM (moe)

| Сейчас | Станет |
|---|---|
| `glm_convert.{cpp,h}` | `glm_moe_convert.{cpp,h}` |
| `glm_inference_load.{cpp,cu}` + `include/glm_inference_load.h` | `glm_moe_load.{cpp,cu}` + `glm_moe_load.h` |
| `src/glm_load_internal.h` | `glm_moe_load_internal.h` |
| `glm_v5_model.{cpp,h}` | `glm_moe_model.{cpp,h}` |
| `glm_inference.cu` | `glm_moe_inference.cu` |
| `glm_device_math.h` | `glm_moe_device_math.h` |
| `glm_expert_kernels.cuh` | `glm_moe_expert_kernels.cuh` |
| `include/glm_expert_cache.h` | `glm_moe_expert_cache.h` |

### Qwen dense

| Сейчас | Станет |
|---|---|
| `qwen_convert.{cpp,h}` | `qwen_dense_convert.{cpp,h}` |
| `qwen_v5_model.{cpp,h}` | `qwen_dense_model.{cpp,h}` |
| `qwen_3_8_27b_model.{cpp,h}` | `qwen_dense_27b_model.{cpp,h}` |

**Уточнение по верификации (планирование):** строка `qwen_inference.cu → qwen_dense_inference.cu`
удалена — grep показал включения `qwen_inference.h` из `qwen_moe_inference.{h,cu}`,
`qwen_3_5_moe_model.h` и MoE-тестов, т.е. файл кросс-типовый → переехал в allowlist (п.4).

### Qwen MoE

| Сейчас | Станет |
|---|---|
| `qwen_moe_inference_load.{cpp,h}` | `qwen_moe_load.{cpp,h}` |
| `qwen_3_5_moe_model.{cpp,h}` | `qwen_moe_model.{cpp,h}` |

`qwen_moe_{convert,inference}.{cpp,cu}`, `qwen_moe_constants.h` уже соответствуют схеме.
`qwen_moe_inference.h` (include) остаётся.

### Gemma (moe)

| Сейчас | Станет |
|---|---|
| `gemma4_convert.{cpp,h}` | `gemma_moe_convert.{cpp,h}` |
| `gemma4_inference_load.{cpp,h}` | `gemma_moe_load.{cpp,h}` |
| `include/gemma4_inference.h` | `gemma_moe_inference.h` |
| `gemma_4_26b_a4b_model.{cpp,h}` | `gemma_moe_model.{cpp,h}` |
| `gemma_inference.cu` | `gemma_moe_inference.cu` |

### Kimi (moe)

| Сейчас | Станет |
|---|---|
| `kimi_convert.{cpp,h}` | `kimi_moe_convert.{cpp,h}` |
| `kimi_inference_load.{cpp,h}` | `kimi_moe_load.{cpp,h}` |
| `kimi_v5_model.{cpp,h}` | `kimi_moe_model.{cpp,h}` |
| `kimi_inference.cu` | `kimi_moe_inference.cu` |
| `kimi_inference_kernels.h` | `kimi_moe_kernels.h` |

### DeepSeek (moe)

| Сейчас | Станет |
|---|---|
| `deepseek_v4_flash_model.{cpp,h}` | `deepseek_moe_model.{cpp,h}` |
| `deepseek_inference_load.{cu,cuh}` | `deepseek_moe_load.{cu,cuh}` |
| `deepseek_inference_{attn,csa,head,moe,pipeline}.cu` | `deepseek_moe_inference_{attn,csa,head,moe,pipeline}.cu` |
| `deepseek_norm_kernels.{cu,h}` | `deepseek_moe_norm_kernels.{cu,h}` |

## 4. Файлы вне схемы (allowlist guard-теста)

Не переименовываются (общая инфраструктура либо кросс-типовые файлы семейства):

- общая инфраструктура: `lsx_kernels_*`, `safetensors_gpu.*`, `safetensors.*`, `bf16_convert.cu`,
  `nvfp4_*`, `model_*`, `converter.*`, `tokenizer*`, `pretokenizer.cpp`, `unicode*`,
  `schema.cpp`, `sampling.*`, `benchmark_gpu.*`, `path_registry.*`, `resource_governor.*`,
  `runtime_tuner.*`, `vram_estimate.*`, `lifecycle_strategy.*`, `generation_cli.*`,
  `prompt_framing.*`, `log.cpp`, `common.*`, `core.cpp`, `expert_store.*`, `fast_load.*`,
  `inference_gpu.h`, `diagnostics.h`, `shared_experts.*`, `byte_unicode.h` и т.п.;
- кросс-типовые файлы Qwen: `qwen_base_model.{cpp,h}`, `qwen_inference_load.{cpp,h}`,
  `qwen_inference.{h,cu}`, `qwen_device_math.h`, `qwen_reduce.cuh`, `qwen_tc_gemm.{cu,h}`;
- база Gemma: `gemma_base_model.{cpp,h}`.

Принадлежность каждого «пограничного» файла верифицируется grep'ом включений на этапе
плана реализации: если выяснится, что файл используется ровно одной моделью одного типа,
он переносится из allowlist под схему.

## 5. Тесты

Тесты зеркалируют схему: `test_<family>_<type>_<topic>.cpp/.cu`, фикстуры
`<family>_<type>_test_fixture.h`. Соответствия:

- `test_glm_*` → `test_glm_moe_*`; `glm_test_fixture.h` → `glm_moe_test_fixture.h`;
- `test_qwen_selfcheck`, `test_qwen_loader`, `test_qwen_model`, … → `test_qwen_dense_*`;
- `test_deepseek_*` → `test_deepseek_moe_*`; `test_kimi_*` → `test_kimi_moe_*`;
  `test_gemma4_*` → `test_gemma_moe_*`;
- qwen MoE тесты остаются (`test_qwen_moe_*`);
- принадлежность общих qwen-kernel тестов (`test_qwen_nvfp4_*`, `test_qwen_rotary`, …)
  определяется по фактически тестируемому TU: только dense → `_dense_`; оба → остаётся
  в allowlist без маркера;
- фильтры `ctest -R "qwen|gemma"` из AGENTS.md продолжают работать.

## 6. Постоянный контроль конвенции (guard-тест)

Новый host-only Catch2 тест `tests/test_code_naming_guard.cpp`:

- рекурсивно сканирует `libs/lsxcommon/src/**` и `libs/lsxcommon/include/lsxcommon/**`
  (корень репозитория передаётся через compile-time макрос `LSX_SOURCE_DIR` из
  `tests/CMakeLists.txt`);
- файл, чей стем начинается с известного family, обязан матчить regex схемы из п.2 с
  расширениями `{cpp, h, cu, cuh}`; невыполнение — провал с точной диагностикой
  (имя файла, ожидаемый шаблон, ближайший корректный вариант);
- stem'ы вне известных family и перечисленные исключения должны состоять в allowlist,
  заданном списком литералов с обоснованием в комментарии;
- реестры family / role / subsystem / variant живут константами внутри теста и
  расширяются при добавлении новой модели **в том же коммите**, что и новые файлы.

Таким образом TDD работает и для будущих моделей: создание файла в обход схемы роняет CI
в момент добавления.

## 7. Сопутствующие правки

1. Все `#include "..."` по всему дереву (src, include, apps/, tests/) — текстовая замена имён заголовков.
2. `libs/lsxcommon/CMakeLists.txt` — списки исходников OBJECT/CUDA-библиотек.
3. `tests/CMakeLists.txt` — цели тестов, фикстуры, макрос `LSX_SOURCE_DIR`.
4. AGENTS.md: секция Structure (пути), эталонные якоря Model Ingestion Protocol, новый раздел
   «Model Code File Naming Convention» со схемой, словарём ролей и чек-листом новой модели.

## 8. Порядок работ (TDD)

1. Написать guard-тест против текущего дерева → красный, список нарушений совпадает с картой
   ренейма (тест = исполняемая спецификация).
2. Механический ренейм всех файлов + правки include/CMakeLists/AGENTS.md → guard зелёный.
3. Полная пересборка windows-debug + `ctest` (Windows: только фильтр `ctest -R "qwen|gemma"`),
   подтверждение битовой идентичности поведения — меняются только имена файлов.
4. Коммиты: единый атомарный `refactor:` (guard + ренейм вместе — CI никогда не видит красное
   состояние); AGENTS.md допускается отдельным `docs:`-коммитом следом.

## 9. Критерии готовности

- В `src/`+`include/` не осталось файла модели без маркера `moe`/`dense` (кроме allowlist).
- Guard-тест зелёный и включён в стандартный прогон ctest.
- Все цельевые сборки (windows-debug) проходят; модельные тесты выдают те же результаты.
- AGENTS.md содержит конвенцию и обновлённые пути.
