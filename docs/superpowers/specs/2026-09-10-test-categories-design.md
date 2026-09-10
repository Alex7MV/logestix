# Тест-категории: fast / inference / perf

Дата: 2026-09-10
Статус: approved (design), реализация по TDD-плану

## Проблема

109 позиций `add_test` в `tests/CMakeLists.txt` (125 зарегистрированных тестов
с учётом foreach-циклов) выполняются одним монолитным прогоном
`ctest --preset <preset>` — часы на машине с чекпоинтами. Фильтрация
существует только по имени (`ctest -R`), категорий нет.

## Цели

1. Разделить тесты на три категории:
   - `fast` — запускается при каждой сборке (host-only, минуты);
   - `inference` — контур инференса (CUDA-ядра, загрузчики, модели,
     селфчеки); запускается автоматически, когда изменился контур
     инференса (git diff);
   - `perf` — perf-бенчмарки на реальных весах; только отдельной командой,
     никогда не входит в автопрогон.
2. Базовый `ctest --preset <preset>` гоняет только `fast`.
3. Автодетект «изменился контур инференса» — по `git diff`.
4. Унифицировать именование файлов в `tests/` (всё `test_*`).
5. Дисциплина категорий защищена guard-тестом (TDD).

## Немеханизмы (не делаем)

- Никакого физического разноса файлов по каталогам (`tests/fast/` и т.п.) —
  labels решают задачу без переноса 100+ файлов.
- Никаких переименований 125 тестовых таргетов — префиксы в именах
  (`fast_*`/`inf_*`) сломали бы рецепты `ctest -R glm` из AGENTS.md.
- CI-логика вне скоупа (в репо нет CI-конфига); роутер работает локально
  на обеих OS.

## Дизайн

### 1. Категории = CTest LABELS

Каждый тест получает ровно один label: `fast` | `inference` | `perf`.
Присвоение — тремя каноническими блоками в конце `tests/CMakeLists.txt`
(после всех `add_test`), в регулярной форме, удобной для статического
парсинга guard-тестом:

```cmake
# ── Test category labels (fast | inference | perf) ────────────────────────
set_tests_properties(<fast tests...>     PROPERTIES LABELS fast)
set_tests_properties(<inference tests...> PROPERTIES LABELS inference)
set_tests_properties(<perf tests...>      PROPERTIES LABELS perf)
```

Существующие свойства (`SKIP_RETURN_CODE 4`, `ENVIRONMENT`) не трогаются.
`${qt}`-имена из foreach-циклов перечисляются в блоках развёрнуто.

### 2. testPresets

К каждому из 5 базовых пресетов добавляются три testPreset-варианта
(итого +15):

- `<preset>-fast` — `filter.include.label = "^fast$"` (совпадает с базовым
  пресетом по смыслу; базовые пресеты также переводятся на `^fast$`);
- `<preset>-inference` — `label = "^inference$"`;
- `<preset>-perf` — `label = "^perf$"`.

Базовые testPresets (`linux-debug` и др.) получают тот же фильтр `^fast$` —
дефолтный прогон становится быстрым. Полный прогон доступен вручную:
`ctest -L "^(fast|inference|perf)$"`.

Ограничение Windows (только модели qwen|gemma): пресеты — чисто label-фильтры;
правило `ctest -R "qwen|gemma"` для inference-прогона на Windows применяет
роутер-скрипт (см. ниже) и остаётся документированной практикой в AGENTS.md.

### 3. Роутер `tools/test_router.cmake`

CMake script mode (кроссплатформенно, без компиляции):

```
cmake -P tools/test_router.cmake -- --preset linux-debug [--base origin/main] [--all] [--list]
```

Алгоритм:

1. `git diff --name-only <base>...HEAD` (base по умолчанию — merge-base с
   `origin/main`; нет git/нет диффа → только `fast`).
2. Классификация каждого изменённого файла по упорядоченной таблице правил
   (первое совпадение выигрывает), метки накапливаются в множество:
   - `\.cuh?$` в любом месте → `inference`;
   - `libs/lsxcommon/(src|include)/...` → `inference`, КРОМЕ файлов,
     подпадающих под fast-лексикон
     `(tokenizer|unicode|pretokenizer|path_registry|schema|converter|_convert|orchestrator)`
     (консервативный дефолт: неизвестный новый файл lsxcommon → inference);
   - `libs/lsxnetwork/**`, `apps/**` → `fast`;
   - `tests/**` → `fast` + `inference` (инфра тестов менялась — прогоняем оба);
   - `docs/**`, `*.md`, `CMakeLists.txt`, `CMakePresets.json`, `tools/**` → `fast`;
   - прочее → `fast`.
3. Итог: `ctest --preset <preset> -L "^(fast|inference)$"`.
   На WIN32 к inference-прогону добавляется `-R "^(test_qwen|test_gemma)"`.
4. Флаги: `--all` — форс `fast+inference`; `--list` — только показать
   классификацию файлов и выбранные метки, ctest не запускать.

Таблица правил живёт в начале скрипта списком `lsx_router_rule(<regex> <labels>)`
— явная, ревьюабельная, проверяемая guard-тестом. Perf в автопрогон не
попадает never: роутер умеет только `fast` и `inference`.

### 4. Guard-тест `tests/test_ctest_labels_guard.cpp` (TDD-якорь)

Host-only Catch2-тест по образцу `test_code_naming_guard` (сканирует
`LSX_SOURCE_DIR`). Утверждает:

1. Каждый тест (все имена из `add_test(NAME ...)` плюс все элементы
   foreach-списков `tests/CMakeLists.txt`) встречается ровно в одном
   `set_tests_properties(... PROPERTIES LABELS <cat>)` с категорией из
   {fast, inference, perf}.
2. Каждый perf-тест имеет `SKIP_RETURN_CODE 4` в своём
   `set_tests_properties`.
3. `CMakePresets.json`: все testPreset-фильтры `label` ссылаются только на
   валидные категории; для каждого configurePreset существуют `-fast`,
   `-inference`, `-perf` варианты.
4. `tools/test_router.cmake`: все `lsx_router_rule` метки ∈ {fast, inference};
   метка `perf` в правилах запрещена.
5. Именование `tests/`: каждый файл соответствует `^test_.+\.(cpp|cu|h)$`.

Плюс runtime-пин: мини-парсер (выделен в тестируемую функцию) на
синтетическом CMake-сниппете — дубликат метки, неизвестная метка,
пропущенный тест → FAILURE. Это делает guard самопроверяемым, а не
только «регэксп по файлу».

### 5. Унификация имён фикстур

5 заголовков `<family>_test_fixture.h` → `test_<family>_fixture.h`:

- `glm_moe_test_fixture.h` → `test_glm_moe_fixture.h`
- `deepseek_moe_test_fixture.h` → `test_deepseek_moe_fixture.h`
- `gemma_moe_test_fixture.h` → `test_gemma_moe_fixture.h`
- `qwen_dense_test_fixture.h` → `test_qwen_dense_fixture.h`
- `qwen_moe_test_fixture.h` → `test_qwen_moe_fixture.h`
- `qwen_moe_exp_test_fixture.h` → `test_qwen_moe_exp_fixture.h`
- `kimi_moe_test_fixture.h` → `test_kimi_moe_fixture.h`

(7 файлов; обновить ~30 `#include` в тестах. Механический refactor без
изменения поведения; пункт 5 guard-теста после этого зелёный.)

### 6. Документация

AGENTS.md, раздел Test: таблица категорий, команды
(`ctest --preset <p>` = fast; `ctest -L inference`; `ctest -L perf`;
`cmake -P tools/test_router.cmake -- --preset <p>`), правило
«новый тест обязан получить label в том же коммите» (штраф — красный
guard в CI).

## Классификация (полная, 125 + 1 новый)

### perf (6)

test_blackwell_perf_glm, test_blackwell_perf_dense,
test_blackwell_perf_moe_qwen, test_blackwell_perf_moe_gemma,
test_blackwell_perf_deepseek, test_kimi_moe_perf

### fast (41 + 1 новый guard)

test_core, test_fast_load, test_geometry_guard, test_gpu_perf,
test_schema_views, test_deepseek_moe_loader, test_expert_store,
test_qwen_dense_convert, test_glm_moe_expert_cache, test_model_integrity,
test_qwen_dense_converter_integrity, test_glm_moe_convert,
test_glm_moe_flash_convert, test_kimi_moe_convert, test_model_factory,
test_generation_cli, test_glm_moe_prompt_framing, test_gemma_prompt_framing,
test_gemma_moe_convert, test_model_runtime, test_model_engine,
test_tokenizer, test_mosaic_network, test_mosaic_frontend_route,
test_token_translator, test_network_translator_weld, test_integrated_pipeline,
test_path_registry, test_vram_estimate, test_resource_governor,
test_hardware_profile, test_lifecycle_plan, test_runtime_tuner,
test_kimi_moe_tokenizer, test_code_naming_guard, test_absl_format_guard,
test_kimi_moe_expert_cache, test_qwen_moe_convert, test_qwen_moe_36_convert,
test_qwen_moe_exp_convert, test_model_independence_guard,
**test_ctest_labels_guard** (новый)

### inference (78)

**CUDA-ядра и model-neutral (.cu):** test_sampling, test_lsx_nvfp4_dispatch,
test_perf_knobs_apply, test_blackwell_tc_mma, test_lsx_kernels,
test_blackwell_micro_gemm, test_nvfp4_pack,
test_deepseek_moe_inference_kernels, test_deepseek_moe_moe,
test_deepseek_moe_arena, test_qwen_moe_kernels, test_qwen_moe_exp_kernels,
test_glm_moe_flash_kernels, test_kimi_moe_int4_dequant, test_kimi_moe_int4_real,
test_kimi_moe_router, test_kimi_moe_attention, test_kimi_moe_experts,
test_kimi_moe_tc_batch, test_glm_moe_load_fast, test_deepseek_moe_load_fast,
test_glm_moe_model_lifecycle

**Qwen dense:** test_qwen_dense_loader, test_qwen_dense_model,
test_qwen_dense_27b_model, test_qwen_dense_kv_cache, test_qwen_dense_kv_parity,
test_qwen_dense_selfcheck, test_qwen_dense_27b_selfcheck,
test_qwen_dense_norm, test_qwen_dense_rotary, test_qwen_dense_shuffle_reduce,
test_qwen_dense_mlp, test_qwen_dense_nvfp4, test_qwen_nvfp4_tile_gemm,
test_qwen_nvfp4_gemv, test_qwen_dense_full_attn,
test_qwen_dense_full_attn_parallel, test_qwen_dense_linear_attn,
test_qwen_dense_fused_mlp, test_qwen_dense_fused_linear_attn

**Qwen MoE (3.5/3.6/exp):** test_qwen_moe_loader, test_qwen_moe_model,
test_qwen_moe_selfcheck, test_qwen_moe_36_model, test_qwen_moe_36_selfcheck,
test_qwen_moe_exp_loader, test_qwen_moe_exp_model, test_qwen_moe_exp_selfcheck

**GLM-5.2:** test_glm_moe_loader, test_glm_moe_model, test_glm_moe_selfcheck,
test_glm_moe_inference, test_glm_moe_full_compute, test_glm_moe_prefill_seqp,
test_glm_moe_prompt_output, test_glm_moe_attention, test_glm_moe_dense_compute,
test_glm_moe_layer_compute, test_glm_moe_experts, test_glm_moe_moe_router,
test_glm_moe_moe_compute

**GLM-5.3-Flash:** test_glm_moe_flash_loader, test_glm_moe_flash_model,
test_glm_moe_flash_selfcheck

**DeepSeek:** test_deepseek_moe_prompt_output, test_deepseek_moe_selfcheck

**Gemma:** test_gemma_moe_loader, test_gemma_moe_model, test_gemma_moe_selfcheck,
test_gemma_moe_norm, test_gemma_moe_rotary, test_gemma_moe_attn,
test_gemma_moe_moe

**Kimi:** test_kimi_moe_loader, test_kimi_moe_model, test_kimi_moe_selfcheck,
test_kimi_moe_inference

Принцип для спорных случаев: тест, исполняющий CUDA-контур (даже на
синтетической фикстуре), — `inference`; host-only — `fast`.
`test_integrated_pipeline` и `test_model_engine` — host-only (моки), → fast.
`test_expert_store`/`test_glm_moe_expert_cache`/`test_kimi_moe_expert_cache`
— host bookkeeping → fast.

## TDD-план (кратко; детально — в implementation plan)

1. **Cycle 1 — guard RED→GREEN.** Написать `test_ctest_labels_guard.cpp`
   (пп. 1–4 выше) → RED (ни одного LABELS). Добавить label-блоки,
   testPresets, роутер → GREEN.
2. **Cycle 2 — naming RED→GREEN.** Добавить в guard пункт 5 → RED
   (фикстуры). Переименовать 6 фикстур + include-ы → GREEN.
3. **Верификация:** configure (схема пресетов валидируется CMake),
   `ctest -L fast` зелёный, `ctest -L perf` выбирает ровно 6, роутер
   `--list` корректно классифицирует тестовый дифф.
