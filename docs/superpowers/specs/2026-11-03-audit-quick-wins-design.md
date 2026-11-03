# Быстрый аудит-пакет: дедупликация конвертеров, guard-инфраструктура, шум сборки — дизайн

Дата: 2026-11-03
Ветка: `audit/2026-11-03-quick-wins`
Статус: дизайн на ревью оператора; реализация — по TDD.

## 1. Задача

Аудит кода от 2026-11-03 дал топ-10 рекомендаций. Эта ветка берёт три пункта с
наибольшим отношением «эффект / стоимость» и закрывает их TDD-циклом:

1. **M1 — дедупликация «механизма» конвертеров.** Три I/O-хелпера, которые
   по-прежнему скопированы по семействам, выносятся в единый слой
   `lsxcommon`. Полный инвентарь (с учётом snake_case-вариантов) — **39
   определений в 20 файлах**. Это тот же класс дефекта, из-за которого 26
   копий `ReadTensorData` разошлись в валидации (закрыто 2026-11-03).
2. **M2 — guard-тесты инфраструктуры.** Централизация CMake-путей, архива и
   общего ридера тензоров сейчас держится дисциплиной; ничто не мешает
   вернуть `${CMAKE_BINARY_DIR}/_deps`, литерал `120-real;121-real` или
   приватную копию хелпера.
3. **M3 — шум сборки.** Full build даёт 51 408 предупреждений и 159 487 строк
   лога; в сэмпле нет ни одного содержательного предупреждения из нашего кода —
   только шум третьих сторон и `-Wpedantic` в host-проходе NVCC. После правки
   реальные предупреждения должны быть видны.

Наблюдаемая база (2026-11-03, `linux-debug`, GCC 15):

| метрика | значение |
|---|---|
| предупреждений в full build | 51 408 (Arrow 4 807, absl 5 730, …) |
| «style of line directive» в наших файлах | 2024 в `test_glm_moe_flash_kernels.cu`, 834 в `expert_store.h`, … |
| строк лога сборки | 159 487 |
| `WriteBatch` / `WriteExpertFile` копий | 14 / 13 (Pascal + snake_case) |
| `QuantizeSlice` копий | 12 |
| `converter.cpp` (эталон цены TU) | 14.4 s, 1.1 GiB RSS |

## 2. Границы

### Входит

- M1: `WriteBatch`, `WriteExpertArrow` (плюс `ExpertArrowPath`), `QuantizeSlice`
  (2 перегрузки) — единые реализации и миграция 39 копий.
- M2: три статических guard-проверки (пути `_deps`, единственность
  определений, литералы `CUDA_ARCHITECTURES`) + реестр guard-тестов.
- M3: `SYSTEM`-include для зависимостей, `-Wpedantic` только для C/CXX,
  измерение до/после.
- Документация: `AGENTS.md` (CMake Key Points), `test-lifecycle` (таблица
  guard'ов), `cuda-cmake-build-gotchas` (строка «симптом → причина»).

### НЕ входит (осознанно)

- `auto bad =` (15 файлов): лямбда захватывает локальный `why`; свободная
  функция меняет ~40 call sites на файл при нулевом выигрыше в надёжности.
- Прелюдия чтения `config.json` (28 файлов): формы разные (плоский корень,
  `text_config`), смешаны `ondemand` (48 файлов) и `dom` (21); унификация —
  отдельная задача с собственным риском, не «быстрый» пакет.
- `-Werror`: после M3 включать глобально нельзя (Arrow/absl собираются теми же
  флагами); это следующий шаг, не эта ветка.
- Пункты 4–10 аудита: политика ошибок, реестр env-кнубов, fuzz прочих входов,
  PCH/ccache, слоистость `lsxcommon`, документация, платформенная матрица.

## 3. M1 — единые хелперы конвертеров

### 3.1 Инвентарь дублей

| хелпер | копий | расхождения |
|---|---|---|
| `WriteBatch` / `write_batch` | 14 | нет; Pascal в 6 файлах, snake_case в 8 |
| `WriteExpertFile` / `write_expert_file` | 13 | 11 — `(base, layer, expert)` с `MakeBatch`; `qwen_moe_exp` — `MakeBatchView`; `glm_moe_flash` — zero-padded `layer_%03d`/`exp_%03d`; `alice` — `(dir, expert)` без слоя |
| `QuantizeSlice` / `quantize_slice` | 12 | 4 с параметром `dtype` (step/flint/hy/beam); 2 у `minimax` (полная с E8M0 и удобная); 5 BF16-only (qwen_moe/ornith/alice/nemotron/qwen_moe_exp); 1 у `kimi_k3` берёт dtype из `TensorInfo` |

Рядом уже есть `SaveExpertToArrow`: та же запись, но через `MakeBatchView`
(zero-copy) + `diag::ValidateBatchBuffers` + info-лог на каждый файл.
Логировать на каждый эксперт нельзя (десятки тысяч файлов), поэтому общий
хелпер берёт zero-copy запись и валидацию, но не лог; `SaveExpertToArrow`
остаётся тонкой обёрткой с логом. Побочный эффект: 11 семейств перестают
копировать payload эксперта перед записью (`minimax`/`qwen_moe_exp` уже так
делают).

Имя файла эксперта — часть формата семейства, поэтому общий хелпер принимает
**готовый путь**, а не `(base, layer, expert)`: `glm_moe_flash` пишет
zero-padded `layer_%03d/exp_%03d.arrow` (и его загрузчик это ожидает), `alice`
получает каталог слоя от вызывающего кода. Канонический путь для остальных 11
семейств строит маленький `ExpertArrowPath`.

### 3.2 API

Объявления — в `lsxcommon/converter.h` (это уже общий заголовок конвертеров;
`.cu` его не включает — проверено: 0 включающих), реализации — в
`converter.cpp`:

```cpp
// Unwraps a MakeBatch() result and writes it (14 identical copies).
arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path);

// <model_root>/experts/layer_<layer>/exp_<expert>.arrow — the canonical layout
// of the 11 families that do not pad names.
std::filesystem::path ExpertArrowPath(const std::filesystem::path &model_root,
                                      int layer, int expert);

// Creates the parent directory, builds a zero-copy batch over `expert`,
// validates it and writes it. The path carries the family's naming format
// (glm_moe_flash pads, alice passes the layer directory). No per-file log.
arrow::Status WriteExpertArrow(const std::filesystem::path &path,
                               const PackedExpert &expert);

// Quantizes a contiguous slice to NVFP4. The full form carries the optional
// E8M0 companion scales (minimax); the short form passes nullptr and uses
// `out.shape`.
bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   const uint8_t *e8m0, const std::vector<int64_t> &shape,
                   const std::vector<int64_t> &e8m0_shape,
                   DenseNvfp4Record &out);
bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   DenseNvfp4Record &out);
```

Семантика вывода сохраняется побайтово: путь файла остаётся за семейством
(`ExpertArrowPath` или собственный), запись идёт через `MakeBatchView` +
`ValidateBatchBuffers`; `QuantizeSlice` — тонкая обёртка над
`gpu::quantize_tensor_gpu` без изменения аргументов.

### 3.3 TDD

RED — новый `tests/test_converter_helpers.cpp` (label `fast`, host-only):

1. `WriteBatch` возвращает статус ошибочного `arrow::Result` и не создаёт файл.
2. `WriteBatch` на валидном `DenseNvfp4Batch` пишет файл, читаемый
   `ReadRecordBatchFromFile(path, 6)` с теми же payload-байтами.
3. `WriteExpertArrow` создаёт каталог слоя и round-trip через
   `ExpertSchema::UnpackBatch` (w1/w2/w3 packed + scales + global scales);
   `ExpertArrowPath` даёт канонический путь.
4. `WriteExpertArrow` возвращает non-OK (и не бросает), если каталог создать
   нельзя (под `model_root` лежит обычный файл), и не оставляет файла.

`QuantizeSlice` — GPU-путь: кейс в `tests/test_lsx_nvfp4_dispatch.cpp`
(label `inference`): BF16-срез → `packed == ceil(n/2)`, `scales == ceil(n/16)`,
`global_scale` конечен и > 0; повтор детерминирован.

GREEN: реализации в `converter.cpp`, удаление 39 копий, `SaveExpertToArrow`
делегирует в `WriteExpertArrow`. Проверка: `ctest -L convert` (37/37) —
конвертерные тесты сверяют содержимое, не только коды возврата.

## 4. M2 — guard-тесты инфраструктуры

Новый `tests/test_infrastructure_guard.cpp` (label `fast`, только Catch2).
Проверки — чистые функции над текстом (`Scan*`), чтобы каждую можно было
протестировать на синтетических строках, плюс по одному тесту «дерево чисто».

1. **Пути зависимостей.** После вырезания `#`-комментариев запрещены
   `${CMAKE_BINARY_DIR}/_deps` и `_deps/` внутри аргументов
   `target_include_directories` / `include_directories`. Сканируются корневой
   `CMakeLists.txt`, `cmake/*.cmake`, `libs/**/CMakeLists.txt`,
   `apps/**/CMakeLists.txt`, `tests/CMakeLists.txt`. `third_party/` не
   сканируется. `${FETCHCONTENT_BASE_DIR}` разрешён только в
   `lsx_dep_includes` (корневой файл).
2. **Единственность определений.** В `libs/` и `apps/` (production-код;
   тестовые фикстуры не сканируются) запрещены определения
   `ReadTensorData`/`read_tensor_data`, `WriteBatch`/`write_batch`,
   `WriteExpertArrow`/`WriteExpertFile`/`write_expert_file`,
   `ExpertArrowPath`, `QuantizeSlice`/`quantize_slice` вне
   `libs/lsxcommon/src/converter.cpp`.
3. **Архитектура кубиков.** В CMake-файлах запрещены литералы
   `CUDA_ARCHITECTURES "90"|"89"|"80"|"<числа>-real..."`; разрешены только
   `${LSX_CUDA_ARCH}` и `OFF` (не-CUDA цели).

Каждый `Scan*` получает позитивные и негативные кейсы на строках; отдельные
`TEST_CASE` проверяют реальное дерево. Мутационная проверка (обязательна, в
плане): внедрить нарушение каждого типа → guard красный → откатить.

Регистрация: `lsx_add_test` (как остальные guard'ы) + имя в группе `fast`.

## 5. M3 — шум сборки

### 5.1 Правки

```cmake
# root CMakeLists.txt, вместо add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(-Wall -Wextra)
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>)

# lsx_dep_includes: зависимости становятся системными заголовками
target_include_directories(lsx_dep_includes SYSTEM INTERFACE ...)
```

Причина: `build.ninja` показывает `-Wall -Wextra -Wpedantic` прямо в FLAGS
`.cu`-правила (`gemma_moe_inference.cu.o`); nvcc передаёт их host-компилятору,
и GCC ругается на собственные `#line`-маркеры nvcc («style of line directive
is a GCC extension»). `SYSTEM` убирает и педантичный шум Arrow/absl
(`__int128` под `-Wpedantic`), не меняя порядок поиска заголовков.

### 5.2 Проверка

- До/после: полный `cmake --build --preset linux-debug > log`; считаем
  `warning:` всего и отдельно по нашим файлам (`libs/`, `apps/`, `tests/`).
- Приёмка: третьи стороны и NVCC-pedantic — **0**; всего — ≤ 5 % от 51 408;
  полная сборка exit 0; `ctest -L fast` и `-L convert` зелёные.

**Уточнение по факту (2026-11-03).** После снятия шума стало видно **238
предупреждений в нашем коде** (170 сайтов в 83 файлах), которые раньше были
погребены: 53 `-Wunused-result`, 42 `-Wint-in-bool-context` (одна макрос-цепочка),
39 `-Wrestrict` — все на **in-place** вызовах активаций, где аргумент алиасит
`__restrict__`-параметр (формально UB, находка аудита), 26 `-Wunused-parameter`,
24 `-Wmissing-field-initializers`, 19 мёртвых `static`-функций и 33 прочих.
Итог замера: 51 408 → **484** строки предупреждений (238 наших + 246 из
vendored-таргетов). Обещание «наших 0» требует отдельного прохода (включая
правку `__restrict__` в общих ядрах и повторный прогон kernel-сьютов), поэтому
оно вынесено в `docs/OPEN_TASKS.md` § W со разбивкой по категориям; `-Werror`
включается только после его закрытия.
- Риск: `-isystem` скрывает предупреждения из шаблонов, инстанцируемых нашим
  кодом. Митигация: сравнение не только счётчика, но и списка оставшихся
  предупреждений (они должны быть только из наших файлов); при неожиданной
  потере — откат точечно для конкретного заголовка.

## 6. Процесс и коммиты

- Ветка `audit/2026-11-03-quick-wins` (от `main` @ `803412af`).
- Порядок: M1 RED → GREEN (коммит `test:` + `refactor:`), M2 (коммит `test:`),
  M3 (коммит `build:` + docs).
- Каждый коммит: полная сборка + `-L fast` (+ `-L convert` для M1).
- TDD-дисциплина: тест пишется и **падает** до реализации; для guard'ов
  «падение» доказывается мутацией дерева.

## 7. Критерии приёмки ветки

1. 0 определений трёх хелперов и ридера тензоров вне `converter.cpp`.
2. `_deps`-путей и arch-литералов в CMake нет; guard это доказывает.
3. Наших предупреждений в full build — 0; лог сборки пригоден для чтения.
4. `ctest -L fast` зелёный (132 + 2 новых: `test_converter_helpers`,
   `test_infrastructure_guard`), `-L convert` 37/37.
5. Документация обновлена в тех же коммитах.
