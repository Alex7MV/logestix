# Фаза 2: DeepSeek V4 Flash/Pro на одном драйвере — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. **Subagents are forbidden in this repository** (AGENTS.md § Autonomous Mode): every step is executed inline by the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Свести 12 файлов-копий `deepseek_moe_pro_*` к одному набору трансляционных единиц, параметризованных паспортом и типом residual-потока, сохранив побитовую идентичность обеих моделей.

**Architecture:** Flash-файлы становятся шаблонами `template <class T>`, где `T` — паспорт (`lsxcommon::deepseek::Passport` или `lsxcommon::deepseek_pro::Passport`), а `typename T::ResidualT` — тип HC-residual (`__half` у Flash, `float` у Pro). Обе инстанцирования живут в тех же `.cu`: у каждого файла внизу набор явных инстанцирований. Pro-копии удаляются.

**Tech Stack:** C++20, CUDA 13.x, NVCC (шаблоны в device-коде), CMake OBJECT-библиотеки, Catch2 v3.16.1.

**Spec:** `docs/superpowers/specs/2026-11-03-code-duplication-consolidation-design.md` §2.1, §4, §5 (Ф2), §7.

## Global Constraints

- Критерий приёмки — **побитовая идентичность**: `logits_hash` V4-Flash и V4-Pro (снятый baseline) не меняется, FSI/CODE/DET не хуже, perf-строки (`test_perf_baseline_deepseek_v4_flash`, `test_perf_baseline_deepseek_v4_pro`) в пределах шума. Пере-сид `tests/perf/perf_baselines.json` запрещён.
- **Перенос, а не переписывание**: арифметика не редактируется. Единственное разрешённое смысловое изменение — замена литерального типа/констант на `T::` и `ResidualT` в местах, перечисленных в Task 4, шаг 2.
- Геометрия и per-layer таблицы (`compress_ratio`, hash/router-слои, indexer-слои) — данные времени исполнения из загрузчика (`pw.layers[i]`); в шаблон уходят только константы паспорта.
- Заголовки, достижимые из `.cu`, не включают `<absl/container/*>`.
- Сборка — **0 предупреждений** в наших файлах; цель — не увеличить время сборки более чем на 30 % (замеряется в Task 7).
- Коммиты: английские, `refactor:`/`test:`/`docs:`. Субагенты запрещены.
- Деревья vault не пере-конвертируются; `lsxforge --verify` обязан давать те же `files/footer_ok/deep/problems=0`.
- Процедура с двумя моделями в одном процессе (повторная загрузка после `Free()`) остаётся рабочей: общий `Free()` освобождает shared-expert блобы (~4 GiB, `docs/OPEN_TASKS.md` §R3).

---

### Task 0: Снять baseline (measure first)

**Files:**
- Create: `/tmp/opencode/deepseek-baseline.txt` (вне репозитория, артефакт сессии)

**Interfaces:**
- Produces: эталонные `logits_hash`, top-1 и t/s для Flash и Pro, с которыми сравниваются все последующие шаги.

- [ ] **Step 1: Снять SelfCheck обоих вариантов**

```bash
ctest --preset linux-debug-inference -R "test_deepseek_moe_selfcheck" --output-on-failure 2>&1 | tee /tmp/opencode/ds-flash-selfcheck.txt
ctest --preset linux-debug-inference -R "test_deepseek_moe_pro_selfcheck" --output-on-failure 2>&1 | tee /tmp/opencode/ds-pro-selfcheck.txt
```

Записать в `/tmp/opencode/deepseek-baseline.txt` строки вида `flash logits_hash=<...> top1=<...>` и `pro logits_hash=<...> top1=<...>`. Если тест не печатает хеш, взять его из вывода `lsxbenchmark --selfcheck` тем же способом, каким он снимался для §R3.

- [ ] **Step 2: Снять perf-строки**

```bash
ctest --preset linux-debug-perf -R "test_perf_baseline_deepseek_v4_(flash|pro)" --output-on-failure 2>&1 | tee /tmp/opencode/ds-perf.txt
```

Дописать t/s обоих вариантов в `/tmp/opencode/deepseek-baseline.txt`. Известные значения для сверки: Flash ≈ 0.77 t/s (perf) / 0.79 (r3), Pro ≈ 0.77 / 0.79; TTFT ≈ 9 s.

- [ ] **Step 3: Зафиксировать время сборки**

```bash
/usr/bin/time -f "flash-build %e s" cmake --build --preset linux-debug --target lsxcommon_deepseek_cu 2>&1 | tail -3
```

Записать результат. (Имя OBJECT-цели уточнить по `libs/lsxcommon/CMakeLists.txt`: `rg -n "deepseek.*_cu OBJECT" libs/lsxcommon/CMakeLists.txt`.)

---

### Task 1: Паспорт Flash и обновление реестра гвардов

**Files:**
- Create: `libs/lsxcommon/src/deepseek_moe_constants.h`
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cuh` (константы уходят в паспорт)
- Modify: `tests/test_model_independence_guard.cpp::kPassports` (`deepseek-v4-flash`)
- Modify: `libs/lsxcommon/src/deepseek_moe_pro_constants.h` (добавить `ResidualT`, привести к общей форме)

**Interfaces:**
- Produces: `lsxcommon::deepseek::Passport` и `lsxcommon::deepseek_pro::Passport` — оба с одинаковыми именами членов: `kHidden`, `kLayers`, `kHeads`, `kHeadDim`, `kRopeHeadDim`, `kQLoraRank`, `kOLoraRank`, `kOGroups`, `kNumRouted`, `kTopK`, `kMoeIntermediate`, `kSharedIntermediate`, `kHcMult`, `kHcMix`, `kHcDim`, `kSinkhornIters`, `kRmsEps`, `kAttnWindow`, `kIndexNHeads`, `kIndexHeadDim`, `kIndexTopk`, `kHashLayers`, плюс `using ResidualT = __half;` (Flash) / `= float;` (Pro). Отсутствующие у Flash значения сверить с `deepseek_moe_pro_constants.h` и с фактическим `inference/config.json` дерева (шаг 2).

- [ ] **Step 1: Написать падающий guard-тест**

В `tests/test_model_independence_guard.cpp` две таблицы:

`kPassports` — заменить строку

```cpp
{"deepseek-v4-flash", "deepseek_moe_load.cuh", "lsxcommon::gpu"},
```

на

```cpp
{"deepseek-v4-flash", "deepseek_moe_constants.h", "lsxcommon::deepseek"},
```

и в `TEST_CASE("each passport defines its geometry in its own namespace or
class")` (карта `owner`) заменить

```cpp
{"deepseek_moe_load.cuh", "namespace gpu {"},
```

на

```cpp
{"deepseek_moe_constants.h", "namespace deepseek {"},
```

- [ ] **Step 2: Собрать и убедиться, что гвард падает**

```bash
cmake --build --preset linux-debug --target test_model_independence_guard 2>&1 | tail -5
./build/linux-debug/tests/test_model_independence_guard 2>&1 | tail -15
```

Expected: FAIL — паспорт `deepseek_moe_constants.h` не найден.

- [ ] **Step 3: Создать паспорт Flash**

Перенести 18 `constexpr` из `deepseek_moe_load.cuh:23-59` в `lsxcommon::deepseek`, переименовав в pro-стиль (`kHiddenDim` → `kHidden`, `kNumLayers` → `kLayers`, `kAttnHeads` → `kHeads`, `kIntermediateDim` → `kMoeIntermediate`), добавить недостающие члены из pro-паспорта со значениями, проверенными по `inference/config.json` дерева Flash:

```json
{"hidden_size":4096,"num_hidden_layers":43,"num_attention_heads":64,
 "head_dim":512,"q_lora_rank":1024,"o_lora_rank":1024,"o_groups":8,
 "n_routed_experts":256,"num_experts_per_tok":6,"moe_intermediate_size":2048,
 "shared_expert_intermediate_size":2048,"hc_mult":4,"sinkhorn_iters":20,
 "rms_norm_eps":1e-6,"sliding_window":128,"index_n_heads":?,
 "index_head_dim":128,"index_topk":?,"num_hash_layers":?}
```

Поля, помеченные `?`, берутся из реального `config.json` (команда ниже) — **не угадывать**:

```bash
python3 -c "import json;c=json.load(open('/mnt/fast_storage/huggingface/DeepSeek-V4-Flash/inference/config.json'));print({k:v for k,v in c.items() if 'index' in k or 'hash' in k or 'nope' in k})"
```

Определение структуры:

```cpp
#pragma once

// DeepSeek-V4-Flash frozen passport (stage 0 triage froze the geometry; the
// driver is shared with V4-Pro-0813, which only changes the values and the
// Hyper-Connections residual type).

#include <cuda_fp16.h>

#include <cstdint>

namespace lsxcommon {
namespace deepseek {

struct Passport {
    using ResidualT = __half;

    static constexpr int kHidden = 4096;
    static constexpr int kLayers = 43;
    // ... остальные члены, значения из config.json / прежних констант ...
};

inline constexpr char kType[] = "deepseek-v4-flash";

} // namespace deepseek
} // namespace lsxcommon
```

`deepseek_moe_load.cuh` после этого содержит только `#include "deepseek_moe_constants.h"` и `using namespace`-совместимые псевдонимы, если они нужны остальному коду; сами `constexpr` удаляются.

- [ ] **Step 4: Привести pro-паспорт к той же форме**

В `libs/lsxcommon/src/deepseek_moe_pro_constants.h`: переименовать `namespace deepseek_pro` остаётся, но члены переезжают в `struct Passport` с теми же именами, что у Flash, и `using ResidualT = float;`. Добавить `static constexpr int kIndexNHeads`, `kIndexHeadDim`, `kIndexTopk`, `kHashLayers`, `kAttnWindow`, `kSharedIntermediate`, если они сейчас лежат плоскими константами.

- [ ] **Step 5: Прогнать гвард и быстрый набор**

```bash
cmake --build --preset linux-debug --target test_model_independence_guard 2>&1 | tail -5
./build/linux-debug/tests/test_model_independence_guard
ctest --preset linux-debug-fast
```

Expected: гвард PASS (7 assertions на паспорт), `fast` 134/134.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_constants.h \
        libs/lsxcommon/src/deepseek_moe_pro_constants.h \
        libs/lsxcommon/src/deepseek_moe_load.cuh \
        tests/test_model_independence_guard.cpp
git commit -m "refactor(deepseek): give V4-Flash a passport shaped like V4-Pro's"
```

---

### Task 2: `norm_kernels` и `inference_head` — это чистые копии, шаблон не нужен

Измерено при исполнении: `deepseek_moe_pro_norm_kernels.cu` отличается от
flash-версии **только** строкой `#include` и лишним `namespace deepseek {`
(файлы проверены `diff` с нормализацией `deepseek_pro` → `deepseek`); то же у
`inference_head.cu`. Обе функции (`rmsnorm_float`, `run_lm_head`) принимают
геометрию аргументами и не читают ни одной константы паспорта, поэтому их не
надо темплейтить — достаточно удалить pro-копии и перевести вызовы на
`lsxcommon::gpu::*`. Это сильнее шаблона: расхождение исключено
конструкцией. Задача ниже заменяет первоначальный «доказать механизм».

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_norm_kernels.cu`
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_head.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/deepseek_moe_norm_kernels.h`
- Reference (read-only): `libs/lsxcommon/src/deepseek_moe_pro_norm_kernels.cu`, `deepseek_moe_pro_inference_head.cu`

**Interfaces:**
- Produces: правило, по которому переделываются остальные файлы: `template <class T> cudaError_t rmsnorm_float(...)`, `template <class T> cudaError_t run_lm_head(...)` + явные инстанцирования для `deepseek::Passport` и `deepseek_pro::Passport`.

- [ ] **Step 1: Доказать, что pro-копии идентичны (побайтово, не только телами)**

```bash
diff <(sed 's/deepseek_pro/deepseek/g' libs/lsxcommon/src/deepseek_moe_pro_norm_kernels.cu) \
     libs/lsxcommon/src/deepseek_moe_norm_kernels.cu
diff <(sed 's/deepseek_pro/deepseek/g' libs/lsxcommon/src/deepseek_moe_pro_inference_head.cu) \
     libs/lsxcommon/src/deepseek_moe_inference_head.cu
```

Expected: в каждом — только строка `#include` и строка `namespace deepseek {`
(`} // namespace deepseek`). Любое иное расхождение останавливает задачу.

- [ ] **Step 2: Удалить pro-копии и перевести вызовы**

```bash
git rm libs/lsxcommon/src/deepseek_moe_pro_norm_kernels.cu \
       libs/lsxcommon/src/deepseek_moe_pro_inference_head.cu \
       libs/lsxcommon/include/lsxcommon/deepseek_moe_pro_norm_kernels.h
```

В `deepseek_moe_pro_inference_{attn,moe,pipeline}.cu` заменить вызовы
`rmsnorm_float(...)` / `run_lm_head(...)` (они резолвились в
`deepseek_pro::gpu`) на `lsxcommon::gpu::rmsnorm_float(...)` /
`lsxcommon::gpu::run_lm_head(...)`. Поиск незакрытых ссылок:

```bash
rg -n "deepseek_pro::gpu::(rmsnorm_float|run_lm_head)" libs/lsxcommon tests
```

Expected: пусто после правки.

- [ ] **Step 3: Проверить**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head -20
ctest --preset linux-debug-inference -R "test_deepseek_moe_(norm_kernels|inference_kernels)" --output-on-failure
```

Expected: сборка без ошибок и предупреждений в наших файлах, оба теста PASS **на обеих моделях** (тест-файлы уже инстанцируют pro-вариант через `deepseek_moe_pro_*` символы — до удаления копий они должны оставаться рабочими; если тест ссылается на namespace `deepseek_pro::gpu`, добавить переадресацию на шаблон в Task 7).

- [ ] **Step 4: Commit**

```bash
git rm libs/lsxcommon/src/deepseek_moe_pro_norm_kernels.cu \
       libs/lsxcommon/src/deepseek_moe_pro_inference_head.cu \
       libs/lsxcommon/include/lsxcommon/deepseek_moe_pro_norm_kernels.h
git add libs/lsxcommon/src/deepseek_moe_pro_inference_attn.cu \
        libs/lsxcommon/src/deepseek_moe_pro_inference_moe.cu \
        libs/lsxcommon/src/deepseek_moe_pro_inference_pipeline.cu
git commit -m "refactor(deepseek): drop the norm/head copies and call the shared kernels"
```

---

### Task 3: `inference_moe`

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu`
- Reference (read-only): `libs/lsxcommon/src/deepseek_moe_pro_inference_moe.cu`

**Interfaces:**
- Consumes: правило из Task 2.
- Produces: `template <class T>` для `run_shared_experts_ffn`, `run_routed_expert_ffn`, `route_token*`, `run_routed_experts*`, `issue_routed_staging` + два инстанцирования.

- [ ] **Step 1: Сделать файл шаблонным**

Механически: все ссылки на константы → `T::k*`; каждая свободная функция → `template <class T>`; внизу блока — по одной строке явного инстанцирования на функцию для каждого паспорта:

```cpp
template cudaError_t run_shared_experts_ffn<deepseek::Passport>(...полный список типов...);
template cudaError_t run_shared_experts_ffn<deepseek_pro::Passport>(...);
```

Аргументы в строке инстанцирования копируются из объявления; ошибка компиляции здесь означает несовпадение списка — исправить список, не сигнатуру.

- [ ] **Step 2: Проверить, что арифметика не сдвинулась**

```bash
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/deepseek_moe_pro_inference_moe.cu \
    --to   libs/lsxcommon/src/deepseek_moe_inference_moe.cu \
    --rename deepseek_pro=deepseek
```

Expected: `missing in target: 0  unexpected in target: 0`.

- [ ] **Step 3: Собрать и прогнать MoE-тесты**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head -20
ctest --preset linux-debug-inference -R "test_deepseek_moe_moe" --output-on-failure
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_inference_moe.cu
git commit -m "refactor(deepseek): template the routed-MoE driver on the passport"
```

---

### Task 4: `inference_attn` — Hyper-Connections, MLA, индексер

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_attn.cu`
- Reference (read-only): `libs/lsxcommon/src/deepseek_moe_pro_inference_attn.cu` (диф — 15 строк)

**Interfaces:**
- Consumes: паспорта из Task 1.
- Produces: `template <class T> run_hc_pre / run_hc_post / run_hc_head / run_expand_hc / run_mla_attention / run_csa_attention`; `typename T::ResidualT` в трёх местах.

- [ ] **Step 1: Сделать файл шаблонным**

Все константы → `T::k*`; функции → `template <class T>`; два инстанцирования внизу. Ядра остаются в анонимном пространстве имён (`__global__` внутри шаблонной функции не нужен — параметризация идёт через аргументы, кроме `kHcMix`/`kHcDim`, которые становятся `T::`).

- [ ] **Step 2: Заменить тип residual на `ResidualT` — ровно в трёх местах**

Это единственное разрешённое смысловое изменение. Текущая разница Flash/Pro:

```cpp
// Flash
float acc = ...;
y[i] = __float2half(acc);
// Pro
y[i] = acc;            // FP32 residual: HC-сумма превышает диапазон FP16
```

Становится:

```cpp
using ResidualT = typename T::ResidualT;
ResidualT *d_single,   // [dim]
...
y[i] = static_cast<ResidualT>(acc);
```

Проверка эквивалентности: для `ResidualT = __half` выражение обязано давать тот же бит-паттерн, что `__float2half(acc)` (это то же самое преобразование); для `ResidualT = float` — identity. Если компилятор жалуется на `static_cast` для `__half`, использовать `__float2half_rn(acc)` через `if constexpr (std::is_same_v<ResidualT, __half>)` и `static_cast<float>(acc)` иначе — но только при доказанной необходимости, зафиксировав причину в комментарии.

- [ ] **Step 3: Доказать, что разница ровно та, что ожидалась**

```bash
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/deepseek_moe_pro_inference_attn.cu \
    --to   libs/lsxcommon/src/deepseek_moe_inference_attn.cu \
    --rename deepseek_pro=deepseek
```

Expected до правки шага 2: ровно три «missing/extra» пары (три места residual). После правки — 0, потому что обе стороны теперь используют `ResidualT`; если после правки остались расхождения, это признак случайной правки тела.

- [ ] **Step 4: Собрать и прогнать attention-гейты**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head -20
ctest --preset linux-debug-inference -R "test_deepseek_moe_(inference_kernels|arena)" --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_inference_attn.cu
git commit -m "refactor(deepseek): template the HC/MLA attention on the passport and the residual type"
```

---

### Task 5: `inference_csa` (compressor + indexer)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_csa.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/deepseek_moe_inference_csa.h`
- Reference (read-only): `libs/lsxcommon/src/deepseek_moe_pro_inference_csa.cu`, `deepseek_moe_pro_inference_csa.h`

**Interfaces:**
- Consumes: правило Task 2.
- Produces: шаблонные `run_csa_*`; заголовок получает шаблонные объявления вместо `namespace deepseek_pro`-копии.

- [ ] **Step 1: Сделать CSA шаблонным**

Константы → `T::`, функции → `template <class T>`, два инстанцирования. Заголовок: убрать pro-специфичные объявления, оставить один шаблон.

- [ ] **Step 2: Проверить**

```bash
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/deepseek_moe_pro_inference_csa.cu \
    --to   libs/lsxcommon/src/deepseek_moe_inference_csa.cu \
    --rename deepseek_pro=deepseek
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head
ctest --preset linux-debug-inference -R "test_deepseek_moe_(inference_kernels|arena)" --output-on-failure
```

Expected: provenance 0/0; сборка чистая; тесты PASS.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_inference_csa.cu \
        libs/lsxcommon/include/lsxcommon/deepseek_moe_inference_csa.h
git commit -m "refactor(deepseek): template the compressed-KV/indexer driver on the passport"
```

---

### Task 6: `inference_pipeline` и `load`

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu`
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cu`, `deepseek_moe_load.cuh`
- Reference (read-only): `deepseek_moe_pro_inference_pipeline.cu`, `deepseek_moe_pro_load.cu`, `deepseek_moe_pro_load.cuh`

**Interfaces:**
- Consumes: шаблоны Task 2–5.
- Produces: `template <class T> load_pinned_weights / RunDeepSeekV4Generation`; модель вызывает `...<Passport>`.

- [ ] **Step 1: Сделать pipeline шаблонным**

Константы → `T::`; `run_csa_attention<T>(...)` и прочие кросс-файловые вызовы получают явный параметр шаблона. Два инстанцирования внизу файла.

- [ ] **Step 2: Сделать загрузчик шаблонным**

`load_pinned_weights`, `UploadLayerJobs`, `ValidateArenaJob` → `template <class T = deepseek::Passport>` (default-аргумент сохраняет неизменным вызов из flash-glue `deepseek_moe_model.cpp` — его в этой задаче не трогаем); структуры `PinnedWeights`/`LayerWeights` остаются общими (они без семейных констант). Per-layer таблицы (`compress_ratio`, hash/router/indexer-слои) читаются как раньше из топологии. Два инстанцирования на функцию.

Особое внимание: `Free()` (в `deepseek_moe_model.cpp` / загрузчике) освобождает shared-expert блобы ~4 GiB — убедиться, что после шаблонизации путь освобождения не изменился (он вне шаблона, если структура общая).

- [ ] **Step 3: Проверить**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head -20
ctest --preset linux-debug-inference -R "test_deepseek_moe_(loader|load_fast|model)" --output-on-failure
```

Expected: PASS для Flash; Pro-тесты пока ещё идут через свои копии (удаляются в Task 7).

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu \
        libs/lsxcommon/src/deepseek_moe_load.cu \
        libs/lsxcommon/src/deepseek_moe_load.cuh
git commit -m "refactor(deepseek): template the generation pipeline and the loader on the passport"
```

---

### Task 7: Удалить 12 pro-копий, перевести Pro-модель на шаблон

**Files:**
- Delete: `libs/lsxcommon/src/deepseek_moe_pro_inference_{attn,csa,head,moe,pipeline}.cu`, `deepseek_moe_pro_load.{cuh,cu}`, `deepseek_moe_pro_norm_kernels.cu`, `libs/lsxcommon/include/lsxcommon/deepseek_moe_pro_inference_csa.h`, `deepseek_moe_pro_norm_kernels.h`
- Keep: `libs/lsxcommon/src/deepseek_moe_pro_constants.h`, `deepseek_moe_pro_convert.{h,cpp}`, `deepseek_moe_pro_model.{h,cpp}` (family glue: `kType`, регистрация, `SelfCheck`; гварды `kLifecycle`/`consumers` адресуют модель по этому файлу)
- Modify: `libs/lsxcommon/CMakeLists.txt` (убрать удалённые источники из OBJECT-либ)
- Modify: `tests/test_deepseek_moe_pro_kernels.cu`, `tests/test_deepseek_moe_pro_loader.cpp`, `tests/test_deepseek_moe_pro_model.cpp`, `tests/test_deepseek_moe_pro_fixture.h` (перевод на шаблоны)

**Interfaces:**
- Consumes: все шаблоны Task 2–6.
- Produces: `DeepSeekV4ProModel`, инстанцирующая `V4Driver<deepseek_pro::Passport>`; публичный API реестра не меняется (`memory`/`kType` те же).

- [ ] **Step 1: Удалить копии и перевести Pro-модель**

```bash
git rm libs/lsxcommon/src/deepseek_moe_pro_inference_attn.cu \
       libs/lsxcommon/src/deepseek_moe_pro_inference_csa.cu \
       libs/lsxcommon/src/deepseek_moe_pro_inference_head.cu \
       libs/lsxcommon/src/deepseek_moe_pro_inference_moe.cu \
       libs/lsxcommon/src/deepseek_moe_pro_inference_pipeline.cu \
       libs/lsxcommon/src/deepseek_moe_pro_load.cu \
       libs/lsxcommon/src/deepseek_moe_pro_load.cuh \
       libs/lsxcommon/src/deepseek_moe_pro_norm_kernels.cu \
       libs/lsxcommon/include/lsxcommon/deepseek_moe_pro_inference_csa.h \
       libs/lsxcommon/include/lsxcommon/deepseek_moe_pro_norm_kernels.h
```

`DeepSeekV4ProModel` остаётся в своих `deepseek_moe_pro_model.{h,cpp}` (family glue, ~155 строк, 90 из которых не совпадают с flash-аналогом); записи `kLifecycle` и `consumers` в гварде не меняются.

- [ ] **Step 1b: Перевести Pro-glue на общие типы и явный параметр шаблона**

В `deepseek_moe_pro_model.cpp` заменить провайдера `deepseek_pro::gpu::` на общий `gpu::` и передать паспорт явно. Было:

```cpp
auto topo = deepseek_pro::gpu::load_topology(cfg_.model_path);
if (gpu::load_pinned_weights(cfg_.model_path, pw_, stream()) ...);
return deepseek_pro::gpu::RunInferenceModel(...);
```

Стало:

```cpp
auto topo = gpu::load_topology(cfg_.model_path);
using Pro = deepseek_pro::Passport;
if (topo.num_layers != Pro::kLayers || topo.experts_per_layer != Pro::kNumRouted) { ... }
if (gpu::load_pinned_weights<Pro>(cfg_.model_path, pw_, stream()) ...);
return gpu::RunInferenceModel<Pro>(...);
```

`PinnedWeights`, `load_topology`, `kIndexFormatNvfp4` — общие (`gpu::`), они не зависят от геометрии. Все обращения к `deepseek_pro::k*` вне паспорта заменяются на `Pro::k*`. Проверка:

```bash
rg -n "deepseek_pro::gpu" libs/lsxcommon/src/deepseek_moe_pro_model.cpp
```

Expected: пустой вывод.

- [ ] **Step 2: Перевести pro-тесты на шаблоны**

Pro-тесты обращаются к символам удаляемого namespace напрямую. Перечислить их:

```bash
rg -n "deepseek_pro::gpu::" tests/ | sed 's/:.*deepseek_pro::gpu:://' | sed 's/[^A-Za-z0-9_:].*//' | sort -u
```

Для `tests/test_deepseek_moe_pro_kernels.cu`, `test_deepseek_moe_pro_loader.cpp`,
`test_deepseek_moe_pro_model.cpp`, `test_deepseek_moe_pro_fixture.h` выполнить
две замены:

```text
deepseek_pro::gpu::X   ->  lsxcommon::gpu::X<deepseek_pro::Passport>      (функции, CSAState, любые шаблонные типы)
deepseek_pro::gpu::Y   ->  lsxcommon::gpu::Y                             (не-шаблонные: PinnedWeights, LayerWeights, load_topology, kIndexFormatNvfp4)
```

Правило выбора: если `Y` упоминает в объявлении константу паспорта или
`ResidualT` — это `X` (шаблон), иначе `Y`. Проверять по заголовку
`deepseek_moe_load.cuh` после Task 6, а не по памяти.

- [ ] **Step 3: Обновить CMake и реестры**

Убрать удалённые источники из OBJECT-библиотек; убедиться, что `test_infrastructure_guard` не жалуется на «один definition на общий хелпер».

- [ ] **Step 4: Прогнать гварды и быстрый набор**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head -20
ctest --preset linux-debug-fast
ctest --preset linux-debug-convert
```

Expected: 0 предупреждений; `fast` 134/134; `convert` 37/37.

- [ ] **Step 5: Побитовый гейт обеих моделей**

```bash
ctest --preset linux-debug-inference -R "test_deepseek_moe_selfcheck" --output-on-failure
ctest --preset linux-debug-inference -R "test_deepseek_moe_pro_selfcheck" --output-on-failure
ctest --preset linux-debug-inference -R "test_deepseek_moe_(fsi|code|det)_selfcheck" --output-on-failure
ctest --preset linux-debug-inference -R "test_deepseek_moe_pro_(fsi|code|det)_selfcheck" --output-on-failure
```

Expected: `logits_hash` совпадает с `/tmp/opencode/deepseek-baseline.txt` для обеих моделей; FSI/CODE/DET — не хуже baseline (у Pro FSI-02 остаётся RED).

- [ ] **Step 6: Проверить, что деревья не пострадали**

```bash
./build/linux-debug/apps/lsxforge/lsxforge --verify --input /mnt/fast_storage/huggingface/DeepSeek-V4-Flash 2>&1 | tail -3
./build/linux-debug/apps/lsxforge/lsxforge --verify --input /mnt/fast_storage/huggingface/DeepSeek-V4-Pro-0813 2>&1 | tail -3
```

Expected: `problems=0`, `footer_ok = files`, exit 0 для обоих.

- [ ] **Step 7: Commit**

```bash
git add -A libs/lsxcommon tests
git commit -m "refactor(deepseek): delete the V4-Pro copies and run both variants on one driver"
```

---

### Task 8: Полный барьер, документация, отчёт фазы

**Files:**
- Modify: `AGENTS.md` § Structure (DeepSeek: один драйвер, паспорта)
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`
- Modify: `CHANGELOG.md`
- Create: `docs/open-tasks/2026-11-03-deepseek-v4-shared-driver-closed.md`

- [ ] **Step 1: Замерить сборку и perf после фазы**

```bash
/usr/bin/time -f "after %e s" cmake --build --preset linux-debug --target lsxcommon_deepseek_cu 2>&1 | tail -3
ctest --preset linux-debug-perf -R "test_perf_baseline_deepseek_v4_(flash|pro)" --output-on-failure
```

Expected: время сборки не более +30 % к замеру Task 0 (иначе перейти к «ТУ на инстанцирование» — выделить `<ProTraits>`-инстанцирования в отдельный `.cu`); perf в пределах шума.

- [ ] **Step 2: Записать артефакты**

В `docs/open-tasks/2026-11-03-deepseek-v4-shared-driver-closed.md`: входящие/исходящие LOC (−5 300 ±), вывод provenance-проверок, baseline и после (хеши, top-1, t/s), время сборки до/после, число удалённых файлов.

`AGENTS.md`: заменить абзац про `deepseek_moe_pro_*` на «V4-Flash и V4-Pro-0813 делят один набор ТУ, параметризованный паспортом (`deepseek::Passport`/`deepseek_pro::Passport`) и `ResidualT` (`__half`/`float`); Pro отличается геометрией и FP32-residual».

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md .opencode/skills/logestix-repo-map/SKILL.md CHANGELOG.md \
        docs/open-tasks/2026-11-03-deepseek-v4-shared-driver-closed.md
git commit -m "docs: record the DeepSeek V4 shared driver and update the registries"
```

---

## Приёмка фазы

- 10 файлов-копий удалены; `deepseek_moe_pro_constants.h`, `convert.{h,cpp}` и `model.{h,cpp}` остались.
- Provenance-проверки: 0 расхождений по всем шести парам после правки residual.
- `logits_hash` Flash и Pro совпадают с baseline; FSI/CODE/DET не хуже.
- t/s в пределах шума; сборка не выросла больше чем на 30 %.
- `lsxforge --verify` обоих деревьев — `problems=0`.
- Ветка `refactor/deepseek-v4-shared-driver`, мерж — только по явному запросу оператора.
