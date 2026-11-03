# Консолидация дублирования в `libs/lsxcommon`: нейтральные слои — дизайн

Дата: 2026-11-03
Статус: дизайн одобрен оператором по секциям; реализация — по плану
(`docs/superpowers/plans/`), фаза за фазой.

## 1. Задача и критерий приёмки

Убрать измеренное дублирование кода между семействами моделей, вынеся
общий функционал в нейтральные слои. Критерий приёмки жёсткий и задан
оператором: **побитовая идентичность поведения**. После каждого шага
`logits_hash` всех затронутых моделей, все FSI/CODE/DET-гейты и все
perf-строки обязаны остаться прежними; любое расхождение трактуется как
дефект рефакторинга (шаг откатывается), а не как «новая базовая линия».
Пере-сид `tests/perf/perf_baselines.json` в этом цикле запрещён.

Из этого ограничения следует главное правило дизайна: **перенос, а не
переписывание**. Переносится существующий код (тела ядер, guard-ветки,
константы), меняются только имена и границы трансляционных единиц. Ни одна
формула, ни один порог и ни один код ошибки не редактируются.

## 2. Измеренная картина (до изменений)

Метрики сняты на `main` @ `caa90982`:

| область | LOC |
|---|---|
| `libs/` (cpp/cu/h/cuh) | 160 921 |
| `tests/` | 105 542 |
| `apps/` | 4 855 |
| `tools/` | 6 045 |

Автоматический скан повторяющихся `__global__`/`__device__` определений
(тела сравниваются после нормализации идентификаторов, комментариев и
пробелов) даёт **39 групп дубликатов ≥ 8 строк, 3 050 строк** в
продублированных определениях. Это нижняя оценка: скан не видит
хост-кода и не видит копий, отличающихся на одну строку.

Три кластера:

### 2.1 DeepSeek V4 Flash ↔ V4 Pro — 5 595 LOC

Pro-вариант — 15 файлов (12 в `src/`, 3 в `include/`), из которых **12 —
пофайловые копии** flash-варианта (10 в `src/` + `inference_csa.h`,
`norm_kernels.h`), а 3 копиями не являются: `deepseek_moe_pro_constants.h`
(паспорт, 133 строки), `convert.{h,cpp}` (30-строчный шим на общий
`Converter` с `SetSkipPrefixes({"mtp."})`). Копии отличаются на
единицы-десятки строк:

| файл | flash | pro | различающихся строк (после нормализации имён) |
|---|---|---|---|
| `inference_attn.cu` | 1 216 | 1 223 | **15** |
| `inference_moe.cu` | 925 | 931 | 10 |
| `inference_pipeline.cu` | 950 | 968 | 58 |
| `inference_csa.cu` | — | — | 2 |
| `inference_head.cu` | — | — | 2 |
| `norm_kernels.cu` | — | — | 2 |
| `load.cu` | 1 187 | 1 256 | 119 |
| `load.cuh` | — | — | 58 |
| `model.cpp` | 160 | 153 | 90 |
| `model.h` | — | — | 13 |

Единственное смысловое расхождение в горячем пути — тип residual-потока
Hyper-Connections: Pro держит его в FP32 (`float* d_single`, `y[i] = acc`,
потому что HC-сумма превышает диапазон FP16), flash — в FP16
(`__float2half(acc)`). Остальное — геометрия и идентификаторы реестра.

Копии уже расходились: `bd0ecb7c` (parallel sinkhorn, perf flash 17.1 →
17.8 t/s), `c576ce61` и `87b9096f` правили flash-файлы и не касались pro.
То есть дублирование — не теоретический, а уже сработавший риск.

### 2.2 Dense-трио `granite_dense` / `granite_dense_switch` / `phi_dense` — ~1.8k LOC

Ядра внимания и нормализации в трёх файлах — один и тот же код под тремя
именами (скан совпадений по нормализованным телам):

| ядро | строк | копий |
|---|---|---|
| `*_attn_split_kernel` | 95 | 3 |
| `*_attn_kernel` | 77 | 3 |
| `*_rmsnorm_kernel` | 42 | 3 |
| `*_rope_kernel` | 33 | 3 |
| `*_attn_combine_kernel` | 32 | 3 |
| `*_swiglu_fused_kernel` | 14 | 3 |
| `*_split_qkv_kernel` | 19 | 3 |

Масштаб дублирования целых файлов: `granite_dense_inference.cu` (1 014) и
`phi_dense_inference.cu` (1 066) различаются на **86 строк** суммарно
(~92 % совпадения); `granite_dense_switch_inference.cu` (1 223) отличается
от `granite_dense_inference.cu` на 755 строк (машинерия aLoRA) — но все семь
ядер и их лаунчеры совпадают и там. Заголовки
`granite_dense_inference.h` и `phi_dense_inference.h` структурно идентичны
(`RunGranite*` / `RunPhi*`). Тесты ядер — та же картина:
`test_granite_dense_kernels.cu` (433) против `test_phi_dense_kernels.cu`
(446) различаются на 45 строк.

Удаляемое (а не переносимое) — копии в phi (~980 строк) и в switch
(~470 = 1 223 − 755 уникальных): ≈ 1.45k в `libs/` и ≈ 0.4k в `tests/`.

### 2.3 Остатки — ~250 LOC

- `depth_mix_kernel` (AttnRes depth-softmax, блок 4) — 42 строки, побайтово
  одинаков в `alice_moe_kernels.cu` и `kimi_moe_k3_kernels.cu`.
- Примитивы конверсии и масштабирования размазаны по семействам вместо
  `lsx_kernels_*.cuh`: `f32_to_f16_kernel` — 8 файлов, `f16_to_f32_kernel` —
  6, `half_to_float_kernel` / `float_to_half_kernel` — по 10,
  `silu_mul_kernel` — 2 копии вне `lsx_kernels_act.cuh`,
  `scale_half_kernel` / `scale_float_kernel` — по 2–3.

## 3. Рассмотренные подходы

**A. Нейтральный слой, фаза за фазой (выбран).** Каждый кластер — отдельная
ветка с полным барьером верификации. Охват максимальный (≈ 8k LOC), каждый
шаг независимо проверяем и откатываем; шаблон уже опробован в репозитории
(minimax m3/m3.1 и hy/hy_simulmt переиспользуют родительские
loader/kernels/pipeline через паспорт).

**B. Всё одним пакетом.** Меньше циклов ревью, но регрессия в любой фазе
блокирует все, а bisect по 8k LOC неинформативен.

**C. Только вынос ядер.** Перенести дублирующиеся `__global__`-тела в
заголовки, драйверы не трогать. Дёшево, но 5.4k LOC клона DeepSeek и
92 %-идентичные драйверы dense-трио остаются — задача не решена.

Выбран **A**.

## 4. Архитектура

| модуль | namespace | содержимое | потребители |
|---|---|---|---|
| `include/lsxcommon/lsx_dense_inference.h` + `src/lsx_dense_inference.cu` (новый) | `lsxcommon::dense` | RMSNorm, RoPE, attn decode, attn split + combine, split-QKV, SwiGLU — **алгоритм текущего granite**, тела переносятся дословно | granite, granite-switch, phi |
| `src/deepseek_moe_constants.h` (новый) + `deepseek_moe_*` (пере-темплейчены) | `lsxcommon::deepseek` (паспорт) / `lsxcommon::gpu` (драйвер) | весь V4-контур: HC/Sinkhorn, MLA + compressed KV + indexer, hash/router MoE, head | DeepSeek-V4-Flash, V4-Pro-0813 |
| `lsx_attn_inference.{h,cu}`, `lsx_kernels_common.cuh`, `lsx_kernels_act.cuh` (расширение) | `lsxcommon::attn` / `lsxcommon::kernels` | `RunAttnResDepthMix`, `F32ToF16`/`F16ToF32`/`HalfToFloat`/`FloatToHalf`/`SiluMul`/`ScaleHalf`/`ScaleFloat` | alice, kimi-k3 и 8 семейств с конверсиями |

### 4.1 Границы

- Нейтральный слой знает только форму тензоров и числовые параметры; он не
  знает паспортов, format-тегов и имён семейств. Имена ядер нейтральные
  (`dense_attn_split_kernel`, `attn_res_depth_mix_kernel`), без семейных
  стемов — это же требование уже проверяет `test_model_independence_guard`.
- Семейные заголовки остаются публичным API: `RunGranite*`, `RunPhi*`,
  `RunGraniteSwitch*` превращаются в однострочные inline-делегаты на
  `dense::*`. `kLifecycle` и `Suites()` продолжают указывать на те же ТУ.
- Драйверы слоёв (проекции, KV-кеш, цикл генерации) остаются семейными:
  раскладка весов у трёх dense-моделей разная (phi — fused `qkv`/`gate_up`,
  switch — aLoRA-плоскости), и их объединение изменило бы поведение.

### 4.2 API `lsxcommon::dense`

```cpp
namespace lsxcommon::dense {
cudaError_t RunDenseRmsNormKernel(cudaStream_t, __half* out, const __half* x,
                                  const __half* w, int rows, int hidden, float eps);
cudaError_t RunDenseRopeKernel(cudaStream_t, __half* q, __half* k,
                               const int32_t* pos, int seq, int nh, int nkv,
                               int hd, double theta);
cudaError_t RunDenseAttentionKernel(cudaStream_t, const __half* q, const __half* k,
                                    const __half* v, __half* out, float* partials,
                                    int seq, int nh, int nkv, int hd, float scale,
                                    int kv_base);          // split/legacy-диспетчер
cudaError_t RunDenseAttentionSplitKernel(cudaStream_t, const __half* q,
                                         const __half* k, const __half* v,
                                         __half* out, float* partials, int seq,
                                         int nh, int nkv, int hd, float scale,
                                         int kv_base, int splits);
cudaError_t RunDenseSplitQKVKernel(cudaStream_t, const __half* qkv, __half* q,
                                   __half* k, __half* v, int seq, int qn, int kn);
cudaError_t RunDenseSwiGLUKernel(cudaStream_t, const __half* gu, __half* out,
                                 int I, int64_t n);
}
```

Внутрь переезжают без правок: `kThreads`, `kAttnSplitMax`, `kAttnSplitKeys`,
`kAttnSplitMaxSeq = 8`, `kAttnSplitMinKv = 256`, `kNormThreads = 256`,
порог `shmem > 48 * 1024 → cudaErrorInvalidValue` и fallback
`kernels::RmsNormTreeKernel<__half, kernels::NormWeight::Direct>`.

## 5. Фазы

### Ф1 — dense-трио → `lsxcommon::dense`

Ветка `refactor/dense-trio-neutral-layer`.

1. Новый модуль `lsx_dense_inference.{h,cu}` с шестью лаунчерами §4.2 и
   семью ядрами (`dense_rmsnorm_kernel`, `dense_rope_kernel`,
   `dense_attn_kernel`, `dense_attn_split_kernel`, `dense_attn_combine_kernel`,
   `dense_split_qkv_kernel`, `dense_swiglu_fused_kernel`), перенесёнными из
   `granite_dense_inference.cu`. Отдельная OBJECT-библиотека
   (`lsxcommon_dense_cu`) по образцу прочих `.cu`.
2. Нейтральный тест `tests/test_dense_kernels.cu` — объединение общих кейсов
   `test_granite_dense_kernels.cu` и `test_phi_dense_kernels.cu`.
3. Семейные `RunGranite*`/`RunPhi*`/`RunGraniteSwitch*` становятся
   inline-делегатами; из `phi_dense_inference.cu` удаляются ~980 строк
   (ядра + дублирующие лаунчеры), из `granite_dense_switch_inference.cu` —
   ~470; `granite_dense_inference.cu` отдаёт ядра нейтральному модулю и
   сохраняет ~86 строк семейного кода, отличающего его от phi.
4. Семейные тесты ядер сохраняют только то, что действительно семейное
   (раскладка весов, fused-проекции, aLoRA-срезы).

Ожидаемая дельта: ≈ −1 400 LOC в `libs/`, ≈ −400 LOC в `tests/`.

### Ф2 — DeepSeek V4: один шаблонный драйвер

Ветка `refactor/deepseek-v4-shared-driver`.

1. `src/deepseek_moe_constants.h`: паспорт flash в форме
   `deepseek_moe_pro_constants.h` (namespace `lsxcommon::deepseek`); 18
   `constexpr` переезжают из `deepseek_moe_load.cuh` (сейчас `namespace gpu`),
   включая производные `kHcMix = (2+kHcMult)*kHcMult` и
   `kHcDim = kHcMult*kHiddenDim`.
2. Шесть общих ТУ (`inference_attn`, `inference_csa`, `inference_head`,
   `inference_moe`, `inference_pipeline`, `norm_kernels`) и `load.{cuh,cu}`
   параметризуются шаблоном:

   ```cpp
   template <class P, typename ResidualT> struct V4Driver;
   // P: deepseek::Passport | deepseek_pro::Passport
   // ResidualT: __half (flash) | float (pro)
   ```

   Файловые `constexpr`-алиасы заменяются на `P::k*`; параметр шаблона
   `ResidualT` заменяет `__half`/`float` ровно в трёх местах (`y[i] = acc`,
   `d_single` и его потребление). Per-layer таблицы
   (`compress_ratio`, hash/router-слои, indexer-слои) уже приходят из
   загрузчика в рантайме (`pw.layers[i]`) — их темплейтить не нужно.
3. Явные инстанцирования в одном ТУ:
   `template struct V4Driver<deepseek::Passport, __half>;` и
   `<deepseek_pro::Passport, float>;`.
4. Удаляются 12 файлов-копий (`deepseek_moe_pro_inference_{attn,csa,head,moe,pipeline}.cu`,
   `deepseek_moe_pro_load.{cuh,cu}`, `deepseek_moe_pro_norm_kernels.cu`,
   `deepseek_moe_pro_model.{h,cpp}`, `deepseek_moe_pro_inference_csa.h`,
   `deepseek_moe_pro_norm_kernels.h`); `constants.h` и `convert.{h,cpp}`
   остаются. `DeepSeekV4ProModel` остаётся тонкой обёрткой (registry id,
   паспорт, `PinnedWeights pw_`), но инстанцирует общий шаблон.

Ожидаемая дельта: ≈ −5 300 LOC.

### Ф3 — остатки

Ветка `refactor/kernel-primitives`.

1. `depth_mix_kernel` → `attn::RunAttnResDepthMix` в `lsx_attn_inference.{h,cu}`
   (alice и kimi-k3 переключаются, семейные копии удаляются).
2. `f32_to_f16`/`f16_to_f32`/`half_to_float`/`float_to_half`/`silu_mul`/
   `scale_half`/`scale_float` → `lsx_kernels_common.cuh` /
   `lsx_kernels_act.cuh`; семейные копии удаляются, включая заголовки,
   которые их реэкспортировали.

Ожидаемая дельта: ≈ −250 LOC.

## 6. Поток данных и обработка ошибок

Поток данных не меняется ни в одной фазе:
`safetensors → converter → Arrow-дерево → loader (PinnedWeights) → pipeline
→ tokens`. Меняются только адресаты вызовов.

Инварианты обработки ошибок (проверяются диффом, а не обещанием):

- каждый лаунчер возвращает `cudaError_t` из `cudaGetLastError()` и не
  оборачивает его; семейные inline-делегаты возвращают значение напрямую,
  чтобы код ошибки не подменялся;
- guard-ветки переносятся дословно: `shmem > 48*1024 → cudaErrorInvalidValue`,
  `vec_ok = (hidden % 2) == 0 && hidden >= 64` с fallback на
  `RmsNormTreeKernel`, clamp `splits ∈ [1, kAttnSplitMax]`, диспетчер
  `seq <= kAttnSplitMaxSeq && kv_base + 1 >= kAttnSplitMinKv`;
- в DeepSeek ошибки остаются в текущем виде
  `out.error_msg += absl::StrCat(" (", cudaGetErrorString(e), ")")`;
  `ResidualT` влияет только на присваивание, не на арифметику;
- новые ветки отказа, новые коды возврата и новые исключения не появляются.

## 7. Верификация

### Уровень 0 — «это перенос»

Скрипт сравнивает `git show <base>:<file>` с новым файлом после
нормализации имён (семейный стем → нейтральный; `granite`/`phi`/`switch` →
`X`). Требование: **0 различий** в перенесённых телах. Скрипт кладётся в
`tools/` рядом с прочими проверками и запускается вручную в каждой фазе;
его вывод — часть артефактов фазы.

### Уровень 1 — быстрые гейты

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug   # 0 warnings в наших файлах
ctest --preset linux-debug-fast                                     # 134/134
ctest --preset linux-debug-convert                                  # 37/37
```

Плюс целевые наборы фазы:

- Ф1: `test_granite_dense_kernels`, `test_phi_dense_kernels`,
  `test_granite_switch_kernels`, `test_dense_kernels` (новый).
- Ф2: `test_deepseek_moe_inference_kernels`, `test_deepseek_moe_moe`,
  `test_deepseek_moe_norm_kernels`, `test_deepseek_moe_arena`,
  `test_deepseek_moe_pro_*` (второе инстанцирование: `_convert`, `_loader`, `_model`, `_kernels`, `_dequant`, `_actquant`, `_arena`).
- Ф3: `test_alice_moe_kernels`, `test_kimi_k3_kernels` и все тесты,
  включающие мигрированные примитивы.

### Уровень 2 — реальные веса, побитово

Перед началом каждой фазы на этой машине снимается baseline: `logits_hash`,
top-1 и t/s (perf-строка) для каждой затронутой модели. Требование —
совпадение после.

| фаза | модели и гейты |
|---|---|
| Ф1 | `test_granite_dense_selfcheck` (эталон `c165e80fde6401cf`, top-1 12366, Paris ×3), `test_granite_switch_selfcheck` (эталон `fdc632042c30919c`), `test_phi_dense_selfcheck` (baseline снимается перед фазой); FSI/CODE/DET всех трёх (label `inference`, SKIP 4); `test_perf_baseline_granite_4_2_30b`, `test_perf_baseline_granite_switch_4_1_30b`, `test_perf_baseline_phi_4` |
| Ф2 | `test_deepseek_moe_selfcheck` и `test_deepseek_moe_pro_selfcheck` (Paris ×3; baseline-хеш снимается перед фазой, т.к. в доках он не зафиксирован — §R3 хранит только «identical» и t/s); FSI/CODE/DET обоих (у Pro FSI-02 RED — это зафиксированный дефект модели, критерий «не стало хуже»); `test_perf_baseline_deepseek_v4_flash`, `test_perf_baseline_deepseek_v4_pro` |
| Ф3 | SelfCheck + FSI/CODE/DET alice и kimi-k3, плюс SelfCheck всех семейств, чьи примитивы затронуты конверсиями. Правило: список моделей берётся из `git diff --name-only` по фазе (каждый файл, где удалена семейная копия примитива), и перечисляется в отчёте фазы — на момент написания спеки это granite, phi, switch, mistral, mistral-large4, xing, glm-flash, gigachat, nemotron |

Дополнительно в Ф2: `lsxforge --verify --input <flash-tree>` и то же для
pro-дерева — `problems=0`, `footer_ok = files`, exit 0. Деревья из vault
не пере-конвертируются.

### Критерий приёмки фазы

Все три уровня зелёные; ни один `logits_hash` не изменился; ни одна
perf-строка не вышла за шум; уровень 0 показал 0 различий.

## 8. CMake, реестры и гварды

Обновляется в том же коммите, что и переезд:

- `libs/lsxcommon/CMakeLists.txt`: новая OBJECT-библиотека
  `lsxcommon_dense_cu`, удаление pro-объектов в Ф2.
- `tests/CMakeLists.txt`: `lsx_add_cuda_test(test_dense_kernels ...)`,
  правки списков источников.
- `test_model_suite_guard.cpp::Suites()`: переименование/удаление
  тест-таргетов Pro в Ф2, появление `test_dense_kernels` в Ф1.
- `test_model_independence_guard.cpp`: `kPassports` — запись
  `deepseek-v4-flash` переезжает на `deepseek_moe_constants.h` /
  `lsxcommon::deepseek`; `kLifecycle` — ТУ остаются те же.
- `test_absl_container_guard.cpp`: новый `lsx_dense_inference.h` достижим
  из `.cu`, abseil-контейнеров в нём быть не должно (проверяется
  автоматически).
- `test_infrastructure_guard.cpp`: один definition на общий хелпер, никаких
  `${CMAKE_BINARY_DIR}/_deps`, arch только из `LSX_CUDA_ARCH`.

## 9. Ветки и коммиты

Три последовательные ветки, каждая мержится только после зелёного барьера:

1. `refactor/dense-trio-neutral-layer`
2. `refactor/deepseek-v4-shared-driver`
3. `refactor/kernel-primitives`

Внутри ветки — три коммита: (1) нейтральный модуль + тест, (2) переключение
потребителей, (3) удаление копий + реестры + CMake. Префиксы: `refactor:`
для переноса, `test:` для нейтральных тестов и реестров, `docs:` для карты
репозитория и AGENTS. Коммиты и мержи — только по явному запросу оператора.

## 10. Риски и их снятие

| риск | снятие |
|---|---|
| время сборки растёт от двух инстанцирований шаблона (Ф2) | явные инстанцирования в одном ТУ; при заметном росте — по ТУ на инстанцирование; build time до/после фиксируется в отчёте фазы |
| гварды кодируют имена (`kPassports`, `Suites()`, naming) | обновляются в том же коммите; падение гварда из-за нового нейтрального имени — сигнал, что слой получил семейное имя, а не повод ослабить гвард |
| незаметный числовой дрейф | уровень 0 (нормализованный diff = 0) + `logits_hash`; шаг не принимается «по глазам» |
| новый заголовок ломает NVCC (abseil-контейнеры) | `test_absl_container_guard` + правило «headers reachable from .cu не включают `<absl/container/*>`» |
| ошибки в порядке free при удалении pro-файлов | Ф2 удаляет только копии; `Free()`-контракт (включая освобождение shared-expert блобов ~4 GiB, §R3) остаётся в общем ТУ и проверяется повторной загрузкой двух моделей в одном процессе |
| perf-регресс от изменения инлайнинга | побитовый барьер уровня 2 + perf-строки; регресс = откат |

## 11. Что не меняется

- Публичные заголовки семейств, `kType`/`kIndexFormat`/format-теги.
- Раскладка Arrow-деревьев и vault: пере-конвертация не нужна,
  `lsxforge --verify` даёт те же `files/footer_ok/deep/problems`.
- CLI, флаги, env-knobs, поведение стриминга и отмены.
- Числовая семантика: ни одна формула, порог или код ошибки не
  редактируются.

## 12. Документация и артефакты

В тех же ветках: `AGENTS.md` §Structure (новый нейтральный слой, исчезнувшие
pro-файлы), skill `logestix-repo-map` (per-file inventory), `CHANGELOG.md`.
По завершении каждой фазы — запись в `docs/open-tasks/YYYY-MM-DD-<slug>.md`
с измеренными числами: сколько LOC ушло, какие хеши сняты до/после,
build time до/после, и вывод уровня 0.

## 13. Вне объёма (YAGNI)

Осознанно не входит в этот цикл; при необходимости — отдельные спеки:

- RAII для device-памяти и общий `LSX_CUDA_CHECK` (462 `cudaMalloc` /
  785 `cudaFree` / 3 212 ручных проверок) — это класс «защита от дефектов»,
  а не дублирование;
- разбиение god-файлов (`gemma_moe_inference.cu` 3 272,
  `glm_moe_flash_inference.cu` 3 073) — размер, а не дублирование;
- `.clang-format`/`clang-tidy`, сантайзер-стадия в CI, расширение фаззинга;
- слияние `deepseek_moe_v41_*` с общим драйвером: v41 структурно отличается
  (нормализованный diff — 1.5–1.8k строк на файл), это отдельный цикл.
