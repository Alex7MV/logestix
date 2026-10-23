# Аудит инференса logestix — 2026-10-22

Ревью кода инференса + прогон доступных реальновесовых гейтов на вольте
`/mnt/llm_models/models`. Хост: RTX 5090 32 GiB, driver 595.99.02, Linux.
Сборка: `linux-release-znver4-aocc` (AOCC clang 17, CUDA 13.4, CC 12.0).

Внешний оракул — офлайн-выгрузка NVIDIA-документации
`/home/alexmv2025/projects/cuda/docs/` (NVFP4/Memory, Hopper, Ampere, Rubin)
и `docs/NVIDIA_DOCS.md`; инварианты из неё (E2M1/UE4M3, 48 warps/SM, 100 KB
smem, один поток/порядок редукций, правила `cudaMallocAsync`) использованы
в чек-листе. Runtime-оракулы: Paris/FSI/CODE/DET и хеши `final_logits`.

---

## 1. Что проверено

| Слой | Файлы |
|---|---|
| NVFP4 dispatch / GEMV / TC | `lsx_inference_nvfp4.{h,cu}`, `lsx_tc_gemm.cu`, `lsx_gemv.cu`, `nvfp4_*` |
| attention / MLP / KV | `lsx_attn_inference.{h,cu}`, `lsx_kv_cache.h`, `qwen_dense_inference_pipeline.cu` |
| routed MoE | `lsx_moe_inference.{h,cu}`, `expert_device_cache.h`, `expert_store.h` |
| per-family pipelines | 22 зарегистрированных семейства |
| lifecycle / engine | `model_base`, `model_engine`, `model_factory`, `resource_governor` (обзорно) |
| сборка/тесты | fast-набор, реальновесовые selfcheck/FSI/CODE/DET |

17 моделей в вольте; сконвертированное `logestix/`-дерево есть у 13.
Без дерева (только исходники или отсутствуют): DeepSeek-V4-Flash,
DeepSeek-V4.1-Flash, GigaChat-3.5-432B, GLM-5.3-Flash, Qwen3.5-35B-A3B,
Kimi-K2.7-Code, MiMo-V2.6-Flash-RL, GLM-5.2. Для них — только статический
аудит.

## 1a. Сверка с CUDA-документацией

Проверено против `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md`
(и `NVIDIA_DOCS.md` из репозитория):

- **Кодек.** E2M1: nibble 0..7/8..F = 0,0.5,1,1.5,2,3,4,6 и знак; упаковка
  «lo nibble first» (`nvfp4_codec.h`, `lsx_kernels_quant.cuh`). UE4M3-скейлы —
  полная таблица E4M3FN, включая субнормали (`e==0 → m·2^-9`, ноль только
  `0x00`), бит 7 не выставляется. Совпадает с §1/§B2 документа; отклонений не
  найдено.
- **tcgen05 отсутствует на CC 12.x** — репозиторий использует
  `mma.sync.aligned...kind::mxf4nvf4.block_scale` (`lsx_tc_gemm.cu`), что
  соответствует §B4 (sm_120a/sm_121a не имеют tensor-memory MMA). TMA в
  аудированных decode-путях не используется.
- **48 warps/SM, 100 KB smem** — максимальные блоки в аудированных ядрах
  ≤1024 потоков, динамическая smem далеко ниже лимита; `__launch_bounds__(256)`
  на warp-per-row GEMV. Нарушений не найдено.
- **Детерминизм.** На пути к `final_logits` нет `atomicAdd`; редукции —
  thread-0/shuffle/shmem с барьерами, порядок фиксирован (исключение —
  латентная гонка F-10). `atomicAdd` в `xing_moe_inference.cu` — только в
  debug-счётчике non-finite, не в логарифмах.
- **`__device__`-символы.** В аудированных путях адреса резолвятся через
  `cudaGetSymbolAddress` (gigachat `RouterScratchDev`); передачи символа в
  launch не найдено.
- **`cudaMallocAsync`.** Освобождение — на том же стриме, где аллокация
  (split-KV attention, router). Единственное отклонение — общий
  `g_splitk_partials` (F-12).

## 1b. Сводка находок

| ID | P | Область | Суть |
|---|---|---|---|
| F-01 | P1 | Mistral/Xing | продолжение сессии пишет KV/`scores` за границу кэша (подтверждено memcheck) |
| F-15 | P1 | нейтральный MoE | fused w13 пишет `2I` в `d_yg` размером `I`; Mistral/standalone — реальный OOB (memcheck) |
| F-02 | P1 | 4 семейства | семантика continuation расходится; Mistral подаёт позицию вместо token id; Granite сбрасывает контекст |
| F-03 | P1 | E1 guard | счётчик `FastMathDepth` — по копии на TU; ложные VIOLATION в каждом FSI/DET, реальную конкуренцию может не увидеть |
| F-04 | P1 | численный контракт | `--bit-exact` использует `gemv_large`; хеши меняются от env-ручки |
| F-05 | P2 | GLM-Flash/Kimi/Gemma | процесс-глобальные кэши не сбрасываются на reload/втором инстансе |
| F-06 | P2 | MoE handshake | busy-poll проглатывает CUDA-ошибку, нет abort/таймаута |
| F-07 | P2 | decode perf | `cudaMallocAsync` на слой/токен, `std::vector pos` на слой, `getenv` в цикле |
| F-08 | P2 | сборка/запуск | относительный RPATH пресета ломает все app-бинарники |
| F-09…F-14 | P3 | attention/KV/perf/docs | валидация `head_dim`, гонка GDN, fallback swiglu, общий split-K буфер, drift §V, zero-copy без проверки |

Главный вывод: **ядерная математика NVFP4/attention/MoE на реальных весах
здорова** (12/13 Paris — единственный FAIL это известный granite §U; все
прогнанные FSI/CODE/DET, кроме трёх известных RED, зелёные; memcheck
GEMV/dispatch — 0 ошибок). Основные риски — не в арифметике, а в
**жизненном цикле сессии/памяти** (F-01, F-15, F-02) и в **инструментах,
которые должны были это поймать** (F-03, F-04). Оба P1-дефекта памяти
найдены только `compute-sanitizer`, а не зелёным selfcheck.

## 2. Сборка и fast-набор

`cmake --build --preset linux-release-znver4-aocc` — 569/569, без ошибок.

`ctest --preset linux-release-znver4-aocc -L fast` **без** `LD_LIBRARY_PATH`
даёт 82/94: падают все 12 app-тестов с
`libarrow_cuda.so.2600: cannot open shared object file`. С
`LD_LIBRARY_PATH=<build>/_deps/arrow-build/release` — **94/94**.

Это не код инференса, а дефект пресета (F-08): `CMakePresets.json:77-78`
задаёт `CMAKE_INSTALL_RPATH="$ORIGIN/../_deps/arrow-build/release"`, что
верно только для бинарников в `build/<preset>/tests/`; бинарники в
`apps/<app>/` и `apps/<app>/tests/` видят `apps/_deps` и `apps/<app>/_deps`.
Скилл `building-the-project` требует **абсолютный** путь.

## 3. Runtime-матрица (реальные веса)

`PASS`/`SKIP`/`FAIL` — ctest-вердикт; `SKIP` = нет сконвертированного дерева
или голдена. GPU эксклюзивно, тесты последовательно.

| Семейство (vault) | Paris | FSI | CODE | DET |
|---|---|---|---|---|
| qwen3.5-9b | PASS | PASS | PASS | PASS |
| qwen3.8-27b | PASS | PASS | PASS | PASS |
| granite-4.2-30b | PASS* | PASS | PASS | PASS |
| phi-4 | PASS | PASS | PASS | **FAIL** |
| gemma-4-31b-it (dense) | PASS | **FAIL** | PASS | PASS |
| qwen3.6-35b-a3b | PASS | PASS | PASS | PASS |
| ornith-1.5-35b-a3b | PASS | PASS | PASS | PASS |
| gemma-4-26b-a4b-it (MoE) | PASS | **FAIL** | **FAIL** | PASS |
| nemotron-3.5-lightning-30b-a3b-nvfp4 | PASS | SKIP | SKIP | SKIP |
| mistral-small-4-119b | PASS | SKIP | SKIP | SKIP |
| qwen3.8-flash-next | PASS | PASS | PASS | PASS |
| xing4.0-29b-a4b | PASS | PASS | PASS | PASS |
| alice-foundation-80b-a3b | PASS | PASS | PASS | PASS |

CODE/DET у alice прогнаны 2026-10-23 после включения streaming-GEMV в strict
режиме: оба зелёные (CODE 151 s, DET 94 s). `PASS` у nemotron/mistral в колонке `Paris` —
это базовый selfcheck. `SKIP` у nemotron/mistral — задекларированный golden-debt
(OPEN_TASKS §K): файлов `tests/fixtures/{fsi,code,det}/<model>.json` нет,
тест не запускает модель. Остальные `FAIL` воспроизводят известные открытые
пункты, новых среди гейтов нет:

- **granite**: на момент аудита Paris проходил с `logits_hash=3d1604aa75c99a1f`,
  но tool-round-trip давал `get_weather.x` вместо `get_weather` — §U.
  Исправлено в этой сессии нативным `granite_tool_format` (см. §5), хеш
  не изменился; `test_granite_dense_selfcheck` снова зелёный.
- **gemma-4-26b (MoE)**: FSI-TEST-01/02 — голдены требуют символьную
  формулу, модель выдаёт численную; CODE — тот же класс расхождения —
  OPEN_TASKS §J.
- **phi-4**: DET-TEST-02 ожидает `CATASTROPHIC`, модель выдаёт `CRITICAL`
  (детерминированно, 3/3) — OPEN_TASKS §O.
- **gemma-4-31b (dense)**: FSI-TEST-01 — арифметическая формула не
  конкретизирована (`PV=CF*(...)` вместо `PV=2500000*(...)`) — OPEN_TASKS §R.
  CODE/DET зелёные.

## 4. Дефекты

Приоритеты: **P1** — численная/память/контракт, исправлять до следующего
релиза; **P2** — латентный дефект или существенная стоимость; **P3** —
hardening/долг.

### F-01 (P1) — Продолжение сессии у Mistral и Xing пишет за границу KV-кэша

Обе модели переиспользуют KV-кэш между запросами через `DecideSession`
(`qwen::SessionDecision`), но, в отличие от Qwen (`QwenKvCache::EnsureCapacity`),
**не растят** его. Granite `EnsureCapacity` вызывает, но с неверным `needed`
(см. F-02).

- `mistral_moe_model.cpp:205` считает `capacity` для нового запроса, но
  использует её только при `!session.is_continuation`; на продолжении
  `session_` не переаллоцируется (строки 223-242), а `pos` начинается с
  `session_->length` (`:255`).
- `MistralDecodeStep` пишет KV без проверки `pos < capacity`:
  `mistral_moe_inference.cu:249` (`K[kg.Offset(layer,h,pos,d)] = k`),
  `V[...]` там же; `mla_attn_kernel` пишет `sc[p]` при `p <= pos`, где
  `sc = scores + h*capacity` (`:276`). При `pos == capacity` это запись в
  первую строку следующего KV-слоя, а для последнего слоя/последней головы —
  за пределы аллокации. Строка вызова — `:724`.
- Xing: `xing_write_cache_kernel` (`xing_moe_inference.cu:287-310`, вызов
  `:999`) пишет `kb`/`vb` по `kgeom.Offset(0,h,pos)` **до** проверки
  `capacity < n_pos` в `LaunchXingAttention` (`xing_moe_kernels.cu:802-805`).
  Проверка возвращает `InvalidValue` уже после записи; сама запись
  разрушает следующий слой KV.
- Роста нет и у Xing: `xing_moe_model.cpp:198-229` переаллоцирует сессию
  только при `!is_continuation`.

Репро (синтетический Mistral, `MistralDecodeStep` на `pos == capacity`)
подтверждён `compute-sanitizer --tool memcheck` — см. Приложение A.3.

**Исправление:** добавить `EnsureCapacity`-эквивалент (по образцу
`QwenKvCache::EnsureCapacity`) либо жёсткий `pos + seq <= capacity` до
любой записи; для Xing — проверку до `xing_write_cache_kernel`.

### F-02 (P1) — Семантика продолжения сессии расходится между семействами

`DecideSession` (`lsx_attn_inference.h:26-37`) подставляет
`gen_input = {kv_last_tok}` — один токен, который надо дописать в кэш.
Корректность зависит от того, что цикл генерации **не** декодировал
последний сэмплированный токен.

| Семейство | `session_tail_` | Что подаётся на продолжении | Проблема |
|---|---|---|---|
| qwen3.5-9b / qwen3.6 / ornith / qwen3.8-flash-next | `gen_tokens` (`qwen_base_model.cpp:159`, `qwen_moe_model.cpp:313`, …) | последний токен, `pos = kv_rows` | корректно: последний токен не декодирован |
| granite-4.2 | `gen_tokens` (`granite_dense_model.cpp:156`) | последний токен | `EnsureCapacity` зовётся с `P + max_new` (P=1 на продолжении), а не `kv_rows + …` (`granite_dense_inference.cu:867-874`); префил идёт на **pos 0**, `kv_rows = seq` (`:900-930`) → контекст сессии затирается/сбрасывается |
| mistral | `feed + tokens` (`mistral_moe_model.cpp:318-319`) | `session_->length - 1` (`:222`) | в `kv_last_tok` уходит **позиция**, а не id токена; плюс дубль последнего токена (он уже декодирован в цикле, `:312`) |
| xing | `feed + tokens` (`xing_moe_model.cpp:401-402`) | `session_->token_id` (корректный id) | последний токен уже декодирован в цикле (`:395`) → он попадает в контекст дважды |

Дополнительно: сам триггер продолжения **несовместим между семействами** —
Qwen/Granite требуют `input_ids == gen_tokens`, Xing/Mistral —
`input_ids == prompt + gen_tokens`. Один и тот же клиентский запрос не может
корректно продолжить сессию Qwen и Xing.

Ни один тест не покрывает продолжение: `test_*_model` вызывают `Infer`
один раз (или с тем же `input_ids`, что не совпадает с `session_tail_`).
Контракт стоит либо закрепить тестом на каждое семейство, либо убрать
`DecideSession` из публичного пути до нормализации.

### F-03 (P1) — Guard E1 (`LsxFastMathScope`) не является process-global и даёт ложные срабатывания

`Detail::FastMathDepth()` определён в **анонимном namespace в заголовке**
(`lsx_inference_nvfp4.h:174-181`), т.е. каждая единица трансляции получает
свой `static std::atomic<int>`. В бинарнике `test_gemma_dense_fsi_selfcheck`
`nm -C` показывает **66** разных `...::(anonymous namespace)::FastMathDepth()`.
Счётчик не является process-global: `EnterFastMathScope`/`LeaveFastMathScope`
inline-функции, но обращаются к TU-локальной копии; в зависимости от
инлайна/слияния COMDAT вход и выход скоупа могут попасть в разные копии.

Наблюдаемый эффект на реальных прогонах (один поток!):

- `test_gemma_dense_fsi_selfcheck`, 3 последовательных `Infer` →
  `VIOLATION: 2`, затем `VIOLATION: 3`;
- `test_phi_dense_det_selfcheck`, 3 промпта × 3 stress-прогона = 9
  `Infer` → `VIOLATION: 2,3,4,5,6,7,8,9`.

То есть каждый mutating-скоуп «утекает»: счётчик монотонно растёт на число
`Infer`, хотя вложенности нет. Это ложные срабатывания, а не конкуренция;
одновременно это означает, что настоящую конкуренцию (два потока) guard
может **не увидеть**, если потоки используют разные копии. Guard-тест в
`tests/test_lsx_nvfp4_dispatch.cu` проходит только потому, что он сам живёт
в одной TU: enter/leave там резолвятся в один и тот же локальный счётчик.

Дополнительно реализация не различает вложенность и второй поток (проверка
только `d > 1`), хотя `docs/open-tasks/2026-10-17-lifecycle-e1-e2-closed.md`
утверждает обратное.

**Исправление:** один объект с external linkage —
`inline std::atomic<int> g_lsx_fast_math_depth{0};` в `namespace Detail`
(C++17 inline variable), либо определение в одном `.cpp/.cu`; при желании
хранить `std::thread::id` владельца, чтобы отличать вложенность от второго
потока.

### F-04 (P1) — `--bit-exact` (`kOff`) не выключает `gemv_large`

`ResolvePerfKnobs(kOff)` (`gpu_perf.cpp:152-165`) сбрасывает только
`tc_fast_math_default` и `seq1_gemv_default`; `gemv_large_default` остаётся
`true`. `LsxFastMathScope(kOff)` (`lsx_inference_nvfp4.h:142-160`) также не
трогает `gemv_large`. А диспетчер `RunLsxNvfp4Gemm`
(`lsx_inference_nvfp4.cu:894-898`) выбирает `gemv_large` без проверки
fast-math:

```cpp
const bool gemv_large =
    LsxGemmLargeActive() && seq == 1 && ... && K >= 4096 && N >= 4096;
```

Поэтому в строгом режиме для форм `K>=4096 && N>=4096` выполняется
tolerance-class streaming GEMV вместо скалярного/тайлового пути.

Эмпирическое подтверждение (Qwen3.5-9B, `--selfcheck --bit-exact`):

| окружение | `logits_hash` | токен |
|---|---|---|
| по умолчанию (policy `gemv_large=true`) | `c58a9e99ed47cbbb` | Paris |
| `LSX_GEMV_LARGE=0` | `3e44e44d8375058a` | Paris |

Один и тот же `--bit-exact` даёт разные биты в зависимости от perf-ручки.
`tests/test_gpu_perf.cpp:119-127` проверяет только `tc`/`seq1`, поэтому
гейт зелёный.

Тот же класс — env-ручка `LSX_GEMV_HEAD`: `kGemvHead` вычисляется из
`getenv` без проверки fast-math (`lsx_inference_nvfp4.cu:879-880, 903-904`) и тоже
включает tolerance-class GEMV в `kOff`.

**Исправление:** в `kOff` (и в legacy-режиме `LSX_PERF_AUTO=0`) очищать
`gemv_large_default`; как минимум добавить `LsxTcFastMath()` в условия
`gemv_large` и `gemv_head`, расширить `test_gpu_perf`. После исправления
перезаписать `r3`-хеши в `tests/perf/perf_baselines.json`.

### F-15 (P1) — Fused w13 GEMM пишет 2I в буфер `d_yg` размером I

`RunMoERoutedFFN` в TC-ветке `tc_fused` делает **один** GEMM `w13 = [w1|w3]`
с `N = 2*I` в `d_yg` (`lsx_moe_inference.cu:578-586`), а затем SwiGLU читает
`d_yg` (gate) и `d_yu` (up) (`:629-630`). Контракт вызывающего кода —
`d_yg` длиной `I`:

- Qwen-MoE/Ornith/Qwen3.8-Flash-Next: `d_yg` и `d_yu` — соседние срезы одной arena
  (`half_sz[21]=I`, `half_sz[22]=I`; `qwen_moe_inference_pipeline.cu:173-174`),
  поэтому «перелив» 2I из `d_yg` физически попадает в `d_yu`. Это работает,
  но держится на раскладке arena и нигде не задокументировано.
- Mistral: `d_yg` и `d_yu` — **отдельные** `cudaMalloc`
  (`mistral_moe_inference.cu:468-471`, по `I` половин каждый). Запись
  `2I` выходит за `d_yg`; `compute-sanitizer` на синтетическом Mistral:

```
Invalid __global__ write of size 2 bytes
  at TcGemmKernel(...) in lsx_tc_gemm.cu:720
  Access ... is 1 bytes after the nearest allocation of size 64 bytes
  Host: RunMoERoutedFFN -> MistralDecodeStep ...
ERROR SUMMARY: 68 errors
```

`64 bytes` — это `d_yg` для `I=32`; в продакшене `I=2048`, т.е. перелив
4 KB. Selfcheck Mistral проходит только потому, что два одинаковых
`cudaMalloc` обычно оказываются рядом; при другой раскладке (или на другой
платформе) up-половина читается из мусора. Это же относится к любым
standalone-вызовам `RunMoERoutedFFN` с раздельными `d_yg`/`d_yu`.

**Исправление:** либо выделять `d_yg` как `2I` во всех вызывающих
(и документировать это в сигнатуре), либо передавать в fused-ветку отдельный
`d_ygu`/использовать `d_yg + I`, а не `d_yu`.

### F-05 (P2) — Процесс-глобальные кэши GLM-5.3-Flash не сбрасываются при reload

`glm_moe_flash_inference.cu` держит между вызовами `RunFlashGeneration`:

- `d_embed_h` (`:1537`): пересоздаётся только если `embed_elems < V*H`;
  смена чекпоинта с той же геометрией оставит **старую** таблицу эмбеддингов;
  `Unload` её не освобождает;
- `kda_states` (`:1554`), `dsa_kv_k/v` + `dsa_cap` (`:1562-1580`), `arena`
  (`:1583`): живут до конца процесса; второй экземпляр модели или повторный
  `Load` переиспользует чужие буферы;
- `d_rscratch` (`:1320`): grow-only, `cudaFree` при росте может выполняться,
  пока предыдущий поток ещё читает буфер.

Модель не сконвертирована в вольте, поэтому runtime-оракула нет; это
статическая находка. Тот же класс — `Kimi` (`kimi_moe_inference.cu:527-580`,
grow-only scratch), `gemma_moe` (`GemmaExpertDb *g_expert_db`,
`GemmaGenContext *g_gen_ctx`, `gemma_moe_inference.cu:1426/1519`),
`lsx_tc_gemm.cu:868` (`g_splitk_partials`).

### F-06 (P2) — Busy-poll на `cudaEventQuery` проглатывает ошибки и не проверяет abort

Шесть мест: `qwen_moe_inference_pipeline.cu:500`, `ornith_moe_inference_pipeline.cu:513`,
`qwen_moe_exp_inference.cu:2094`, `gemma_moe_inference.cu:2044`,
`kimi_moe_inference.cu:218/1928`.

```cpp
cudaError_t q;
while ((q = cudaEventQuery(tok_ready)) == cudaErrorNotReady) {}
(void)q;
```

Если записанный ранее kernel упал, `q` будет, например,
`cudaErrorLaunchFailure`; цикл выходит, ошибка отбрасывается, и дальше
читаются `h_slots` (роутер) или `next` (сэмплер) — возможно, устаревшие.
Таймаута/проверки `gs.Aborted()` в спин-цикле нет. `qwen_moe_exp` хотя бы
валидирует `eid` после этого (`:2097-2100`), остальные — нет.

### F-07 (P2) — Аллокации и `getenv` на пути декода (стоимость/латентность)

- `RunAttnFullAttnKernelSplitKV` — `cudaMallocAsync`+`cudaFreeAsync` на
  каждый вызов (`lsx_attn_inference.cu:490-510`), т.е. на слой на токен при
  контексте > 512.
- `RunMoERouterTopK(seq==1)` — `cudaMallocAsync` на слой на токен
  (`lsx_moe_inference.cu:320-335`).
- `RunAttnFusedMLP` — две `cudaMallocAsync` на слой на токен в
  prequant-ветке (`lsx_attn_inference.cu:1598-1602`); для qwen3.5-9B/27B она
  не срабатывает (их формы попадают в `down_warprow`), но сработает для
  любой dense-формы с `I < 4096` или `H < 2048`.
- `Kimi` читает `KIMI_TRACE_FILE`/`KIMI_DIAG_*` внутри цикла по токенам
  (`kimi_moe_inference.cu:1351-1418`).
- `granite`/`phi`/`gemma dense`/`gemma moe`/qwen-dense full-recompute
  создают `std::vector<int32_t> pos` **на каждый слой** и копируют его
  `cudaMemcpyAsync` в `ws.pos` (`granite_dense_inference.cu:884`,
  `phi_dense_inference.cu:884`, `gemma_dense_inference.cu:989`,
  `gemma_moe_inference.cu:1854`, `qwen_dense_inference_pipeline.cu:355`).
  Для pageable-памяти драйвер делает staging синхронно, поэтому UAF нет, но
  это лишний host-alloc и синхронная подготовка на каждом слое; в
  incremental-пути Qwen тот же паттерн уже вынесен в pinned `h_step`.

Ни один из этих пунктов не меняет численный результат; это стоимость
латентности/CPU. Для 5090 измеримо только профайлером (nsys), но паттерн
противоречит правилу «одна стрим-упорядоченная арена на запрос».

### F-08 (P2) — Пресет `linux-release-znver4-aocc` ломает запуск app-бинарников

См. §2. `CMakePresets.json:77-78`: относительный `$ORIGIN` RPATH.
`cmake --preset … && ctest --preset … -L fast` в чистом окружении даёт 12
loader-ошибок. Лечится либо абсолютным `CMAKE_INSTALL_RPATH` (как в скилле
`building-the-project`), либо `$ORIGIN/../..` для app-целей, либо
`LD_LIBRARY_PATH` в test-пресете. Это блокирует `lsxbenchmark`/`lsxfabric`
при обычном запуске без ручной настройки окружения.

### F-09 (P3) — Нейтральный non-split attention не валидирует `head_dim` как степень двойки

`RunAttnFullAttnKernel` (`lsx_attn_inference.cu:271-289`) принимает любой
`head_dim` и запускает `threads = min(head_dim, 512)`, а
`full_attn_softmax_kernel` делает tree-редукцию суммы по `threads/2`.
Для `head_dim`, не являющегося степенью двойки (например, 192 у MLA-моделей),
дерево не покрывает все элементы → неверный softmax без ошибки. Split-KV
вход эту проверку делает (`:468-490`: `(head_dim & (head_dim-1)) != 0`).
Сейчас нейтральный путь вызывают Qwen/Ornith с `HD=256`, так что дефект
латентный, но публичный entry допускает 192.

### F-10 (P3) — Гонка состояния GatedDeltaNet при `seq > 1` + `col_seed`

`qwen_gated_delta_kernel` (`lsx_attn_inference.cu:778-833`) читает
`col_seed` в начале каждого блока и пишет финальное состояние из блока
`t == seq - 1`. При `seq > 1` и непустом `col_seed` нет grid-barrier между
чтением и записью: блок `t = seq-1` может завершиться раньше, чем другой
блок прочитает исходное состояние. Сейчас все пайплайны вызывают
linear-attention с `seq == 1` (Qwen/Ornith/Alice), а full-recompute — с
`col_seed == nullptr`, поэтому гонка не воспроизводится. Нейтральный API
об этом не предупреждает; нужен либо контракт «`col_seed` ⇒ `seq==1`», либо
раздельная запись состояния на отдельном шаге.

### F-11 (P3) — Двойная активация в fallback `RunAttnFusedMLP`

`lsx_attn_inference.cu:1588-1616`: `qwen_swiglu_quant_kernel` **перезаписывает**
`d_gate` активированным значением, затем при
`NotSupported/InvalidValue` от `RunLsxTcGemmPreQuant` код «падает в fp16
путь» и снова вызывает `qwen_swiglu_kernel(d_gate, d_up, d_gate)`, т.е.
считает `silu(silu(gate)*up) * up`. Сейчас `TcGemmDispatch` возвращает
`NotSupported` только при `prop.major != 12`, а сборка/политика нацелены на
CC 12, поэтому на 5090 путь недостижим; на будущей архитектуре (Rubin,
CC 11) или при форсированном `LSX_TEST_FAST_MATH` — это тихая порча
результата. Fallback должен восстанавливать pre-activation значения
(или вычислять swiglu заново из `d_gate`/`d_up` до перезаписи).

### F-12 (P3) — `g_splitk_partials` разделяется между стримами и освобождается при росте

`lsx_tc_gemm.cu:865-882`: один grow-only буфер на процесс; комментарий
утверждает «each GEMM's partials region is disjoint», но регион у всех
вызовов один и тот же. Два split-K GEMM на разных стримах перезапишут
partials друг друга; `cudaFree` при росте может освободить буфер, пока
предыдущий kernel ещё пишет. При текущем sequential-dispatch и одном
compute-стриме не воспроизводится, но комментарий вводит в заблуждение и
мутекс не защищает in-flight kernel. Нужен либо per-stream буфер, либо
явный контракт «split-K только на одном стриме».

### F-13 (P3) — Документация противоречит коду: §V и qwen3.8 pinned arena

`docs/OPEN_TASKS.md` §V (строки 1055-1065) заявляет, что арена убрана и
`qwen3.8` занимает 10.9 GB с загрузкой 9.9 s. В том же файле §V1
(`open-tasks/2026-10-14-qwen38-arena-required-negative.md`) фиксирует
обратное: qwen3.8 **остаётся на pinned arena 63.6 GB**, views дают 9.2 t/s.
Код это подтверждает: `qwen_moe_exp_model.cpp:125` вызывает
`LoadQwen4ExpExpertStore`, который делает два прохода по 24 576 файлам и
`cudaMallocHost` всего дерева (`qwen_moe_exp_load.cpp:874-1040`); наш
прогон `test_qwen_moe_exp_selfcheck` длится 257 s, из которых загрузка —
доминирующая часть. Стоит привести §V в соответствие с §V1.

### F-14 (P3) — Zero-copy Arrow-загрузка не проверяется

`LoadTensorBatchViews`/`LoadDenseNvfp4BatchViews`/`LoadSharedExpertBatchViews`
(`schema.cpp:831-945`) берут `LargeBinaryArray::GetView()` и сохраняют
указатели, но не проверяют, что буферы лежат внутри mmap (ср. `in_range`
в `tokenizer.cpp`, которого требует скилл `nvfp4-blackwell-conventions`).
Lifetime безопасен, потому что `TensorBatchViews::keepalive` держит
`RecordBatch`, но утверждение «zero-copy» не верифицировано; при Arrow,
вернувшем heap-копию, это останется незамеченным (только тише/медленнее).

F-01, F-15, F-02 (continuation), F-03 и F-04 **не значатся** в
`docs/OPEN_TASKS.md` на 2026-10-22 — это новые находки этого аудита; их
стоит завести как отдельные пункты после решения по приоритету. F-05…F-14 —
частично известный долг (§V/§U/§K) либо hardening, зафиксированы здесь,
чтобы не потерять контекст.

---

## 5. Известные RED, подтверждённые аудитом

| Пункт | Гейт | Симптом | Статус |
|---|---|---|---|
| OPEN_TASKS §U | granite Paris | `get_weather.x` вместо `get_weather`; хеш Paris стабилен | воспроизведён |
| OPEN_TASKS §O | phi DET-02 | `CATASTROPHIC` → `CRITICAL`, детерминированно | воспроизведён |
| OPEN_TASKS §R | gemma-4-31b FSI-01 | формула не конкретизирована | воспроизведён |

Отдельно: AGENTS.md утверждает, что `test_granite_dense_selfcheck` проходит;
на этом дереве он **падает** (см. таблицу §3). Это расхождение документации
с реальностью, а не новый дефект.

## 6. Пробелы покрытия

- 8 семейств без сконвертированного дерева (см. §1) — только чтение кода.
- Нет теста продолжения сессии ни у одного семейства (F-02), поэтому
  дефекты continuation не видны в зелёном наборе.
- Длинный контекст (>512) проверяется только split-KV-путём в selfcheck
  (Paris ~ десяток токенов); нужны отдельные гейты на 2k/8k/32k токенов,
  особенно для sliding-window (gemma), YaRN (xing/mistral/gigachat) и
  compress-attention (deepseek-v41).
- `compute-sanitizer` не входит в CI; найденный F-01 без него не
  обнаруживается (запись идёт в соседний слой, ошибки CUDA нет).
- Perf-бюджеты проверяют отдельные ядра, но не декод end-to-end на
  «одна арена/запрос»; F-07 виден только в nsys.

## 7. Рекомендованный порядок

1. F-01/F-15/F-02 — `EnsureCapacity`/границы + корректный `kv_last_tok` +
   единый контракт `session_tail_`; `d_yg` = 2I (или отдельный `d_ygu`).
   Регрессионные тесты «второй Infer продолжает первый» для
   qwen/granite/mistral/xing + `compute-sanitizer` в CI.
2. F-03 — вынести счётчик E1 в один external-linkage объект; тест на двух
   потоках и на вложенность в **разных** TU.
3. F-04 — выключить `gemv_large` в `kOff`; обновить `test_gpu_perf` и
   перезаписать `r3`-хеши.
4. F-08 — починить RPATH пресета (или добавить env в test-пресет).
5. F-05/F-06/F-11/F-12 — статические hardening-правки; для F-05 нужен
   vault GLM-5.3-Flash, чтобы закрыть runtime-оракулом.
6. F-09/F-10/F-13/F-14 — по мере касания файлов.

## Приложение A — команды и evidence

### A.1 Сборка и fast

```bash
cmake --build --preset linux-release-znver4-aocc -j 16        # 569/569
# без LD_LIBRARY_PATH: 82/94 (12 app-тестов, loader)
LD_LIBRARY_PATH=$PWD/build/linux-release-znver4-aocc/_deps/arrow-build/release \
  ctest --preset linux-release-znver4-aocc -L fast            # 94/94
```

### A.2 kOff / gemv_large

```bash
B=build/linux-release-znver4-aocc/apps/lsxbenchmark/lsxbenchmark
M=/mnt/llm_models/models/Qwen3.5-9B
$B --model $M --selfcheck --bit-exact
# logits_hash=c58a9e99ed47cbbb
LSX_GEMV_LARGE=0 $B --model $M --selfcheck --bit-exact
# logits_hash=3e44e44d8375058a
```

### A.3 Continuation OOB (синтетический Mistral)

Временный тест-пробник (не входит в коммит; удалён после прогона) вызывал
`Infer` дважды: первый запрос `{5,7,9}`/`max_gen=2`, второй — с
`input_ids = prompt + gen_tokens` (ровно `session_tail_`). Второй вызов
начинается с `pos = session.length = capacity`.

```bash
B=build/linux-release-znver4-aocc
export LD_LIBRARY_PATH=$PWD/$B/_deps/arrow-build/release
# 1) default policy (TC on): finds F-15 first
LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 \
  compute-sanitizer --tool memcheck --launch-timeout 120 \
  $B/tests/test_mistral_moe_model "[audit]"
# 2) legacy policy (TC off): isolates F-01
LSX_PERF_AUTO=0 LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 \
  compute-sanitizer --tool memcheck --launch-timeout 120 \
  $B/tests/test_mistral_moe_model "[audit]"
```

Результат (2): `Invalid __global__ write of size 4 bytes`,
`at lsxcommon::mistral_moe::<unnamed>::mla_attn_kernel` — адрес на 1 байт
за 40-байтовой аллокацией (`scores = heads*capacity*4`, `2*5*4 = 40`).
`ERROR SUMMARY: 146 errors` (включая `kv_write_kernel`); тест при этом
печатает `All tests passed`, потому что запись попала в соседнюю память и
не дала fault. Это и есть цена отсутствия `EnsureCapacity`/проверки
`pos < capacity`.

Результат (1): `Invalid __global__ write of size 2 bytes`
в `TcGemmKernel` (`lsx_tc_gemm.cu:720`) — F-15. Первый прогон «отравляет»
контекст (`cudaErrorLaunchFailure`), поэтому F-01 виден только при
выключенном TC-пути.

### A.4 E1 guard

```bash
nm -C build/linux-release-znver4-aocc/tests/test_gemma_dense_fsi_selfcheck \
  | grep -c FastMathDepth          # 66
# журнал прогона: после трёх последовательных Infer —
# VIOLATION: 2 concurrent ...; VIOLATION: 3 concurrent ...
```

### A.6 Sanitizer на ядре NVFP4/attention (негативный контроль)

```bash
compute-sanitizer --tool memcheck ./tests/test_qwen_nvfp4_gemv
# ERROR SUMMARY: 0 errors
compute-sanitizer --tool memcheck ./tests/test_lsx_nvfp4_dispatch
# All tests passed (41410 assertions in 16 test cases); 0 errors
```

То есть GEMV/TC/tile-пути на штатных формах чисты; найденные F-01/F-15 —
именно те случаи, которые штатные тесты не покрывают (продолжение сессии и
раздельные `d_yg`/`d_yu`).

### A.5 Полный прогон

Логи: `/tmp/opencode/selfchecks.*` (Paris), `/tmp/opencode/parity.*`
(FSI/CODE/DET), `/tmp/opencode/build-aocc.log`,
`/tmp/opencode/mistral_cont_sanitizer.log`,
`/tmp/opencode/mistral_cont_sanitizer_legacy.log`,
`/tmp/opencode/gemv_sanitizer.log`, `/tmp/opencode/dispatch_sanitizer.log`.

---

## 5. Статус исправлений (2026-10-22)

Ветка `fix/inference-audit-2026-10-22`. Второй проход закрыл X9/X11/X15/X16. Сборка
`linux-release-znver4-aocc`; fast-набор **98/98 без `LD_LIBRARY_PATH`**.
Dense parity после правок: qwen3.5-9b, qwen3.8-27b, granite-4.2-30b,
а также gemma-4-31b CODE/DET — зелёные; остались только известные красные
(phi DET-02, gemma-4-31b FSI-01, gemma-4-26b FSI/CODE).

| Пункт | Статус | Доказательство |
|---|---|---|
| F-01 / X1 | fixed | capacity-guard + fallback to fresh prefill в Mistral/Xing; continuation-тесты; Mistral memcheck **0 errors** (было 146), плюс найден и исправлен `cudaFree` по `cudaMallocHost`-указателю `h_slots` |
| F-02 / X2 | fixed | `last_tok` (Mistral), no-double-decode (Xing), `pos0 = kv_rows` (Granite); continuation-тесты; granite Paris hash не изменился |
| F-03 / X3 | fixed | process-global `g_fast_math_depth` + TLS nesting; `test_fast_math_guard_cross_tu` |
| F-04 / X4 | fixed | kOff пиннит large-N и <=32 MiB streaming GEMV (`LsxBitExactMode`, env игнорируется), TC/seq1-gemv off; r3: qwen9b 147.0, qwen27b 52.27, qwen3.6 73.39; `test_gpu_perf`, `test_lsx_nvfp4_dispatch` |
| F-15 / X5 | fixed | up-half в `d_yg + I`, все арены 2I; `test_qwen_moe_kernels`, `test_ornith_moe_kernels`; Mistral memcheck без OOB |
| F-08 / X6 | fixed | три `$ORIGIN/.../_deps`; app-тесты без env, `ldd` чист |
| F-09 / X7 | fixed | pow2 `head_dim` + GQA gate; новый кейс в `test_qwen_dense_full_attn` |
| F-10 / X8 | fixed | `d_gd_state && seq > 1` отклоняется; новый кейс в `test_qwen_dense_fused_linear_attn` |
| F-06 / X10 | fixed | `WaitEventChecked` во всех шести циклах; `test_cuda_event_wait`; `cudaEventQuery` в `libs/` больше нет |
| F-11 / X12 | fixed by inspection | prequant только при `LsxTcRuntimeSupported()`, ошибка GEMM возвращается; инъекция TC-отказа на CC 12 недостижима, юнит-теста нет |
| F-13 / X14 | fixed | §V приведён к §V1/коду (63.6 GB arena) |
| F-05 / X9 | fixed | `FlashEmbedCache` по (weight epoch, source) + RAII сброс Gemma; test_glm_moe_flash_kernels |
| F-07 / X11 | fixed | per-stream `StreamScratch` для split-KV/router; getenv вынесен из цикла; test_stream_scratch |
| F-12 / X13 | fixed | split-K partials теперь per-stream; старые dispatch-тесты зелёные |
| F-14 / X15 | fixed | `CheckViewRange`/`CheckListSlice`; malformed-batch тест |
| X16 | fixed | `AttnHalfDotRow` читал половину head (`hd/2` вместо `hd/4`); тест `< 2e-2` зелёный |

Закрытые пункты вынесены в
[`open-tasks/2026-10-22-inference-audit-closed.md`](open-tasks/2026-10-22-inference-audit-closed.md).
