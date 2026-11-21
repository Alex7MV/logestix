# Полный аудит дублирования: закрытие Ф3 + фазы Ф4–Ф7 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. **Subagents are forbidden in this repository** (AGENTS.md § Autonomous Mode): every step is executed inline by the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Убрать доказуемое дублирование по всем 41 моделям (−2.3…2.7k LOC), полностью закрыть Ф3, сохранив побитовую идентичность поведения.

**Architecture:** Нейтральные слои внутри `libs/lsxcommon` (продолжение Ф1/Ф2): `attn`/`kernels` + `lsx_kernels_*.cuh` для ядер, `lsxcommon::tools`/`fast_load_internal.h`/converter-helpers для хоста. Перенос, а не переписывание; шаблонизация по паспорту для file-level клонов (образец Ф2). Приборы: `tools/dedup_provenance.py` (уровень 0) и новый `tools/dedup_scan.py` (абсолютный остаток).

**Tech Stack:** C++20, CUDA 13.x, Catch2 v3.16.1, `lsx_add_test`/`lsx_add_cuda_test`, python3 (приборы).

**Spec:** `docs/superpowers/specs/2026-11-20-dedup-full-audit-design.md`. Старый план Ф3 (`2026-11-03-dedup-phase3-kernel-primitives.md`) заменяется Task 1–3 этого плана (уточнения: шаблонный `MaxSrc` у depth_mix; расширенный охват конверсий; `-fmad`-нейтральность).

## Global Constraints

- Побитовая идентичность: `logits_hash`, FSI/CODE/DET, perf-строки затронутых моделей не меняются; расхождение = откат шага. Пере-сид `tests/perf/perf_baselines.json` запрещён.
- Перенос дословный (provenance 0 missing); единственные санкционированные не-дословные места: (а) grid-stride `int64_t` конверсии Ф3 с тестом эквивалентности, (б) `__fadd_rn/__fmul_rn` в переносимом depth_mix (flag-independent арифметика, паттерн `lsx_kernels_*.cuh`).
- **FMA-инвариант:** ядро, переезжающее из `-fmad=false` ТУ (alice, kimi-k3) в ТУ с contraction, обязано использовать явные `*_rn` интринсики на КАЖДУЮ арифметическую операцию (или остаться в заголовке, инклюзимом семейными ТУ). Простой перенос тела без `*_rn` запрещён.
- Семейные guard-ветки (`n_sources > 64`, `n_src <= 0 || n_src > 16` и т.п.) остаются в семейных лаунчерах; коды ошибок не меняются.
- Заголовки, достижимые из `.cu`, не включают `<absl/container/*>`. Сборка — 0 предупреждений в наших файлах.
- Нейтральные имена без семейных стемов; `kLifecycle`/`consumers`/passport-реестры не меняются (семейные ТУ остаются). Падение гварда из-за нового имени — сигнал ошибки слоя, не повод ослабить гвард.
- GPU: один workload за раз (`nvidia-smi --query-compute-apps` перед каждым GPU-прогоном). Release-барьер: `linux-release-znver4-aocc`.
- Ветка `refactor/dedup-audit-2026-11` (worktree). Коммиты — только по явному запросу оператора (AGENTS.md § Git); шаги «Commit» ниже — точки, где коммит ЗАПРОСИТЬ, не делать молча.
- Временные артефакты — `/mnt/fast_storage/tmp/dedup-audit/`.
- Семейства без сконвертированного дерева (deepseek-pro, gigachat, kimi-k3/k3.1-pro, minimax×2, kimi-k2.7, ling, beam, step, flint, mistral-large, glm-5.2, mimo): барьер = unit/kernel-тесты + provenance + fast framing-тесты; ограничение фиксируется в отчёте фазы, не маскируется под pass.

## Review Focus

1. **FMA-контракция при переезде depth_mix** (alice/kimi-k3 `-fmad=false` → нейтральный ТУ с contraction): тест — реальные веса alice SelfCheck hash `f900547d25be7914` (Task 2) + host-reference kernel-тесты обоих семейств; kimi-k3 дерева нет — kernel-тест + provenance + явные `*_rn`.
2. **grid-stride конверсии при n > grid*block и n=0**: тест эквивалентности `n ∈ {0,1,255,1024,100000}` с capped grid 1024 (Task 3); лаунчеры с `n<=0 → cudaSuccess` guard сохраняют его.
3. **smem-раскладка depth_mix 64 vs 16**: шаблонный `MaxSrc`, инстанцирования `<64>`/`<16>`; kernel-тесты гоняют n_src на границе cap (Task 2).
4. **tool_format-хелперы: escape-семантика** (`PyJsonValue` vs `SerializeJson` — разные форматтеры, НЕ сливать друг с другом): каждый переносится только в группу побайтово идентичных; framing-тесты 25 семейств (Task 4).
5. **ornith `abort_stream`**: параметр политики паспорта, у ornith путь остаётся выключенным; provenance на тело цикла + selfcheck ornith на реальном дереве (Ф6).

---

## Фаза Ф3 (обязательная)

### Task 1: Прибор `tools/dedup_scan.py` + baseline-замер

**Files:**
- Create: `tools/dedup_scan.py` (из `/mnt/fast_storage/tmp/dedup-audit/scan.py`, довести до продакшн-вида: argparse `--root`, `--json <path>`, `--min-lines`, exit 0 всегда — это измеритель, не гейт)
- Create: `tests/test_dedup_scan.py` (unittest, host-only: синтетические фикстуры во временном каталоге — два файла с идентичным ядром после нормализации комментариев/пробелов → группа найдена; разные тела → не найдена; ядро < min-lines → не найдено)
- Modify: `tests/CMakeLists.txt` (add_test `test_dedup_scan` через `${LSX_PYTHON3}`, LABEL fast, TIMEOUT 120 — образец `test_manifest_generator_selftest`)

**Interfaces:**
- Produces: `python3 tools/dedup_scan.py --root . --json <path>` → JSON `{kernel_dup_groups, kernel_wasted_lines, host_dup_groups, host_wasted_lines, kernel_groups[], host_groups[]}`; печать сводки в stdout.

- [ ] Step 1: Написать падающий тест `tests/test_dedup_scan.py` (фикстуры: identical-after-normalization группируется; differing — нет; host-функция ≥12 строк ×2 файла группируется). Запуск: `python3 -m unittest tests.test_dedup_scan` → FAIL (модуля нет).
- [ ] Step 2: Перенести scan.py → `tools/dedup_scan.py`, argparse, без захардкоженных путей. Тест → PASS.
- [ ] Step 3: Зарегистрировать ctest-таргет (LABEL fast). `ctest --preset linux-debug -R test_dedup_scan` → PASS.
- [ ] Step 4: Baseline-замер: `python3 tools/dedup_scan.py --root . --json /mnt/fast_storage/tmp/dedup-audit/baseline.json`. Записать числа в отчёт (ожидание ≈ 32 группы / 562 строки ядра, 26 / 1271 хост).
- [ ] Step 5: Commit (запросить): `test: add dedup_scan.py, the duplication-residue instrument`

### Task 2: `depth_mix_kernel` → нейтральный `lsx_kernels_attnres.cuh`

**Решение против старого плана:** перенос в `lsx_attn_inference.cu` НЕВОЗМОЖЕН дословно — alice/kimi-k3 живут в `-fmad=false` ТУ, а `lsxcommon_qwen_cu` — с contraction. Поэтому: новый заголовок `libs/lsxcommon/src/lsx_kernels_attnres.cuh` (паттерн `lsx_kernels_norm.cuh`), ядро-шаблон инклюзится семейными ТУ — арифметика остаётся под их флагом, плюс явные `*_rn` на каждую операцию (flag-independent, как `lsx_kernels_*.cuh`).

**Files:**
- Create: `libs/lsxcommon/src/lsx_kernels_attnres.cuh`
- Create: `tests/test_attnres_kernels.cu` (нейтральный тест, `lsx_add_cuda_test(NAME test_attnres_kernels FMAD_OFF LABEL inference)`)
- Modify: `libs/lsxcommon/src/alice_moe_kernels.cu` (удалить `depth_mix_kernel`, инклюз заголовка, лаунчер делегирует), `libs/lsxcommon/src/kimi_moe_k3_kernels.cu` (то же)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
```cpp
namespace lsxcommon::kernels {
// AttnRes depth mixer: per-source RMS-dot score, softmax over sources,
// weighted combine. MaxSrc fixes the shared logits array (alice 64, kimi 16).
// Launch: <<<1, 256>>>; caller guards n_src (family caps stay family-side).
// All arithmetic uses explicit *_rn intrinsics: bit-identical under any
// -fmad setting (the alice/kimi TUs are -fmad=false, consumers may not be).
template <int MaxSrc>
__global__ void AttnResDepthMixKernel(const __half *__restrict__ sources,
                                      int n_src,
                                      const __half *__restrict__ norm_w,
                                      const __half *__restrict__ proj_w,
                                      __half *__restrict__ out, int hidden,
                                      float eps);
}
```
- Consumes: ничего (block_sum переносится внутрь заголовка как `__device__` detail-функция с trailing `__syncthreads()` — вариант kimi; для alice добавление trailing barrier не меняет значений, только синхронизацию, и provenance фиксирует это как единственную дельту тела).

- [ ] Step 1: Падающий тест `tests/test_attnres_kernels.cu`: CPU-эталон (формула из kernel-теста alice: RMS-dot → softmax → combine, double-precision reference, допуск 6e-3 как в старом плане), кейсы `<64>` и `<16>`, `n_src ∈ {1, 3, cap}`, `hidden ∈ {8, 64, 2048}`, eps 1e-6/1e-5. Плюс boundary: `n_src = cap` не трогает соседний smem. Зарегистрировать, собрать → FAIL (заголовка нет).
- [ ] Step 2: Перенести тело из `alice_moe_kernels.cu:54-95` в заголовок, `logits[64]` → `logits[MaxSrc]`, каждую `+`/`*`/`/` заменить на `__fadd_rn/__fmul_rn/__fdiv_rn` (rsqrtf/expf/fmaxf — уже интринсики, не трогаются). block_sum — вариант kimi (с trailing barrier).
- [ ] Step 3: Тест → PASS. Provenance: `python3 tools/dedup_provenance.py --from alice_moe_kernels.cu --to lsx_kernels_attnres.cuh --rename depth_mix_kernel=AttnResDepthMixKernel --allow-extra` — ОЖИДАЕМО не 0 (из-за `*_rn` и MaxSrc): вместо него зафиксировать в отчёте построчный diff «только *_rn-обёртки, MaxSrc, trailing barrier» и приложить его. Это санкционированное не-дословное место (Global Constraints).
- [ ] Step 4: Переключить семейства: `RunAliceDepthMix` guard `n_sources > 64` остаётся, вызов `kernels::AttnResDepthMixKernel<64><<<1, kBlock, 0, stream>>>`; `RunKimiK3DepthMix` guard `n_src <= 0 || n_src > 16` остаётся, `<16>`. Удалить семейные тела и ставшие неиспользуемыми `block_sum` (проверить другие потребители в файле!).
- [ ] Step 5: Барьер: сборка 0 warnings; `ctest -R "test_attnres_kernels|test_alice_moe_kernels|test_kimi_k3_kernels"` (inference) → PASS. Release-сборка + `test_alice_moe_selfcheck` на реальном дереве (`LSX_TEST_ALICE_ROOT=/mnt/fast_storage/huggingface/AliceAI-Foundation-80B-A3B-Base`) → hash `f900547d25be7914` без изменений. kimi-k3: дерева нет — kernel-тест + provenance-diff, ограничение в отчёт.
- [ ] Step 6: Commit (запросить): `refactor(attnres): share the depth mixer between alice and kimi-k3`

### Task 3: Конверсионные ядра → `lsx_kernels_common.cuh` + закрытие Ф3

**Охват (свежий скан):** `f32_to_f16`/`f16_to_f32` — gigachat×2, glm-flash, mistral, mistral-large4, xing; `float_to_half`/`half_to_float` — nemotron×2, qwen-exp, deepseek `load.cuh` (static, инклюзится flash+pro пайплайнами — один перенос закрывает обе пары); `f32_to_bf16` — gigachat×2 (`_rn`), mistral (`__float2bfloat16`), mistral-large4; `bf16_to_f16` — mistral, mistral-large4. **НЕ трогаем:** glm-flash `bf16_to_f16` (uint8-битовый вход — другая сигнатура/семантика), `flash_f16_to_f32` (другой порядок аргументов + дебаг), `scale_f32_to_f16` (умножение — не cast), deepseek `clamp_*` (clamp — не cast), gemma `gemma_f32_to_bf16` (проверить тело: если идентично — включить, иначе оставить).

**Files:**
- Modify: `libs/lsxcommon/src/lsx_kernels_common.cuh` (4+2 ядра)
- Create: `tests/test_kernels_common.cu` (`lsx_add_cuda_test(NAME test_kernels_common LABEL inference)`)
- Modify: 10 семейных файлов (список выше) — удалить локальные определения, заменить вызовы
- Modify: `tests/CMakeLists.txt`, `AGENTS.md`, `.opencode/skills/logestix-repo-map/SKILL.md`, `CHANGELOG.md`
- Create: `docs/open-tasks/2026-11-20-kernel-primitives-closed.md`

**Interfaces:**
- Produces (все — grid-stride `int64_t`, паттерн старого плана Ф3 Task 2):
```cpp
namespace lsxcommon::kernels {
__global__ void F32ToF16Kernel(const float *__restrict__ src, __half *__restrict__ dst, int64_t n);
__global__ void F16ToF32Kernel(const __half *__restrict__ src, float *__restrict__ dst, int64_t n);
__global__ void F32ToBf16Kernel(const float *__restrict__ src, __nv_bfloat16 *__restrict__ dst, int64_t n); // __float2bfloat16_rn
__global__ void Bf16ToF16Kernel(const __nv_bfloat16 *__restrict__ src, __half *__restrict__ dst, int64_t n); // __float2half(__bfloat162float(x))
}
```
Вызовы сохраняют прежние `blocks/threads/stream` (grid-stride покрывает любой grid; one-shot вызовы с полным grid дают тот же результат поэлементно). `n` приводится к `int64_t`. Guard-ы лаунчеров (`n<=0 → cudaSuccess` у xing) остаются.

- [ ] Step 1: Падающий тест эквивалентности: все 4 направления × `n ∈ {0,1,255,1024,100000}`, grid capped 1024 (проверяет grid-stride), значения `0.001f*((i*7919)%100003)-50.0f`, побитовое сравнение с `__float2half`/`__float2bfloat16_rn`/композицией на хосте. Собрать → FAIL.
- [ ] Step 2: Добавить ядра в `lsx_kernels_common.cuh` (тела — из старого плана Ф3 Task 2 Step 3, плюс bf16-направления через `detail::CastOut`/`CastIn`). Тест → PASS.
- [ ] Step 3: Удалить семейные копии, перевести вызовы (10 файлов). Проверка: `rg -n "void (f32_to_f16|f16_to_f32|half_to_float|float_to_half|f32_to_bf16|bf16_to_f16)_kernel" libs/lsxcommon/src | rg -v "lsx_kernels_common|glm_moe_flash"` → только glm-flash uint8-вариант (объяснён в отчёте).
- [ ] Step 4: Барьер уровня 1: полная сборка contour (shared .cuh → ВСЕ инклюзящие ТУ, правило test-lifecycle), 0 warnings; fast+convert 100 %; inference — все PASS кроме документированных SKIP/debt.
- [ ] Step 5: Барьер уровня 2 (release-znver4-aocc, GPU по одному): SelfCheck затронутых семейств с деревьями — glm-5.3-flash, mistral-small, xing, nemotron-lightning (обе), qwen3.8-flash-next, deepseek-flash, deepseek-v41 (инклюз load.cuh). Хеши до/после — совпадение; FSI/CODE/DET не хуже. gigachat/mimo/kimi — без деревьев, ограничение в отчёт.
- [ ] Step 6: Повторный `dedup_scan.py` → after.json; отчёт Ф3 `docs/open-tasks/2026-11-20-kernel-primitives-closed.md`: таблица удалённого/оставленного намеренно, provenance/diff-артефакты, хеши, числа до/после. Обновить AGENTS.md/repo-map/CHANGELOG.
- [ ] Step 7: Commit (запросить): `refactor(kernels): one grid-stride cast per direction; close dedup phase 3`

---

## Фаза Ф4: tool_format-хелперы → `lsxcommon::tools`

### Task 4: Перенос побайтово идентичных JSON/framing-хелперов

**Files:**
- Modify: `libs/lsxcommon/src/tool_format.cpp`, `libs/lsxcommon/include/lsxcommon/tool_format.h` (или новый `tool_format_json.h`, если хелперы нужны инлайн в семейных ТУ — решить по факту include-граффа)
- Modify: семейные `*_tool_format.cpp` (flint/hy/muse/deepseek/gemma/gigachat/granite×2/mimo/qwen/minimax/step/mistral/xing/inkling×2/dory — только те, чьи тела доказаны идентичными)
- Test: существующие `test_*_tool_framing` (25, fast) + новый `tests/test_tool_format_json.cpp` (fast) на перенесённые хелперы

**Interfaces:**
- Produces: `lsxcommon::tools::PyJsonValue/PyJsonString/SerializeJson/AppendTextWithSentinels/AppendEncoded/AppendJinjaString/AppendIds` — сигнатуры переносятся дословно из семейных копий (все копии в группе идентичны после нормализации, проверить provenance-скриптом по хост-телам).
- **НЕ сливать** `PyJsonValue` с `SerializeJson` — разные форматтеры (Python-tojson vs compact), группы раздельные.

- [ ] Step 1: Для каждой группы снять proof: нормализованные тела всех членов идентичны (скан уже доказал; приложить вывод). Написать падающий тест на нейтральный `tools::PyJsonValue` (golden-строки из семейных framing-тестов).
- [ ] Step 2: Перенести тела в `tool_format.{h,cpp}`, семейные копии удалить, вызовы перевести. Публичный семейный API (`ParseToolCalls`, framing) не меняется.
- [ ] Step 3: Барьер: fast-свип 100 % (25 framing-тестов + logging/absl guards); FSI/CODE-гейты затронутых семейств с деревьями (hy×2, muse, xing, mistral-small, granite×2, gemma×2, inkling×2, deepseek-flash, qwen×4, dory) — не хуже.
- [ ] Step 4: Отчёт фазы + dedup_scan after. Commit (запросить): `refactor(tools): share the byte-identical tool-format JSON helpers`

---

## Фаза Ф5: мелкие хост-хелперы

### Task 5: get_d / UploadBytes / Free*ExpertCache / Unfence / convert-хелперы

**Files (по группам, только побайтово идентичные тела):**
- `get_d` ×4 (alice/inkling×2/kimi-k3 load.cpp) → `fast_load_internal.h`
- `UploadBytes` ×3 (mistral×2/xing load.cpp) → `fast_load_internal.h`
- `Free*ExpertCache` ×4 (gigachat/mimo/nemotron×2 expert_cache.cpp) → общий хелпер рядом с `expert_device_cache.h` (сигнатуры проверить: могут отличаться типами — переносить только доказуемо идентичные)
- `Unfence` ×3 (fsi/code/det_selfcheck.cpp) → общий selfcheck-хелпер
- `MergeGenerationConfigEos` ×2 (gemma), `ReadGenerationIds` ×2 (hy/minimax), `parse_layer_body` ×2 (alice/kimi-k3), `Bf16ToHalf` ×2, `HyMemoryPassportGeometry` ×2 (внутри семейства) → converter-helpers / семейный общий заголовок
- Test: существующие loader/convert/expert_cache тесты + точечные новые кейсы на нейтральные хелперы (fast)

- [ ] Step 1: Падающие тесты на нейтральные сигнатуры (по группе).
- [ ] Step 2: Перенос + переключение + удаление копий (provenance по хост-телам).
- [ ] Step 3: Барьер: fast+convert 100 %; selfcheck затронутых с деревьями (alice, inkling×2, mistral-small, xing, hy×2, gemma×2, nemotron×2) — хеши без изменений.
- [ ] Step 4: Отчёт + scan. Commit (запросить): `refactor(host): share the byte-identical loader/converter/selfcheck helpers`

---

## Фаза Ф6: Ornith ↔ Qwen MoE — один шаблонный контур

### Task 6: Loader + ParseMoeIndex → шаблон по паспорту

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_moe_load.h`, `libs/lsxcommon/src/qwen_moe_load.cpp` (тела → шаблон `<Passport>`; формат-тег/geometry из паспорта)
- Modify: `libs/lsxcommon/src/ornith_moe_load.cpp` (удалить 100 %-копию, инстанцировать), `libs/lsxcommon/include/lsxcommon/ornith_moe_load.h` (алиасы типов на шаблонные инстанцирования — публичные имена `OrnithMoePinnedWeights`/`LoadOrnithMoeDenseWeights` сохраняются)
- Test: `test_ornith_moe_loader`, `test_qwen_moe_loader` (существующие, synthetic fixtures)

**Interfaces:**
- Produces: `lsxcommon::moe::LoadMoeDenseWeightsT<Passport>` (точное имя/namespace уточнить при реализации по образцу Ф2 `load_pinned_weights<T>`);паспорт-traits: format tag, geometry, expert-store policy.
- Consumes: паспорта `qwen_moe_constants.h`/`ornith_moe_constants.h` (не меняются).

- [ ] Step 1: Baseline: release selfcheck ornith + qwen3.5 + qwen3.6 (деревья есть), записать хеши.
- [ ] Step 2: Падающий тест: орnit-инстанцирование шаблона даёт тот же результат на synthetic fixture (loader-тест уже есть — добавить кейс на общий шаблон).
- [ ] Step 3: Шаблонизация (тела 100 % идентичны — provenance 0 missing), переключение, удаление копии.
- [ ] Step 4: Барьер: loader/model тесты обоих; release selfcheck всех трёх деревьев — хеши совпадают.
- [ ] Step 5: Commit (запросить): `refactor(moe): one templated dense loader for qwen and ornith`

### Task 7: Пайплайн → общий шаблонный драйвер

**Files:**
- Modify: `libs/lsxcommon/src/qwen_moe_inference_pipeline.cu` (→ шаблон), `libs/lsxcommon/src/ornith_moe_inference_pipeline.cu` (→ инстанцирование + семейный glue)
- Test: `test_ornith_moe_kernels`, `test_qwen_moe_kernels`

**Дельты (доказаны сканом, ratio 99.4 %):** (1) `abort_stream` — параметр политики паспорта: qwen `true`, ornith `false` (поведение ornith не меняется — путь остаётся мёртвым, как отсутствие фичи сегодня); (2) `QwenTcFastMath()` vs `lsx_nvfp4::LsxTcFastMath()` — алиасы одного состояния, унифицировать на нейтральное имя; (3) include-порядок/комментарии.

- [ ] Step 1: Падающий тест: ornith-драйвер на synthetic fixture побитово равен прежнему (детерминизм-кейс уже есть в kernels-тесте — расширить на общий шаблон).
- [ ] Step 2: Шаблонизация + политики; provenance на тела ядер (0 missing) и построчный diff драйвера (только 3 дельты выше).
- [ ] Step 3: Барьер: kernels-тесты обоих; release selfcheck ornith/qwen3.5/qwen3.6 + FSI/CODE/DET + perf-строки (perf: ornith и qwen3.5 — в пределах шума).
- [ ] Step 4: Commit (запросить): `refactor(moe): one templated generation driver for qwen and ornith`

### Task 8: Конвертер + model — только идентичные функции

**Files:**
- Modify: `libs/lsxcommon/src/ornith_moe_convert.cpp` (идентичные функции → `ConvertQwenMoeModelCommon(spec)` или общий хелпер; ornith-специфика остаётся), `ornith_moe_model.cpp` (81 % — только идентичные функции; `kType`/регистрация/SelfCheck остаются)
- Test: `test_ornith_moe_convert`, `test_ornith_moe_model`, `test_qwen_moe_*`

- [ ] Step 1: Построчный diff каждой функции-кандидата; переносить только 100 %-совпадения после нормализации имён. Падающий тест на общий путь.
- [ ] Step 2: Перенос + переключение. Барьер: convert/model тесты; release selfcheck ornith (конвертер не пере-конвертирует vault — `lsxforge --verify` на существующем дереве: files/footer_ok/problems=0 без изменений).
- [ ] Step 3: Отчёт Ф6 + scan. Commit (запросить): `refactor(ornith): share the proven-identical converter and model helpers`

---

## Фаза Ф7: Nemotron Lightning ↔ Ultra

### Task 9: Идентичные ядра + expert_cache + UploadOne

**Files:**
- Modify: `libs/lsxcommon/src/nemotron_moe_kernels.cu`, `nemotron_moe_ultra_kernels.cu` (5 идентичных ядер: `ssm_step`, `conv_step`, `moe_weighted_sum`, `relu2_half`, `add_half_to_f32` → общий заголовок `nemotron_moe_shared_kernels.cuh` ИЛИ нейтральный слой — решить по содержанию: ssm/conv — Mamba-примитивы, если нейтральны по форме → `lsx_kernels_*.cuh`; если геометрия зашита → family-shared заголовок, паттерн minimax/hy-simulmt)
- Modify: `nemotron_moe_inference_pipeline.cu`, `nemotron_moe_ultra_inference_pipeline.cu` (6 идентичных ядер; 4 из них — конверсии, уже закрыты Ф3; остальные `rmsnorm_half`, `kv_store`, `embed_gather`, `residual_add` — тем же правилом)
- Modify: `nemotron_moe_expert_cache.cpp`, `nemotron_moe_ultra_expert_cache.cpp` (99 % → общий ТУ/шаблон), `nemotron_moe_load.cpp`/`ultra_load.cpp` (`UploadOne`)
- Test: `test_nemotron_moe_kernels`, `test_nemotron_moe_ultra_kernels`, expert_cache-тесты обоих
- **НЕ трогаются:** `gated_rmsnorm`/`router` (тела различаются), geometry, паспорта.

- [ ] Step 1: Baseline release selfcheck обоих (деревья есть: BF16 18G, Ultra 292G). Падающий тест на общий путь.
- [ ] Step 2: Перенос (provenance 0 missing на ядра), переключение, удаление копий.
- [ ] Step 3: Барьер: kernels/loader/model/expert_cache тесты обоих; release selfcheck + FSI/CODE/DET + perf обоих — хеши совпадают, perf в шуме.
- [ ] Step 4: Отчёт Ф7 + финальный scan (ожидание: группы < 12, строки < 700 — критерий старого плана Ф3, пересмотренный под расширенный объём: каждая оставшаяся группа объяснена). Обновить AGENTS.md/repo-map/CHANGELOG, закрыть запись в docs/OPEN_TASKS.md.
- [ ] Step 5: Commit (запросить): `refactor(nemotron): share the lightning/ultra kernels and expert cache`

---

## Приёмка всего цикла

- Ф3 закрыта полностью: depth_mix одна реализация, конверсии по одной на направление, `dedup_scan.py` в репозитории, отчёт с числами.
- Финальный скан: каждая оставшаяся группа дубликатов объяснена в итоговом отчёте (разные тела / заблокированный Ф2-остаток / намеренно семейное).
- Ни один `logits_hash` не изменился; ни одна perf-строка не вышла за шум; пере-сида не было.
- Все гварды зелёные; fast+convert 100 %; inference — только документированные SKIP/debt.
- DeepSeek flash/pro Task 7 остаётся открытым долгом с ссылкой на блокер (+2.2 GiB) — запись в docs/OPEN_TASKS.md.