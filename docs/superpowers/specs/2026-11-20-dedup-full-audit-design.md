# Полный аудит дублирования (41 модель) и закрытие Ф3 — дизайн

Дата: 2026-11-20
Статус: на утверждение оператору
Предшественник: `2026-11-03-code-duplication-consolidation-design.md` (Ф1 закрыта,
Ф2 partial — Task 7 заблокирован, Ф3 не начата).

## 1. Задача и критерий приёмки

Решение оператора (эта сессия): **Вариант 1 — только доказуемые переносы тел
ядер и хост-хелперов**, всё строго внутри существующего `libs/lsxcommon`
(новые библиотеки `lsxkernels`/`lsxblocks` НЕ создаём; продолжаем архитектуру
Ф1/Ф2: нейтральные слои `attn`/`dense`/`moe`/`kernels` + `lsx_kernels_*.cuh`).
**Ф3 обязательна к полному закрытию** (прямое указание оператора).

Критерий приёмки наследуется без изменений: **побитовая идентичность
поведения**. `logits_hash` всех затронутых моделей, FSI/CODE/DET и perf-строки
остаются прежними; расхождение = дефект рефакторинга (шаг откатывается).
Пере-сид `tests/perf/perf_baselines.json` запрещён. **Перенос, а не
переписывание**: меняются только имена и границы ТУ; единственное
исключение — уже санкционированное дизайном 2026-11-03 слияние конверсионных
ядер Ф3 в один grid-stride `int64_t` вариант с тестом эквивалентности.

## 2. Измеренная картина (свежий скан, main @ 911ba823)

Прибор: read-only скан всех 174 `.cu/.cuh` (658 `__global__`-определений) и
всех `.cpp` в `libs/`+`apps/` — нормализованные тела (комментарии/пробелы
убраны), группировка точных дубликатов; для больших пар — построчный
normalized diff. Артефакты: `/mnt/fast_storage/tmp/dedup-audit/` (scan.py,
report.json). Скан станет `tools/dedup_scan.py` (прибор Ф3, план §Task 3).

Точные дубликаты: **32 группы ядер (562 лишние строки) + 26 групп хост-функций
(1 271 лишняя строка)**. Кластеры по убыванию объёма:

### 2.1 Ornith ↔ Qwen MoE — file-level клоны, ~1.5k LOC (самый большой)

| файл | объём | совпадение после нормализации имён |
|---|---|---|
| `ornith_moe_load.cpp` vs `qwen_moe_load.cpp` | 463L каждый | **100 %** (различие — только namespace) |
| `ornith_moe_inference_pipeline.cu` vs `qwen_moe_inference_pipeline.cu` | 918L / ~920L | **99.4 %**; смысловые дельты: `abort_stream` (есть у qwen, нет у ornith), `QwenTcFastMath()` vs `lsx_nvfp4::LsxTcFastMath()` (алиасы одного состояния) |
| `ornith_moe_model.cpp` vs `qwen_moe_model.cpp` | 420L / ~370L | 81 % |
| `ornith_moe_convert.cpp` vs `qwen_moe_convert.cpp` | 804L / 707L | 66 % (qwen уже факторизован в `ConvertQwenMoeModelCommon(spec)`; ornith — нет) |

Ornith был ингестирован копированием qwen-контура (в шапке ornith-пайплайна до
сих пор комментарий «Qwen3.5-35B-A3B MoE inference kernels»). Формат vault
разный (`logestix-ornithmoe` vs `logestix-qwen35moe`) — формат-тег остаётся
данными паспорта, как в паре flash/pro.

### 2.2 Nemotron Lightning ↔ Ultra — sibling-клоны, ~350–450 LOC

- `expert_cache.cpp`: **99 %** идентичны (включая `Free*ExpertCache`, 16L ×4 —
  третья и четвёртая копии в gigachat/mimo).
- kernels: 7 общих ядер, **5 побайтово идентичны** после переименования
  (`ssm_step`, `conv_step`, `moe_weighted_sum`, `relu2_half`, `add_half_to_f32`);
  `gated_rmsnorm`/`router` различаются (геометрия) — остаются семейными.
- pipeline: 6 общих ядер, **все 6 идентичны** (`rmsnorm_half`, `kv_store`,
  `embed_gather`, `half_to_float`, `float_to_half`, `residual_add`).
- `load.cpp`: 85 % (включая `UploadOne` 18L ×2).

### 2.3 DeepSeek flash/pro — остаток Ф2 (~4.4k LOC копий), ЗАБЛОКИРОВАН

Прошлый цикл оставил Task 7 открытым: общий загрузчик даёт у Pro **+2.2 GiB**
resident против старого (не хватает на последние 2 слоя из 61). Копии
`deepseek_moe_pro_{load.cu,inference_attn,inference_moe,inference_pipeline}.cu`
живы и линкуются. Это не дедуп-задача, а memory-audit-дефект; в Variant 1
исправление учёта памяти = изменение поведения → **вне объёма**, остаётся
задокументированным долгом (как и решено в merge `1e050244`).

### 2.4 Хост-инфраструктура tool_format — ~500 LOC

25 семейных `*_tool_format.cpp`; побайтово идентичные хелперы:
`PyJsonValue` 66L ×3 (flint/hy/muse), `SerializeJson` 43L ×3
(deepseek/gemma/gigachat), `AppendTextWithSentinels` 27L ×4,
`AppendEncoded` 15–17L ×9 (glm-flash/granite/granite-switch/mimo/qwen/hy/
minimax/muse/step/mistral/xing), `PyJsonString` 47L ×2, `AppendJinjaString`
24L ×2, `AppendIds` 20L ×2 (inkling-пара). Нейтральный дом уже существует:
`tool_format.{h,cpp}` (`lsxcommon::tools`) + `lsxnetwork::json::JsonEscape`.
Покрыто 25 family-тестами `test_*_tool_framing` (fast) + FSI/CODE-гейты.

### 2.5 Мелкие хост-дубликаты — ~250 LOC

`Free*ExpertCache` 16L ×4; `get_d` 13L ×4 (alice/inkling×2/kimi-k3);
`UploadBytes` 15L ×3 (mistral×2/xing); `MergeGenerationConfigEos` 23L ×2
(gemma); `ReadGenerationIds` 19L ×2 (hy/minimax); `parse_layer_body` 15L ×2
(alice/kimi-k3); `Bf16ToHalf` 13L ×2; `Unfence` 21L ×3
(fsi/code/det_selfcheck.cpp); `HyMemoryPassportGeometry` 18L ×2 (внутри
семейства); `UploadOne` 18L ×2 (закрывается в 2.2).

### 2.6 Остатки Ф3 (по плану 2026-11-03, всё ещё на месте)

- `depth_mix_kernel`: alice/kimi-k3 — тела идентичны после нормализации
  пробелов; единственная дельта — `__shared__ float logits[64]` vs `[16]`
  (cap глубины). Решение: нейтральное ядро с параметром `max_src`
  (инстанцирования 64/16 сохраняют smem-раскладку побайтово), семейные
  guard-ы (`>64` / `<=0||>16`) остаются в семейных лаунчерах.
- Конверсии: `f32_to_f16`/`f16_to_f32` — 6 копий, 5 различных тел;
  `float_to_half`/`half_to_float` — 3 копии, 2 тела; `f32_to_bf16` — 4 копии,
  3 тела; `bf16_to_f16` — 3 копии, 3 тела; `half_to_float_f2h` — 2 копии
  (flash/pro, умрут с Task 7 Ф2 — не трогаем). План Ф3 санкционирует
  единственное не-дословное слияние: один grid-stride `int64_t` вариант на
  направление + тест эквивалентности `n ∈ {1,255,1024,100000}` + побитовые
  гейты моделей. Округление везде RNE (`__float2half`/`__half2float`),
  отличие копий — только кратность обхода (one-shot vs grid-stride,
  `int` vs `int64_t`), не арифметика.
- `silu_mul` deepseek НЕ сливается с `lsx_kernels_act.cuh` (асимметричный
  clamp — зафиксировано в комментарии `lsx_kernels_act.cuh:14`).

### 2.7 Прочее (мелочь, включается попутно)

`fp8_tile128_to_bf16_kernel` 16L ×2 (gigachat/mimo dequant); gigachat
intra-family конверсии (закрываются Ф3-конверсиями); `embed_gather` — 3 тела,
сливаются только идентичные (nemotron-пара в 2.2), остальные остаются.

## 3. Рассмотренные подходы (в рамках Варианта 1)

**A. Ф3 первой, затем кластеры по убыванию «безопасность × объём» (выбран).**
Ф3 мала, уже спланирована и прямо заказана; её прибор (`dedup_scan.py`) и
provenance-барьер становятся инструментом всех последующих фаз. Далее
хост-инфраструктура (покрыта fast-тестами — самый дешёвый барьер), затем
nemotron-пара (ядра с kernel-тестами), затем ornith↔qwen (самый большой, но
требует шаблонизации по паспорту — образец Ф2 уже есть).

**B. Большие кластеры первыми (ornith, nemotron).** Быстрее убирает основной
LOC, но Ф3-прибор появляется поздно, а самый рискованный перенос (шаблонный
драйвер) идёт без обкатанного барьера. Отклонён.

**C. Только Ф3 + мелкие хелперы, кластеры задокументировать как долг.**
Дёшево, но не соответствует «максимальный объём» — 1.5k LOC ornith-клона
остаются. Отклонён.

## 4. Фазы

Каждая фаза — отдельный коммит-набор в общей ветке
`refactor/dedup-audit-2026-11` (оператор заказал одну ветку), TDD-порядок
«красный тест → перенос → зелёный барьер», откат фазы независим.

### Ф3 (обязательная) — `attn` depth mixer + конверсионные ядра + `dedup_scan.py`

По плану `2026-11-03-dedup-phase3-kernel-primitives.md` с двумя уточнениями
против старого текста: (1) `depth_mix` переносится как
`attn_res_depth_mix_kernel<int MaxSrc>` — инстанцирования `<64>` (alice) и
`<16>` (kimi-k3) сохраняют smem побайтово; (2) в объём конверсий добавляются
`f32_to_bf16`/`bf16_to_f16` (скан показал их после 2026-11-03) — тем же
прибором: одно направление = один grid-stride вариант
(`CastOut`/`CastIn` из `lsx_kernels_common.cuh` уже фиксируют RNE), тест
эквивалентности на все 4 направления × n ∈ {1,255,1024,100000}.
`tools/dedup_scan.py` — коммитится как эталонный прибор «до/после».
Ожидаемо: −250…300 LOC копий, +~120 LOC нейтрального кода и тестов.

### Ф4 — хост-инфраструктура tool_format → `lsxcommon::tools`

Переносятся побайтово идентичные тела (§2.4) в `tool_format.{h,cpp}` (или
новый `tool_format_json.h`, если хелперы нужны в заголовках), имена
нейтральные. Семейные файлы сохраняют публичный API (`ParseToolCalls`,
framing) — гварды `kLifecycle`/naming не затрагиваются. Барьер: 25
`test_*_tool_framing` (fast) + `test_logging_guard`/`test_absl_format_guard` +
FSI/CODE-гейты затронутых семейств на реальных весах (где дерево есть).
Ожидаемо: −450…500 LOC.

### Ф5 — мелкие хост-хелперы (§2.5)

Каждый — в существующий нейтральный дом (`fast_load_internal.h`,
`converter.cpp`-helpers, `expert_device_cache.h`, selfcheck-драйверы),
только побайтово идентичные тела; `ParseMoeIndex` ornith НЕ трогаем здесь
(он уезжает целиком в Ф6). Барьер: соответствующие loader/convert/expert_cache
тесты (fast/convert) + selfcheck затронутых. Ожидаемо: −200…250 LOC.

### Ф6 — Ornith ↔ Qwen MoE: один шаблонный контур (образец Ф2)

1. `LoadQwenMoeDenseWeights`/`ParseMoeIndex`/`Free*` → шаблон по паспорту
   (`qwen_moe::Passport` / `ornith::Passport`), формат-тег и geometry — из
   паспорта; тела 100 % идентичны, доказательство — provenance.
2. Пайплайн: общий шаблонный драйвер; дельта `abort_stream` переносится как
   параметр политики паспорта (у ornith поведение НЕ меняется — флаг
   остаётся выключенным, как отсутствие фичи сегодня; у qwen — включённым).
   `QwenTcFastMath()` — уже алиас нейтрального toggle, унификация имени не
   меняет состояние.
3. `ornith_moe_convert.cpp` → `ConvertQwenMoeModelCommon(spec)` с
   ornith-спеком (только если тела конвертера доказуемо совпадают после
   нормализации; 66 % совпадения означает, что часть орnit-специфики
   остаётся семейной — переносим только идентичные функции, не весь файл).
4. `model.cpp` (81 %) — переносим только идентичные функции; `kType`,
   паспорт, регистрация, SelfCheck остаются семейными (гварды адресуют
   модель по этим файлам).
Барьер: `test_ornith_moe_*` + `test_qwen_moe_*` (все стадии), selfcheck
ornith/qwen3.5/qwen3.6 на реальных деревьях (все три в vault), FSI/CODE/DET,
perf-строки. Ожидаемо: −1 100…1 400 LOC.

### Ф7 — Nemotron Lightning ↔ Ultra

Идентичные ядра (5 kernels + 6 pipeline) → нейтральный слой
(`lsx_kernels_*.cuh` / `lsx_attn_inference.cu` — по содержанию), либо общий
family-neutral `nemotron` слой по образцу minimax/hy-simulmt (родительские
kernels через паспорт — этот паттерн в репозитории уже санкционирован для
siblings). `expert_cache.cpp` (99 %) → шаблон/общий ТУ; `UploadOne`/`get_d`/
`Free*` — в общий хост-хелпер. Различающиеся `gated_rmsnorm`/`router` и
geometry остаются семейными. Барьер: `test_nemotron_moe_*` +
`test_nemotron_moe_ultra_*` все стадии, selfcheck обоих (деревья в vault),
FSI/CODE/DET, perf. Ожидаемо: −350…450 LOC.

### Вне объёма (задокументированный долг)

- DeepSeek flash/pro Task 7 (§2.3) — блокер memory-audit, не дедуп.
- `router_topk`/`swiglu`/`mla_attn`/`indexer_*`/`kda_*` — тела РАЗНЫЕ
  (разная семантика/геометрия), слияние = переписывание → Variant 2.
- RAII/cudaMalloc-обвязка, god-файлы, v4.1 — как в дизайне 2026-11-03 §13.

## 5. Верификация (наследуется из дизайна 2026-11-03 §7)

- **Уровень 0**: `tools/dedup_provenance.py` — 0 missing на каждый перенос;
  `tools/dedup_scan.py` — замер «до/после» одним прибором.
- **Уровень 1**: `linux-debug` сборка 0 warnings в наших файлах; fast-свип
  100 %; convert-свип 100 %; inference-свип — все PASS кроме документированных
  SKIP 4 и declared-debt RED (`docs/SELFCHECK_MATRIX.md`).
- **Уровень 2**: перед каждой фазой — baseline `logits_hash`/top-1/t/s на
  этой машине для затронутых моделей (release-znver4-aocc, GPU по одному);
  после — совпадение. Деревья доступны для: ornith, qwen3.5/3.6 MoE, qwen
  dense, nemotron (обе), alice, glm-flash, gigachat-нет, muse, xing,
  mistral-small, hy×2, inkling×2, granite×3, phi, gemma×2, deepseek-flash,
  deepseek-v41, dory, ernie. **Нет деревьев**: deepseek-pro, gigachat,
  kimi-k3/k3.1-pro, minimax×2, kimi-k2.7, ling, beam, step, flint,
  mistral-large, glm-5.2, mimo, ornith-есть. Для семейств без дерева барьер
  = kernel/unit-тесты + provenance + (для tool_format) fast framing-тесты;
  это фиксируется в отчёте фазы как ограничение, а не «pass».
- Пере-сид perf запрещён; регресс = откат шага.

## 6. Гварды и реестры

Обновляются в том же коммите, что и перенос: `test_model_suite_guard`
(если меняются тест-таргеты), `test_model_independence_guard`
(`kLifecycle`-ТУ не меняются — семейные файлы остаются), naming-guard
(нейтральные имена без семейных стемов), `test_absl_container_guard`
(заголовки из `.cu` — без abseil-контейнеров), `test_infrastructure_guard`
(один definition на хелпер). Падение гварда из-за нейтрального имени —
сигнал ошибки слоя, не повод ослабить гвард.

## 7. Риски

| риск | снятие |
|---|---|
| шаблонизация ornith-пайплайна меняет инлайнинг → perf-дрейф | уровень 2 побитово + perf-строки; регресс = откат |
| `abort_stream`-политика случайно включится у ornith | параметр паспорта + тест: ornith-драйвер без abort-пути побайтово как раньше (provenance на тело цикла) |
| конверсии Ф3: one-shot→grid-stride меняет порядок записи | элементwise-ядра без RMW: каждый индекс пишется ровно один раз тем же значением; тест эквивалентности + побитовые гейты |
| depth_mix smem 64 vs 16 | шаблонный параметр, инстанцирования сохраняют раскладку |
| семейства без vault-деревьев | барьер ограничен unit/provenance; ограничение фиксируется в отчёте, не маскируется |
| время сборки растёт от инстанцирований | явные инстанцирования в одном ТУ (образец Ф2); замер до/после в отчёте |

## 8. Артефакты

По каждой фазе: `docs/open-tasks/YYYY-MM-DD-dedup-<phase>-closed.md` с
числами (LOC до/после по `dedup_scan.py`, provenance-вывод, хеши до/после,
build time). Обновление `AGENTS.md`/skill `logestix-repo-map`/`CHANGELOG.md`.
Ветка: `refactor/dedup-audit-2026-11`. Коммиты — только по явному запросу
оператора (правило репозитория); работа идёт в отдельном worktree.