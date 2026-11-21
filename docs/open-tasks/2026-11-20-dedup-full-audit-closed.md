# Полный аудит дублирования (41 модель): доказуемые переносы — итоговый отчёт

Дата: 2026-11-20
Ветка: `refactor/dedup-audit-2026-11` (от main @ 911ba823)
Дизайн: `docs/superpowers/specs/2026-11-20-dedup-full-audit-design.md`
План: `docs/superpowers/plans/2026-11-20-dedup-full-audit.md`
Критерий оператора: Вариант 1 — только доказуемые переносы тел, побитовая
идентичность поведения, всё внутри `libs/lsxcommon`, Ф3 обязательна к закрытию.

## Итоговые числа

| метрика (прибор `tools/dedup_scan.py`) | baseline | после |
|---|---|---|
| группы дубликатов ядер | 52 | 21 |
| лишние строки ядер | 929 | 578 |
| хост-группы (≥12 строк, cross-file) | 26 | 7 |
| лишние строки хоста | 1271 | 218 |
| всего `__global__`-определений | 660 | 605 |

Дифф ветки: **+1296 / −5558 строк (нетто −4262)**.

## Что перенесено (по фазам)

### Ф3 (закрыта полностью) — см. `2026-11-20-kernel-primitives-closed.md`
- `tools/dedup_scan.py` + `tools/test_dedup_scan.py` (ctest fast).
- `depth_mix_kernel` → `lsx_kernels_attnres.cuh`
  (`kernels::AttnResDepthMixKernel<MaxSrc>`, alice `<64>` / kimi-k3 `<16>`,
  явные `*_rn` — flag-independent; семейные guard-ы сохранены).
- 6 направлений конверсий → `lsx_kernels_common.cuh` (grid-stride `int64_t`,
  шаблонные SrcT/DstT для weak linkage); 29 семейных копий удалены в 12 файлах.
- Тесты: `test_attnres_kernels.cu`, `test_kernels_common.cu` (inference).

### Ф4 — tool-format JSON-хелперы → `lsxcommon::tools`
Замкнутые побайтово идентичные группы (проверка включала ВЫЗОВЫ внутрь
семейства — flint `PyJsonValue` идентичен, но вызывает семейный
`PyJsonString` ensure_ascii=False, поэтому flint остался целиком):
- `PyJsonString`/`PyJsonValue`/`PyJsonDumps` → hy, muse;
- `SerializeJson` → deepseek, gemma, gigachat;
- `AppendJinjaString` → dory, qwen;
- `AppendEncoded` (cap 65536) → glm-flash, granite, granite-switch, mimo, qwen;
- `AppendEncodedLarge` (cap 1<<20) → hy, minimax, muse, step;
- `AppendTextWithSentinels` (с параметром sentinels) → hy, minimax, step;
- `AppendSegmentIds<T>` → inkling-пара.
Тест: `test_tool_format_json.cpp` (fast, goldens трёх РАЗНЫХ escape-семантик).
Барьер: 29/29 framing-тестов PASS.

### Ф5 — мелкие хост-хелперы
- `config_json.h`: `GetDouble` (get_d ×4: alice/inkling×2/kimi-k3),
  `GetInt`, `ParseLayerBody` (alice/kimi-k3 convert),
  `MergeGenerationConfigEos` (gemma-пара через `eos_ids`).
- `selfcheck_common.h`: `Unfence` ×3 (fsi/code/det).
- `fast_load_internal.h`: `UploadBytes` ×3 (mistral×2/xing),
  `Bf16ToHalf` ×2 (alice/kimi-k3), `UploadOne` ×2 (nemotron-пара).
- `expert_device_cache.h`: `FreeExpertCacheCommon<T>` ×4
  (gigachat/mimo/nemotron×2, CUDA-guarded).
- hy: `HyMemoryPassportGeometry` — одна дефиниция (convert.cpp), load вызывает.
Тест: `test_config_json.cpp` (fast; RED доказан мутацией).

### Ф6 — Ornith ↔ Qwen MoE: один шаблонный контур
- `lsx_moe_load.{h,cpp}`: `moe::MoePinnedWeights<MoeTopology>` /
  `LoadMoeDenseWeights<T>` — load.cpp ornith был **100 %** rename-копией
  (доказано нормализованным diff). Семейные заголовки — alias + inline-делегаты.
- `lsx_moe_inference_pipeline.{h,cu}`: `moe::RunMoeGeneration<MoeTopology,
  AbortStream>` — пайплайн ornith был **99.4 %** копией; единственные
  смысловые дельты: `abort_stream` (compile-time политика: qwen `<true>`,
  ornith `<false>` — флаг компилируется out, break идентичен) и алиас
  fast-math toggle. Provenance-diff:
  `/mnt/fast_storage/tmp/dedup-audit/pipeline-provenance-diff.txt`.
- `lifecycle_tu` трёх манифестов (ornith/qwen3.5/qwen3.6) обновлён на
  нейтральный ТУ; logging-allowlist перенесён; гварды зелёные.
- **Не слито (обосновано):** конвертер ornith (ratio 0.71 — per-expert vs
  fused gate_up, MTP-учёт, другой format tag; слияние = переписывание);
  model.cpp — идентичны только тривиальные делегаты (Infer 7L / MsSince 6L /
  ParseToolCalls 6L), вынос не даёт объёма.

### Ф7 — Nemotron Lightning ↔ Ultra
- `nemotron_moe_shared_kernels.cuh`: 5 kernels-ядер (conv_step/ssm_step/
  relu2_half/moe_weighted_sum/add_half_to_f32) + 4 pipeline-ядра
  (embed_gather/rmsnorm_half/residual_add/kv_store) — все побайтово
  идентичны; `ssm_step` получил `dt_min` параметром из паспорта семьи
  (оба 0.001f). `gated_rmsnorm`/`router` НЕ слиты (тела разные).
- `nemotron_moe_expert_cache_impl.h`: общий шаблонный движок
  (Load/Plan/Allocate/Stage/Prewarm/Resolve) со Strings-политикой —
  log-сообщения побайтово сохранены (единственная дельта файлов была
  `log_prefix`). Provenance:
  `/mnt/fast_storage/tmp/dedup-audit/nemotron-cache-provenance.txt`.

### Остаток скана (дополнительно закрыто)
`lsx_kernels_misc.cuh`: `Fp8Tile128ToBf16Kernel` (gigachat/ling/mimo),
`EmbedGatherRowsKernel` (beam/flint/hy), `AddF32Kernel` (flint/inkling×2),
`RelBiasKernel`/`NormF32InKernel`/`LogScaleKernel`/`ScaleToHalfKernel`/
`EmbedGatherRowKernel` (inkling-пара), `RepeatHcKernel` (glm-flash/qwen-exp);
inkling `cast_half`/`half_to_f32` → нейтральные конверсии Ф3.

## Объяснённый остаток (21 группа ядер / 578 строк, 7 хост / 218)

| группа | почему осталась |
|---|---|
| deepseek flash/pro пары (~380 строк: act_quant/compressor/kv_act_quant/rope_apply/clamp/silu_mul/scale/…) | **блокер Ф2 Task 7**: общий загрузчик даёт Pro +2.2 GiB resident (не хватает на 2 слоя из 61). Это дефект учёта памяти, не дедуп; исправление = изменение поведения, запрещено Вариантом 1. Копии живут до закрытия memory-audit (merge `1e050244`). |
| gemma-dense ↔ muse-dense (attn 81L, argmax 35+34L) | тела attn/argmax идентичны, НО ТУ с разными `-fmad` (gemma on, muse off), а split/combine тела РАЗНЫЕ (muse — другая раскладка partials). Слияние attn/argmax потребовало бы `*_rn`-переписывания под общий флаг — вне Варианта 1. |
| gemma_geglu (10L ×2, один файл) | внутрифайловая пара batched/non-batched — разные вызовы, не дублирование между семьями. |
| flint PyJsonValue (66L) | тело идентично нейтральному, но вызывает семейный PyJsonString (ensure_ascii=False) — группа не замкнута. |
| CheckOrnithFrozenGeometry / check_frozen_geometry_36 (50L) | паспортные проверки: `using namespace frozen` читает СВОЙ паспорт семьи — слияние нарушит инвариант независимости геометрии. |
| Infer minimax-пара (27L) / hy-пара (26L) | тонкие делегаты на семейный RenderPolicy — тело идентично, но адресует семейную политику; вынос не даёт объёма. (ParseToolCalls kimi-пара вынесена в `tools::DecodeAndParseToolCalls`.) |
| ReadGenerationIds hy/minimax (19L) | вызывает семейные get_i-лямбды (разные тела) — группа не замкнута. |
| AppendEncoded mistral/xing (17L) | `static_cast<long>` в insert — семантически идентично, но не побайтово; критерий Варианта 1 — побайтово. |
| Free glm/kimi (13L) | разные типы PinnedWeights с разными полями — совпадение тела случайно, слияние требует общего базового типа (переписывание). |

## Верификация

- **Уровень 0 (provenance):** построчные diff-артефакты на каждый
  не-дословный перенос (depth_mix, pipeline, expert_cache) — только
  санкционированные дельты. Дословные переносы доказаны нормализованным
  diff до переключения.
- **Уровень 1:** debug-сборка 0 новых warnings; все гварды PASS
  (independence/suite/manifest/naming/infrastructure/absl×2/logging/labels/
  expert_store); финальный fast-свип (свежий прогон): **100% passed, 0 failed из 156** (3 SKIP — kimi-k3/ling tokenizer parity, expert_store_stage: нет деревьев/условий) (в первом прогоне словлен
  настоящий пропуск ветки: `test_expert_store_guard` legacy_spread — Ф6
  перенёс `MoEExpertStore&` в `lsx_moe_inference_pipeline.{h,cu}`, allowlist
  `kLegacyConsumers` не был обновлён; исправлен с комментарием-провенансом,
  RED→GREEN наблюдался; `test_gpu_perf` — VRAM-проба, флейк при параллельных
  GPU-тестах, поодиночке PASS, файлы не тронуты);
  29/29 tool-framing тестов PASS; новые нейтральные тесты PASS.
- **Уровень 2 (побитовый, release-znver4-aocc): 26/26 MATCH, 0 DIFF.**
  `lsxbenchmark --selfcheck` (per-family `--max-gen` из `selfcheck_max_gen`,
  `--bit-exact` из `selfcheck_bit_exact`) прогнан на OLD-бинаре main @911ba823
  (release-aocc, 19:01, символов ветки нет — проверено nm) и на NEW-бинаре
  ветки для всех 26 затронутых моделей с деревьями. `logits_hash` + first
  token побитово идентичны на каждой. Где хеш записан в матрице/отчёте —
  совпадает с эталоном (alice f900547d25be7914, ornith c3fe24e9dbc55b1e,
  qwen3.5 f4c91be5f2206d56, granite c165e80fde6401cf, granite-switch-8b
  930dd592ea5f5834, qwen3.8-flash-next 2e3bcd090441ad65, dory f100088895adb12d,
  hy-simulmt 44b846591fd64cb9, qwen3.5-9b a2af669548e59c73, qwen3.8-27b
  8bf2a2bf937eca91, deepseek-v41 b759bc84b820b4de). Артефакты:
  `/mnt/fast_storage/tmp/dedup-audit/barrier/` (52 лога) + `barrier-run.log`.
  **Критерий оператора (побитовая идентичность поведения) выполнен.**
- **Perf-барьер (release, 2048 токенов, −3% порог):** 20/26 PASS, 6 DEGRADED:
  alice 40.25 vs 45.7 t/s (−11.9%), mistral-small-4 41.92 vs 45.77 (−8.4%),
  inkling 1.547 vs 1.68 (−7.9%), qwen3.8-flash-next 32.50 vs 33.91 (−4.2%),
  granite-4.2 73.99 vs 76.79 (−3.65%), nemotron-ultra 1.719 vs 1.78 (−3.4%).
  Probe bandwidth во всех прогонах в норме (~1522–1526 vs 1533 GB/s).
  Harness по своей штатной механике (kUpdate) переписал 20 строк
  `perf_baselines.json` — дифф откачен (`git checkout`), re-seeding запрещён.
  **A/B OLD-vs-NEW (app-путь `lsxbenchmark --bench --max-gen 2048`, чистая
  машина, 120 s cooldown, old/new/old/new):**
  alice old 47.1/46.9, new 46.6/46.9 (паритет, ~1% шум);
  qwen3.8-flash-next old 42.2/43.8, new 43.8/43.8 (new ≥ old, оба ВЫШЕ
  baseline 33.91); granite-4.2 old 75.5/75.5, new 75.6/75.6 (паритет;
  app-путь 98.3% baseline — внутри −3% допуска, test-путь 96.35% — за).
  mistral-small-4 old 42.1/44.7, new 44.7/44.5 (new ≥ old);
  nemotron-ultra old 1.7225/1.7520, new 1.7649/1.7481 (new ≥ old);
  inkling old 1.5301/1.6529, new 1.6382/1.6547 (new ≥ old).
  **Ни одного perf-регресса ветки на app-пути (6/6 паритет или лучше).**
  **OLD-test оракул** (main-овский `test_perf_baseline`, та же машина, те же
  baselines): alice/granite/nemotron-ultra PASS; qwen-exp DEGRADED (perf
  41.91/43.39 = 96.6%, r3 30.15/33.91 = 88.9%), mistral DEGRADED (perf
  43.52/45.77 = 95.1%, r3 47.60/49.82 = 95.5%), inkling DEGRADED (perf
  1.5719/1.68 = 93.6%) — **на СТАРОМ main-бинаре**. Ночные NEW-прогоны
  (05:00–08:00) additionally просели по среде (alice test 40.25 ночью vs
  PASS днём на OLD). Итог: все 6 DEGRADED — test-путь/среда/устаревшие
  baselines, присутствующие на main; регрессом ветки не является ни один.
  baselines не пересеваны (kUpdate-мутации откачены в обоих деревьях).
  Артефакты: `/mnt/fast_storage/tmp/dedup-audit/perf-ab/` (24 лога),
  `perf-old-test/` (6 логов).
- Полный inference-свип debug (335 тестов): все падения — задокументированный
  долг либо предсуществующие дефекты main, ни одного нового регресса:
  deepseek-v4.1 FSI/CODE/DET (matrix 1/3, 1/3, 2/3); gemma_moe_model (A3.7,
  synthetic vocab OOB, подтверждён на чистом main); gemma_dense FSI-01;
  glm-flash loader/model (A3.8, missing router bias); granite-switch DET-02 ×2;
  nemotron lightning FSI/CODE/DET (matrix 2/3, 2/3, 1/3, budget-sensitive).
  **Найденный pre-existing RED:** `test_mimo_moe_model` (4/4, «[mimo-model]
  router upload») падает ИДЕНТИЧНО на OLD-бинаре main @911ba823 (19 ноября,
  до изменений ветки) — дефект main, не задокументирован; исправление вне
  объёма дедуп-аудита (Вариант 1). Зафиксирован здесь как найденный.
  **Найденный pre-existing «Not Run»:** `test_gemma_moe_{norm,rotary,attn,moe}`
  зарегистрированы в манифесте, но не собираются — коммит main `ffd26907`
  (6 ноября, миграция манифестов) удалил их build-wiring (`foreach
  add_executable`), оставив регистрацию. Ни main, ни ветка не строят их
  (проверено по build.ninja); exe на main — stale от 6 ноября. Дефект main,
  исправление (вернуть add_executable) вне объёма дедуп-аудита. Тот же дефект
  касается ещё 12 тестов: `test_qwen_dense_*` (full_attn, full_attn_parallel,
  fused_linear_attn, fused_mlp, linear_attn, mlp, norm, nvfp4, rotary,
  shuffle_reduce) и `test_qwen_nvfp4_{gemv,tile_gemm}` — все «Not Run».
  **Найденный pre-existing RED:** `test_tokenizer_vocab_contract` —
  Muse-Glimmer-30B не объявлен в `FindKnown` (падает идентично на OLD-бинаре
  main); 524/525 assertions pass. Дефект main, вне объёма дедуп-аудита.

## Ограничения (честно)

- Семейства без сконвертированного дерева на этой машине (deepseek-pro,
  gigachat, kimi-k3/k3.1-pro, minimax×2, kimi-k2.7, ling, beam, step, flint,
  mistral-large, glm-5.2, mimo): барьер = unit/kernel-тесты + provenance +
  fast framing-тесты. Реальных весов для них нет — это ограничение, не pass.
- Конвертер ornith и model-делегаты остались семейными (обосновано выше).
- Пере-сид `tests/perf/perf_baselines.json` не делался (запрещён критерием).