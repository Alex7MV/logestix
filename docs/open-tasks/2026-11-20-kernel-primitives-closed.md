# Ф3 закрыта: AttnRes depth mixer + конверсионные ядра + прибор dedup_scan

Дата: 2026-11-20
Ветка: `refactor/dedup-audit-2026-11`
План: `docs/superpowers/plans/2026-11-20-dedup-full-audit.md` (Task 1–3),
замещает `2026-11-03-dedup-phase3-kernel-primitives.md`.
Дизайн: `docs/superpowers/specs/2026-11-20-dedup-full-audit-design.md`.

## Что сделано

### Task 1 — прибор `tools/dedup_scan.py`

Эталонный измеритель остатка дублирования (read-only, exit 0 всегда):
нормализованные тела `__global__` (174→180 `.cu/.cuh`) и хост-функций,
группировка побайтово идентичных тел **без учёта имени** (старый скан
группировал только одноимённые — переименованные копии не видел).
Юнит-тест `tools/test_dedup_scan.py` (6 кейсов, ctest `test_dedup_scan`,
label fast) пинит правила группировки на синтетических фикстурах.

Baseline (main @ 911ba823): **52 группы ядер / 929 лишних строк,
26 хост-групп / 1271 строка**.

### Task 2 — `depth_mix_kernel` → `lsx_kernels_attnres.cuh`

- Нейтральное ядро `kernels::AttnResDepthMixKernel<MaxSrc>`; инстанцирования
  `<64>` (alice) и `<16>` (kimi-k3) сохраняют smem-раскладку побайтово.
- **Отступление от старого плана (зафиксировано в дизайне):** перенос в
  `lsx_attn_inference.cu` невозможен дословно — alice/kimi-k3 живут в
  `-fmad=false` ТУ, а `lsxcommon_qwen_cu` — с contraction. Поэтому заголовок
  инклюзится семейными ТУ, и каждая `+`/`*`/`/` заменена на
  `__fadd_rn/__fmul_rn/__fdiv_rn` (flag-independent, паттерн
  `lsx_kernels_*.cuh`)._block_sum перенесён с trailing `__syncthreads()`
  варианта kimi (добавляет барьер, не значение).
- Семейные guard-ы (`n_sources > 64`, `n_src <= 0 || n_src > 16`) остались в
  семейных лаунчерах; коды ошибок не менялись.
- Provenance: построчный diff исходного тела против нейтрального —
  `/mnt/fast_storage/tmp/dedup-audit/depthmix-provenance-diff.txt`
  (только санкционированные дельты: `*_rn`, `MaxSrc`, `scratch[256]`=kBlock,
  trailing barrier, имя/форматирование).
- Тест `tests/test_attnres_kernels.cu` (FMAD_OFF, label inference):
  CPU-эталон double-precision, `n_src ∈ {1,3,5,cap}`, `hidden ∈ {8,32,64,2048}`.

### Task 3 — конверсионные ядра → `lsx_kernels_common.cuh`

Одно grid-stride `int64_t` ядро на направление (шаблон `SrcT/DstT` с
дефолтами — weak linkage, иначе multiple definition host stub при линковке):
`F32ToF16Kernel`, `F16ToF32Kernel`, `F32ToBf16Kernel` (`_rn`),
`Bf16ToF16Kernel`, `Bf16ToF32Kernel`, `F16ToBf16Kernel`.

Удалены 29 семейных копий в 12 файлах: gigachat×2, glm-flash, mistral,
mistral-large4, xing, nemotron×2 (порядок аргументов `(src,n,dst)` сохранён
на вызовах), qwen-exp, deepseek `load.cuh`-пара (static-ядра в заголовке),
deepseek pipeline-пара (`half_to_float_f2h_kernel`).

**Намеренно оставлены:** glm-flash `bf16_to_f16` (uint8-битовый вход —
другая сигнатура), `flash_f16_to_f32` (scale-аргумент), `scale_f32_to_f16`
(умножение — не cast), deepseek `clamp_*` (clamp — не cast),
`silu_mul` deepseek (асимметричный clamp, зафиксировано в
`lsx_kernels_act.cuh:14`).

Тест `tests/test_kernels_common.cu` (label inference): все 6 направлений ×
`n ∈ {1,255,1024,100000}` с capped grid 1024 (grid-stride покрытие),
побитовое сравнение с хостовыми интринсиками; отдельный кейс контракта
`n=0` (лаунчеры guards `n<=0 → cudaSuccess` без запуска).

## Верификация

- Уровень 0: provenance-diff приложены (см. выше); `dedup_provenance.py`
  для точечных переносов даёт «missing 0» на ядрах-источниках.
- Уровень 1: debug-сборка 0 новых warnings; `test_attnres_kernels`,
  `test_kernels_common`, `test_alice_moe_kernels`, `test_kimi_k3_kernels`
  PASS; fast-свип 155/156 (единственный сбой — `test_gpu_perf`,
  VRAM-проба, чувствительная к параллельным GPU-тестам; поодиночке PASS,
  предсуществующая флейк-чувствительность, файлы не тронуты).
- Уровень 2: см. итоговый отчёт аудита
  (`2026-11-20-dedup-full-audit-closed.md`) — общий побитовый барьер
  release-znver4-aocc для всех фаз.

## Числа (прибор dedup_scan, до/после всего цикла)

| метрика | baseline | после |
|---|---|---|
| группы ядер | 52 | 21 |
| лишние строки ядер | 929 | 578 |
| хост-группы | 26 | 8 |
| лишние строки хоста | 1271 | 239 |

Остаток объяснён в итоговом отчёте (deepseek flash/pro пары — блокер Ф2;
gemma-dense↔muse — разные `-fmad` флаги и разные тела combine/split;
паспортные проверки и делегаты — намеренно семейные).