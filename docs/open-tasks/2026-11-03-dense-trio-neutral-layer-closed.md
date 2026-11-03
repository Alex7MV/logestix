# Dense-трио → нейтральный `lsxcommon::dense` — фаза 1 закрыта (2026-11-03)

Ветка `refactor/dense-trio-neutral-layer` (от `main` @ `f9118e03`).
Дизайн: `docs/superpowers/specs/2026-11-03-code-duplication-consolidation-design.md`
§2.2, §5 (Ф1). План: `docs/superpowers/plans/2026-11-03-dedup-phase1-dense-trio.md`.

## Что сделано

`granite_dense_inference.cu` содержал семь ядер (RoPE, causal GQA attention,
split-KV attention, combine, fused split-QKV, fused SwiGLU, vectorized
RMSNorm) и шесть лаунчеров, побайтово повторённых в
`phi_dense_inference.cu` и `granite_dense_switch_inference.cu` под другими
именами. Ядра и лаунчеры перенесены в новый нейтральный модуль
`lsxcommon::dense` (`include/lsxcommon/lsx_dense_inference.h` +
`src/lsx_dense_inference.cu`, отдельная OBJECT-библиотека
`lsxcommon_dense_cu`), семейные `RunGranite*` / `RunGraniteSwitch*` /
`RunPhi*` стали inline-делегатами.

Перенос — дословный, и это проверено прибором, а не глазами:

```
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/granite_dense_inference.cu \
    --to   libs/lsxcommon/src/lsx_dense_inference.cu --rename granite_=dense_
source kernels: 7  target kernels: 7
missing in target: 0  unexpected in target: 0
```

Для phi та же проверка прогнана против **до-рефакторного** состояния
(`git show b2c45cb9:libs/lsxcommon/src/phi_dense_inference.cu`): 7/7, 0
расхождений; то есть код, который исполнялся до фазы, и код, который
исполняется после, — один и тот же текст.

В `granite-switch` остались только его собственные ядра aLoRA
(`switch_scale`, `switch_scale_add`, `switch_lora_shrink`,
`switch_lora_expand_add`) — проверка показывает ровно эти четыре «missing»
и нулевой «extra».

## Измеренная дельта

| файл | до | после |
|---|---|---|
| `granite_dense_inference.cu` | 1 014 | 611 |
| `phi_dense_inference.cu` | 1 066 | 661 |
| `granite_dense_switch_inference.cu` | 1 223 | 824 |
| `lsx_dense_inference.cu` (новый) | — | 478 |
| **итого `libs/`** | **3 303** | **2 574** (−729)** |

| тест | до | после |
|---|---|---|
| `test_granite_dense_kernels.cu` | 433 | 164 |
| `test_phi_dense_kernels.cu` | 446 | 169 |
| `test_dense_kernels.cu` (новый) | — | 340 |
| **итого `tests/`** | **879** | **673** (−206)** |

Пять общих parity-кейсов переехали в нейтральный `test_dense_kernels.cu`;
семейные тесты сохранили NVFP4-budget-кейс и smoke-кейс, который держит
`RunGranite*`/`RunPhi*` на тестируемом пути. План оценивал −1 400 LOC в
`libs/`; фактически −729, потому что нейтральный модуль (478 строк) — это
тоже наш код: реальная экономия считается как «удалённые копии минус новый
модуль», а не как «размер копий».

## Барьер

| проверка | результат |
|---|---|
| `ctest --preset linux-debug-fast` | 134/134 |
| `ctest --preset linux-debug-convert` | 37/37 |
| `test_dense_kernels` | PASS (5 кейсов) |
| `test_granite_dense_kernels/_loader/_model/_selfcheck/_fsi/_code/_det` | 7/7 PASS |
| `test_phi_*` | 7/7 PASS |
| `test_granite_switch_*` | PASS, кроме документированного DET-02 |
| предупреждения сборки в наших файлах | 0 |
| `test_model_suite_guard` | 14/14 |
| `test_model_independence_guard`, `test_infrastructure_guard`, `test_absl_container_guard` | PASS |

Побитовые эталоны после фазы:

- granite: `logits_hash=c165e80fde6401cf`, top-1 12366 `Paris`, ×3 — совпадает
  с записанным в `docs/model-reports/2026-11-03-granite-4.2-30b.md`;
- granite-switch: `logits_hash=fdc632042c30919c` (эталон
  `docs/model-reports/2026-10-31-granite-switch-4.1-30b.md`), FSI 3/3,
  CODE 3/3, DET 2/3 — DET-02 (`CRITICAL` вместо `CATASTROPHIC`) это
  зафиксированный дефект источника (`docs/SELFCHECK_MATRIX.md` §44,
  `docs/OPEN_TASKS.md` §R10), а не регресс фазы;
- phi: `logits_hash=cf51b9bbd5c913dd` — **это первое измерение**: в
  документации хеш phi-4 не зафиксирован. Побитовость доказывается не им, а
  provenance-проверкой против до-рефакторного файла (см. выше); хеш записан
  здесь как baseline для следующих фаз.

## Что осталось открытым

- **perf-строки не измерены.** `test_perf_baseline` требует Release-сборки
  (`NDEBUG`) — в Debug он SKIP'ается, а существующие `build/linux-release*`
  привязаны к старому пути `/home/alexmv2025/projects/cuda/logestix` и
  непригодны после переезда дерева на `/mnt/fast_storage`. Команда для
  закрытия:
  `cmake --preset linux-release && cmake --build --preset linux-release --target test_perf_baseline && ctest --preset linux-release-perf -R "test_perf_baseline_(granite_4_2_30b|granite_switch_4_1_30b|phi_4)"`.
  Код при этом доказанно идентичен (уровень 0), поэтому ожидание — t/s в
  пределах шума; но это ожидание, а не измерение, и записано оно здесь
  именно так.
- `granite-switch` DET-02 и gemma-dense FSI-01 остаются красными по причинам,
  не связанным с этой фазой (`docs/SELFCHECK_MATRIX.md`).

## Ссылки

- Коммиты фазы: `b2c45cb9` (тест и скелет, RED), `bc7e3e5f` (перенос ядер),
  `c4dea989` (granite), `aa788759` (trim семейного теста), `40c144ea` (phi),
  `5652b868` (granite-switch).
- Инструменты: `tools/dedup_provenance.py` (уровень 0: «это перенос»),
  `tools/dedup_scan.py` (фаза 3: абсолютный остаток дублирования).
