# Gemma-4-26B-A4B-it Decode Optimization Roadmap

Date: 2026-09-11
Status: committed (not yet scheduled)
Baseline: 98.1 t/s (2048-tok gen, RTX 5090, EPYC 9V74, preset linux-release-znver4)
Journey: 55.9 (initial after IMA fix) → 74.5 → 77.9 → 98.1 t/s

## Профиль остатка (nsys, 512-tok, prof8-9)

| Ядро | % GPU | мс/токен | GB/s (оценка) | Комментарий |
|---|---|---|---|---|
| `lsx_nvfp4_gemv_kernel<2>` | 26.1% | 2.55 | ~400-420 | attn q/k/v/o + dense gate/up/down, ~185 запусков/токен |
| `RmsNormTreeKernel` (W1) | 13.5% | 1.32 | — | 108k запусков, по 6.2µs |
| `gemma_gemv_pair_batch_kernel<2>` | 13.4% | 1.31 | ~410 | routed-эксперты gate/up (фьюжен уже есть) |
| `gemma_logits_rows_nvfp4_kernel` | 10.3% | 0.99 | ~610 | LM head warp-per-row ILP2 |
| `gemma_split_kv_partials_kernel` | 9.4% | 0.91 | — | чанк 128; avg 30µs, хвост до 45µs |
| `gemma_gemv_down_batch_kernel<1>` | 7.4% | 0.72 | ~410 | routed down |
| `lsx_nvfp4_gemv_kernel<4>` | 3.8% | 0.37 | ~470 | крупные формы (dense) |
| `gemma_router_softmax_topk_kernel` | 2.9% | 0.29 | — | 9.6µs, блок/токен |
| Прочее (rope, embed, f16↔bf16, axpy, scale, combine) | ~13% | ~1.3 | — | мелочь по 0.5-4µs × много |

Итого streaming-GEMV семейство ≈ 51% GPU-времени, все на ~23% пиковой полосы
(1792 GB/s). Wall-цель «железного» потолка ≈ 140-160 t/s.

## Направления (по убыванию отдачи)

### R1. Общий NVFP4 GEMV: поднять 400 → 900+ GB/s  (главный рычаг, ~+25-30 t/s)

Семейство `lsx_nvfp4_gemv_kernel` (общее для qwen/glm/gemma — `lsx_inference_nvfp4.cu`)
держит ~26% времени на ~23% полосы. Причины:
- KP=2: на строку 2 варпа, lanes читают uint4 с шагом 32 — на gemma-геометрии
  (K=2816 → 176 uint4/строку) это 5-6 раундов зависимых загрузок с ILP=1.
- Ни prefetch, ни double-buffer следующей порции весов.

Приёмы (в порядке безопасности контракта бит-точности):
1. **ILP внутри lane** — аккумулировать 2-4 uint4-чанка в независимые
   аккумуляторы (как сделано в gemma head/router): порядок слияния фиксирован
   → бит-точность сохраняется по построению. Проверяется parity-тестами
   `test_lsx_nvfp4_dispatch.cu` (144k assertions) и gemma/qwen host-reference.
2. **uint4×2 (32B) за транзакцию** — 2 смежных uint4, парный дешифр.
3. **cp.async/TMA prefetch** следующего чанка — только если (1)-(2) не дадут
   2×; самый инвазивный вариант.
Оценка: GEMV-семейство 13.9µs → 6-7µs, ~+25-30 t/s.
Риск: затрагивает 3 модели — гонять полный fast + gemma/qwen/glm kernel-сьюты.

### R2. RmsNormTreeKernel: фьюжн в потребителей  (~+10-15 t/s)

13.5% — 6.2µs на запуск при полезной работе <1µs. 108k запусков/512 токенов
= 6-7 норм на слой × 30 слоёв. Варианты:
- **Фьюжен scale+norm** (router pre-scale уже отдельным ядром `gemma_scale_vec_kernel`
  — влить в norm-вызов).
- **Фьюжен norm+bf16_to_half** на входах NVFP4-GEMV веток (сейчас отдельный
  cast-кернел перед каждой цепочкой).
- **Фьюжен residual+norm** (axpy → norm → один кернел).
Каждая пара −1 запуск −2-4µс латентного промежутка; целевой счётчик <40k.

### R3. LM head → TC mxf4nvf4 / сегментный TC путь  (~+8-10 t/s)

10.3%: warp-per-row 0.95мс — упор в скалярный дешифр nibble. Head — единственная
матрица V×H (262144×2816, 368MB NVFP4), это TC-класс. Варианты:
- Диспетчер `RunLsxNvfp4Gemm` с TC-путём (уже есть для seq==1) — но head требует
  argmax, не полный logits: нужен fused TC-GEMM + block-argmax, либо TC GEMM в
  d_logits + отдельный reduce.
- Контракт: argmax winner не меняется от порядка суммирования? НЕТ — плавающая
  сумма меняет биты logits, winner может смениться на near-tie. Нужен
  tolerance-envelope контракт (как split-KV attention) или явное решение
  бит-точности vs точности.

### R4. Split-KV: убрать хвост 45µs  (~+3-5 t/s)

Чанк 128 дал avg 30µs, но max 45µs (при window-полосе → 8 чанков). Причины
хвоста: warp divergence на граничных чанках + два cudaMallocAsync/запуск.
- Пулы `d_part/d_st` в GenWorkspace (убрать mallocasync из горячего пути).
- Chunk 64 при малой полосе (ранние токены): grid 16×30 блоков.
- Балансировка: чанк по полосе, а не фиксированный.

### R5. Router phase-2: блок 256 → warp-scan на 32 треда  (~+1-2 t/s)

9.6µs всё ещё: block-reduce с 8 барьерами. Вариант: 1 warp, ballot-argmax
с lower-index tie-break через `__ffs(__ballot(...))` — 0 барьеров.

### R6. Конвертерная упаковка [gate|up] для dense-MLP + attn qkv  (~с R1, +5-8 t/s)

Как GLM-Flash pair-GEMV: упаковать gate/up в один NVFP4-пак → один запуск
`RunLsxNvfp4GemmGateUpPair` вместо двух. Для dense (I=2112, K=2816) и qkv.
Формат-изменение: новый layout в logestix-gemma индексе, конвертер + loader +
диспетчер. Делать ПОСЛЕ R1 (общее ядро сначала).

### R7. Экспертный batch: больше K в полёте  (инфраструктурно)

`gemma_gemv_pair_batch/down_batch` уже батчат K=8 экспертов. Глубина ILP
внутри батча ограничена shared-редукцией. После R1 переоценить.

## Дисциплина

- Каждый пункт: TDD, бит-точность/детерминизм-контракт явно в тесте.
- R1 затрагивает общий слой — прогон fast 41/41 + все kernel-сьюты 3 моделей
  + real-model selfcheck каждой.
- Ре-профиль после каждого пункта (nsys, 512 tok, один и тот же промпт).
- Фиксация скорости в AGENTS.md (gemma_moe_model entry) после достижения
  стабильного числа на полном прогоне (2048 tok).
