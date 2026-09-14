# SGLang CUDA adoption program — Implementation Plan (umbrella)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Subagents ЗАПРЕЩЕНЫ политикой AGENTS.md («Autonomous Mode») — выполнение
> только inline основной сессией. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перенести в logestix наиболее ценные CUDA-техники из SGLang
(`/home/alexmv2025/projects/sglang`, Apache-2.0; код НЕ копируется — только
идеи/алгоритмы/схемы), не нарушая toolchain, инварианты точности и
одно-GPU/одно-стримовую архитектуру движка.

**Architecture:** Программа из 8 независимых подсистем (workstreams W1–W8).
Это **umbrella-план**: он фиксирует scope, референсы, файлы, критерии приёмки и
порядок. Каждый W* при запуске получает **собственный** детальный план
(`docs/superpowers/plans/YYYY-MM-DD-<w>.md`) после короткого дизайн-прохода
(`brainstorming`), потому что writing-plans требует один план на подсистему и
каждый план должен производить работающее тестируемое ПО сам по себе.

**Tech Stack:** CUDA 13.4 / CC 12.0–12.1 (sm_120a/121a), C++20, NVCC + CCCL,
CMake presets `linux-debug`/`linux-release-znver4`(`-aocc`), Catch2 v3,
lsxbenchmark, spdlog, abseil (host-only). Референс: SGLang CUDA-ярусы AOT/JIT/
CuTe-DSL (`python/sglang/kernels/`).

## Global Constraints

- AGENTS.md: субагенты ЗАПРЕЩЕНЫ — только inline. AI — единственный редактор кода.
- Архитектуры ONLY CC 12.0/12.1. **Нельзя брать sm100a/103a tcgen05/TMEM/2-CTA/
  cluster-ядра** (expert-specialization `es_sm100_mxfp8_blockscaled_*`,
  `nvfp4_gemm_swiglu_nvfp4_quant.py`, `kda_blackwell`/`gdn_blackwell`) — на
  sm_120/121 tcgen05 отсутствует; SGLang сам это подтверждает
  (`ops/attention/flash_mla_sm120.py`: «the flash_mla CUDA kernel is not
  available» на SM120).
- NVFP4-математика — ТОЛЬКО через модельно-нейтральный слой
  `include/lsxcommon/lsx_inference_nvfp4.h` + `std::mdspan`
  (MODEL INGEST INVARIANT, шаг 5). Никаких параллельных named-копий диспетчера.
- Нормы/биасы — FP16/BF16 (ИНВАРИАНТ ТОЧНОСТИ). Численный класс не менять без
  явного гейта; selfcheck Paris ×3 остаётся бит-точным.
- Один стрим end-to-end, без `cudaDeviceSynchronize` в decode-loop; shared-mem
  RMW — только thread 0 под `__syncthreads()`.
- lsxbenchmark/perf — строгий GPU-эксклюзив; одна GPU-нагрузка за раз.
- Логи — только spdlog; `absl::StrFormat` — printf-стиль (`%d`, не `{}`);
  комментарии не добавлять (кроме Doxygen в уже документированных файлах).
- Каждый новый тест получает label в том же коммите (`.opencode/skills/test-lifecycle`).
  Loader/model-стадии не грузят vault (guard `test_model_suite_guard`).
- Коммиты: английский, `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`.

## SGLang reference inventory (что смотреть, что НЕ смотреть)

**Применимо (sm120/sm121 или архитектурно-нейтрально):**
- `python/sglang/kernels/jit/include/sgl_kernel/{tile,mbarrier,vec,type,warp,atomic,cta}.cuh` — примитивы.
- `jit/utils/arch.py` — gating sm_120f/120a/121a (trap-стабы CUTLASS при неверном target).
- `jit/csrc/gemm/sm120_fp8_gemv.cuh` — M=1 FP8 GEMV, shared-стейджинг активации, `__ldcs`, warp-shuffle.
- `jit/csrc/gemm/fp8_blockwise/fp8_blockwise_scaled_mm_sm120.cuh` — FP8 blockwise GEMM sm120.
- `jit/csrc/moe/expert_pack_mxfp4.cuh` — `mxfp4_matvec_kernel`/`mxfp4_matvec_dual_kernel` (прямое чтение квантизованных).
- `jit/csrc/moe/moe_fused_gate.cuh`, `moe_topk_sigmoid.cuh`, `align_single_token.cuh`, `moe_permute_prepare.cu`, `moe_finalize_fuse_shared.cu`, `topk_sum.cuh`.
- `jit/csrc/deepseek_v4/{fused_norm_rope,fused_norm_rope_v2,main_norm_rope,c128*,c4*,c_plan,topk_v1,topk_v2,hash_topk,paged_mqa_metadata,silu_and_mul_masked_post_quant}.cuh`, `deepseek_v32/indexer_k.cuh`.
- `jit/csrc/dsa/{fused_store_index_cache,kpool_topk_transform}.cuh`.
- `jit/csrc/elementwise/{fused_qknorm_rope,qknorm_across_heads,fused_add_rmsnorm,hc_combine,rmsnorm,activation}.cuh`.
- `ops/quantization/{nvfp4_online.py,mxfp8_quant.py,mxfp8_interleave_sf.py,per_token_group_quant_8bit_v2.py}`, `_jit` per-token quant.
- `srt/model_executor/cuda_graph_config.py` + runner — full/breakable/tc_piecewise decode graphs.
- `srt/speculative/dflash_*`, `srt/speculative/...` + MTP — spec-decode v2.
- `srt/layers/quantization/fp4_kv_cache_quant_method.py`, `ops/attention/flash_mla_sm120.py` — FP4/FP8 KV layout.
- `kda_kernels/qwen38_qsa_sm121/` — QSA decode SM121 (mma.sync BF16, split-KV merge).

**Не применимо:** всё `tcgen05`/TMEM/`_tcgen05.py`; FlashInfer/CUTLASS/Triton/CuTe-DSL
как зависимости; distributed EP/all-reduce/IPC/a2a; paged attention; continuous
batching; radix-cache (только если taskmosaic станет multi-request сервером).

## Program map

| ID | Workstream | SGLang ref | Эффект | Риск | Зависит |
|----|-----------|-----------|--------|------|---------|
| W1 | Speculative/MTP-верификация (k токенов на проход) | `dflash_*`, MTP | очень высокий | высокий | — |
| W2 | Прямой квантизованный matvec без bf16-dequant | `mxfp4_matvec*`, `sm120_fp8_gemv` | высокий | средний | — |
| W3 | Fused norm+rope/qknorm/add-rmsnorm/gate+topk | `fused_norm_rope*`, `fused_qknorm_rope` | высокий | низкий | — |
| W4 | CUDA graph на весь decode-step | `cuda_graph_config.py` | средний-высокий | средний | W3 |
| W5 | FP4/FP8 KV-кэш | `fp4_kv_cache_quant_method.py` | средний | средний | — |
| W6 | Grouped GEMM для verify-батча | `moe_permute_prepare`, group GEMM | высокий | высокий | W1 |
| W7 | DeepSeek V4 CSA indexer/compressor сверка | `deepseek_v4/*` | средний | средний | — |
| W8 | Инфра: примитивы/PDL/arch-gating/registry | `sgl_kernel/*`, `arch.py` | базовый | низкий | — |

**Порядок:** W8 (база) → W3 (fusions, дёшево) → W2 (dequant) → W4 (graphs) →
W5 (KV) → W7 (CSA сверка) → W1 (spec) → W6 (grouped GEMM, после W1).
W1/W6 — крупные архитектурные изменения: перед реализацией обязателен
`brainstorming` и отдельный design-spec в `docs/superpowers/specs/`.

---

### W8: Инфраструктурная база (примитивы, PDL, arch-gating)

**Файлы:**
- Modify: `CMakeLists.txt` (арх-набор), `CMakePresets.json`
- Modify/Create: `libs/lsxcommon/src/lsx_kernels_{common,math,norm,quant,act}.cuh`
- Create: `tests/test_kernel_arch_guard.cu` (label fast/inference, по образцу `test_cuda_toolkit_guard.cu`)

**Задачи:**
- [x] **W8.1 Arch-gating (verified, no change).** `libs/lsxcommon/CMakeLists.txt:225-251`
  уже изолирует arch-специфичную `mxf4nvf4`-PTX в `lsxcommon_tc_cu` с
  `--generate-code=compute_120a,code=sm_120a;compute_121a,code=sm_121a`; все
  прочие OBJECT-либы — `120-real;121-real` (валидно без CUTLASS). SGLang-правило
  про `sm_120a/f` применимо только к CUTLASS-атомам, которых у нас нет.
  Evidence: `test_cuda_toolkit_guard` — 14 assertions / 4 cases passed.
- [ ] **W8.2 PDL (Programmatic Dependent Launch).** Добавить RAII-обёртки
  `LxPdlWaitPrimary()`/`LxPdlTriggerSecondary()` (по образцу
  `sgl_kernel/utils.cuh`) в `lsx_kernels_common.cuh`, только для sm_90+ (у нас
  sm_120a). Применить в первую очередь к паре «conv-shift → gemv» и
  «router → expert-chain».
- [ ] **W8.3 Единый primitive-слой.** Унифицировать `tile::Memory`-подобный
  доступ (`load_as/store_as`), `AlignedVector` уже есть — проверить дубли в
  `lsx_kernels_*.cuh`, свести без смены математики.
- [ ] **W8.4 Registry-инвентарь.** Не копировать Python-рантайм; достаточно
  расширить guard-тесты (реестр моделей/ядер), чтобы много-бэкендные пути
  (legacy vs fused vs TC) были перечислимы и тестируемы.

**Приёмка:** `ctest --preset linux-release-znver4 -L fast` зелёный; SASS
проверен (`cuobjdump -sass`) на отсутствие trap-стабов для SM120-ядер;
`test_cuda_toolkit_guard` не регрессировал.

**Риск:** смена `code=sm_120`→`sm_120a` меняет кэш/совместимость; проверить
запуск на обеих наличных CC-версиях.

---

### W3: Fused decode-ядрa (norm+rope, qknorm, add-rmsnorm, gate+topk)

**Файлы:**
- Modify: `libs/lsxcommon/src/glm_moe_flash_inference.cu`
  (`flash_conv_decode_kernel`, `flash_rmsnorm_gated_kernel`, `flash_l2norm_kernel`,
  `flash_kda_gating_kernel`, `flash_router_top8*_kernel`)
- Modify: `libs/lsxcommon/src/deepseek_moe_v41_inference.cu` / `_inference_attn.cu`
- Modify: `libs/lsxcommon/src/qwen_moe_inference.cu` (router softmax→top-8→renorm)
- Modify: `libs/lsxcommon/src/lsx_kernels_norm.cuh` (общие фьюжн-хелперы)
- Test: `tests/test_glm_moe_flash_kernels.cu`, `tests/test_deepseek_moe_v41_kernels.*`,
  `tests/test_qwen_moe_kernels.cu` (device-parity, label fast/inference)

**SGLang-референсы:** `deepseek_v4/fused_norm_rope.cuh` (RMSNorm + RoPE в одном
ядре, режимы CompressExtend/CompressDecode/Default, warp-per-row, `kBlockSize=128`,
PDL-опция), `elementwise/fused_qknorm_rope.cuh`, `qknorm_across_heads.cuh`,
`fused_add_rmsnorm.cuh`, `moe_fused_gate.cuh`.

**Задачи:**
- [ ] **W3.1 GLM-Flash: fused conv1d+qknorm+rope decode.** Один warp-per-row
  kernel по образцу `fused_norm_rope`: conv-shift + RMSNorm-gated + L2norm + KDA
  gating для decode-режима. Условие: побитовая идентичность текущей цепочке
  (device-parity тест с допуском 0, т.к. порядок fp32-аккумуляции сохраняем).
  - [x] **W3.1a (done, 2026-09-14):** post-conv SiLU слит в
    `flash_conv_decode_kernel` (вместо отдельного `flash_silu_kernel`), gate
    activation слита в `flash_rmsnorm_gated_kernel` (вместо
    `flash_gate_act_kernel`). Убрано 2 launch'а на KDA-слой (36 KDA-слоёв
    GLM-Flash). Bit-exact round-trip сохранён (conv → half → silu; gate-act →
    half → norm). Test twins `RunFlashConvDecodeTest`/`RunFlashSiluTest`/
    `RunFlashGateActTest`/`RunFlashRmsnormGatedTest`; тесты
    `flash conv decode + fused silu is bit-exact` и
    `flash rmsnorm gated + fused gate act is bit-exact`
    (`tests/test_glm_moe_flash_kernels.cu`) — `memcmp`-точные. Evidence: весь
    бинарь 2331 assertions / 9 cases passed. Осталось: свернуть 2× l2norm
    в 1 launch; слить conv-shift; замер t/s на vault (не сделан).
- [ ] **W3.2 DeepSeek V4.1: fused norm+rope.** Сверить `_inference.cu` RMSNorm/RoPE
  с `main_norm_rope.cuh`/`fused_norm_rope_v2.cuh`; слить, сохранив hc-пайплайн FP32.
- [ ] **W3.3 Router: fused softmax→top-k→renorm.** Для каждого MoE-семейства
  (glm/deepseek/kimi/qwen/gemma) проверить, что top-k selection совпадает с
  `moe_fused_gate` семантикой (tie → меньший индекс), смерджить softmax+renorm.
- [ ] **W3.4 HF-parity гейт.** Расхождение в tie-breaking/renorm запрещено;
  тесты `test_*_full_compute`/`test_*_selfcheck` остаются зелёными.

**Приёмка:** число launch'ей на decode-слой уменьшилось (nsys-замер), t/s растёт
на `lsxbenchmark --bench`, selfcheck Paris ×3 бит-точен, `-L fast` зелёный.

---

### W2: Прямой квантизованный matvec (устранить bf16-dequant)

**Файлы:**
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu`
  (`RunKimiExpertFfn` ~794, `RunKimiExpertFfnCached` ~885, `RunKimiRoutedTcBatch` ~1071)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu`, `lsx_inference_nvfp4.cu`
- Reference: `jit/csrc/moe/expert_pack_mxfp4.cuh` (`mxfp4_matvec_kernel`),
  `jit/csrc/gemm/sm120_fp8_gemv.cuh`
- Test: `tests/test_kimi_moe_kernels.*`, `tests/test_deepseek_moe_inference_kernels.cu`

**Проблема (зафиксирована):** у Kimi 71% времени ядер — `Nvfp4DequantKernel`
(материализует bf16 перед cublas). Полный W4A4 TC-путь заблокирован численно:
swiglu-выход несовместим с per-16 E2M1 pow2-квантованием down-GEMM
(`test_kimi_moe_tc_diag`, QUANT hq rel=120%). Значит нужен **формат активаций
повышенной точности** для down-GEMM (кандидаты: MXFP8/FP8 per-token-group
активации по SGLang `mxfp8_quant.py`/`per_token_group_quant_8bit_v2.py`).

**Задачи (design-first):**
- [ ] **W2.0 Дизайн:** выбрать формат активации down-GEMM (BF16-act × NVFP4-weight
  GEMV vs FP8-act) и способ его получения без materialization. Отдельный spec.
- [ ] **W2.1 Host-примитив:** per-token-group E4M3/E8M0 квантование в
  `lsx_kernels_quant.cuh` + unit-тест (label fast).
- [ ] **W2.2 Прямой GEMV:** weight остаётся NVFP4, читается `__ldcs`; активация —
  выбранный формат; редукция warp-shuffle. Побитовый порог задать по
  существующему классу точности (тест parity vs dequant-путь с явным допуском).
- [ ] **W2.3 Прокидка** в `RunKimiExpertFfnCached`/TC-batch; legacy остаётся
  fallback и default до прохождения selfcheck.
- [ ] **W2.4 Замер:** nsys до/после, убрать долю `Nvfp4DequantKernel`; perf-baseline.

**Приёмка:** доля dequant-ядра падает, t/s растёт, selfcheck Paris ×3
бит-точен, `-L fast`/`-L inference` зелёные.

---

### W4: CUDA graph на весь decode-step

**Файлы:**
- Modify: `libs/lsxcommon/src/qwen_moe_inference.cu:1641` (существующий per-layer
  FFN graph — обобщить)
- Create: `libs/lsxcommon/include/lsxcommon/decode_graph.h` +
  `libs/lsxcommon/src/decode_graph.cpp` (RAII-обёртка capture/instantiate/replay
  по образцу SGLang `cuda_graph_config.py`: full/breakable/piecewise, disabled)
- Modify: `libs/lsxcommon/src/glm_moe_flash_model.cpp`,
  `deepseek_moe_v41_inference_pipeline.cu`, `kimi_moe_inference.cu:1649`
  (`RunKimiGeneration`), `qwen_moe_model.cpp`
- Test: `tests/test_decode_graph.cpp` (label fast) + guard в selfcheck-тестах

**SGLang-референс:** `srt/model_executor/cuda_graph_config.py` — фазовые
бэкенды (DECODE/PREFILL), `bs`-бакеты, `max_bs`, breakable vs full, disabled-фолбэк.

**Задачи:**
- [ ] **W4.1 RAII-обёртка** `LxDecodeGraph` (capture stream, instantiate,
  replay, `Update()` для меняющихся указателей) — host-only, unit-тестируемо.
- [ ] **W4.2 Пилот на Kimi** (один decode-слой → весь цикл), с сохранением
  device-side token feed (round-25). Обязательное условие: никаких
  `cudaDeviceSynchronize` внутри цикла.
- [ ] **W4.3 Раскатка** на GLM-Flash и Qwen-MoE; DeepSeek V4.1 — последним
  (correctness-first контур).
- [ ] **W4.4 Гейт:** `LSX_DECODE_GRAPH=0` отключает; при несовместимости (expert
  miss/H2D) — eager-фолбэк без падения.

**Приёмка:** launch-overhead снижен (nsys), t/s растёт, детерминизм и selfcheck
не нарушены.

**Риск:** graph capture ломается на dynamic-shape expert staging; держать
per-layer FFN graph как уже доказанный baseline.

---

### W5: FP4/FP8 KV-кэш

**Файлы:**
- Modify: `glm_moe_flash_inference.cu:904` (`__half *d_kv_cache_k/v`) и близкие
  MLA-кэши (`kimi_moe_inference.cu` `KimiMlaState`, `deepseek_moe_v41_inference_attn.cu`)
- Reference: `srt/layers/quantization/fp4_kv_cache_quant_method.py`,
  `ops/attention/flash_mla_sm120.py` (NOPE=448/RoPE=64, block-64 scales,
  separate scale region per page)
- Test: соответствующие `test_*_kernels.*` + selfcheck

**Задачи:**
- [ ] **W5.1 Дизайн layout** KV (NVFP4 vs FP8) с учётом MLA и DSA/CED; отдельный spec.
- [ ] **W5.2 Квантование записи** в KV-append-ядрах; масштабы рядом со страницей.
- [ ] **W5.3 Dequant/consume** — внутри attention-ядра (не materialize).
- [ ] **W5.4 Гейт** `cfg.kv_quant` off/on; selfcheck Paris ×3 допуск.

**Приёмка:** VRAM под KV падает в ~2–4×, длинный контекст быстрее, точность Paris
в пределах согласованного допуска.

---

### W7: DeepSeek V4 CSA indexer/compressor — сверка с SGLang

**Файлы:**
- Modify: `libs/lsxcommon/src/deepseek_moe_v41_inference_attn.cu` (SWA + CED),
  `deepseek_moe_v41_inference.cu` (compressor/indexer/router)
- Reference: `jit/csrc/deepseek_v4/{c128*,c4*,c_plan,topk_v1,topk_v2,hash_topk,paged_mqa_metadata,silu_and_mul_masked_post_quant}.cuh`,
  `deepseek_v32/indexer_k.cuh`, `dsa/fused_store_index_cache.cuh`
- Test: `tests/test_deepseek_moe_v41_pipeline.*`, `test_deepseek_moe_v41_kernels.cu`

**Задачи:**
- [ ] **W7.1 Gap-анализ** алгоритмов: compressor ratio-планирование (`c_plan`),
  top-k варианты, indexer-K, masked post-quant — против текущей реализации.
- [ ] **W7.2 Перенести 1–2 выигрышных фьюжна** (напр. `fused_norm_rope` +
  `silu_and_mul_masked_post_quant`), с device-parity тестами.
- [ ] **W7.3 Замер** на `test_perf_baseline_deepseek_v41_flash` (2048 токенов).

**Приёмка:** parity-тесты зелёные, baseline улучшен, selfcheck 3/3 бит-точных.

---

### W1: Speculative / MTP-верификация (design-first, крупный)

**Файлы:**
- Create: `libs/lsxcommon/include/lsxcommon/spec_decode.h` +
  `libs/lsxcommon/src/spec_decode.cpp` (draft loop, verify, accept/reject)
- Modify: `glm_moe_model.cpp` (MTP-слой 78 уже есть),
  `qwen_moe_convert.cpp` (MTP хранится, не применяется), `model_runtime.cpp`
- Reference: `srt/speculative/dflash_*`, MTP/nextN, `srt/model_executor/...`
- Test: `tests/test_spec_decode.cpp` (label fast) + новый selfcheck-вариант

**Почему приоритет:** движок memory-bound на стриминге экспертов; k
верифицируемых токенов обслуживаются одним проходом по весам ⇒ амортизация
PCIe/DRAM-трафика на принятый токен, ожидаемо 2–4×.

**Задачи:**
- [ ] **W1.0 `brainstorming` + design-spec** (обязательно): тип draft
  (MTP vs ngram vs EAGLE), batch-геометрия verify, взаимодействие с
  expert LRU-кэшем, требования бит-точности (greedy verify должен быть
  детерминированным для принятых токенов).
- [ ] **W1.1** Draft-голова (MTP) как отдельный `IModel`-этап.
- [ ] **W1.2** Verify-батч: тот же frozen-graph, что и одиночный decode, но
  tokens=1+k; приёмка по argmax-совпадению.
- [ ] **W1.3** KV-буферы под verify; откат непринятых.
- [ ] **W1.4** selfcheck-вариант: принятые токены бит-совпадают с последовательным
  greedy decode.

**Приёмка:** t/s на реальном vault ≥ baseline, выходной поток при
`--bit-exact` побитово равен последовательному greedy; spec-baseline в perf.

---

### W6: Grouped GEMM для verify-батча (после W1)

**Файлы:**
- Modify: `libs/lsxcommon/src/lsx_tc_gemm.cu` / `.h` (batch/group режим)
- Modify: chain-драйверы MoE (`qwen_moe_inference.cu`, `kimi_moe_inference.cu`)
- Reference: `jit/csrc/moe/{moe_permute_prepare.cu,expert_pack_mxfp4.cuh}`,
  token-pipeline `align → permute → group GEMM → finalize+shared`
- Test: batch-parity к последовательным single-slot GEMV

**Задачи:**
- [ ] **W6.1** Token sorting/offset (`moe_align`-паттерн) host/device.
- [ ] **W6.2** Свой sm120 group GEMM (НЕ tcgen05): `mma.sync` mxf4nvf4/FP8,
  tile M>1, split-K по правилам `QwenTcBatchSplitK`.
- [ ] **W6.3** Finalize+shared-combine одним ядром.
- [ ] **W6.4** Побитовая parity per-slot; гейт по числу токенов на эксперта.

**Приёмка:** verify-батч ускоряется пропорционально попаданиям в один эксперт;
fallback на per-slot chain сохранён.

---

## Deliverables / acceptance (программа)

- Каждый W* закрывается собственным PR со своим детальным планом, тестами
  (label в том же коммите) и perf-записью там, где меняется скорость.
- Сквозной инвариант: все model-модели остаются `-L fast`/`-L inference`
  зелёными; selfcheck Paris ×3 бит-точен, кроме W5/W2, где допуск фиксируется
  в design-spec заранее.
- Никаких зависимостей от FlashInfer/CUTLASS/Triton/CuTe-DSL и никаких
  sm100a-ядер.

## Риски программы

- **tcgen05-ловушка:** соблазн взять «самые быстрые» sm100-ядра — они физически
  не работают на CC 12.x. Фильтр в Global Constraints обязателен.
- **Дизайн W1/W2/W5:** без `brainstorming`+spec есть риск переусложнить; не
  начинать код до утверждённого spec.
- **Бит-точность:** любой фьюжн/W4A4 может сломать Paris; каждый шаг —
  device-parity тест + selfcheck.
- **GPU-эксклюзив:** замеры только последовательно, lsxbenchmark — последним.

## Self-review

- **Scope:** W1–W8 — независимые подсистемы; umbrella фиксирует порядок и
  требует отдельный план на каждый (Scope Check writing-plans соблюдён).
- **Coverage:** все пункты Tier 1–3 из анализа (`2026-09-14 SGLang CUDA`) имеют
  workstream; N/A-пункты явно исключены.
- **Placeholders:** намеренно отсутствуют фиктивные шаги; W1/W2/W5/W6 помечены
  design-first и не содержат выдуманного кода — детальные TDD-шаги появятся в
  их собственных планах после spec.
- **Consistency:** имена артефактов (`lsx_inference_nvfp4.h`, `LsxDecodeGraph`,
  `RunKimiExpertFfnCached`, `QwenTcBatchSplitK`) взяты из существующего кода
  или объявлены здесь как производимые интерфейсы.

## Execution log

- **2026-09-14 W8.1** — verified only; no code change (arch config already
  correct). Evidence: `test_cuda_toolkit_guard` 14 assertions / 4 cases passed.
- **2026-09-14 W3.1a** — GLM-Flash conv+SiLU and gate-act+RMSNormGated fusions,
  bit-exact. Evidence: `test_glm_moe_flash_kernels` 2331 assertions / 9 cases
  passed. Pre-existing, unrelated failures at `Load()` in
  `test_glm_moe_flash_loader` / `test_glm_moe_flash_model` ("index topology
  disagrees with frozen geometry") reproduced on a clean stash — NOT caused by
  this change. Perf (t/s) on a real vault not yet measured.

