# GLM-5.3-Flash Ingestion Design

Date: 2026-09-04
Status: Approved
Goal: Адаптация GLM-5.3-Flash по Model Ingestion Protocol (8 этапов) с максимально
возможной скоростью генерации на текущем оборудовании (RTX 5090 32GB + EPYC 9V74),
реализация через TDD.

## Инвариант №1: Не трогать GLM-5.2

- Все новые файлы — только `glm_moe_flash_*`. Файлы `glm_moe_*` (GLM-5.2)
  не редактируются вообще.
- Переиспользование существующих примитивов (`RunQwenNvfp4Gemm`, NVFP4 decode-таблицы
  `shared_experts.cu`, hc GatedResidual qwen4_exp, expert cache) — только по ссылке.
  Нужен изменённый вариант — новая функция/перегрузка, не правка.
- Легитимные касания чужих файлов (механические, без изменения семантики):
  - `libs/lsxcommon/src/model_registry.cpp` — одна строка `Register<...>()`
  - `tests/test_code_naming_guard.cpp` — allowlist варианта `flash`
  - `libs/lsxcommon/CMakeLists.txt`, корневой `tests/CMakeLists.txt` — новые таргеты
- Регрессионный щит: 12 `test_glm_*` executables с бит-точными `final_logits`
  в CI уронит любую случайную задетку GLM-5.2.

## Замороженная геометрия (compile-time, static_assert)

Источник истины: `/mnt/llm_models/models/GLM-5.3-Flash/config.json` (text_config,
model_type `glm5_next`) + инвентаризация safetensors (2026-09-04, 62 shard'а, 306 GB,
FP8 e4m3 blockwise 128×128 с `weight_scale_inv`).

```
architectures = Glm5NextForConditionalGeneration (vision отсекается, подсчёт ведётся)
hidden = 4096, vocab = 154880
Слои: 45 основных (0..44) + MTP nextn (45), num_nextn_predict_layers=1
KDA (linear_attention): 34 слоя [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,
  25,26,28,29,30,32,33,34,36,37,38,40,41,42,44]
  64 heads x head_dim 128; short conv1d kernel=4 (q/k/v веса [8192,1,4]);
  A_log F32[64], dt_bias F32[8192]; beta b_proj BF16[64,4096];
  gate low-rank f_a/f_b, g_a/g_b (128 rank); o_norm RMS BF16[128]
DSA (deepseek_sparse_attention): 11 слоёв [3,7,11,15,19,23,27,31,35,39,43]
  MLA: q_lora=1536, kv_lora=512, qk_head_dim=256, qk_nope=256, qk_rope=0,
  v_head_dim=256, 64 heads; mla_use_nope=true
  Indexer: index_n_heads=32, index_head_dim=128, index_topk=2048, index_kpool=4,
  compress=true, always_select_tail=true, rope_interleave=true
Hyper-connections (mhc=true): hc_mult=4, eps=1e-6, sinkhorn_iters=20
  hc_attn_fn / hc_ffn_fn BF16[24,16384]; hc_attn_base / hc_ffn_base F32[24];
  hc_attn_scale / hc_ffn_scale F32[3]
MoE: first_k_dense_replace=3 (слои 0..2 dense MLP I=12288), MoE слои 3..45
  n_routed_experts=288 (I=2048), n_shared_experts=1 (I=2048)
  Router: scoring_func=sigmoid, topk_method=noaux_tc, n_group=1, topk_group=1,
  num_experts_per_tok=8, norm_topk_prob=true, routed_scaling_factor=2.5,
  moe_router_dtype=float32; gate BF16[288,4096] + e_score_correction_bias F32[288]
eos_token_id=[154820,154827,154829], pad_token_id=154820
swiglu_limit=10.0
MTP: store-not-apply (решение оператора 2026-09-04)
```

## Решения оператора

| Вопрос | Решение |
|---|---|
| MTP | Store-not-apply (бит-точный авторегрессионный декод) |
| Размещение экспертов | Pinned-host store всех 43×288 экспертов + LRU device-слоты (механика GLM-5.2 / Qwen3.5-35B-A3B) |
| Конвертация | Послойные shards (dense-part + 43 MoE-layer прохода), без OOM RAM |
| Attention | Кастомные KDA-ядра: GDN из qwen4_exp + short conv1d; DSA/Indexer из GLM-5.2 |

## Подход: гибридный клон проверенных стеков

Композиция бит-точных компонентов без их модификации:
- KDA-слои: механика `qwen_gated_delta_kernel` (qwen4_exp) + hc GatedResidual + новый
  short conv1d компонент.
- DSA-слои: Indexer/IndexShare из `glm_moe_inference.cu` (GLM-5.2).
- MoE: роутер sigmoid/noaux_tc/top-8 (по образцу `qwen_moe_inference.cu`), FFN на
  NVFP4 GEMM-пути, shared expert + coefficient combine.
- Квантование: FP8 (e4m3, scale_inv 128×128) → dequant FP32 → requant NVFP4
  (E2M1 nibbles + E4M3 per-block scales + global scale). Нормы/biases/embed/lm_head/
  conv1d/hc/gates — FP16/BF16 без квантования (инвариант точности протокола).

## Компоненты (naming: family=glm, type=moe, variant=flash)

| Файл | Роль |
|---|---|
| `libs/lsxcommon/src/glm_moe_flash_constants.h` | frozen-геометрия + static_assert |
| `libs/lsxcommon/src/glm_moe_flash_convert.cpp` + `include/lsxcommon/glm_moe_flash_convert.h` | `ConvertGlm5NextModel` — классификатор, послойные shards, формат `logestix-glm5next` |
| `libs/lsxcommon/src/glm_moe_flash_load.cpp` + `include/lsxcommon/glm_moe_flash_load.h` | pinned dense upload, expert store 43×288, LRU device cache |
| `libs/lsxcommon/src/glm_moe_flash_inference.cu` + `include/lsxcommon/glm_moe_flash_inference.h` | KDA-ядро (conv1d + gated delta), DSA/Indexer, MoE FFN, hc GatedResidual, генерационный драйвер `RunGlmFlashGeneration` |
| `libs/lsxcommon/src/glm_moe_flash_device_math.h` | host-референсы KDA/hc/router для parity-тестов |
| `libs/lsxcommon/src/glm_moe_flash_model.cpp` + `.h` | `Glm5NextFlashModel` (ModelBase), kType=`glm-5.3-flash` |

Дополнительно (этапы 4, 7 протокола):
- Токенизатор: `BuildTokenizerArrowFiles` из `tokenizer.json` (BPE byte-level,
  bos/eos/pad из конфига: eos 154820/154827/154829, pad 154820).
- Passport: `model-metadata/passport/passport.bin` через `lsxnetwork::BuildPassport`,
  `LsxTokenTranslator` приватным членом IoWorker (загрузка после pin, до set_value).

## Метаданные

Манифест и индексы — через `lsxcommon::lsx_model_metadata(rel)` в
`/mnt/llm_models/models/model-metadata/` (Linux) / `D:\Programs\huggingface.co\model-metadata`
(Windows). Никаких хардкодов путей в конвертере.

## Генерационный драйвер

- Единый сквозной `cudaStream_t` (инвариант стрима протокола), без
  `cudaDeviceSynchronize` в цикле генерации.
- KDA: рекуррентное per-head delta state, conv1d ring-buffer со stride-арифметикой
  (инвариант шага протокола).
- DSA: KV-кэш + indexer top-2048 выбор блоков.
- RMW в shared memory: только thread 0 под `__syncthreads()` (инвариант песочницы).
- Expert residency: LRU device-слоты, residency обеспечивается до GEMM слоя.

## Тесты (TDD — сначала тесты, потом реализация)

| Файл | Содержание |
|---|---|
| `tests/test_glm_moe_flash_convert.cpp` | классификатор архитектуры, FP8→NVFP4 parity против host-dequant эталона, dense/MTP/vision-политика |
| `tests/test_glm_moe_flash_loader.cpp` | roundtrip, strictness отсутствующих тензоров, LRU-слоты |
| `tests/test_glm_moe_flash_kernels.cu` | conv1d, KDA gated-delta parity с host-референсом, hc GatedResidual, router sigmoid top-8 + scaling 2.5, indexer top-k |
| `tests/test_glm_moe_flash_selfcheck.cpp` | реальный селфчек "The capital of France is" → "Paris", 3 бит-точных цикла, `LSX_TEST_GLM_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4 |
| `tests/test_code_naming_guard.cpp` | расширение реестров (variant `flash`) |

Регресс: полные `ctest -R glm` (GLM-5.2 не задет), затем весь suite.

## Скорость (фаза 2, после зелёного селфчека)

Итеративный тюнинг на живом железе: occupancy-подбор KDA/DSA ядер (48 warps/SM max,
Blackwell), tile-варианты NVFP4 GEMM, LRU hit-rate метрики, замер t/s (`--benchmark`).
Ускорение только через новые перегрузки/ядра — GLM-5.2 не затрагивается. Финальная
метрика фиксируется в README-отчёте коммита.

## Риски

- FP8→NVFP4 requant меняет квант-шум относительно оригинала — парируется
  parity-тестами против host-эталона и селфчеком.
- Sinkhorn (hc, 20 итераций) на decode-пути — считаем в F32, при дороговизне
  выносим в precomputed per-layer масштабы (проверка parity).
- Indexer top-2048 при длинном контексте — бюджет фиксируем frozen-константой,
  парируем тестом degenerate-случая.
