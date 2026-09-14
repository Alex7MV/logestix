# logestix — AGENTS.md

## Project

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C23. CMake 3.25+ (CMakePresets).
Target architecture: NVIDIA Blackwell (CC 12.0 / 12.1). Active use of 5th-gen Tensor Cores and TMA.
Max warps per SM on Blackwell: 48 (factor into occupancy calculations).
Forbidden architectures: Hopper (90), Ada (89), Ampere (80).
No backward compatibility — any code targeting pre-Blackwell (sm_90, sm_80) is rejected.
CUDA Toolkit 13.4 (nvcc V13.4.59; минимум 13.2+), full native C++23 support in NVCC.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.1 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.17.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.6.11 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | 3.3 (latest) | submodule (`third_party/cccl`) | Header-only |
| abseil-cpp | 20260526.0 | FetchContent (`20260526.0`, LTS May 2026) | Header-only + static libs |
| CUDA cuBLAS | 13.2+ | system (CUDA Toolkit) | Dynamic (`cublas.dll`) |
| Catch2 | v3.16.0 | FetchContent (test only) | Header-only |

## Build

Use CMakePresets — ready-made for both supported OSes. Replace `debug` with
`release` in the preset name for an optimization build.

Toolchain notes (tested, higher than the minimums below also work):
- CMake 4.2.3 / 3.25+ (CMakePresets, `CMAKE_POLICY_VERSION_MINIMUM=3.5`)
- GCC 15.2 (Ubuntu 26.04) — C++20 baseline; GCC 15+ supports C++23, but the
  project pins to C++20 deliberately
- NVCC 13.4.59 (Linux, `/usr/local/cuda`; Windows — 13.3, см. ниже) — CUDA
  `LANGUAGES ... CUDA` enabled at top level
  (`find_package(CUDAToolkit)`), not `enable_language()` in a subdirectory
  (CMake 4.x breaks when CUDA is enabled in a nested dir)
- Arrow `BUILD_WARNING_LEVEL=PRODUCTION` — GCC 15 treats previously-ignored
  warnings as errors (`-Werror=unused-parameter` in Arrow 25.0.0 under the
  default CHECKIN debug level); PRODUCTION never adds `-Werror`

### Linux (Ubuntu 26.04, Ninja + GCC)

```bash
git submodule update --init --recursive
cmake --preset linux-debug
cmake --build --preset linux-debug
```

### Linux release (tuned for AMD EPYC 9V74, Zen 4)

```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4   # Ninja -j160
ctest --preset linux-release-znver4
```

Enables `LSX_ARCH_ZNVER4=ON` + `LSX_LTO=ON`: `lsx_tune` INTERFACE library
applies `-march=znver4` (-mtune=znver4, -O3, -fno-math-errno,
-falign-loops=16, -funroll-loops) and LTO (`-flto` GCC / `-flto=thin`
Clang-AOCC) to C/C++ targets only. CUDA (`lsxcommon_cu`), dependencies and
the Windows/MSVC build are never affected.

### Linux release — AOCC (AMD Optimizing C/C++ 5.2.0)

Альтернативный хост-компилятор (clang 17.0.6) для znver4-сборки; NVCC
использует его как `-ccbin`. Проверено 2026-08 (Linux, RTX 5090, EPYC 9V74):
конфигурация, полная сборка и 110/110 тестов зелёные.

```bash
AOCC=/home/alexmv2025/projects/amd/aocc-compiler-5.2.0/bin
cmake -S . -B build/linux-release-znver4-aocc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLSX_ARCH_ZNVER4=ON -DLSX_LTO=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_COMPILER=$AOCC/clang -DCMAKE_CXX_COMPILER=$AOCC/clang++ \
  -DCMAKE_CUDA_HOST_COMPILER=$AOCC/clang++
cmake --build build/linux-release-znver4-aocc -j 160
ctest --test-dir build/linux-release-znver4-aocc
```

Особенности AOCC (проверено на 5.2.0):
- `-flto=thin` НЕ поддерживается — clang падает на `-flto=full` (warning
  "Unsupported lto mode -flto=thin, falling back to -flto=full"), сборка
  при этом корректна.
- `__has_attribute(enable_if)` у AOCC = 1, поэтому abseil compile-time
  format checker активируется (у GCC = 0, checker инертен). Его
  `ValidFormatImpl` понимает только printf-`%`-конверсии и отвергает
  fmt-style `{}`, используемый проектом повсеместно. Лечится идемпотентным
  патчем abseil в корневом `CMakeLists.txt` (guard `#if 0 &&`), приводящим
  checker к GCC-поведению. Не убирать при сборке под AOCC/Clang.
- Host-only тест `test_kimi_moe_expert_cache` требует, чтобы CUDA-поверхность
  `kimi_moe_expert_cache.h` была под `#if defined(LOGESTIX_HAS_CUDA) ||
  defined(__CUDACC__)` — иначе host-TU без CUDA include-путей не соберётся.
- В `.cu`-тестах не использовать Windows-only `_putenv` без guard'а — только
  `_putenv_s`/`setenv` под `#ifdef _WIN32` (эталон: `test_qwen_nvfp4_gemv.cu`).

### Windows (Visual Studio 2026)

```powershell
git submodule update --init --recursive
cmake --preset windows-debug
cmake --build --preset windows-debug
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD`.

Arrow's xsimd sub-dependency requires `CMAKE_POLICY_VERSION_MINIMUM=3.5`; the
variable is now applied automatically through `CMakePresets.json` — no manual
`$env:CMAKE_POLICY_VERSION_MINIMUM` needed.

### Windows — расположение тулчейна (VS 2026, MSVC 14.51, CUDA 13.3)

`cmake`/`ctest`/`ninja` НЕ в PATH — их поставляет Visual Studio 2026
(`18\Community`), CUDA Toolkit 13.3 лежит отдельно. Кэш сборки — `build\`
уже сконфигурирован (пресет `windows-debug`, `WIN32` generator). Точные пути
(актуально для этой машины; проверил в 2026-08):

- CMake 4.3: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"`
- CTest: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"`
- Ninja: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"`
- MSVC `cl.exe` (Hostx64): `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"`
- NVCC 13.3: `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"`
  (в PATH уже есть; CMake находит через `find_package(CUDAToolkit)`)

PowerShell-вызовы (без добавления в PATH):

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --build build --config Debug --target <target>
$cmake --preset windows-debug && $cmake --build --preset windows-debug
```

Если VS обновится (новый `MSVC\<ver>`), сверь номер каталога `14.51.36231`
через `Get-ChildItem "$vs\VC\Tools\MSVC"`. Альтернативный CMake 4.2.3 в
системе НЕ установлен — используется только встроенный в VS 4.3.

## Resources (host)

Политика расхода ресурсов хост-машины. Показатели относятся к **Windows**
dev box (машина со свежим тулчейном даёт ещё меньшую загрузку):

- Средняя утилизация CPU при сборках и работе инструментов ≈ **10%**.
  Процессорное время экономить НЕ нужно: параллельность не ограничивать,
  отдельные треды/ядра не сдерживать.
- RAM занята примерно на **35%**. Память тоже экономить НЕ нужно — никаких
  ухищрений по снижению потребления в обычном инструментальном коде.
- Под **Linux** (Ubuntu 26.04 dev box) ничего экономить не нужно вообще:
  загрузка там ещё ниже, чем под Windows, поэтому ограничений дополнительных
  не вводить — обычные правила (подробности выше) не трактовать жёстче.

ИСКЛЮЧЕНИЕ (оба OS-профиля): когда запускается **инференс** или **тесты
инференса** (`--bench`/`--selfcheck` lsxbenchmark, `--testprompt`, infer-тесты,
CUDA-ядра на реальных весах), потребление памяти и GPU-ресурсов считается
критичным: не запускать параллельные тяжёлые нагрузки, следить за VRAM/RAM
и освобождать модели по завершении. Работающие инференсы не дублировать.

**lsxbenchmark — строгий GPU-эксклюзив**: `lsxbenchmark` занимает GPU целиком
и его ЗАПРЕЩЕНО запускать параллельно с другими GPU-нагрузками (инференс,
CUDA-тесты, другой lsxbenchmark, perf-сьюты `ctest -L perf`). Конкуренция за
VRAM и SM искажает замер t/s (измерено: ~2x разброс при разделении VRAM
несколькими моделями) — замер становится непригодным для perf-справочника.
Одна GPU-нагрузка за раз; perf-тест `test_perf_baseline_*` при обращении к
справочнику `tests/perf/perf_baselines.json` обязан гоняться после
завершения всех прочих GPU-задач.

## Test

`BUILD_TESTING=ON` is included in every preset. Тесты разбиты на 4 категории
(CTest labels; дисциплину проверяет guard `test_ctest_labels_guard`):

| Label | Что это | Когда запускается |
|---|---|---|
| `fast` (37+4 app) | host-only тесты | каждый `ctest --preset` (дефолт всех пресетов) |
| `convert` (10) | конвертеры моделей | при правке конвертера (роутер) или отладке модели |
| `inference` (77) | CUDA/модельный контур | когда изменился контур (роутер) или вручную |
| `perf` (6) | perf-бенчмарки на реальных весах | только отдельной командой |

```bash
ctest --preset linux-debug                  # fast only (Ubuntu 26.04)
ctest --preset linux-debug-convert          # convert sweep
ctest --preset linux-debug-inference        # inference sweep
ctest --preset linux-debug-perf             # perf, отдельная команда
ctest -L "^(fast|convert|inference|perf)$"  # полный прогон

# Автодетект по git diff (fast всегда; convert/inference — при изменении
# соответствующего контура):
cmake -DLSX_PRESET=linux-debug -P tools/test_router.cmake
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake  # dry-run
```

Дисциплина: новый тест обязан получить label в том же коммите (иначе красный
`test_ctest_labels_guard`). Правила классификации путей — таблица
`lsx_router_rule(...)` в `tools/test_router.cmake`.
Процедура подъёма/замера скорости (nsys-профиль, budget-тесты, нобы, перф-базлайн)
— скил `perf-tuning-a-model`.

### Converter test rules (фиксированные правила тестов конвертеров)

Эталон: `tests/test_glm_moe_flash_convert.cpp` (synthetic e2e) +
`tests/test_qwen_dense_fixture.h` (`FakeTensor`/`WriteSafetensors`).

1. **Label `convert` в том же коммите**; TIMEOUT обязателен и ограничен
   (≤600 s; канонично 120 s) — guard `CheckConvertTimeout`. TIMEOUT —
   механическая защита от регрессии «тест конвертирует модель».
2. **Запрещены конвертации реальных моделей в тестах.** E2E-пайплайн
   конвертера гоняется на синтетическом чекпоинте: `config.json` +
   `*.safetensors` через `qwentest::WriteSafetensors` во временном каталоге.
3. **Синтетическая геометрия**: frozen-гейты конвертера снимаются
   `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` — это runtime env, задаётся через
   `set_tests_properties(... PROPERTIES ENVIRONMENT ...)`, а НЕ
   compile-define (compile-define env не выставляет).
4. **Фикстура обязана покрывать все пути пайплайна**: globals
   (embed/lm_head/final norm — без них конвертер ошибается), BF16→NVFP4,
   FP8→NVFP4 с scale-тайлами, raw-тензоры, routed-эксперты (все 3
   проекции), пустые слои frozen-диапазона, layout `model.index.json`
   (формат-тег + kda/dsa/moe списки).
5. **FP8-тензоры фикстуры**: стороны кратны 128 (GPU-kernel), scale-тайл
   `[ceil(R/128), ceil(C/128)]` F32; dtype-строки safetensors: `BF16`,
   `F8_E4M3`, `F32`.
6. **Реальный чекпоинт допустим только для чтения одного лёгкого файла**
   (например `config.json` в `ParseFlashConfig`) с `SKIP()` без модели;
   полное чтение/конвертация весов — никогда. Синтетикой не заменяют тесты,
   валидирующие реальные имена полей формата.

**WINDOWS-ОГРАНИЧЕНИЕ**: под Windows запускать тесты ТОЛЬКО для моделей
Qwen3.5-9B и gemma-4-26B-A4B-it (фильтр `ctest -R "qwen|gemma"`); остальные
model-тесты (GLM, DeepSeek и т.д.) на этой OS не запускать.

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `libs/lsxcommon/src/core.cpp` — ядро (логирование, инициализация)
  - `libs/lsxcommon/include/lsxcommon/core.h` — публичный заголовок
  - `libs/lsxcommon/src/common.cpp` — version + GPU info (spdlog)
  - `libs/lsxcommon/src/schema.cpp` — ExpertSchema (9 полей: weights, E4M3 per-block scales uint8[], float32 global_scale), PackedExpert
  - `libs/lsxcommon/src/converter.cpp` — GPU-only NVFP4 квантизация (per-block E4M3 + global_scale), парсинг safetensors, сериализация в Arrow IPC
  - `libs/lsxcommon/src/loader.cpp` — загрузка экспертов из .arrow, CUDA pinned регистрация, E4M3 scale decoding
  - `libs/lsxcommon/src/glm_moe_convert.cpp` + `include/lsxcommon/glm_moe_convert.h` — GLM-5.2 классификатор, Topology, `ConvertGlmModel`, `SaveGlmPinnedTensors`, `WriteGlmModelIndex` (extended `logestix-glm` формат)
  - `libs/lsxcommon/src/glm_moe_load.cpp` + `include/lsxcommon/glm_moe_load.h` + `src/glm_moe_load_internal.h` — host-only GLM loader: `LoadGlmPinnedWeights` (быстрый путь: GPU in-place BF16→FP16 через `GlmBf16ToHalfInPlace`, pinned staging ring, per-layer device-арены, строгие ошибки на отсутствующие файлы слоёв/битые тензоры), `GlmLayerWeights`/`GlmPinnedWeights`
  - `libs/lsxcommon/src/glm_moe_model.cpp` + `src/glm_moe_model.h` — `GlmV5Model` (IModel, registry `glm-5.2`); Infer: полный 79-слойный цикл (dense 0-2, IndexShare attention + MoE router + NVFP4 expert stacks 3-77, MTP epilogue layer 78); SelfCheck: инкрементальный 1→2-токенный селфчек (форс `max(2, max_gen_tokens)`, декод 1/2-токенного префикса против "Paris"), публикует `gen_tokens` + `final_logits` в `InferenceResult`
  - `libs/lsxcommon/src/glm_moe_inference.cu` — GLM-5.2 CUDA-ядра (router, IndexShare attention, NVFP4 expert decode/SwiGLU)
  - `libs/lsxcommon/src/lsx_inference_nvfp4.cu` + `include/lsxcommon/lsx_inference_nvfp4.h` — МОДЕЛЬНО-НЕЙТРАЛЬНЫЙ NVFP4-слой (namespace `lsx_nvfp4`, объект `lsxcommon_cu`): `LsxNvfp4Weight`, `RunLsxNvfp4Gemm` (диспетчер: seq==1 weight-streaming GEMV / TC mxf4nvf4 / scalar tile), `RunLsxNvfp4GemmSeq1GemmV`, `RunLsxNvfp4GemmGateUpPair` (фьюжен gate/up pair-GEMV для смежных [w1|w3] паков, пошаговый global scale — бит-идентичен двум одиночным GEMV), routing-тумблеры `SetLsxTcFastMath`/`SetLsxSeq1Gemv` + RAII `LsxFastMathScope`. Qwen-имена (`RunQwenNvfp4Gemm`, `QwenNvfp4Weight`, …) — inline-шимы над этим слоем. ИНВАРИАНТ НЕЗАВИСИМОСТИ МОДЕЛЕЙ: каждая модель открывает `LsxFastMathScope(cfg_.fast_math, ...)` внутри своего `Infer` (политика per-call, не «последний созданный конструктор»); контроль — `tests/test_model_independence_guard.cpp` (список consumers расширять в коммите новой модели)
  - `libs/lsxcommon/src/glm_moe_flash_convert.cpp` + `include/lsxcommon/glm_moe_flash_convert.h` — GLM-5.3-Flash конвертер `ConvertGlm5NextModel` (формат `logestix-glm5next`): KDA q/k/v/o → NVFP4 из BF16-источника, эксперты/DSA/MLP из fp8+scale-tiles, `BuildFlashPassportTokens` (шаг 7: мост GLM vocab + DeepSeek-V4-Flash vault vocab, byte-0 unk-якоря); frozen-геометрия в `src/glm_moe_flash_constants.h`
  - `libs/lsxcommon/src/glm_moe_flash_load.cpp` + `include/lsxcommon/glm_moe_flash_load.h` — GLM-5.3-Flash loader: dense NVFP4/FP16 upload, expert store 45×288 pinned slabs, LRU-кэш из свободной VRAM (`LSX_FLASH_SLOTS` override, резерв 3 GiB), KDA q/k/v/o NVFP4
  - `libs/lsxcommon/src/glm_moe_flash_model.cpp` + `.h` — `Glm5NextFlashModel` (ModelBase, registry `glm-5.3-flash`); Infer: 45-слойный токен-за-токеном цикл (hc FP32-пайплайн, KDA/DSA attention, MoE router top-8, dual-stream H2D expert staging с event-пломбингом), SelfCheck через RunParisSelfCheck; decode 32.1 t/s на RTX 5090 (pair-GEMV w13, 2026-09)
  - `libs/lsxcommon/src/glm_moe_flash_inference.cu` + `glm_moe_flash_device_math.h` — GLM-5.3-Flash CUDA-ядра (router top-8 warp-per-expert uint4, KDA decode NVFP4 GEMV, DSA MLA, hc_pre/post reuse из deepseek_moe_load.cuh с defaulted rms_eps)
  - `libs/lsxcommon/src/glm_moe_flash_constants.h` — frozen-геометрия GLM-5.3-Flash (hidden=4096, 45 слоёв, 36 KDA + 9 DSA, E=288, top-8, moe_i=2048, V=154880)
  - `libs/lsxcommon/src/glm_moe_device_math.h` — device-математика, бит-точная репликация glibc-функций для GLM-5.2
  - `libs/lsxcommon/src/tokenizer_convert.cpp` — tokenizer arrows; merges в форматах string `"a b"` И array `["a","b"]` (GLM-5.2), byte_to_token_id, bos/eos/pad
  - `libs/lsxcommon/src/tokenizer.cpp` + `pretokenizer.cpp` + `unicode.cpp` — BPE-токенизатор, word-level претокенизация, Unicode-классификация
  - `libs/lsxcommon/src/deepseek_moe_model.cpp` — `DeepSeekV4FlashModel` (IModel, registry `deepseek-v4-flash`); NVFP4-контур: формат `logestix-v1-nvfp4` (pow2-E4M3 per-16 шкалы экспертов + per-tensor global_scale, строгий гейт формата в Load), routed FFN на `lsx_nvfp4` pair-GEMV (кэш-слот [w1|w3|w2]/[s1|s3|s2]), expert LRU-кэш на всю свободную VRAM, dual-stream staging, pinned expert store (11008 экспертов / 148.6 GiB locked host, pair-layout), decode **21.4 tok/s** на RTX 5090 (baseline сессии 4.43); SelfCheck Paris
  - `libs/lsxcommon/src/deepseek_moe_repack.cpp` + `include/lsxcommon/deepseek_moe_repack.h` — оффлайн-репак экспертов E8M0/32 → NVFP4 E4M3/16 (`lsxforge --deepseek_nvfp4`); конвертация из safetensors — дефолтный `Converter::Run` (PackExpertNVFP4 I8/U8-ветка репачит шкалы, fold global_scale = 2^m, `nvfp4_scale.h`)
  - `libs/lsxcommon/src/deepseek_moe_v41_*` — DeepSeek-V4.1-Flash (registry `deepseek-v4.1-flash`, формат `logestix-v41-nvfp4`), АВТОНОМНЫЙ контур (никаких V4-Flash-заголовков): `_constants.h` (frozen-геометрия hidden=5120, 40 слоёв, E=384/top-6, CED compress_ratios `[0,0,2×18,1×20]`, kv-source {2,8,14,20}), `_convert.{h,cpp}` (`ConvertDeepSeekV41Model`, `lsxforge --deepseek_v41`, FP8 32×32 dense → FP16 при загрузке, эксперты I8 E8M0/32 → NVFP4 E4M3/16, tokenizer-мост), `_load.{h,cpp}`/`_load.cu` (топология/формат-гейт, pinned dense, expert store), `_dequant.{h,cu}` (BF16→FP16, FP8 dequant, FP4/FP8 act-quant), `_inference.{h,cu}` (RMSNorm/RoPE/HC-Sinkhorn/sparse-attn/compressor/indexer/router/SwiGLU/GEMM/head), `_inference_attn.cu` (SWA + CED decode), `_inference_pipeline.cu` (`RunDeepSeekV41Inference`, токен-за-токеном цикл hc FP32), `_model.{h,cpp}` (`DeepSeekV41FlashModel`), `_inference_csa.h`/`_device_math.h`; OBJECT-либа `lsxcommon_deepseek_v41_cu` (`-fmad=false`). Синтетический e2e `test_deepseek_moe_v41_pipeline` (SWA + CED ratio-1); реальный vault сконвертирован (`lsxforge --deepseek_v41`), Paris-селфчек 3/3 бит-точных цикла, perf-baseline `test_perf_baseline_deepseek_v41_flash` (2048 токенов, label perf), decode **2.6 tok/s** на RTX 5090 (AOCC znver4 release, `--max-gen 2048`, 15.3 GiB; короткий `--max-gen 16` даёт 1.1 t/s — не прогревает steady-state; без device expert-cache — correctness-first); engram/DSpark/vision — см. `docs/superpowers/plans/2026-09-13-deepseek-v41-flash-ingestion.md`
  - `libs/lsxcommon/src/gpu_perf.cpp` — perf-политика: TC-глобалы lsx_nvfp4/lsx_tc + expert-slots_pct; DeepSeek expert-кэш дефолт = вся свободная VRAM минус резерв
  - `libs/lsxcommon/src/model_factory.cpp` + `model_registry.cpp` + `include/lsxcommon/model_factory.h` — `LsxModelFactory` + IModel registry
  - `libs/lsxcommon/src/models/lsx_orchestrator.cpp` + `include/lsxcommon/lsx_orchestrator.h` — TaskMosaic оркестратор: lock-free SPSC dispatch GLM-5.2 / DeepSeek / unknown
  - `libs/lsxcommon/src/deepseek_inference_*.cu` + `deepseek_moe_load.cuh` + `deepseek_moe_norm_kernels.cu` — DeepSeek-V4-Flash inference kernels: load, norm, moe, head, attn, csa, pipeline
  - `libs/lsxcommon/src/qwen_moe_convert.cpp` + `include/lsxcommon/qwen_moe_convert.h` — Qwen3.5-35B-A3B MoE конвертер `ConvertQwen35MoeModel` (формат `logestix-qwen35moe`): fused-expert split gate_up [E,2I,H] → w1/w3 (HF parity: gate=rows[0,I)), per-expert `experts/layer_N/exp_E.arrow`, router BF16, shared expert NVFP4 + FP16 scalar gate, MTP stored-not-applied, vision отсекается с подсчётом; frozen-геометрия в `src/qwen_moe_constants.h`. Тело вынесено в общее ядро `ConvertQwenMoeModelCommon(QwenMoeConvertSpec)` — формат-агностичный конвейер, разделяемый 3.5/3.6
  - `libs/lsxcommon/src/qwen_moe_36_convert.cpp` + `include/lsxcommon/qwen_moe_36_convert.h` — Qwen3.6-35B-A3B конвертер `ConvertQwen36MoeModel` (формат `logestix-qwen36moe`): общее ядро с 3.5, свой frozen-гейт + bos_token_id мост в индекс (config 3.6, `src/qwen_moe_36_constants.h`); MTP fused-эксперты идут generic Mtp-веткой
  - `libs/lsxcommon/src/qwen_moe_36_constants.h` — frozen-геометрия Qwen3.6-35B-A3B (значения совпадают с 3.5; независимый паспорт `qwen_moe_36::frozen`, kBosTokenId=248044)
  - `libs/lsxcommon/src/qwen_moe_exp_convert.cpp` + `include/lsxcommon/qwen_moe_exp_convert.h` — Qwen3.8-Flash-Next (qwen4_exp) конвертер `ConvertQwen38FlashNextModel` (формат `logestix-qwen38flashnext`): 48 слоёв GDN/QSA, hyper-connections (GatedResidual hc_count=4), PLE n-gram shards (mmap-only), splitmix64-мультипликаторы бит-точно против чекпоинта, prime-таблица 16 голов; frozen-геометрия в `src/qwen_moe_exp_constants.h`
  - `libs/lsxcommon/src/qwen_moe_exp_load.cpp` + `include/lsxcommon/qwen_moe_exp_load.h` — qwen4_exp loader: pinned dense (hc norms BF16, projections NVFP4), expert store 48×512, PLE mmap shards, LRU-кэш слотов
  - `libs/lsxcommon/src/qwen_moe_exp_inference.cu` + `include/lsxcommon/qwen_moe_exp_inference.h` — qwen4_exp ядра: hc GatedResidual (F32-ядро + NVFP4 wrapper), hc-scatter, QSA indexer, PLE inject, router top-10, генерационный драйвер `RunExpGeneration`
  - `libs/lsxcommon/src/qwen_moe_exp_model.cpp` + `.h` — `Qwen38FlashNextModel` (ModelBase, registry `qwen3.8-flash-next`); 48-слойный Infer, SelfCheck через RunParisSelfCheck
  - `libs/lsxcommon/src/qwen_moe_exp_device_math.h` — host-референс GatedResidual/PLE gate для parity-тестов
  - `libs/lsxcommon/src/kimi_moe_convert.cpp` + `include/lsxcommon/kimi_moe_convert.h` — Kimi-K2.7-Code конвертер (`lsxforge --kimi`, формат `logestix-kimi-r2`): INT4 group-32 эксперты → GPU-dequant BF16 → `PackExpertNVFP4` (routed + shared), MLA-проекции q_a/q_b/kv_a/kv_b/o → NVFP4 через `quantize_tensor_gpu`, norms/router/dense-MLP BF16, `mla_nvfp4` global scales в model.index.json
  - `libs/lsxcommon/src/kimi_moe_load.cpp` + `include/lsxcommon/kimi_moe_load.h` — Kimi loader: frozen-паспорт (hidden=7168, 61 слой, E=384, top-8, MLA 64 голов, V=163840), pinned dense upload, MLA NVFP4 пары, **pinned routed-expert store**: 23040 экспертов / 542 GiB в 34×16 GiB чанках через общий `ExpertStore::PinChunked<kSplitFence>` (mmap-чанки → параллельный fault-at-write copy из горячего page cache → `cudaHostRegister` последним; driver fails single >300 GiB cudaHostAlloc, его zerofill 332 с сериализован и выдавливал page cache), layout блока [w1|w3][s1|s3][w2][s2] под split-fence upload, `LSX_EXPERT_PIN=0` отключает
  - `libs/lsxcommon/src/kimi_moe_inference.cu` + `kimi_moe_kernels.h` — Kimi ядра: MLA absorbed attention (kv_b dequant кэшируется per-layer — static-memo слот трэшил 60×/токен), noaux_tc router split-phase, routed FFN legacy dequant-GEMV chain (дефолт, бит-точный) / batched TC chain `RunKimiRoutedTcBatch` (opt-in `LSX_KIMI_TC=1`; ROOT CAUSE зафиксирован 2026-09-12, `test_kimi_moe_tc_diag`: swiglu-выход Kimi несовместим с per-16 E2M1 pow2-квантованием — QUANT hq rel=120%, down-выход повреждён на ~70%, top-1 мусор; TC-ядра при этом корректны (stage-чеки <0.2%); фикс требует формата активаций повышенной точности для down-GEMM), split-fence upload (ev13/ev2: w13 GEMM перекрывает down-upload), expert LRU-кэш из свободной VRAM (резерв 2 GiB → 748 слотов ПОСЛЕ shared-арены), decode **5.1 tok/s** на RTX 5090 @2048 бит-точно (baseline round-2: 1.3)
  - Kimi decode построен на двух замерах nsys (2026-09-14): H2D 203 ms/токен (маршрутизированные промахи 277/токен × 24.78 MiB по PCIe на **57 GB/s** — потолок шины, + shared 84 ms НА COMPUTE-стриме) и 141 ms/токен ядер (71% — `Nvfp4DequantKernel`, материализующий bf16). Отсюда два рычага: (1) **перекрытие copy/compute** — per-entry readiness `ready_slot[8]` (`KimiMoeCacheBatch::wait_miss_pos` говорит, какой именно upload гейтит запись) + per-slot last-read `slot_read[slots]` вместо batch-wide fence, который заставлял copy-стрим ждать ВСЁ compute; (2) **shared-expert device arena** (61 слоёв × `KimiSharedSlabLayout` = 1441 MiB, загружается один раз, `KimiSharedSlotView` использует ту же layout-функцию, что и upload) — убирает per-layer pageable H2D 84 ms. Остаток: `LSX_KIMI_TC=1` (W4A4) не включается из-за численного root cause; следующий рычаг — устранение dequant через модельно-нейтральный NVFP4 GEMV (71% времени ядер). **LFU-политика кэша (frequency-eviction + ageing) проверена и ОТКЛОНЕНА (2026-09-14)**: на 128 токенах даёт +15% (4.7→5.4 t/s), но на длинном прогоне регрессирует (3.7 t/s и early-EOS на 1386 токенов вместо 2048) — LFU вытесняет недавно прочитанные «редкие» слоты, из-за чего copy-стрим постоянно ждёт `slot_read` последнего compute; плюс расхождение выходного потока. Возврат к LRU воспроизвёл 2048 токенов @5.1 t/s бит-в-бит (398.4 s). Повторную попытку делать только с fence-совместимым правилом (учёт «времени последнего чтения» при выборе жертвы) и обязательной валидацией на 2048 токенах. Для офлайн-анализа политик — env-хук `KIMI_TRACE_FILE=<path>` в `KimiLayerStack` (пишет {turn,layer,top-8} на каждый decode-слой; на реальном 125-токенном trace LRU = 283 промаха/токен, offline-оптимум Belady = 158 при 748 слотах)
  - `libs/lsxcommon/src/kimi_moe_model.cpp` + `.h` — `KimiK27CodeModel` (ModelBase, registry `kimi-k2.7-code`); Infer резолвит fast-math из tri-state `EffectiveFastMathMode` (legacy `cfg_.fast_math` bool не включал TC — root cause 3.0→5.5), SelfCheck через RunParisSelfCheck
  - `libs/lsxcommon/include/lsxcommon/expert_store.h` — `PinChunked<kStream|kSplitFence>`: общий chunked pin для ExpertStore-моделей (plan → parallel mmap-чанки 16 GiB → fault-at-write copy → cudaHostRegister последним); kimi [w1|w3][s1|s3][w2][s2] (split-fence), glm/gemma [w1|w3|w2][s1|s3|s2] (stream); GLM/Gemma pinned по умолчанию (Gemma 3840 экспертов ≈ 12.5 GiB за 2.7 c), DeepSeek/Qwen — собственные пины
  - `libs/lsxcommon/src/qwen_moe_load.cpp` + `include/lsxcommon/qwen_moe_load.h` — MoE loader: dense pinned upload (reuse qwen ring machinery), router/shared device upload, pinned expert store (`LoadQwenMoeExpertStore`) + LRU device cache alloc/residency (`AllocateQwenMoeExpertCache`, `EnsureQwenMoeExpertResident`)
  - `libs/lsxcommon/include/lsxcommon/qwen_moe_expert_cache.h` — LRU device-slot кэш экспертов (адаптация GlmExpertDeviceCache), host-bookkeeping unit-testable без CUDA
  - `libs/lsxcommon/src/qwen_3_5_moe_model.{h,cpp}` — `Qwen35MoeModel` (ModelBase, registry `qwen3.5-35b-a3b`); static_assert геометрия (hidden=2048, layers=40, E=256, top-k=8, moe_i=512); incremental session policy как QwenBaseModel; SelfCheck через runtime::RunParisSelfCheck
  - `libs/lsxcommon/src/qwen_moe_model.{h,cpp}` — shared runtime-graph `QwenMoeModelBase` (Load/Infer-ядро/SelfCheck/Benchmark, expert-cache sizing, governor) для 3.5+3.6; листовые модели держат только kType, свой frozen-гейт и собственный `Infer` c `LsxFastMathScope`
  - `libs/lsxcommon/src/qwen_moe_36_model.{h,cpp}` — `Qwen36MoeModel` (ModelBase, registry `qwen3.6-35b-a3b`); геометрия идентична 3.5, общий `QwenMoeModelBase`; SelfCheck через RunParisSelfCheck (Paris, 3 бит-точных цикла)
  - `libs/lsxcommon/src/qwen_moe_inference.cu` (OBJECT `lsxcommon_qwen_moe_cu`, `-fmad=false`) — router softmax→top-8→renorm ядро (HF parity, thread-0 selection), routed FFN slot chain на RunQwenNvfp4Gemm + swiglu/coeff-add, shared-gate combine, инкрементальный single-stream драйвер `RunQwenMoeGeneration`
  - `libs/lsxcommon/src/shared_experts.cu` — NVFP4 decode-таблицы (E2M1/E8M0), routed-expert FFN
  - `libs/lsxcommon/src/safetensors_gpu.cu` — GPU-ускоренная квантизация (3 CUDA-ядра: convert, block_maxabs, quantize_pack_perblock; CCCL thrust для global max)
  - `libs/lsxcommon/src/model_runtime.cpp` — `runtime::RunCliBenchmark` (lsxbenchmark --bench): реальная генерация через IModel::Infer на args.fast_math_mode, отчёт tokens/t/s/ttft/resident
- `libs/lsxnetwork/` — shared STATIC library; namespace `lsxnetwork`; zero-alloc async сетевой слой TaskMosaic CE (FrameQueue SPSC, ZMQ context, I/O worker, OpenAI endpoint, affinity)
- `apps/lsxfabric/` — B2B inference engine, всегда максимальная скорость (`--localmodel`, `--validate`, `--testprompt`; CLI parse в `src/cli.{h,cpp}`); замер скорости и бит-точность переехали в lsxbenchmark
  - `apps/lsxfabric/src/validator.cpp` — валидатор NVFP4/E4M3/global-scale в собранной модели
  - `apps/lsxfabric/tests/` — 12 test cases (1 lifecycle + 11 validator)
- `apps/lsxforge/` — data preprocessing utility; `--glm` for the GLM-5.2 pipeline (`lsxcommon::glm::ConvertGlmModel`), `--tokenizer`, repack-флаги
  - `apps/lsxforge/tests/` — 1 test case, 1 assertion
- `apps/taskmosaic/` — task orchestrator (CLI-обёртка над `lsxcommon::LsxOrchestrator`)
  - `apps/taskmosaic/tests/` — 1 test case, 1 assertion
- `apps/lsxcli/` — CLI client (`--model`, `--info`, `--infofull`)
  - `apps/lsxcli/tests/` — 1 test case, 1 assertion
- `apps/lsxbenchmark/` — замер скорости генерации + бит-точность (`--bench` — дефолтный режим, реальная генерация `IModel::Infer`; `--selfcheck` — Paris-проверка; `--bit-exact` — `FastMathMode::kOff` для активного режима; `--model`, `--prompt`, `--max-gen`; CLI parse в `src/cli.{h,cpp}`); единственное приложение с численным режимом — остальные всегда на максимальной скорости
  - `apps/lsxbenchmark/tests/` — 5 test cases (CLI parser, label fast)
- `apps/lsxhome/` — Windows-native x64 desktop shell (`lsxhome`, WIN32 subsystem, D3D12 + Dear ImGui docking branch). Только Windows — блок в CMakeLists обёрнут в `if(NOT WIN32) return()`. Compute thread публикует токены в lock-free SPSC `GuiBridge` → UI thread дренирует их каждый кадр.
  - `src/main_win32.cpp` — WinMain, окно, render loop, SPSC compute→UI bridge, sidebar + ImPlot график
  - `src/d3d12_renderer.cpp` — `D3D12Renderer` (device/queue/swapchain, SRV-дескрипторный пул, ImGui Win32+DX12 backend, fence), `CreateContext()` для ImGui+ImPlot+ImNodes
  - `src/gui_renderer.cpp` + `include/lsxhome/gui_renderer.h` — Phase 1 theme-слой: `ApplyBlackwellCoworkTheme()` (матовый Blackwell/Cowork, WindowBg `0.08`, ChildBg `0.10`, Text `0.88`, все rounding ровно 6.0f) и `BuildWorkspaceSkeleton()` (`ImGui::DockSpaceOverViewport`)
  - `src/font_loader.cpp` + `include/lsxhome/font_loader.h` — imgui_freetype LCD-растеризатор, встроенные TTF (Inter + JetBrains Mono)
  - `include/lsxhome/blackwell_theme.h` — палитра `BlackwellPalette` + `Geometry` (kPalette/kGeometry)
  - `include/lsxhome/gui_bridge.h` — `TokenPayload` (одна cache-line, 64B) + `GuiBridge` (SPSC publish/drop/poll/drain, zero-alloc)
  - `include/lsxhome/spsc_token_ring.h` — lock-free SPSC ring
  - `include/lsxhome/imnodes_offsetof_shim.h` — shim для imnodes v0.5 против obsolete IM_OFFSETOF
  - `tests/` — `test_lsxhome_gui` (5 test cases: payload alignment, SPSC zero-alloc/backpressure/burst) + `test_lsxhome_font` (3 test cases)
  - GUI-стек (Dear ImGui 1.92.9-docking, ImPlot 1.0, ImNodes 0.5, FreeType 2.14.3) — FetchContent, статические архивы (`lsxhome_imgui`/`lsxhome_implot`/`lsxhome_imnodes`/`lsxhome_font`/`lsxhome_mono`)
  - **GUI-инварианты**: `IMGUI_DISABLE_OBSOLETE_FUNCTIONS` НЕ используется (меняет layout ImGuiIO/ImGuiStyle → десинхрон `sizeof` с app TU → assert "Mismatched struct layout!" в `IMGUI_CHECKVERSION`); ImPlot/ImNodes требуют своих `CreateContext()`/`DestroyContext()` (иначе assert `GImPlot == nullptr`)
- `tests/` — Catch2 test suite; GLM и DeepSeek тесты автономны и запускаются по отдельности (`ctest -R glm` / `ctest -R deepseek`)
  - `tests/test_core.cpp` — интеграционные тесты зависимостей (Arrow, CCCL, FlatBuffers, ZMQ, simdjson); 11 test cases, 101 assertions
  - `tests/test_deepseek_moe_loader.cpp` — 3 test cases, 64 assertions (round-trip, empty fields, E4M3 encoding)
  - `tests/test_deepseek_moe_model.cpp` — DeepSeek-V4-Flash model-стадия (host-only): `detect_model_type`/`LsxModelFactory::DetectType` + `load_topology` (слои/эксперты/VRAM-профиль + error-пути) на синтетической logestix-фикстуре; convert-стадии у семейства by design нет (конвертера в репо нет)
  - `tests/test_deepseek_moe_inference_kernels.cu` — DeepSeek CUDA inference kernel unit tests; 12 test cases, 4470 assertions (E2M1/E8M0, act_quant, RMSNorm, SwiGLU, CSA Compressor, routed-expert FFN, NVFP4 codec)
  - `tests/test_deepseek_moe_v41_{model,loader,convert,kernels,dequant,load_fast,actquant,pipeline,selfcheck}.*` + `test_deepseek_moe_v41_fixture.h` — DeepSeek-V4.1-Flash suite: frozen-паспорт + CED-классификация, топология/формат-гейт, синтетический конвертер (`logestix-v41-nvfp4`), device-parity ядер (RMSNorm/RoPE/HC/attn/compressor/indexer/router/MoE/head), `test_deepseek_moe_v41_pipeline` — production `RunDeepSeekV41Inference` на синтетической ПОЛНО-ГЕОМЕТРИЧЕСКОЙ модели (SWA ratio-0 + CED ratio-1) с бит-детерминизмом повторов; `test_deepseek_moe_v41_selfcheck` — Paris 3 цикла (env `LSX_TEST_DEEPSEEK_V41_ROOT`, SKIP_RETURN_CODE 4); perf-reference `test_perf_baseline_deepseek_v41_flash` (2048 токенов, label perf, env `LSX_PERF_VAULT_SUBDIR=DeepSeek-V4.1-Flash`)
  - `tests/test_model_factory.cpp` — LsxModelFactory/registry; 5 test cases, 4 assertions (real-model tests SKIP без модели)
  - `tests/test_tokenizer.cpp` — токенизатор (unicode, schema, arrow-конвертация, pre-tokenizer, encode/decode); 14 test cases, 236 assertions
  - `tests/test_glm_*.cpp` — GLM-5.2 suite (12 executables): convert, loader, model, compute, dense_compute, moe_router, experts, moe_compute, attention, inference, full_compute, selfcheck (~541k assertions, real-model tests SKIP без GLM; test_glm_selfcheck — 3-цикловый stress SelfCheck с бит-точной детерминированностью `final_logits` (3 циклов достаточно для подачи нагрузки; переопределяется `LSX_TEST_GLM_SELFCHECK_CYCLES`), exit 0)
  - `tests/test_qwen_moe_{convert,loader,model}.cpp`, `tests/test_qwen_moe_kernels.cu`, `tests/test_qwen_moe_selfcheck.cpp` + `qwen_moe_test_fixture.h` — Qwen3.5-35B-A3B suite: конвертер/классификатор/расщепление fused gate_up (119 assertions), loader roundtrip+strictness (49), LRU-кэш/роутер/routed-FFN/shared-combine/generation-driver детерминизм (~500 assertions в kernels), реальный селфчек "Paris" 3 бит-точных цикла (`LSX_TEST_QWEN_MOE_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4)
  - `tests/test_qwen_moe_36_{convert,model,selfcheck}.cpp` — Qwen3.6-35B-A3B suite: классификатор MTP fused-грамматики + bos-мост, реестр/геометрия/детерминизм, реальный селфчек Paris 3 бит-точных цикла (`LSX_TEST_QWEN36_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4)
  - `tests/test_mosaic_network.cpp` — lsxnetwork (FrameQueue SPSC, zero-copy); 8 test cases, ~2M assertions
  - `tests/test_integrated_pipeline.cpp` — TaskMosaic оркестратор (GLM + DeepSeek burst); 4 test cases, 8026 assertions
  - `tests/test_glm_moe_flash_{convert,loader,model,selfcheck}.cpp` + `test_glm_moe_flash_kernels.cu` + `test_glm_moe_flash_fixture.h` — GLM-5.3-Flash suite (6 файлов, ~2300 assertions): общая фикстура `ConvertedFlashFixture` (синтетический KDA-0 + DSA/MoE-3 чекпойнт → `ConvertGlm5NextModel` → полный logestix-layout); конвертер/лоадер/модель гоняются ТОЛЬКО на ней (loader-гейт `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` в `glm_moe_flash_load.cpp` открывает frozen-геометрию и пропуск пустых frozen-слоёв; vault не грузится никогда — 0.5-2 с вместо минут); геометрия/классификатор NVFP4-политики, loader strictness + LRU-кэш, router top-8 (raw-sigmoid coeff) / KDA / DSA / dual-stream staging ядра, реальный селфчек Paris 3 бит-точных цикла (env `LSX_TEST_GLM_FLASH_ROOT`, SKIP без модели)
  - `tests/test_model_independence_guard.cpp` — независимость моделей: уникальные паспорта констант, без cross-includes, routing-тумблеры в lsx_nvfp4 (одно определение), каждый consumer-модель открывает `LsxFastMathScope(cfg_.fast_math, ...)` в Infer (список consumers расширять в коммите новой модели)
  - `tests/test_model_suite_guard.cpp` — автономность+полнота тест-сьют моделей (зеркало model-independence для тестов): реестр lifecycle-стадий на модель (convert → load → model → kernels → selfcheck; каждая стадия обязана иметь зарегистрированный ctest-таргет, шаринг стадий между сиблингами — явно в реестре), SKIP_RETURN_CODE 4 на selfcheck-таргетах, запрет кросс-семейственных include тест-заголовков (family-stem: glm/qwen/deepseek/gemma/kimi) и ЗАПРЕТ загрузки model-vault в loader/model-стадиях (`test_*_loader*`/`test_*_load_fast*`/`test_<family>_model.cpp` — только синтетические конвертированные фикстуры; vault-интеграция живёт в selfcheck/perf-сьют; env-var opt-in + SKIP допустим вне loader-файлов). Модельно-нейтральные примитивы синтетических чекпойнтов (FakeTensor/WriteSafetensors/MakeBf16*/TensorNamesIn) живут ТОЛЬКО в `tests/test_synth_checkpoint.h` (namespace `synthtest`) — family-фикстуры реэкспортируют их, заимствовать чужую family-фикстуру запрещено. Реестр сьют расширять в том же коммите, что и тесты новой модели
  - `tests/test_absl_format_guard.cpp` — host-only guard: `absl::StrFormat`/`StrAppendFormat`/`StreamFormat`/`Format` — printf-стиль (`%s/%d/%u/%f`); fmt-плейсхолдеры `{}` в format-строках запрещены (печатается литеральный `{}`; compile-time checker abseil инертен на GCC и заглушен `#if 0` для AOCC в корневом CMakeLists). Шпаргалка конверсий: строки → `%s` (path → `.string()`), signed int (вкл. int64_t) → `%d`, unsigned (вкл. size_t) → `%u`, float/double → `%f`/`%.Nf`; спеки `{:03d}`/`{:<N}`/`{:>N}`/`{:.Nf}` → `%03d`/`%-Ns`/`%Ns`/`%.Nf`. Плюс runtime-пин семантики этих пар (checker выключен — компилятор типы не проверяет). Сканирует `libs/` и `apps/`, обязан быть зелёным в CI
  - `tests/test_cuda_toolkit_guard.cu` — CUDA-тулчейн guard (label inference): рантайм 13.x (пол 13.2, AGENTS.md), header/runtime версии совпадают, семантика `cuda_fp4.h` E2M1 (RNE ties-to-even, SATFINITE→±MAXNORM, NaN→+6) и pack-порядок float2-API (`x.x`→low nibble — измерено на 13.4, см. справочник §B3), наличные GPU — только Blackwell CC 12.x
- `third_party/cccl/` — NVIDIA CCCL (сабмодуль)
- `third_party/flatbuffers/` — Google FlatBuffers v25.12.19 (сабмодуль)
- `third_party/libzmq/` — ZeroMQ libzmq v4.3.5 (сабмодуль)
- `third_party/simdjson/` — simdjson v4.6.11 (сабмодуль)

## Model Ingestion Protocol

Добавление/адаптация новой модели — скил **`ingesting-a-model`**
(`.opencode/skills/ingesting-a-model/SKILL.md`): 8-этапный MODEL INGEST INVARIANT
(lsxforge-конвертер → NVFP4-квантование → регистрация в `LsxModelFactory` +
frozen geometry → tokenizer-мост → CUDA-ядра → shared-mem sandbox → passport →
SelfCheck/тесты), соглашение об именах файлов моделей
(`<family>_<type>[_<variant>]_<role>[_<subsystem>]`) и реестры guard-тестов
(`test_code_naming_guard`, `test_model_independence_guard`,
`test_model_suite_guard`), которые обязательно расширяются в том же коммите, что и
файлы модели. Единственный источник истины — SKILL.md; здесь оставлен только якорь,
потому что протокол цитируется в коде как «Model Ingestion Protocol step N».

## CMake Key Points

- `lsxcommon` — STATIC library в `libs/lsxcommon/CMakeLists.txt`; линкуется PUBLIC с `spdlog::spdlog`, `flatbuffers`, `libzmq-static`, `cudart`, `lsxnetwork`; PRIVATE с `arrow_shared`, `arrow_cuda_shared`, `simdjson::simdjson`
- CUDA `.cu` файлы вынесены в отдельные OBJECT библиотеки: `lsxcommon_cu` (DeepSeek inference + safetensors), `lsxcommon_glm_cu` (GLM-5.2 kernels, `-fmad=false` для бит-точной математики) и `lsxcommon_qwen_moe_cu` (Qwen3.5-35B-A3B MoE kernels, `-fmad=false`) — линкуются в `lsxcommon` PUBLIC. Это изолирует NVCC от MSVC-флагов
- `lsxcommon_cu`/`lsxcommon_glm_cu` линкуются с `cudart` и `cublas` (CUDA Toolkit system libs). Пути к CUDA
  Toolkit находятся через `find_package(CUDAToolkit)` (работает и в Linux, и в Windows),
  а не через жёсткие пути `C:/Program Files/...
- `lsxnetwork` — STATIC library (`libs/lsxnetwork/CMakeLists.txt`); simdjson, libzmq-static и Threads — PUBLIC (STATIC-библиотека ре-экспортирует link-зависимости потребителям); spdlog — PRIVATE
- Each app in `apps/*/` links `lsxcommon` PRIVATE
- Catch2 v3.16.0 fetched at root level (via FetchContent) when `BUILD_TESTING=ON`
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` установлен в `${CMAKE_BINARY_DIR}/bin` (до `add_subdirectory`)
- `add_custom_command(TARGET ... POST_BUILD)` копирует `arrow.dll` + `arrow_cuda.dll` в директорию exe (только Win32); applies to all 4 app targets, all test targets (кроме `test_mosaic_network`)
- **MSVC flags**: `/W4 /utf-8 /MP /arch:AVX512` через `CMAKE_CXX_FLAGS` (НЕ `add_compile_options`, чтобы не попадали в NVCC). `/arch:AVX512` включает AVX-512F/BW/CD/DQ/VL. `WIN32_LEAN_AND_MEAN NOMINMAX` (suppress Windows macro conflicts with CCCL). DO NOT use `/Zc:preprocessor` (breaks CCCL/thrust). Use `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` + `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP` for CCCL/MSVC compat.
- **CMAKE_POLICY_VERSION_MINIMUM**: задаётся через `CMakePresets.json` (`cacheVariables`), ручная установка env-переменной не требуется. Arrow's xsimd требует значения 3.5.
- **GCC/Clang Linux flags**: НЕ глобальные. Тюнинг под EPYC 9V74 (Zen 4) живёт в
  INTERFACE-библиотеке `lsx_tune` (root `CMakeLists.txt`) и включается через
  `LSX_ARCH_ZNVER4`/`LSX_LTO` (см. пресет `linux-release-znver4`): `-march=znver4`,
  `-mtune=znver4`, `-O3` (Release), `-fno-math-errno`, `-falign-loops=16`,
  `-funroll-loops`, LTO `-flto` (GCC) / `-flto=thin` (Clang/AOCC). Всё обёрнуто в
  `$<$<PLATFORM_ID:Linux>...$<COMPILE_LANGUAGE:C,CXX>>`, LTO ещё и в
  `$<$<LINK_LANGUAGE:CXX> $<CONFIG:Release>>` — в NVCC и MSVC не протекает.
  Обычные `linux-debug`/`linux-release` пресеты остаются generic-сборками без
  `-march`.
- **Arrow 25.0.0 + Boost patch**: On MSVC, Boost 1.88.0's `literals.hpp` fails with C++20 due to `operator""_cppi` / `_cppui` sized literal redefinition. Workaround: remove the 4 `BOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL(...)` invocations after Arrow download (see `CMakeLists.txt` lines 85-98).
- Arrow include paths добавлены PUBLIC напрямую к `lsxcommon` (`_deps/arrow-src/cpp/src`, `_deps/arrow-build/src`)
- CUDA: архитектуры `120-real;121-real` (Blackwell only; Hopper/Ada/Ampere запрещены), флаг `--extended-lambda` для thrust `__host__ __device__` лямбд

## Conventions

- C++20, C23
- Все логирование через spdlog (никаких `std::cout` / `printf`)
- `#include` paths без лишних комментариев
- `target_include_directories` с генераторными выражениями `$<BUILD_INTERFACE:...>`
- Новые `.cpp` добавлять в `apps/*/CMakeLists.txt` в `add_executable`, логику — в `libs/lsxcommon/`
- **CUDA async pipeline**: Минимизируй `cudaDeviceSynchronize()`. Весь конвейер на неблокирующих `cudaStream_t`, `cudaEvent_t`, `cudaMemcpyAsync`. Используй cooperative groups для синхронизации внутри грида.
- **CUDA Tile C++**: Для вычислительных ядер используй тайловое программирование (Tile Programming) вместо ручного Shared Memory и явных TMA-интринсиков. Компилятор сам раскладывает тайлы под Blackwell. На MSVC: черновые reductions через shared memory (CUB/MSVC несовместим). Компилируй на Linux с CUDA 13.3+ для `cub::BlockReduce`/`DeviceReduce`.
- **CompileIQ**: В CMake активируй фреймворк автотюнинга CompileIQ для подбора агрессивных параметров оптимизации критических ядер. Флаг `--compilation-database` доступен на Linux (CUDA 13.3+). На MSVC передаётся как `-Xcompiler` при поддержке хост-компилятора.
- **NVFP4**: Матричные операции реализуй нативно в NVFP4 через CCCL 3.3 + `std::mdspan`.
- **CCCL priority**: Для редукции/сортировки на GPU — `cub::DeviceReduce` / `cub::BlockReduce` вместо кастомных циклов (если не требуется экстремальный тюнинг регистров). На MSVC: используй thrust как fallback (thrust internally uses CUB/cuDPP).
- **Arrow + CUDA**: Используй `arrow::cuda::CudaBuffer`. Избегай лишних Host→Device копий. Zero-Copy через `CudaMemoryManager`.
- **simdjson + libzmq**: Zero-Copy обёртки — `zmq::message_t` напрямую в `simdjson::onn_demand::parser` без промежуточного `std::string`.
- **Никаких галлюцинаций**: Не уверен в сигнатуре метода Arrow/CCCL — проверь `.h`/`.cuh` в репозитории или запусти компиляцию.
- **Alignment**: Проверяй выравнивание данных для CUDA-структур, контролируй Shared Memory на блок (occupancy). Blackwell: 48 warps/SM max.
- **Комментарии**: Сохраняй все Doxygen/JSDoc при рефакторинге. Адаптируй для проекта.

## absl:: Priority (Container/String policy)

Host-only C++ (НЕ компилируемый NVCC) предпочитает `absl::`: `unordered_map/set` →
`flat_hash_map/set`, `map/set` → `btree_map/set`, простые конкатенации →
`absl::StrCat`/`StrAppend`. Таблица приоритетов, исключения (что остаётся `std::`),
NVCC-инкомпатибельность контейнеров с правилом достижимости и recipe-проверкой,
printf-контракт `absl::StrFormat` и CMake-якоря (ABSL_PROPAGATE_CXX_STD, патчи
crc32 и format-checker) — скил **`absl-in-cuda-cpp`**
(`.opencode/skills/absl-in-cuda-cpp/SKILL.md`). Здесь остаётся незыблемое:

- Заголовки, достижимые из `.cu`/`.cuh`, НЕ должны включать `<absl/container/*>`
  (NVCC падает в raw_hash_map: `C2794`/`C2938`); в `.cu`/`.cuh` контейнеры и
  стриммы остаются `std::`. Сейчас container-заголовок включают только
  `lsx_orchestrator.h` и `model_integrity.h` (host-only), нарушений из `.cu` нет;
  проверка транзитивного графа — в скиле (guard это НЕ автоматизирует).
- `absl::StrFormat`/`StrAppendFormat`/`Format` — printf-стиль (`%s/%d/%u/%f`);
  `{}` печатается литерально (guard `test_absl_format_guard`).
- Остаются `std::`: `string_view`/`optional`/`variant`/`function`/`unique_ptr`,
  потоковые builders с `setw`/`setfill`, кастомные хэшер-функторы.

## NVIDIA Docs Reference (обязательный справочник)

- **`docs/NVIDIA_DOCS_NVFP4_MEMORY.md`** — офлайн-зеркало документации NVIDIA
  (nvidia-cuda-docs MCP): лимиты CC 12.x, альтернативные FP-форматы PTX
  (E2M1/E4M3/UE4M3/UE8M0), семантика cuda_fp4/cuda_fp8 конверсий,
  tcgen05 block-scaled MMA, cuBLASLt NVFP4, TMA/bulk-copies, stream-ordered
  память + производные инварианты кодека репозитория.
- ПРИ РАБОТЕ с CUDA/Blackwell/NVFP4-кодом КАЖДЫЙ агент ОБЯЗАН:
  1. Перед изменением NVFP4/CUDA-ядер — сверить семантику форматов и
     аппаратные лимиты со справочником; MCP доступен не на всех машинах —
     справочник является единым источником истины в офлайне.
  2. При получении НОВЫХ фактов из nvidia-cuda-docs MCP — дополнять
     справочник в том же коммите, что и использующий их код.
   3. Не расходиться со справочником молча: изменение семантики кодека
      (например, zero-check по полю e4, ceil-округление pow2-E4M3) обязано
      сопровождаться обновлением соответствующего раздела и регрессионного
      теста.

## GPU Perf Auto-Tuning

Вся скоростная работа — профилирование decode-токена (nsys), budget-тесты по TDD,
нобы `gpu_perf`, перф-базлайн — описана в скиле **`perf-tuning-a-model`**
(`.opencode/skills/perf-tuning-a-model/SKILL.md`). Здесь остаётся незыблемое:

- **CLI speed/perf-флагов нет**: движок всегда гоняет per-GPU auto-политику;
  единственный численный флаг запуска — `--bit-exact` (lsxbenchmark), он маппится
  в `FastMathMode::kOff`. `LSX_PERF_AUTO=0` возвращает legacy bit-exact дефолты,
  `LSX_TEST_FAST_MATH` форсирует TC. Тесты, пинящие численный класс, ставят `kOff`.
- `gpu_perf` (`include/lsxcommon/gpu_perf.h`) резолвит нобы на `InitializeEngine`
  (TC fast-math, лимит GEMV MiB, порог split-K, % VRAM под expert-кэш, dual-stream
  staging); приоритет разрешения и потребители ноб — в скиле.
- `lsxbenchmark` — строгий GPU-эксклюзив (см. Resources).

## Git

- Английский язык сообщений коммитов
- Префиксы: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`, `CMakeLists.txt`).
- The operator manages the process exclusively through high-level prompts and changes to instruction files.
- If manual code changes are detected outside AI sessions, perform a full diff audit and reconcile with the operator before proceeding.
- **ЗАПРЕТ на параллельных/субагентов**: использование Task-субагентов (explore/general/parallel agents) запрещено — они допускают много ошибок. Все исследования кодовой базы, аудит и правки выполняются только основной сессией напрямую (Read/Grep/Edit/etc.).
  - `tests/test_qwen_moe_exp_{convert,loader,model}.cpp`, `tests/test_qwen_moe_exp_kernels.cu`, `tests/test_qwen_moe_exp_selfcheck.cpp` + `qwen_moe_exp_test_fixture.h` — Qwen3.8-Flash-Next (qwen4_exp) suite: классификация/NVFP4-политика/splitmix64+prime таблицы (parity против чекпоинта), loader strictness+LRU 512-экспертных слотов, hc GatedResidual/scatter бит-точные против host-реплики, QSA indexer degenerate/top-k, router top-10 parity, реальный селфчек Paris (`LSX_TEST_QWEN38_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4)
