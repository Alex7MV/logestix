# logestix — AGENTS.md

## Project

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C23. CMake 3.25+ (CMakePresets).
Target architecture: NVIDIA Blackwell (CC 12.0 / 12.1). Active use of 5th-gen Tensor Cores and TMA.
Max warps per SM on Blackwell: 48 (factor into occupancy calculations).
Forbidden architectures: Hopper (90), Ada (89), Ampere (80).
No backward compatibility — any code targeting pre-Blackwell (sm_90, sm_80) is rejected.
CUDA Toolkit 13.2+ (nvcc 13.2.78; 13.3 также поддерживается), full native C++23 support in NVCC.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.0 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.4.2 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | 3.3 (latest) | submodule (`third_party/cccl`) | Header-only |
| abseil-cpp | 20260526.0 | FetchContent (`20260526.0`, LTS May 2026) | Header-only + static libs |
| CUDA cuBLAS | 13.2+ | system (CUDA Toolkit) | Dynamic (`cublas.dll`) |
| Catch2 | v3.7.0 | FetchContent (test only) | Header-only |

## Build

Use CMakePresets — ready-made for both supported OSes. Replace `debug` with
`release` in the preset name for an optimization build.

Toolchain notes (tested, higher than the minimums below also work):
- CMake 4.2.3 / 3.25+ (CMakePresets, `CMAKE_POLICY_VERSION_MINIMUM=3.5`)
- GCC 15.2 (Ubuntu 26.04) — C++20 baseline; GCC 15+ supports C++23, but the
  project pins to C++20 deliberately
- NVCC 13.2.78 — CUDA `LANGUAGES ... CUDA` enabled at top level
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
инференса** (`--selfcheck`, `--benchmark`, `--testprompt`, infer-тесты,
CUDA-ядра на реальных весах), потребление памяти и GPU-ресурсов считается
критичным: не запускать параллельные тяжёлые нагрузки, следить за VRAM/RAM
и освобождать модели по завершении. Работающие инференсы не дублировать.

## Test

`BUILD_TESTING=ON` is included in every preset.

```bash
ctest --preset linux-debug      # Ubuntu 26.04
```

```powershell
ctest --preset windows-debug    # Windows 11
```

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `libs/lsxcommon/src/core.cpp` — ядро (логирование, инициализация)
  - `libs/lsxcommon/include/lsxcommon/core.h` — публичный заголовок
  - `libs/lsxcommon/src/common.cpp` — version + GPU info (spdlog)
  - `libs/lsxcommon/src/schema.cpp` — ExpertSchema (9 полей: weights, E4M3 per-block scales uint8[], float32 global_scale), PackedExpert
  - `libs/lsxcommon/src/converter.cpp` — GPU-only NVFP4 квантизация (per-block E4M3 + global_scale), парсинг safetensors, сериализация в Arrow IPC
  - `libs/lsxcommon/src/loader.cpp` — загрузка экспертов из .arrow, CUDA pinned регистрация, E4M3 scale decoding
  - `libs/lsxcommon/src/glm_convert.cpp` + `include/lsxcommon/glm_convert.h` — GLM-5.2 классификатор, Topology, `ConvertGlmModel`, `SaveGlmPinnedTensors`, `WriteGlmModelIndex` (extended `logestix-glm` формат)
  - `libs/lsxcommon/src/glm_inference_load.cpp` + `include/lsxcommon/glm_inference_load.h` — host-only GLM loader: `LoadGlmPinnedWeights` (BF16→FP16 upload, router bias host pinned, indexer bias device), `GlmLayerWeights`/`GlmPinnedWeights`
  - `libs/lsxcommon/src/glm_v5_model.cpp` + `src/glm_v5_model.h` — `GlmV5Model` (IModel, registry `glm-5.2`); Infer: полный 79-слойный цикл (dense 0-2, IndexShare attention + MoE router + NVFP4 expert stacks 3-77, MTP epilogue layer 78); SelfCheck: инкрементальный 1→2-токенный селфчек (форс `max(2, max_gen_tokens)`, декод 1/2-токенного префикса против "Paris"), публикует `gen_tokens` + `final_logits` в `InferenceResult`
  - `libs/lsxcommon/src/glm_inference.cu` — GLM-5.2 CUDA-ядра (router, IndexShare attention, NVFP4 expert decode/SwiGLU)
  - `libs/lsxcommon/src/glm_device_math.h` — device-математика, бит-точная репликация glibc-функций для GLM-5.2
  - `libs/lsxcommon/src/tokenizer_convert.cpp` — tokenizer arrows; merges в форматах string `"a b"` И array `["a","b"]` (GLM-5.2), byte_to_token_id, bos/eos/pad
  - `libs/lsxcommon/src/tokenizer.cpp` + `pretokenizer.cpp` + `unicode.cpp` — BPE-токенизатор, word-level претокенизация, Unicode-классификация
  - `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` — `DeepSeekV4FlashModel` (IModel, registry `deepseek-v4-flash`)
  - `libs/lsxcommon/src/model_factory.cpp` + `model_registry.cpp` + `include/lsxcommon/model_factory.h` — `LsxModelFactory` + IModel registry
  - `libs/lsxcommon/src/models/lsx_orchestrator.cpp` + `include/lsxcommon/lsx_orchestrator.h` — TaskMosaic оркестратор: lock-free SPSC dispatch GLM-5.2 / DeepSeek / unknown
  - `libs/lsxcommon/src/deepseek_inference_*.cu` + `deepseek_inference_load.cuh` + `deepseek_norm_kernels.cu` — DeepSeek-V4-Flash inference kernels: load, norm, moe, head, attn, csa, pipeline
  - `libs/lsxcommon/src/shared_experts.cu` — NVFP4 decode-таблицы (E2M1/E8M0), routed-expert FFN
  - `libs/lsxcommon/src/safetensors_gpu.cu` — GPU-ускоренная квантизация (3 CUDA-ядра: convert, block_maxabs, quantize_pack_perblock; CCCL thrust для global max)
  - `libs/lsxcommon/src/benchmark_gpu.cu` + `include/lsxcommon/benchmark_gpu.h` — `--benchmark` pipeline: NVFP4 ping-pong benchmark (init_hidden_states kernel, cuBLAS GEMM, dual-stream overlap, pinned staging), `lsxcommon::gpu::RunBenchmark()`, BenchmarkResult
- `libs/lsxnetwork/` — shared STATIC library; namespace `lsxnetwork`; zero-alloc async сетевой слой TaskMosaic CE (FrameQueue SPSC, ZMQ context, I/O worker, OpenAI endpoint, affinity)
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`, `--benchmark`, `--testprompt`, `--selfcheck`)
  - `apps/lsxfabric/src/validator.cpp` — валидатор NVFP4/E4M3/global-scale в собранной модели
  - `apps/lsxfabric/tests/` — 12 test cases (1 lifecycle + 11 validator)
- `apps/lsxforge/` — data preprocessing utility; `--glm` for the GLM-5.2 pipeline (`lsxcommon::glm::ConvertGlmModel`), `--tokenizer`, repack-флаги
  - `apps/lsxforge/tests/` — 1 test case, 1 assertion
- `apps/taskmosaic/` — task orchestrator (CLI-обёртка над `lsxcommon::LsxOrchestrator`)
  - `apps/taskmosaic/tests/` — 1 test case, 1 assertion
- `apps/lsxcli/` — CLI client (`--model`, `--info`, `--infofull`)
  - `apps/lsxcli/tests/` — 1 test case, 1 assertion
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
  - `tests/test_deepseek_loader.cpp` — 3 test cases, 64 assertions (round-trip, empty fields, E4M3 encoding)
  - `tests/test_deepseek_inference_kernels.cu` — DeepSeek CUDA inference kernel unit tests; 12 test cases, 4470 assertions (E2M1/E8M0, act_quant, RMSNorm, SwiGLU, CSA Compressor, routed-expert FFN, NVFP4 codec)
  - `tests/test_model_factory.cpp` — LsxModelFactory/registry; 5 test cases, 4 assertions (real-model tests SKIP без модели)
  - `tests/test_tokenizer.cpp` — токенизатор (unicode, schema, arrow-конвертация, pre-tokenizer, encode/decode); 14 test cases, 236 assertions
  - `tests/test_glm_*.cpp` — GLM-5.2 suite (12 executables): convert, loader, model, compute, dense_compute, moe_router, experts, moe_compute, attention, inference, full_compute, selfcheck (~541k assertions, real-model tests SKIP без GLM; test_glm_selfcheck — 3-цикловый stress SelfCheck с бит-точной детерминированностью `final_logits` (3 циклов достаточно для подачи нагрузки; переопределяется `LSX_TEST_GLM_SELFCHECK_CYCLES`), exit 0)
  - `tests/test_mosaic_network.cpp` — lsxnetwork (FrameQueue SPSC, zero-copy); 8 test cases, ~2M assertions
  - `tests/test_integrated_pipeline.cpp` — TaskMosaic оркестратор (GLM + DeepSeek burst); 4 test cases, 8026 assertions
- `third_party/cccl/` — NVIDIA CCCL (сабмодуль)
- `third_party/flatbuffers/` — Google FlatBuffers v25.12.19 (сабмодуль)
- `third_party/libzmq/` — ZeroMQ libzmq v4.3.5 (сабмодуль)
- `third_party/simdjson/` — simdjson v4.4.2 (сабмодуль)

## Model Ingestion Protocol

Унифицированный 8-этапный протокол добавления новых моделей (MODEL INGEST
INVARIANT). Каждый агент ОБЯЗАН строго следовать порядку этапов при интеграции
любой новой архитектуры. Нарушение порядка или пропуск шагов классифицируется
как дефект сборки (broken build). Эталоны реализации: GLM-5.2, DeepSeek-V4-Flash,
Qwen3.5-9B (перечисленные пути — актуальные якоря кода, не плейсхолдеры).

### 1. Топология и паспорт структуры (lsxforge)

- Расширить C++-конвейер `apps/lsxforge` (не Python-инструмент) парсингом
  оригинального конфигурационного файла (config.json) и safetensors
  (аналогично `lsxcommon::qwen::ConvertQwenModel` в `libs/lsxcommon/src/qwen_convert.cpp`).
- Сгенерировать очищенный от мультимодального жира структурный манифест.
- НАПРАВЛЕНИЕ ПУТЕЙ: Результат обязан детерминировано сохраняться в
  централизованный каталог метаданных `model-metadata/` через единый authority
  путей `lsxcommon::lsx_model_metadata(rel)` / `lsx_model_metadata_root()`
  (`libs/lsxcommon/src/path_registry.cpp`, Windows `D:\Programs\huggingface.co`
  / Linux `/mnt/llm_models/models`). Инвариант vault-root, переключатели путей —
  только в `path_registry`, никаких хардкодов путей в конвертере.

### 2. Квантование и сжатие (NVFP4)

- Настроить движок квантования на тотальное 4-битное квантование плотных
  линейных матриц: E2M1-нибблы (nibble-packed, 2 элемента на байт) + поблочные
  E4M3-шкалы + float global_scale (см. `QwenNvfp4Weight` в
  `libs/lsxcommon/include/lsxcommon/qwen_inference_load.h`).
- ИНВАРИАНТ ТОЧНОСТИ: Layers нормирования (Norms) и векторы смещений (Biases)
  ОБЯЗАНЫ оставаться в высоком разрешении FP16/BF16, не квантизуются.

### 3. Регистрация в фабрике (LsxModelFactory)

- Зарегистрировать строковый идентификатор модели в
  `libs/lsxcommon/src/model_registry.cpp` через `LsxModelFactory::Register(kType, ...)`
  (пример: `QwenV5Model::kType = "qwen3.5-9b"`, `libs/lsxcommon/src/qwen_v5_model.h`).
- Реализовать класс `IModel` (final, наследуется от `lsxcommon::IModel`), реализовать
  `Load/Infer/Unload/Inspect/SelfCheck/Tokenizer`.
- НОВОЕ ТРЕБОВАНИЕ для новых моделей: параметры геометрии (hidden_size,
  layers_count, heads) жёстко блокировать compile-time макросами и `static_assert`
  на этапе загрузки (текущие модели читают их из конфига в рантайме).

### 4. Побайтовая токенизация (Token Bridge)

- Зарегистрировать BPE-словарь токенизатора через `BuildTokenizerArrowFiles`
  (`libs/lsxcommon/src/tokenizer_convert.cpp`); merges-форматы: string `"a b"`
  И array `["a","b"]`.
- ИНВАРИАНТ ВЫРАВНИВАНИЯ: `bos_id`/`eos_id`/`pad_id` читать из конфига модели
  (`tokenizer_convert.cpp` default-ы), а НЕ хардкодить в ядре; служебные префиксы
  сгенерировать ровно как в исходном токенизаторе, исключая мусорные смещения
  при кодировании промпта. `bos_id >= 0` → реальный ID начала предложения
  (`libs/lsxcommon/src/tokenizer.cpp`), иначе `bos_id = -1`.

### 5. Низкоуровневые CUDA-ядра (inference.cu)

- Написать/расширить кастомные CUDA-ядра инференса под специфику внимания модели
  (пример гибридного макета: GatedDeltaNet + Full Attention —
  `libs/lsxcommon/src/qwen_inference.cu`: `qwen_gated_delta_kernel`,
  `full_attn_softmax_kernel`).
- ИНВАРИАНТ СТРИМА: Весь цикл генерации от Prefill до Decode выполняется внутри
  ОДНОГО сквозного `cudaStream_t` без возврата управления хосту на каждом токене
  (без `cudaDeviceSynchronize` в цикле; только финальная синхронизация).
- ИНВАРИАНТ ШАГА (Stride): Все смещения адресов Q, K, V вычислять строго по
  физической геометрии тензоров, исключая наложение соседних временных интервалов
  (stride-арифметика в strided conv/recurrence бафферах).

### 6. Синхронизация памяти (Shared Memory Sandbox)

- Любая операция Read-Modify-Write (RMW) внутри Shared Memory ОБЯЗАНА быть
  изолирована: запись разрешена только `thread 0` под защитой жёстких барьеров
  `__syncthreads()`, после барьера остальные потоки читают только-на-чтение
  (Эталон: `full_attn_softmax_kernel`, `qwen_inference.cu`).
- Это ликвидирует нетерминизм варпов Independent Thread Scheduling
  (data race в shared памяти → NaN → недетерминированная генерация).

### 7. Поточно-изолированный маппинг (LsxTokenTranslator)

- Обновить бинарный паспорт трансляции токенов `model-metadata/passport/passport.bin`
  (vault: `lsxcommon::lsx_model_metadata("passport/passport.bin")`, генерация через
  `lsxnetwork::BuildPassport` / `WritePassport`).
- Интегрировать экземпляр транслятора как приватный член `alignas(64)` внутри
  сетевого воркера `IoWorker` (`libs/lsxnetwork/include/lsxnetwork/io_worker.h`).
- Загрузка `Load()` через Win32 `MapViewOfFile(PAGE_READONLY)` / POSIX
  `mmap(PROT_READ)` обязана происходить строго ПОСЛЕ аппаратной привязки потока
  к ядру (`CpuAffinity::PinToCores0To3`) И ДО сигнального флага готовности
  `set_value()` (`libs/lsxnetwork/src/io_worker.cpp`).

### 8. ТДД и селфчек (SelfCheck Harness)

- Реализовать инкрементальный селфчек генерации (1→2 токена) через метод
  `SelfCheck()` (эталон: `QwenV5Model::SelfCheck`, `libs/lsxcommon/src/qwen_v5_model.cpp`).
- Контрольный промпт `"The capital of France is"` обязан выдавать детерминированный
  эталонный ответ `"Paris"` (top-1, декод первого токена/короткого префикса),
  `r.selfcheck_passed == true`.
- НОВОЕ ТРЕБОВАНИЕ для новых моделей: добавить `tests/test_<model>_selfcheck.cpp`
  с инкрементальным 1→2-токенным селфчеком подряд минимум 3 циклов стресс-теста
  (для подачи нагрузки достаточно 3-х раз; количество переопределяется
  `LSX_TEST_GLM_SELFCHECK_CYCLES`) с бит-точной детерминированностью логитов
  и `exit 0`.

## CMake Key Points

- `lsxcommon` — STATIC library в `libs/lsxcommon/CMakeLists.txt`; линкуется PUBLIC с `spdlog::spdlog`, `flatbuffers`, `libzmq-static`, `cudart`, `lsxnetwork`; PRIVATE с `arrow_shared`, `arrow_cuda_shared`, `simdjson::simdjson`
- CUDA `.cu` файлы вынесены в отдельные OBJECT библиотеки: `lsxcommon_cu` (DeepSeek inference + safetensors + benchmark) и `lsxcommon_glm_cu` (GLM-5.2 kernels, `-fmad=false` для бит-точной математики) — линкуются в `lsxcommon` PUBLIC. Это изолирует NVCC от MSVC-флагов
- `lsxcommon_cu`/`lsxcommon_glm_cu` линкуются с `cudart` и `cublas` (CUDA Toolkit system libs). Пути к CUDA
  Toolkit находятся через `find_package(CUDAToolkit)` (работает и в Linux, и в Windows),
  а не через жёсткие пути `C:/Program Files/...
- `lsxnetwork` — STATIC library (`libs/lsxnetwork/CMakeLists.txt`); simdjson, libzmq-static и Threads — PUBLIC (STATIC-библиотека ре-экспортирует link-зависимости потребителям); spdlog — PRIVATE
- Each app in `apps/*/` links `lsxcommon` PRIVATE
- Catch2 v3.7.0 fetched at root level (via FetchContent) when `BUILD_TESTING=ON`
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

Приоритет `absl::` над `std::` в host-only C++ коде (НЕ компилируемом NVCC):

| Приоритет | std:: | → absl:: | Заголовок |
|---|---|---|---|
| ДОЛЖЕН | `std::unordered_map` | `absl::flat_hash_map` | `<absl/container/flat_hash_map.h>` |
| ДОЛЖЕН | `std::unordered_set` | `absl::flat_hash_set` | `<absl/container/flat_hash_set.h>` |
| ДОЛЖЕН | `std::map` | `absl::btree_map` | `<absl/container/btree_map.h>` |
| ДОЛЖЕН | `std::set` | `absl::btree_set` | `<absl/container/btree_set.h>` |
| МОЖНО | `std::ostringstream` (простые конкатенации) | `absl::StrCat` / `absl::StrAppend` | `<absl/strings/str_cat.h>` |

Исключения — остаётся `std::`:
- `std::string_view`, `std::optional`, `std::variant`, `std::function`, `std::unique_ptr` —
  для C++17+ abseil сам рекомендует `std::*-версии (absl-версии — полифилы).
- Потоковый JSON/отчёт builder с цикловым мемпингом (`std::ostringstream` c setw/setfill
  и т.п.) — легитимный потоковый use, НЕ конвертируется (эталон: `qwen_convert.cpp`
  `ConvertQwenModel`, билдер model.index.json).
- Кастомные хэшер-функторы (`ExpertKeyHash`, `GlmExpertKeyHash`) — передаются как
  Hasher параметр напрямую, менять не нужно.

NVCC ИНКОМПАТИБИЛЬНОСТЬ (критично):
- abseil контейнеры (flat_, btree_ и все `absl/container/internal/*`) НЕ компилируются
  NVCC-ом (raw_hash_map ломается в `.cu`: `C2794`/`C2938` MSVC-in-NVCC).
- ЗАПРЕЩЕНО добавлять `#include <absl/container/*>` какие-либо в заголовки, достижимые
  транзитивно из `.cu`/`.cuh` TU (эталоны безопасных заголовков: `schema.h`,
  `tokenizer.h`, `shared_experts.h`; НЕбезопасные — только host-заголовки:
  `model_integrity.h`, `lsx_orchestrator.h`, `converter.h`, `http_router.h`).
- В `.cu`/`.cuh` файлах контейнеры и стриммы остаются `std::`; при сомнении проверь
  транзитивной граф: `rg 'include "[^"]+"' libs/lsxcommon/src/*.cu`.
- `absl::strings` подтягивается к объект-библиотекам `lsxcommon_cu`/`glm_cu`/`qwen_cu`
  через `target_link_libraries(... absl::strings)` — это даёт include-пути, но
  НЕ вносит контейнерных заголов во `.cu`.

CMake-якоря:
- `absl::flat_hash_map` / `absl::span` / `absl::strings` — PUBLIC линк `lsxcommon`
  и `lsxhome` (apps/lsxhome PRIVATE), `absl::flat_hash_map` — PUBLIC `lsxnetwork`;
  реализация в lsxcommon_cu/glm_cu/qwen_cu линкует `absl::strings`.
- `ABSL_PROPAGATE_CXX_STD=ON` выравнивает abseil по стандарту проекта (C++20);
  `CMAKE_CXX_STANDARD`/`CMAKE_CXX_FLAGS` вынужденно CACHE для probe-try_compile
  (`/std:c++20` нужен, чтобы `ABSL_INTERNAL_AT_LEAST_CXX20` в under-проекте).
- LTS-тег фиксируется на `20260526.0`; после FetchContent применяется идемпотентный
  патч `crc32_x86_arm_combined_simd.h` (MSVC+`/arch:AVX512` → G-asm не парсится;
  патч отключает MSVC-ветку SIMD в CRC).

## Git

- Английский язык сообщений коммитов
- Префиксы: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`, `CMakeLists.txt`).
- The operator manages the process exclusively through high-level prompts and changes to instruction files.
- If manual code changes are detected outside AI sessions, perform a full diff audit and reconcile with the operator before proceeding.
