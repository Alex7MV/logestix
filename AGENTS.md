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
| NVIDIA CCCL | 3.4.2 | submodule (`third_party/cccl`) | Header-only |
| abseil-cpp | 20260817.0 | FetchContent (`20260817.0`, LTS August 2026) | Header-only + static libs |
| CUDA cuBLAS | 13.2+ | system (CUDA Toolkit) | Dynamic (`cublas.dll`) |
| Catch2 | v3.16.0 | FetchContent (test only) | Header-only |

Версии принуждаются кодом: `tests/test_core.cpp` → `TEST_CASE("pinned dependency
versions")` (`STATIC_REQUIRE` для Arrow, spdlog, simdjson, Catch2, CCCL, abseil —
последний через `ABSL_LTS_RELEASE_VERSION`, так что не-LTS пин не соберётся); у
`flatbuffers`/`libzmq`/cuBLAS guard'а нет. Процедура обновления
зависимости (pin сабмодуля/FetchContent `GIT_TAG` + guard + таблицы AGENTS/README
+ перепроверка патчей-after-fetch) — скил `cuda-cmake-build-gotchas`.

## Build

Пресеты, AOCC-сборка, Windows-тулчейн (VS 2026) с точными путями, тулчейн-минимумы
и пути бинарников — скил **`building-the-project`**
(`.opencode/skills/building-the-project/SKILL.md`).

Быстрая проверка:

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                     # fast only
```

Неочевидные грабли сборки — скил **`cuda-cmake-build-gotchas`**
(`.opencode/skills/cuda-cmake-build-gotchas/SKILL.md`): CMake 4.x и
`enable_language(CUDA)` во вложенном каталоге, флаги хоста, протекающие в NVCC,
`/Zc:preprocessor` vs CCCL/thrust, `CMAKE_POLICY_VERSION_MINIMUM` для xsimd,
патчи Arrow/Boost/abseil, `lsx_tune`/LTO-гейтинг, `cc12x`-only arch set, host-only
CUDA-поверхности и `_putenv`-guard, bump зависимостей. Там же — таблица
«симптом → причина».

## Resources (host)

Политика расхода ресурсов хост-машины, исключение для инференса и правило
«одна GPU-нагрузка за раз» (включая строгую GPU-эксклюзивность `lsxbenchmark` и
запуск perf последним) — раздел «Host resource policy and GPU scheduling» скила
**`running-a-model`**
(`.opencode/skills/running-a-model/SKILL.md`).

## Test

`BUILD_TESTING=ON` включён во все пресеты. Категории тестов (labels
`fast`/`convert`/`inference`/`perf`), пресеты, роутер путей
(`tools/test_router.cmake`, `LSX_PRESET`/`LSX_LIST`/`LSX_DIFF`), дисциплина
label'ов в том же коммите, реестр guard-тестов и WINDOWS-ограничение
(`ctest -R "qwen|gemma"`) — скил **`test-lifecycle`**
(`.opencode/skills/test-lifecycle/SKILL.md`).

Правила тестов конвертеров (label `convert`, bounded TIMEOUT, синтетический
чекпоинт, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` как runtime-env) — раздел
«Converter test rules» в `ingesting-a-model`; процедура замера/подъёма скорости —
`perf-tuning-a-model`.

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`. Ядро:
  `core.cpp` (логирование/init), `schema.cpp` (ExpertSchema/PackedExpert),
  `converter.cpp` (GPU NVFP4-квантование), `loader.cpp`, `tokenizer.cpp` +
  `pretokenizer.cpp`/`unicode.cpp`, `model_factory.cpp`/`model_registry.cpp`,
  `model_runtime.cpp` (lsxbenchmark --bench), `gpu_perf.cpp`, модельно-нейтральный
  слой `lsx_inference_nvfp4.cu`, `expert_store.h`. Модели: glm / qwen / deepseek /
  gemma / kimi.
- `libs/lsxnetwork/` — shared STATIC library; namespace `lsxnetwork`; zero-alloc
  async сетевой слой TaskMosaic CE (FrameQueue SPSC, ZMQ context, I/O worker,
  OpenAI endpoint, affinity).
- `apps/lsxfabric/` — B2B inference engine, всегда максимальная скорость
  (`--localmodel`, `--validate`, `--testprompt`); замер скорости и бит-точность
  переехали в lsxbenchmark. `apps/lsxforge/` — data preprocessing utility
  (`--glm`/GLM-5.2 pipeline, `--tokenizer`, repack-флаги). `apps/taskmosaic/` —
  CLI-обёртка над `lsxcommon::LsxOrchestrator`. `apps/lsxcli/` — CLI client
  (`--model`, `--info`, `--infofull`).
- `apps/lsxbenchmark/` — замер скорости генерации + бит-точность: `--bench`
  (реальная генерация `IModel::Infer`), `--selfcheck` (Paris), `--bit-exact`
  (`FastMathMode::kOff`); единственное приложение с численным режимом.
- `apps/lsxhome/` — Windows-native desktop shell (D3D12 + Dear ImGui docking
  branch, WIN32 subsystem, только Windows); compute-тред публикует токены в
  lock-free SPSC `GuiBridge` → UI-тред дренирует их каждый кадр.
- `tests/` — Catch2 v3.16.0 suite; label'ы `fast`/`convert`/`inference`/`perf`
  (см. § Test) + guard-тесты моделей/именования/absl/тулчейна.
- `third_party/cccl/`, `third_party/flatbuffers/`, `third_party/libzmq/`,
  `third_party/simdjson/` — сабмодули.

Полный пофайловый инвентарь (по семействам и подсистемам: состав сьютов и
assertions, env-переменные selfcheck'ов, GUI-инварианты lsxhome, перф-детали
моделей) — скил **`logestix-repo-map`**
(`.opencode/skills/logestix-repo-map/SKILL.md`). Его машино-проверяемые
эквиваленты: `test_model_suite_guard.cpp::Suites()` (модель → стадии →
ctest-таргеты) и `test_model_independence_guard.cpp::kPassports` (модель →
паспорт-заголовок → namespace) — при расхождении верны они.

Запуск и эксплуатация инференса — скил **`running-a-model`**
(`.opencode/skills/running-a-model/SKILL.md`): приложения и их флаги
(`lsxbenchmark --bench/--selfcheck`, `lsxfabric --localmodel --testprompt/--validate`,
`lsxforge`, `lsxcli`), vault-пути (`path_registry`; CLI принимает КОРЕНЬ модели,
не `logestix/`), lifecycle движка (`InferenceEngine::Session`), prompt
framing/sampling, ресурсный слой (governor/VRAM/LRU-планирование) и каталог
runtime env-нобов `LSX_*`/`KIMI_*`.

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

- `lsxcommon` — STATIC (`libs/lsxcommon/CMakeLists.txt`); PUBLIC линк со spdlog,
  flatbuffers, libzmq-static, cudart, lsxnetwork; PRIVATE — arrow_shared,
  arrow_cuda_shared, simdjson. Arrow include-пути добавлены PUBLIC напрямую к
  `lsxcommon` (`_deps/arrow-src/cpp/src`, `_deps/arrow-build/src`).
- CUDA `.cu` вынесены в отдельные OBJECT-библиотеки (`lsxcommon_cu`,
  `lsxcommon_glm_cu`, `lsxcommon_qwen_moe_cu`, ...), линкуются в `lsxcommon`
  PUBLIC — это изолирует NVCC от MSVC-флагов. `cudart`/`cublas` находятся через
  `find_package(CUDAToolkit)` (Linux и Windows), не жёсткими путями.
- `lsxnetwork` — STATIC; simdjson, libzmq-static и Threads PUBLIC
  (STATIC-библиотека ре-экспортирует link-зависимости), spdlog PRIVATE.
- Каждое app из `apps/*/` линкует `lsxcommon` PRIVATE; Catch2 v3.16.0
  через FetchContent при `BUILD_TESTING=ON`.
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` = `${CMAKE_BINARY_DIR}/bin` (до
  `add_subdirectory`); POST_BUILD копирует `arrow.dll`/`arrow_cuda.dll` (Win32).
- Флаги (MSVC/NVCC), гейтинг `lsx_tune`/LTO, патчи зависимостей и CUDA-архитектуры
  — скил `cuda-cmake-build-gotchas`.

## Conventions

Язык C++20/C23, логирование только через spdlog (никаких `std::cout`/`printf`),
стиль `#include`, `target_include_directories` с `$<BUILD_INTERFACE:...>`, место
новых `.cpp` (app `add_executable` vs `libs/lsxcommon/`) и сохранение
Doxygen/JSDoc — скил **`coding-conventions`**
(`.opencode/skills/coding-conventions/SKILL.md`).

Kernel/Blackwell-правила — скил **`nvfp4-blackwell-conventions`**
(`.opencode/skills/nvfp4-blackwell-conventions/SKILL.md`): один стрим и минимум
`cudaDeviceSynchronize`, shared-mem sandbox (thread 0 + барьеры), NVFP4 только через
нейтральный `lsx_inference_nvfp4` + `std::mdspan`, CCCL-приоритет (cub, thrust
fallback на MSVC), Tile C++, CompileIQ, alignment/48 warps, zero-copy Arrow+CUDA и
simdjson+libzmq, «не уверен в сигнатуре — проверь `.h`/`.cuh`».

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
  (E2M1/E4M3/UE4M3/UE8M0), семантика cuda_fp4/cuda_fp8 конверсий, tcgen05
  block-scaled MMA, cuBLASLt NVFP4, TMA/bulk-copies, stream-ordered память +
  производные инварианты кодека репозитория. Единый источник истины в офлайне.
- Дисциплина работы с ним (сверять семантику ПЕРЕД правкой NVFP4/CUDA-ядер,
  дополнять при новых фактах в том же коммите, не расходиться молча) — скил
  `nvfp4-blackwell-conventions`.

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

Английский язык сообщений коммитов и префиксы `build:`/`feat:`/`fix:`/`refactor:`/
`test:`/`docs:` (плюс правило «коммит только по явной просьбе») — скил
**`coding-conventions`** (`.opencode/skills/coding-conventions/SKILL.md`).

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`, `CMakeLists.txt`).
- The operator manages the process exclusively through high-level prompts and changes to instruction files.
- If manual code changes are detected outside AI sessions, perform a full diff audit and reconcile with the operator before proceeding.
- **ЗАПРЕТ на параллельных/субагентов**: использование Task-субагентов (explore/general/parallel agents) запрещено — они допускают много ошибок. Все исследования кодовой базы, аудит и правки выполняются только основной сессией напрямую (Read/Grep/Edit/etc.).
