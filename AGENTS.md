# logestix — AGENTS.md

## Project

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C23. CMake 3.15+.
Target architecture: NVIDIA Blackwell (CC 12.0 / 12.1). Active use of 5th-gen Tensor Cores and TMA.
Max warps per SM on Blackwell: 48 (factor into occupancy calculations).
Forbidden architectures: Hopper (90), Ada (89), Ampere (80).
No backward compatibility — any code targeting pre-Blackwell (sm_90, sm_80) is rejected.
CUDA Toolkit 13.3+ (nvcc 13.3), full native C++23 support in NVCC.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.0 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.4.2 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | 3.3 (latest) | submodule (`third_party/cccl`) | Header-only |
| Catch2 | v3.7.0 | FetchContent (test only) | Header-only |

## Build

### Linux (GCC/Clang)

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Windows (Visual Studio 2026)

Arrow's xsimd sub-dependency requires `CMAKE_POLICY_VERSION_MINIMUM=3.5`
for CMake 4.x. Set the env var before `cmake --build` (it's lost after configure).

```powershell
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026"
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD`.

## Test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows без `-DCMAKE_BUILD_TYPE` (используется `--config Debug`):

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug
```

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `libs/lsxcommon/src/core.cpp` — ядро (логирование, инициализация)
  - `libs/lsxcommon/include/lsxcommon/core.h` — публичный заголовок
  - `libs/lsxcommon/src/common.cpp` — version + GPU info (spdlog)
  - `libs/lsxcommon/src/schema.cpp` — ExpertSchema (9 полей: weights, E4M3 per-block scales uint8[], float32 global_scale), PackedExpert
  - `libs/lsxcommon/src/converter.cpp` — GPU-only NVFP4 квантизация (per-block E4M3 + global_scale), парсинг safetensors, сериализация в Arrow IPC
  - `libs/lsxcommon/src/loader.cpp` — загрузка экспертов из .arrow, CUDA pinned регистрация, E4M3 scale decoding
  - `libs/lsxcommon/src/safetensors_gpu.cu` — GPU-ускоренная квантизация (3 CUDA-ядра: convert, block_maxabs, quantize_pack_perblock; CCCL thrust для global max)
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`)
  - `apps/lsxfabric/src/validator.cpp` — валидатор NVFP4/E4M3/global-scale в собранной модели
  - `apps/lsxfabric/tests/` — 12 test cases (1 lifecycle + 11 validator)
- `apps/lsxforge/` — data preprocessing utility (placeholder)
  - `apps/lsxforge/tests/` — 1 test case, 1 assertion
- `apps/taskmosaic/` — task orchestrator (placeholder)
  - `apps/taskmosaic/tests/` — 1 test case, 1 assertion
- `apps/lsxcli/` — CLI client (`--model`)
  - `apps/lsxcli/tests/` — 1 test case, 1 assertion
- `tests/test_core.cpp` — интеграционные тесты; 11 test cases, ~100 assertions
- `tests/test_loader.cpp` — 3 test cases — round-trip, empty fields, E4M3 encoding (69 assertions)
- `third_party/cccl/` — NVIDIA CCCL (сабмодуль)
- `third_party/flatbuffers/` — Google FlatBuffers v25.12.19 (сабмодуль)
- `third_party/libzmq/` — ZeroMQ libzmq v4.3.5 (сабмодуль)
- `third_party/simdjson/` — simdjson v4.4.2 (сабмодуль)

## CMake Key Points

- `lsxcommon` — STATIC library в `libs/lsxcommon/CMakeLists.txt`; линкуется PUBLIC с `spdlog::spdlog`, `flatbuffers`, `libzmq-static`, `cudart`; PRIVATE с `arrow_shared`, `arrow_cuda_shared`, `simdjson::simdjson`
- CUDA `.cu` файл вынесен в отдельную OBJECT библиотеку `lsxcommon_cu` (линкуется в `lsxcommon` PUBLIC) — это изолирует NVCC от MSVC-флагов
- Each app in `apps/*/` links `lsxcommon` PRIVATE
- Catch2 v3.7.0 fetched at root level (via FetchContent) when `BUILD_TESTING=ON`
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` установлен в `${CMAKE_BINARY_DIR}/bin` (до `add_subdirectory`)
- `add_custom_command(TARGET ... POST_BUILD)` копирует `arrow.dll` + `arrow_cuda.dll` в директорию exe (только Win32); applies to all 4 app targets, all 5 test targets
- **MSVC flags**: `/W4 /utf-8 /MP /arch:AVX512` через `CMAKE_CXX_FLAGS` (НЕ `add_compile_options`, чтобы не попадали в NVCC). `/arch:AVX512` включает AVX-512F/BW/CD/DQ/VL. `WIN32_LEAN_AND_MEAN NOMINMAX` (suppress Windows macro conflicts with CCCL). DO NOT use `/Zc:preprocessor` (breaks CCCL/thrust). Use `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` + `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP` for CCCL/MSVC compat.
- **GCC/Clang flags**: `-march=native` в `CMAKE_CXX_FLAGS` (а не `add_compile_options`, чтобы не протекало в NVCC). Включает все ISA + тюнинг хостового CPU.
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

## Git

- Английский язык сообщений коммитов
- Префиксы: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`, `CMakeLists.txt`).
- The operator manages the process exclusively through high-level prompts and changes to instruction files.
- If manual code changes are detected outside AI sessions, perform a full diff audit and reconcile with the operator before proceeding.
