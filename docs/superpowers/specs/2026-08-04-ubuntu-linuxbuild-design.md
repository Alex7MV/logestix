# Адаптация сборки под Ubuntu 26.04 + максимальная CPU-оптимизация

Дата: 2026-08-04
Ветка: `feat/linuxbuild`
Статус: approved

## Цель

Адаптировать CMake-сборку проекта под Ubuntu 26.04 LTS и добиться максимальной
производительности на целевой машине: AMD EPYC 9V74 (Zen 4 / Genoa-X, полный
набор AVX-512 включая bf16/vnni/vbmi2). Привязка только к этой машине —
переносимость не требуется. Windows-сборка (MSVC) должна остаться нетронутой.

## Окружение

- Ubuntu 26.04 LTS, GCC 15.2.0, CMake 4.2.3, CUDA 13.2
- CPU: AMD EPYC 9V74 — `znver4`

## Текущее состояние

- `CMakeLists.txt:30` — Linux: `-march=native` в `CMAKE_CXX_FLAGS`
- `libs/lsxcommon/CMakeLists.txt:49` — безусловный `-Xcompiler=/utf-8` (MSVC-флаг,
  ломает GCC)
- LTO не включён нигде

## Дизайн

### 1. Глобальные Linux-флаги (root `CMakeLists.txt`, ветка `else()`)

Заменить `-march=native` на детерминированную привязку `znver4` + агрессивную
оптимизацию:

```cmake
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -march=znver4 -mtune=znver4 -O3 -funroll-loops -fno-plt -fno-semantic-interposition")
endif()
```

- `-march=znver4` включает весь AVX-512 автоматически (bf16, vnni, vbmi2).
- `-O3` явно — работает при любом `CMAKE_BUILD_TYPE`.
- Флаги в `CMAKE_CXX_FLAGS` применяются и к нашему коду, и к bundled Arrow /
  simdjson (решение: «везде»).
- Ветка `MSVC` не изменяется.

### 2. LTO — только наши C++-цели, через CMake API

Без глобального `-flto` (зацепил бы CUDA и третьи библиотеки):

```cmake
include(CheckIPOSupported)
option(LOGESTIX_LTO "Enable LTO for our C++ targets" ON)
if(LOGESTIX_LTO)
    check_ipo_supported(RESULT _lto_ok OUTPUT _lto_err)
    if(_lto_ok)
        set_property(TARGET lsxcommon APPEND PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        # + каждый app в apps/*
    endif()
endif()
```

- `INTERPROCEDURAL_OPTIMIZATION TRUE` только на `lsxcommon` и `apps/*`.
- `lsxcommon_cu` (NVCC-объекты) и third-party код исключены.
- Переключатель `LOGESTIX_LTO` (по умолчанию ON) как фолбэк.

### 3. Хост-код CUDA (`libs/lsxcommon/CMakeLists.txt`)

Исправить нерабочую на Linux строку 49 и добавить хост-флаги NVCC:

- Обернуть `-Xcompiler=/utf-8` в `if(MSVC)`.
- Для Linux добавить хост-оптимизацию без LTO:
  ```cmake
  target_compile_options(lsxcommon_cu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=znver4;-Xcompiler=-mtune=znver4;-Xcompiler=-O3>)
  ```

### 4. Опции CMake

- `LOGESTIX_LTO` (ON) — переключатель LTO.
- `LOGESTIX_FAST_MATH` (OFF, выключен по умолчанию) — `-ffast-math`.
  Не включаем по умолчанию: рискованно для точности NVFP4-декванта.

## Проверка

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Вне объёма

- Windows/MSVC-ветка флагов — без изменений.
- Рантайм-диспетчеризация SIMD (x86-64-v3 + AVX-512) — не требуется.
- `-ffast-math` — выключен по умолчанию.
