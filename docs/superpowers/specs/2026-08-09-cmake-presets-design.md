# CMakePresets for Windows 11 & Ubuntu 26.04 — Design Spec

## 1. Overview

Настройки сборки по умолчанию для двух целевых ОС — Windows 11 (Visual Studio
2026, multi-config) и Ubuntu 26.04 (Ninja + GCC) — через единый `CMakePresets.json`
в корне репозитория. Цель: убрать ручные флаги/переменные окружения из
ежедневного флоу разработки.

Сейчас сборка требует вручную (README/AGENTS.md):

- Windows: `-G "Visual Studio 18 2026"` + `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"` (env-var теряется после configure, ставится перед каждой сборкой) + `--config Debug` + `-C Debug`.
- Ubuntu: `-DCMAKE_BUILD_TYPE=Debug` + `-march=native` (внутренний флаг, не меняется).

Проект: два пресета (Debug/Release) на каждую ОС, команды сводятся к
`cmake --preset <name>` / `cmake --build --preset <name>` / `ctest --preset <name>`.

## 2. Constraints

- **Отсутствие нового кода** — только конфигурационные и документационные файлы.
- `CMakePresets.json` едет в git; личные оверрайды — `CMakeUserPresets.json` (уже в `.gitignore`, правило сохраняется).
- `.gitignore` покрывает `/build/` — новые каталоги `build/windows-debug` и пр. автоматически игнорируются, запасного правила не требуется.
- Внутренние флаги сборки (`/arch:AVX512`, `-march=native`, Boost patch, CCCL-дефайны) не меняются — правки только во внешнем слое вызова.
- `CMAKE_POLICY_VERSION_MINIMUM=3.5` решается через cache-variable в пресете, а не env-var; ручная установка env более не требуется.

## 3. CMakePresets.json

Новый файл `CMakePresets.json` в корне, `version: 6` (CMake 3.25+).

### Configure-пресеты

| name | displayName | generator | arch | binaryDir | cacheVariables |
|---|---|---|---|---|---|
| `windows-debug` | Windows 11 (VS 2026) - Debug | `Visual Studio 18 2026` | `x64` | `${sourceDir}/build/windows-debug` | `CMAKE_POLICY_VERSION_MINIMUM=3.5`, `BUILD_TESTING=ON` |
| `windows-release` | Windows 11 (VS 2026) - Release | `Visual Studio 18 2026` | `x64` | `${sourceDir}/build/windows-release` | `CMAKE_POLICY_VERSION_MINIMUM=3.5`, `BUILD_TESTING=ON` |
| `linux-debug` | Ubuntu 26.04 (Ninja/GCC) - Debug | `Ninja` | — | `${sourceDir}/build/linux-debug` | `CMAKE_BUILD_TYPE=Debug`, `BUILD_TESTING=ON` |
| `linux-release` | Ubuntu 26.04 (Ninja/GCC) - Release | `Ninja` | — | `${sourceDir}/build/linux-release` | `CMAKE_BUILD_TYPE=Release`, `BUILD_TESTING=ON` |

JSON-скелет (см. секцию «Реализация» для полного файла):

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },
  "configurePresets": [ ...4 записи... ],
  "buildPresets": [ ...4 записи... ],
  "testPresets": [ ...4 записи... ]
}
```

## 4. Build и Test пресеты

`buildPresets` — деривативы от configure-пресетов (используют их binaryDir через `configurePreset`):

```json
{ "name": "windows-debug", "configurePreset": "windows-debug" },
{ "name": "windows-release", "configurePreset": "windows-release" },
{ "name": "linux-debug", "configurePreset": "linux-debug" },
{ "name": "linux-release", "configurePreset": "linux-release" }
```

`testPresets` — деривативы от build-пресетов (наследуют buildDir) с `output.outputOnFailure: true`:

```json
{ "name": "windows-debug", "configurePreset": "windows-debug",
  "output": { "outputOnFailure": true } }
// ... и остальные 3
```

`jobCount` в buildPresets не задаётся — CMake использует дефолтные потоки.

## 5. Команды разработчика

| Действие | Windows 11 | Ubuntu 26.04 |
|---|---|---|
| Configure | `cmake --preset windows-debug` | `cmake --preset linux-debug` |
| Build | `cmake --build --preset windows-debug` | `cmake --build --preset linux-debug` |
| Test | `ctest --preset windows-debug` | `ctest --preset linux-debug` |

- `$env:CMAKE_POLICY_VERSION_MINIMUM` больше не нужен на Windows — политика задаётся cache-переменной в пресете.
- `--config` / `-C` не нужны: multi-config generator берёт конфиг из имени buildPreset; для Release заменить `debug` → `release` в имени пресета.
- Выполняемые файлы на Windows: `build/windows-debug/bin/<config>/`, на Linux: `build/linux-debug/bin/`.

## 6. Обновление документации

### README.md

Секции «Build» и «Test» заменяются на пресет-флоу:

- Windows: `cmake --preset windows-debug` + `cmake --build --preset windows-debug`.
- Ubuntu: `cmake --preset linux-debug` + `cmake --build --preset linux-debug`.
- Тесты: `ctest --preset windows-debug` / `ctest --preset linux-debug`.
- Примечание: заменить `debug` → `release` для Release-сборки; `BUILD_TESTING=ON` уже в пресетах.
- Удаляется всё упоминание `$env:CMAKE_POLICY_VERSION_MINIMUM` и `-G "Visual Studio 18 2026"` из команд.
- Раздел «Prerequisites»: явно указать CMake ≥ 3.25 (для presets), GCC 13+ (Ubuntu 26.04) и наличие Ninja (Linux) / VS 2026 (Windows).

### AGENTS.md

- Разделы Build/Test для Windows и Linux приводятся к пресет-командам.
- Пункт про `CMAKE_POLICY_VERSION_MINIMUM` в «CMake Key Points» переформулировать: переменная задаётся через `CMakePresets.json` (cacheVariables), ручная установка env не требуется.
- Прочее (флаги компилятора, Boost patch, требования CUDA) не меняется.

## 7. Non-goals

- Build-скрипты (`build.ps1` / `build.sh`), CI (.github/workflows), .editorconfig / .clang-format / VS Code настройки.
- Изменение внутренних флагов компиляции и структуры CMakeLists.
- Поддержка macOS / ARM / не-NVIDIA GPU.

## 8. Реализация (ориентир для плана)

`CMakePresets.json` (полный файл) — 4 configure, 4 build, 4 test пресета по скелетам выше. Правки только в `README.md` и `AGENTS.md`. Проверка:

- Windows: `cmake --preset windows-debug` (новый каталог `build/windows-debug`), `cmake --build --preset windows-debug`, `ctest --preset windows-debug` — все тесты проходят.
- Ubuntu: те же команды с `linux-debug`.
- `cmake --list-presets` показывает 12 пресетов.