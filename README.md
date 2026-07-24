# logestix

An enterprise-grade industrial inference system designed for high-performance AI workloads.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 16.1.0 | FetchContent (BUNDLED) | Shared |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| NVIDIA CCCL | latest | submodule (`third_party/cccl`) | Header-only |

## Prerequisites

- CMake 3.15+
- C++17/C11 compiler (GCC 9+, Clang 12+, MSVC 2019+)
- Git

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-org>/logestix.git
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Build

### Linux (GCC/Clang)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Windows (Visual Studio 2026)

> **Note:** Arrow's xsimd dependency requires `CMAKE_POLICY_VERSION_MINIMUM=3.5`
> for CMake 4.x compatibility. Set the environment variable before building.

```powershell
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026"
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
```

The executable `lsxfabric.exe` and all required DLLs are placed in `build/bin/Debug/`.

## Test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -DBUILD_TESTING=ON
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug
```

## Structure

- `src/core.cpp` — engine core; object library `lsxfabric_objects`
- `src/main.cpp` — entry point; executable `lsxfabric`
- `tests/test_core.cpp` — Catch2 v3 tests; executable `test_core`
- `third_party/` — Git submodules (cccl, flatbuffers, libzmq)
- `include/lsxfabric/` — public headers

## Licensing

This project is dual-licensed to accommodate both open-source development and commercial production environments.

### Open Source License
The open-source version of **logestix** is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE).

### Commercial License (Enterprise)
For commercial deployments, enterprise infrastructure, closed-source integration, or hosted services, the AGPL-3.0 limitations can be removed.

To obtain a commercial license or request a private evaluation copy (PoC), please contact us at: **<your-sales-email@example.com>**

---

## Contributing

We welcome community contributions! However, to ensure we can protect our dual-licensing model, all contributors are required to sign our **Contributor License Agreement (CLA)** before any Pull Request can be merged.

The CLA signing process is automated via GitHub as soon as you open a Pull Request.
