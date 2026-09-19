# Dependencies

This file records the provenance of every third-party dependency used by
logestix: pinned version, source repository, integration mode, and whether
the upstream source is modified locally. The corresponding license texts
and copyright notices are collected in [`THIRD_PARTY_NOTICES`](./THIRD_PARTY_NOTICES).

Reproduce the exact submodule pins with:

```bash
git submodule status
git ls-tree HEAD third_party/
```

---

## libzmq

| | |
|---|---|
| **Version** | 4.3.5 (`ZMQ_VERSION_MAJOR.Minor.Patch = 4.3.5`) |
| **Source** | https://github.com/zeromq/libzmq.git |
| **Integration** | Git submodule `third_party/libzmq`, built statically via `add_subdirectory(third_party/libzmq)` |
| **Pinned commit** | `622fc6dde99ee172ebaa9c8628d85a7a1995a21d` |
| **License** | Mozilla Public License 2.0 (MPL-2.0) |
| **Local modifications** | **None** |

### No modifications

logestix does **not** modify libzmq. The submodule is consumed as an
unmodified upstream source tree; no patches, file rewrites, or source
edits are applied to it. Only build-time CMake cache options are set
before `add_subdirectory`, and they do not alter the source:

- `BUILD_SHARED=OFF`
- `BUILD_STATIC=ON`
- `BUILD_TESTS=OFF`
- `WITH_PERF_TOOL=OFF`
- `WITH_LIBSODIUM=OFF`
- `ENABLE_DRAFTS=OFF`

Because the pinned source is unmodified, the Corresponding Source for the
MPL-2.0-covered files is exactly the upstream libzmq v4.3.5 tree at the
commit above, available from the repository link above or the matching
`third_party/libzmq` submodule checkout.

### Source disclosure (MPL-2.0)

MPL-2.0 is a file-level copyleft license: proprietary code may be linked
against libzmq, but the MPL-covered source must remain available and may
be relicensed only under MPL-2.0. Whenever logestix binaries are
distributed, ship or offer the libzmq Corresponding Source:

1. Since the pinned tree is unmodified, the Corresponding Source is the
   upstream v4.3.5 tarball or the `third_party/libzmq` submodule at
   commit `622fc6dde99ee172ebaa9c8628d85a7a1995a21d`.
2. Reproduce the MPL-2.0 text and the notices from
   [`THIRD_PARTY_NOTICES`](./THIRD_PARTY_NOTICES) section 4 alongside the
   binary (this is already automated: `THIRD_PARTY_NOTICES` is copied
   next to every executable and into `install/bin`).
3. Keep the copyright/attribution headers of libzmq and its bundled
   `external/` components intact; do not relicense libzmq code under the
   logestix commercial EULA.

Fetch the exact source for a release with:

```bash
git submodule update --init third_party/libzmq
git -C third_party/libzmq archive --format=tar.gz -o libzmq-4.3.5.tar.gz \
   622fc6dde99ee172ebaa9c8628d85a7a1995a21d
```

### Bundled libzmq components

libzmq vendors a few small components with separate permissive licenses
(texts are reproduced in `THIRD_PARTY_NOTICES`):

- `external/unity` — MIT
- `external/wepoll` — BSD-2-Clause
- `external/sha1` — BSD-3-Clause (WIDE Project)

---

## Dependency inventory

| Dependency | Version | Source | Integration | Local modifications |
|---|---|---|---|---|
| Apache Arrow | 25.0.1 | https://github.com/apache/arrow.git (`apache-arrow-25.0.1`) | FetchContent (shared) | no |
| NVIDIA CCCL (Thrust/libcu++/cudax/CUB) | 3.4.2 | https://github.com/NVIDIA/cccl.git | submodule `third_party/cccl` (`d3601220…`) | no |
| FlatBuffers | v25.12.19 | https://github.com/google/flatbuffers.git | submodule `third_party/flatbuffers` (`7e163021…`) | no |
| libzmq | 4.3.5 | https://github.com/zeromq/libzmq.git | submodule `third_party/libzmq` (`622fc6dd…`) | no |
| simdjson | v4.6.11 | https://github.com/simdjson/simdjson.git | submodule `third_party/simdjson` (`f5de14f0…`) | no |
| spdlog | 1.17.0 | https://github.com/gabime/spdlog.git | FetchContent | yes — bundled `fmt/format.h` patched for NVCC 13.x |
| abseil-cpp | 20260817.0 | https://github.com/abseil/abseil-cpp.git | FetchContent | yes — `crc32_x86_arm_combined_simd.h`, `str_format/checker.h` |
| Catch2 | v3.16.0 | https://github.com/catchorg/Catch2.git | FetchContent (test-only) | no |
| Boost | 1.88.0 (bundled by Arrow) | https://github.com/boostorg/boost.git | FetchContent (transitive) | yes — `multiprecision/.../literals.hpp` |
| xsimd | (bundled by Arrow) | https://github.com/xtensor-stack/xsimd.git | FetchContent (transitive) | no |
| Dear ImGui | v1.92.9-docking | https://github.com/ocornut/imgui.git | FetchContent (`lsxhome`, Windows) | no |
| ImPlot | v1.0 | https://github.com/epezent/implot.git | FetchContent (`lsxhome`, Windows) | no |
| ImNodes | v0.5 | https://github.com/Nelarius/imnodes.git | FetchContent (`lsxhome`, Windows) | yes — `imnodes.cpp` texture-ref fix |
| FreeType | 2.14.3 (`VER-2-14-3`) | https://gitlab.freedesktop.org/freetype/freetype.git | FetchContent (`lsxhome`, Windows) | yes — `ftoption.h` subpixel rendering |
| Inter | — | https://github.com/rsms/inter | asset `apps/lsxhome/assets/Inter-Regular.ttf` | no |
| JetBrains Mono | — | https://github.com/JetBrains/JetBrainsMono | asset `apps/lsxhome/assets/JetBrainsMono-Regular.ttf` | no |
| Unicode Character Database | 16.0.0 | https://www.unicode.org/Public/16.0.0/ucd | generated `libs/lsxcommon/src/unicode_tables.inc` | derived data (see `tools/generate_unicode_tables.ps1`) |

### Notes on modified dependencies

All local modifications above are limited to the permissive licenses of
those components (Apache-2.0, MIT, BSD, FTL) and are limited to build/toolchain
compatibility. They do not affect libzmq, which remains unmodified as
required for MPL-2.0 compliance.
