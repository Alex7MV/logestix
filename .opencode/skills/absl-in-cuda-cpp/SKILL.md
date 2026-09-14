---
name: absl-in-cuda-cpp
description: Use when choosing containers or strings in logestix host C++, adding includes to headers reachable from .cu/.cuh, hitting abseil compile errors under NVCC (C2794/C2938 in raw_hash_map), or writing absl::StrFormat calls. Covers the std:: to absl priority table, the NVCC container incompatibility and its reachability rule, the printf-vs-fmt format contract, and the CMake wiring.
---

# Abseil in a CUDA C++ project

## Overview

Two independent rules plus build wiring. Both rules exist because NVCC — not
the host compiler — compiles the same tree:

1. **Host C++ prefers `absl::` containers/strings.**
2. **Device TUs (`\*.cu`/`\*.cuh`) must never see an abseil container.**

## 1. Container and string policy (host-only C++)

| Priority | `std::` | → `absl::` | Header |
|---|---|---|---|
| MUST | `std::unordered_map` | `absl::flat_hash_map` | `<absl/container/flat_hash_map.h>` |
| MUST | `std::unordered_set` | `absl::flat_hash_set` | `<absl/container/flat_hash_set.h>` |
| MUST | `std::map` | `absl::btree_map` | `<absl/container/btree_map.h>` |
| MUST | `std::set` | `absl::btree_set` | `<absl/container/btree_set.h>` |
| MAY | `std::ostringstream` (simple joins) | `absl::StrCat` / `absl::StrAppend` | `<absl/strings/str_cat.h>` |

Stays `std::` (do not "fix" these):

- `std::string_view`, `std::optional`, `std::variant`, `std::function`,
  `std::unique_ptr` — for C++17+ abseil itself recommends the `std::` versions;
  the absl ones are polyfills.
- Streaming JSON/report builders with loop `setw`/`setfill` — a legitimate
  stream use (reference: `ConvertQwenModel`'s `model.index.json` builder in
  `qwen_dense_convert.cpp`).
- Custom hasher functors (`ExpertKeyHash`, `GlmExpertKeyHash`) — they are
  passed as the `Hasher` template argument directly.

## 2. The NVCC incompatibility (critical)

Abseil containers (`flat_*`, `btree_*`, everything under
`absl/container/internal/*`) do **not** compile under NVCC: `raw_hash_map`
breaks inside a `.cu` with MSVC-in-NVCC `C2794`/`C2938`.

**Rule:** never let `#include <absl/container/*>` become transitively
reachable from a `.cu`/`.cuh` translation unit. A header is "safe for device"
iff its whole include closure is free of absl container headers.

- Device TUs keep `std::` containers and streams.
- `absl/strings` is fine for device TUs: linking `absl::strings` into the CUDA
  OBJECT libraries (`lsxcommon_cu` / `lsxcommon_glm_cu` / `lsxcommon_qwen_moe_cu`
  / the kimi TC lib) grants include paths **without** pulling container headers
  into `.cu`.
- **No guard automates this yet** — verify by hand when you add includes.
  Verified state (2026-09-14): exactly two headers include
  `<absl/container/...>` — `include/lsxcommon/lsx_orchestrator.h` and
  `include/lsxcommon/model_integrity.h` (both host-only) — and **zero**
  `.cu`/`.cuh` units reach them; all other including units are `.cpp`.

Verification recipe (transitive include closure over `libs/`, `apps/`, `tests/`;
resolves includes by basename, so it over-approximates — 0 hits is sound):

```bash
python3 - <<'PY'
import re, pathlib, collections
roots = [pathlib.Path(r) for r in ('libs','apps','tests')]
files = [p for r in roots if r.exists() for p in r.rglob('*')
         if p.suffix in ('.h','.hpp','.cuh','.cu','.cpp')]
inc_re = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)
def deps(p):
    return [q for q in files for m in inc_re.findall(p.read_text(errors='ignore'))
            if q.name == pathlib.Path(m).name]
def closure(p):
    seen, st = set(), [p]
    while st:
        f = st.pop()
        if f in seen: continue
        seen.add(f); st += deps(f)
    return seen
absloop = {p for p in files if re.search(r'#\s*include\s*<absl/container',
                                         p.read_text(errors='ignore'))}
for p in files:
    if p.suffix in ('.cu','.cuh') and (closure(p) & absloop):
        print('VIOLATION', p)
PY
```
`rg 'include "[^"]+"' libs/lsxcommon/src/*.cu` is the quick eyeball version of
the same check.

Consider automating this as a guard next to `test_absl_format_guard.cpp` if it
ever regresses (host-only static scan; label `fast`).

## 3. Format strings: printf-style, never `{}`

`absl::StrFormat` / `StrAppendFormat` / `StreamFormat` / `Format` take
**printf** conversions. A `{}` prints literally as `{}` — silently wrong.

- Conversions: string → `%s` (a `path` → `.string()`), signed int (incl.
  `int64_t`) → `%d`, unsigned (incl. `size_t`) → `%u`, float/double →
  `%f`/`%.Nf`. Specs: `{:03d}`→`%03d`, `{:<N}`→`%-Ns`, `{:>N}`→`%Ns`,
  `{:.Nf}`→`%.Nf`.
- The abseil **compile-time** checker does not save you: GCC leaves
  `ABSL_INTERNAL_ENABLE_FORMAT_CHECKER` undefined (no
  `__attribute__((enable_if))`), and the root CMakeLists patches the checker to
  `#if 0 &&` for Clang/AOCC (see §4) because its `ValidFormatImpl` understands
  only `%` conversions. Enforcement is therefore the runtime/static guard
  `tests/test_absl_format_guard.cpp` (scans `libs/` and `apps/`).

## 4. CMake wiring (root `CMakeLists.txt`)

- FetchContent, pinned LTS tag **`20260526.0`**; `ABSL_BUILD_TESTING` /
  `_TEST_HELPERS` / `_EXAMPLES` off; **`ABSL_PROPAGATE_CXX_STD ON`** (CACHE BOOL
  FORCE) so every abseil target inherits the project's C++20.
  `CMAKE_CXX_STANDARD`/`CMAKE_CXX_FLAGS` are forced to CACHE because abseil's
  nested probe `try_compile` needs `/std:c++20` to define
  `ABSL_INTERNAL_AT_LEAST_CXX20` in the under-project.
- **CRC patch** (idempotent, applied every configure because the pinned tag
  re-fetches): the project-wide MSVC `/arch:AVX512` defines `__AVX__`, which
  activates abseil's GCC/Clang-only VPCLMULQDQ inline-asm path in
  `absl/crc/internal/crc32_x86_arm_combined_simd.h`, and MSVC cannot parse that
  `__asm__`. The patch appends `&& 0` to the MSVC/`_M_AMD64` SIMD branch so CRC
  falls back to the generic intrinsics.
- **Format-checker patch** (idempotent): rewrites
  `#if ABSL_HAVE_ATTRIBUTE(enable_if) && !defined(__INTELLISENSE__)` to
  `#if 0 && ...` in `absl/strings/internal/str_format/checker.h`, matching GCC
  behaviour for Clang/AOCC. **Do not remove it** — an AOCC build treats a stray
  `{}` as a hard compile error without it.
- Links: `absl::flat_hash_map` + `absl::span` + `absl::strings` are PUBLIC on
  `lsxcommon` (and `lsxhome`; `apps/lsxhome` is PRIVATE);
  `absl::flat_hash_map` is PUBLIC on `lsxnetwork`; each CUDA OBJECT library
  links `absl::strings`.

## Migration recipe

1. Replace the container and its header: `unordered_map`→`flat_hash_map`,
   `unordered_set`→`flat_hash_set`, `map`→`btree_map`, `set`→`btree_set`.
2. Keep the element type and any custom hasher argument as-is.
3. Replace simple `ostringstream` joins with `absl::StrCat`/`StrAppend`; leave
   `setw`/`setfill` builders alone.
4. If the file is (or feeds) a device TU, stop — containers stay `std::`; move
   the absl-using code into a host `.cpp` behind a header that has no absl
   includes.

## Common mistakes

- Adding `#include <absl/container/flat_hash_map.h>` to a header that a `.cu`
  transitively includes → NVCC `C2794`/`C2938` in `raw_hash_map`.
- "Fixing" `std::string_view`/`std::optional`/`std::unique_ptr` into absl
  polyfills.
- Writing `absl::StrFormat("x=%s", path)` with a `path` — needs `.string()`.
- Using `{}` placeholders in `absl::StrFormat` and shipping a literal `{}`.
- Removing the CMake format-checker patch because "the build worked once".
