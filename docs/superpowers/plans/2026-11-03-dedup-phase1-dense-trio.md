# Фаза 1: dense-трио → `lsxcommon::dense` — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. **Subagents are forbidden in this repository** (AGENTS.md § Autonomous Mode): every step is executed inline by the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перенести семь идентичных `__global__`-ядер и их лаунчеры из `granite_dense_inference.cu` в новый нейтральный модуль `lsxcommon::dense`, переключить на него granite, granite-switch и phi и удалить две копии — без изменения единого бита.

**Architecture:** Новый модуль `include/lsxcommon/lsx_dense_inference.h` + `src/lsx_dense_inference.cu` (отдельная OBJECT-библиотека `lsxcommon_dense_cu`) содержит алгоритм, который сегодня исполняет granite. Семейные заголовки сохраняют публичный API: `RunGranite*`, `RunGraniteSwitch*`, `RunPhi*` становятся inline-делегатами на `dense::*`. Драйверы слоёв и KV-сессий остаются семейными.

**Tech Stack:** C++20, CUDA 13.x, NVCC, CMake presets (`linux-debug`), Catch2 v3.16.1, ctest-пресеты с label-фильтрами.

**Spec:** `docs/superpowers/specs/2026-11-03-code-duplication-consolidation-design.md` §2.2, §4, §5 (Ф1), §7.

## Global Constraints

- Критерий приёмки — **побитовая идентичность**: `logits_hash` granite (`c165e80fde6401cf`), granite-switch (`fdc632042c30919c`) и снятый baseline phi не меняются; FSI/CODE/DET остаются зелёными; perf-строки не выходят за шум. Пере-сид `tests/perf/perf_baselines.json` запрещён.
- **Перенос, а не переписывание**: тела ядер, guard-ветки, константы и коды ошибок переносятся дословно. Ни одна формула и ни один порог не редактируются.
- Нейтральный слой не знает паспортов, format-тегов и имён семейств; имена ядер — `dense_*`, без семейных стемов.
- Заголовки, достижимые из `.cu`, не включают `<absl/container/*>` (NVCC падает в `raw_hash_map`: `C2794`/`C2938`).
- Сборка обязана давать **0 предупреждений** в наших файлах (предупреждения vendored-зависимостей не считаются).
- Коммиты: английские, префиксы `refactor:`/`test:`/`docs:`. Никаких комментариев в коде, кроме уже существующих или объясняющих неочевидный инвариант. Логирование только через spdlog.
- Никаких фоновых/параллельных субагентов; все правки — в текущей сессии.
- Модельные деревья уже сконвертированы в vault; пере-конвертация не требуется.

---

### Task 1: Нейтральный модуль — заголовок, скелет, инструмент доказательства и падающий тест

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/lsx_dense_inference.h`
- Create: `libs/lsxcommon/src/lsx_dense_inference.cu`
- Create: `tools/dedup_provenance.py`
- Create: `tests/test_dense_kernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (новая OBJECT-библиотека + список линкуемых объектов)
- Modify: `tests/CMakeLists.txt` (регистрация `test_dense_kernels`)

**Interfaces:**
- Produces: `lsxcommon::dense::RunDenseRmsNormKernel`, `RunDenseRopeKernel`, `RunDenseAttentionKernel`, `RunDenseAttentionSplitKernel`, `RunDenseSplitQKVKernel`, `RunDenseSwiGLUKernel` — точные сигнатуры в шаге 1.
- Consumes: ничего из предыдущих задач.

- [ ] **Step 1: Написать заголовок нейтрального слоя**

Создать `libs/lsxcommon/include/lsxcommon/lsx_dense_inference.h`:

```cpp
#pragma once

// Model-neutral dense-decoder kernel layer (RMSNorm, RoPE, causal GQA
// attention with the split-KV flash-decode path, fused split-QKV, fused
// SwiGLU). The algorithm is the one Granite-4.2-30B has always used; the
// consumers (granite, granite-switch, phi-4) keep their own layer drivers and
// only delegate here. Header must stay free of abseil containers: it is
// included from .cu translation units.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace lsxcommon {
namespace dense {

// out = x * rsqrt(mean(x^2) + eps) * w, row-wise over `hidden`.
cudaError_t RunDenseRmsNormKernel(cudaStream_t stream, __half *out,
                                  const __half *x, const __half *w, int rows,
                                  int hidden, float eps);

// Full-dimension rotate-half RoPE; positions are absolute.
cudaError_t RunDenseRopeKernel(cudaStream_t stream, __half *q, __half *k,
                               const int32_t *pos, int seq, int nh, int nkv,
                               int hd, double theta);

// Causal GQA attention with a persistent KV cache: split-KV when the caller
// provides `partials` and the shape pays off, legacy block kernel otherwise.
cudaError_t RunDenseAttentionKernel(cudaStream_t stream, const __half *q,
                                    const __half *k, const __half *v,
                                    __half *out, float *partials, int seq,
                                    int nh, int nkv, int hd, float scale,
                                    int kv_base);

cudaError_t RunDenseAttentionSplitKernel(cudaStream_t stream, const __half *q,
                                         const __half *k, const __half *v,
                                         __half *out, float *partials, int seq,
                                         int nh, int nkv, int hd, float scale,
                                         int kv_base, int splits);

// Unpack an interleaved [seq, qn + 2*kn] fused projection.
cudaError_t RunDenseSplitQKVKernel(cudaStream_t stream, const __half *qkv,
                                   __half *q, __half *k, __half *v, int seq,
                                   int qn, int kn);

// out = silu(g) * u elementwise over `n`, where g/u are interleaved in `gu`.
cudaError_t RunDenseSwiGLUKernel(cudaStream_t stream, const __half *gu,
                                 __half *out, int I, int64_t n);

} // namespace dense
} // namespace lsxcommon
```

- [ ] **Step 2: Написать падающий нейтральный тест**

Создать `tests/test_dense_kernels.cu` — перенести четыре кейса из `tests/test_granite_dense_kernels.cu` (строки 39–343: rmsnorm, vectorized rmsnorm, rope, causal GQA attention, split-KV attention) и заменить вызовы `RunGranite*` на `dense::RunDense*`. Начало файла:

```cpp
// Neutral dense-decoder kernel parity tests (Model Ingestion Protocol step 5).
// Host-only FP64/FP32 CPU references against the model-neutral launchers in
// lsxcommon::dense. The granite/phi smoke tests keep only what is family
// specific (weight layouts, fused projections).
#include <catch2/catch_all.hpp>

#include "test_kernel_budget.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <lsxcommon/lsx_dense_inference.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::dense;
namespace lsxg = lsxcommon::lsx_nvfp4;

namespace {

__half H(float v) { return __float2half(v); }

float lcg(int64_t &s)
{
    s = (s * 6364136223846793005LL + 1442695040888963407LL);
    return static_cast<float>((s >> 40) & 0xffff) / 65536.0f - 0.5f;
}

std::vector<__half> ToHalf(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

} // namespace
```

Тела кейсов переносятся дословно; меняются только имена тестов (`"granite rmsnorm kernel matches a CPU reference"` → `"dense rmsnorm kernel matches a CPU reference"`) и теги `[granite][kernels]` → `[dense][kernels]`.

- [ ] **Step 3: Зарегистрировать модуль и тест в CMake**

В `libs/lsxcommon/CMakeLists.txt` перед блоком `lsxcommon_granite_cu` добавить по образцу соседних OBJECT-библиотек:

```cmake
# Model-neutral dense-decoder kernels: RMSNorm / RoPE / causal GQA attention /
# fused split-QKV / fused SwiGLU. Algorithm frozen as the Granite-4.2 runtime
# always ran it; consumers delegate here.
add_library(lsxcommon_dense_cu OBJECT
    src/lsx_dense_inference.cu
)
set_target_properties(lsxcommon_dense_cu PROPERTIES
    CUDA_ARCHITECTURES "${LSX_CUDA_ARCH}"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_dense_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_dense_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_dense_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_dense_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)
```

Добавить `lsxcommon_dense_cu` в список OBJECT-целей, которые линкуются в `lsxcommon` (найти вызов `target_link_libraries(lsxcommon ...)` и добавить имя рядом с `lsxcommon_granite_cu`).

В `tests/CMakeLists.txt` рядом с регистрацией granite-тестов добавить:

```cmake
# ── Neutral dense-decoder kernel parity tests (.cu) ──
lsx_add_cuda_test(NAME test_dense_kernels
    SOURCES test_dense_kernels.cu
    LINK lsxcommon
    LABEL inference)
```

- [ ] **Step 4: Заложить инструмент доказательства «это перенос»**

Создать `tools/dedup_provenance.py`:

```python
#!/usr/bin/env python3
"""Level-0 gate for the deduplication phases: prove a move is a move.

Compares every top-level ``__global__`` definition in --from against every one
in --to after stripping comments, collapsing whitespace and applying the
``--rename old=new`` substitutions. A move is valid only when both sides
produce the same multiset of normalized bodies: every source kernel has an
identical twin in the target and the target adds nothing new.

    python3 tools/dedup_provenance.py \
        --from libs/lsxcommon/src/granite_dense_inference.cu \
        --to   libs/lsxcommon/src/lsx_dense_inference.cu \
        --rename granite_=dense_

Exit status 0 on a proven move, 1 otherwise (prints the unmatched bodies).
"""
import argparse
import re
import sys


def kernels(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("__global__"):
            start, depth, seen, body = i, 0, False, []
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                seen = seen or "{" in lines[i]
                body.append(lines[i])
                if seen and depth <= 0:
                    break
                i += 1
            out.append("".join(body))
        i += 1
    return out


def normalize(body, renames):
    for old, new in renames:
        body = body.replace(old, new)
    return re.sub(r"\s+", "", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--rename", action="append", default=[])
    args = ap.parse_args()
    renames = [tuple(r.split("=", 1)) for r in args.rename]

    src = [normalize(b, renames) for b in kernels(args.src)]
    dst = [normalize(b, renames) for b in kernels(args.dst)]
    missing = [b for b in src if b not in dst]
    extra = [b for b in dst if b not in src]
    print(f"source kernels: {len(src)}  target kernels: {len(dst)}")
    print(f"missing in target: {len(missing)}  unexpected in target: {len(extra)}")
    for body in missing:
        print("  MISSING:", body[:120])
    for body in extra:
        print("  EXTRA:  ", body[:120])
    return 0 if not missing and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Собрать и убедиться, что тест падает**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug --target test_dense_kernels 2>&1 | tail -20
```

Expected: FAIL на этапе сборки — `undefined reference to lsxcommon::dense::RunDenseRmsNormKernel` (модуль пуст). Тело `src/lsx_dense_inference.cu` на этом шаге содержит только `#include "lsxcommon/lsx_dense_inference.h"`.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/lsx_dense_inference.h \
        libs/lsxcommon/src/lsx_dense_inference.cu \
        libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt \
        tests/test_dense_kernels.cu tools/dedup_provenance.py
git commit -m "test(dense): scaffold the neutral dense layer and its parity test (RED)"
```

---

### Task 2: Перенести семь ядер и шесть лаунчеров

**Files:**
- Modify: `libs/lsxcommon/src/lsx_dense_inference.cu`
- Reference (read-only): `libs/lsxcommon/src/granite_dense_inference.cu:36-341`, `:397-493`

**Interfaces:**
- Consumes: заголовок из Task 1.
- Produces: рабочие `dense::RunDense*` — точные сигнатуры уже зафиксированы в заголовке.

- [ ] **Step 1: Перенести ядра дословно**

Взять из `granite_dense_inference.cu` блок `namespace { ... }` строками 36–341 (константы `kThreads`, раздел `Kernels`, `kAttnSplitMax`/`kAttnSplitKeys`, split-KV ядра, `granite_swiglu_fused_kernel`, `granite_split_qkv_kernel`, `granite_rmsnorm_kernel` со строки 354) и вставить в `lsx_dense_inference.cu` внутри `namespace lsxcommon { namespace dense { namespace { ... } } }`.

Переименования (только имена, тела не трогать):

```text
granite_rope_kernel            -> dense_rope_kernel
granite_attn_kernel            -> dense_attn_kernel
granite_attn_split_kernel      -> dense_attn_split_kernel
granite_attn_combine_kernel    -> dense_attn_combine_kernel
granite_swiglu_fused_kernel    -> dense_swiglu_fused_kernel
granite_split_qkv_kernel       -> dense_split_qkv_kernel
granite_rmsnorm_kernel         -> dense_rmsnorm_kernel
```

Шапка файла:

```cpp
// Model-neutral dense-decoder kernels. Bodies are the Granite-4.2-30B runtime
// kernels moved verbatim; see tools/dedup_provenance.py for the proof.

#include "lsxcommon/lsx_dense_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
```

- [ ] **Step 2: Перенести шесть лаунчеров**

Из `granite_dense_inference.cu:397-493` перенести тела `GraniteRmsNorm`, `GraniteRope`, `GraniteAttentionSplit`, `GraniteAttention`, `GraniteSplitQKV`, `GraniteSwiGLU`, вместе с `kAttnSplitMaxSeq` и `kAttnSplitMinKv`, в `namespace lsxcommon::dense` (вне анонимного пространства имён), переименовав в публичные имена:

```text
GraniteRmsNorm          -> RunDenseRmsNormKernel
GraniteRope             -> RunDenseRopeKernel
GraniteAttentionSplit   -> RunDenseAttentionSplitKernel
GraniteAttention        -> RunDenseAttentionKernel
GraniteSplitQKV         -> RunDenseSplitQKVKernel
GraniteSwiGLU           -> RunDenseSwiGLUKernel
```

Внутренние вызовы (`GraniteAttentionSplit(...)` внутри `GraniteAttention`) переименовать соответственно. Комментарии сохранить дословно.

- [ ] **Step 3: Доказать, что перенос побитовый**

```bash
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/granite_dense_inference.cu \
    --to   libs/lsxcommon/src/lsx_dense_inference.cu \
    --rename granite_=dense_
```

Expected:

```text
source kernels: 7  target kernels: 7
missing in target: 0  unexpected in target: 0
```

Если `source kernels` не 7 — перенесён не весь блок; если `missing` не 0 — тело отредактировано, вернуть дословный текст.

- [ ] **Step 4: Собрать и прогнать нейтральный тест**

```bash
cmake --build --preset linux-debug --target test_dense_kernels 2>&1 | tail -5
ctest --preset linux-debug-inference -R test_dense_kernels --output-on-failure
```

Expected: `100% tests passed` (5 кейсов, плюс budget-кейс, если он перенесён).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/lsx_dense_inference.cu
git commit -m "refactor(dense): move the seven dense-decoder kernels and launchers into the neutral layer"
```

---

### Task 3: Переключить granite и удалить его копии

**Files:**
- Modify: `libs/lsxcommon/src/granite_dense_inference.cu` (удалить ядра 36–393 и лаунчеры 397–493, добавить делегаты)
- Modify: `libs/lsxcommon/include/lsxcommon/granite_dense_inference.h` (inline-делегаты)
- Modify: `tests/test_granite_dense_kernels.cu` (кейсы, уже покрытые нейтральным тестом, удаляются; остаются семейные)

**Interfaces:**
- Consumes: `dense::RunDense*` (Task 2).
- Produces: неизменный публичный API `lsxcommon::granite::RunGranite*`.

- [ ] **Step 1: Заменить семейные лаунчеры делегатами**

В `granite_dense_inference.cu` удалить перенесённые ядра и тела лаунчеров, оставив только тонкие обёртки:

```cpp
cudaError_t GraniteRmsNorm(cudaStream_t stream, __half *out, const __half *x,
                           const __half *w, int rows, int hidden, float eps)
{
    return dense::RunDenseRmsNormKernel(stream, out, x, w, rows, hidden, eps);
}

cudaError_t GraniteRope(cudaStream_t stream, __half *q, __half *k,
                        const int32_t *pos, int seq, int nh, int nkv, int hd,
                        double theta)
{
    return dense::RunDenseRopeKernel(stream, q, k, pos, seq, nh, nkv, hd, theta);
}

cudaError_t GraniteAttentionSplit(cudaStream_t stream, const __half *q,
                                  const __half *k, const __half *v, __half *out,
                                  float *partials, int seq, int nh, int nkv,
                                  int hd, float scale, int kv_base, int splits)
{
    return dense::RunDenseAttentionSplitKernel(stream, q, k, v, out, partials,
                                               seq, nh, nkv, hd, scale, kv_base,
                                               splits);
}

cudaError_t GraniteAttention(cudaStream_t stream, const __half *q,
                             const __half *k, const __half *v, __half *out,
                             float *partials, int seq, int nh, int nkv, int hd,
                             float scale, int kv_base)
{
    return dense::RunDenseAttentionKernel(stream, q, k, v, out, partials, seq,
                                          nh, nkv, hd, scale, kv_base);
}

cudaError_t GraniteSplitQKV(cudaStream_t stream, const __half *qkv, __half *q,
                            __half *k, __half *v, int seq, int qn, int kn)
{
    return dense::RunDenseSplitQKVKernel(stream, qkv, q, k, v, seq, qn, kn);
}

cudaError_t GraniteSwiGLU(cudaStream_t stream, const __half *gu, __half *out,
                          int I, int64_t n)
{
    return dense::RunDenseSwiGLUKernel(stream, gu, out, I, n);
}
```

Значения по умолчанию в заголовке (`int kv_base = 0`) сохранить как есть — делегаты принимают их явно.

- [ ] **Step 2: Убрать ставшие ненужными include**

Из `granite_dense_inference.cu` удалить `#include "lsx_kernels_act.cuh"`, `"lsx_kernels_math.cuh"`, `"lsx_kernels_norm.cuh"`, если после удаления ядер они больше не используются (проверить `grep -n "kernels::" libs/lsxcommon/src/granite_dense_inference.cu`). `"lsx_kernels_common.cuh"` может остаться, если используется драйвером.

- [ ] **Step 3: Собрать и прогнать семейные гейты**

```bash
cmake --build --preset linux-debug 2>&1 | rg "warning:" | rg -v "_deps|third_party" | head
ctest --preset linux-debug-inference -R "test_granite_dense" --output-on-failure
```

Expected: 0 предупреждений в наших файлах; `test_granite_dense_kernels`, `test_granite_dense_loader`, `test_granite_dense_model`, `test_granite_dense_selfcheck` — PASS.

- [ ] **Step 4: Побитовый гейт на реальных весах**

```bash
ctest --preset linux-debug-inference -R "test_granite_dense_selfcheck" --output-on-failure
ctest --preset linux-debug-inference -R "test_granite_dense_(fsi|code|det)_selfcheck" --output-on-failure
```

Expected: SelfCheck `logits_hash=c165e80fde6401cf`, top-1 12366, Paris ×3; FSI 3/3, CODE 3/3, DET 3/3.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/granite_dense_inference.cu \
        libs/lsxcommon/include/lsxcommon/granite_dense_inference.h \
        tests/test_granite_dense_kernels.cu
git commit -m "refactor(granite): delegate the dense-decoder kernels to the neutral layer"
```

---

### Task 4: Переключить phi-4

**Files:**
- Modify: `libs/lsxcommon/src/phi_dense_inference.cu` (удалить ядра 41–343 и лаунчеры 398–494)
- Modify: `libs/lsxcommon/include/lsxcommon/phi_dense_inference.h`
- Modify: `tests/test_phi_dense_kernels.cu`

**Interfaces:**
- Consumes: `dense::RunDense*`.
- Produces: неизменный публичный API `lsxcommon::phi::RunPhi*`.

- [ ] **Step 1: Заменить лаунчеры делегатами**

Те же шесть обёрток, что в Task 3, с именами `PhiRmsNorm`, `PhiRope`, `PhiAttentionSplit`, `PhiAttention`, `PhiSplitQKV`, `PhiSwiGLU` → `dense::RunDense*`. Удалить тела ядер (строки 41–343) и старые тела лаунчеров (398–494); `phi_split_qkv_kernel` входит в перенесённый набор и удаляется.

- [ ] **Step 2: Доказать, что phi теперь не содержит копий**

```bash
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/phi_dense_inference.cu \
    --to   libs/lsxcommon/src/lsx_dense_inference.cu \
    --rename phi_=dense_
```

Expected:

```text
source kernels: 0  target kernels: 7
```

(`source kernels: 0` — в phi больше нет семейных `__global__`-ядер из этого набора; `phi_swiglu_fused_kernel` и `phi_split_qkv_kernel` удалены. Если остались `0`-ядерные хвосты — они проверяются следующим шагом.)

- [ ] **Step 3: Собрать и прогнать гейты phi**

```bash
cmake --build --preset linux-debug 2>&1 | rg "warning:" | rg -v "_deps|third_party" | head
ctest --preset linux-debug-inference -R "test_phi" --output-on-failure
```

Expected: 0 предупреждений; `test_phi_dense_kernels`, `test_phi_dense_loader`, `test_phi_dense_model`, `test_phi_dense_selfcheck` — PASS; хеш совпадает со снятым baseline.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/phi_dense_inference.cu \
        libs/lsxcommon/include/lsxcommon/phi_dense_inference.h \
        tests/test_phi_dense_kernels.cu
git commit -m "refactor(phi): delegate the dense-decoder kernels to the neutral layer"
```

---

### Task 5: Переключить granite-switch

**Files:**
- Modify: `libs/lsxcommon/src/granite_dense_switch_inference.cu` (удалить ядра 39–341 и лаунчеры 396–492; блок `Switch-specific kernels` 494–596 остаётся)
- Modify: `libs/lsxcommon/include/lsxcommon/granite_dense_switch_inference.h`
- Modify: `tests/test_granite_switch_kernels.cu`

**Interfaces:**
- Consumes: `dense::RunDense*`.
- Produces: неизменный публичный API `lsxcommon::granite_switch::RunGraniteSwitch*`.

- [ ] **Step 1: Заменить лаунчеры делегатами**

Шесть обёрток `GraniteSwitchRmsNorm`, `GraniteSwitchRope`, `GraniteSwitchAttentionSplit`, `GraniteSwitchAttention`, `GraniteSwitchSplitQKV`, `GraniteSwitchSwiGLU` → `dense::RunDense*`. Ядра aLoRA (`switch_scale_kernel`, `switch_scale_add_kernel`, `switch_lora_shrink_kernel`, `switch_lora_expand_add_kernel`) и их лаунчеры **не трогать**.

- [ ] **Step 2: Собрать и прогнать гейты switch**

```bash
cmake --build --preset linux-debug 2>&1 | rg "warning:" | rg -v "_deps|third_party" | head
ctest --preset linux-debug-inference -R "test_granite_switch" --output-on-failure
```

Expected: 0 предупреждений; `test_granite_switch_kernels`, `_loader`, `_model`, `_adapter_parity`, `_selfcheck`, `_fsi/_code/_det_selfcheck`, `_tokenizer_parity`, `_tool_framing` — PASS; `logits_hash=fdc632042c30919c`.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/granite_dense_switch_inference.cu \
        libs/lsxcommon/include/lsxcommon/granite_dense_switch_inference.h \
        tests/test_granite_switch_kernels.cu
git commit -m "refactor(granite-switch): delegate the dense-decoder kernels to the neutral layer"
```

---

### Task 6: Полный барьер фазы, реестры, документация

**Files:**
- Modify: `tests/test_model_suite_guard.cpp` (добавить `test_dense_kernels` в `Suites()` для трёх моделей)
- Modify: `AGENTS.md` § Structure (новый нейтральный слой)
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md` (per-file inventory)
- Modify: `CHANGELOG.md`
- Create: `docs/open-tasks/2026-11-03-dense-trio-neutral-layer-closed.md`

**Interfaces:**
- Consumes: всё предыдущее.
- Produces: замкнутая фаза с задокументированными числами.

- [ ] **Step 1: Обновить реестр suite-guard**

В `tests/test_model_suite_guard.cpp::Suites()` добавить `"test_dense_kernels"` в stage `kernels` для `granite-4.2-30b`, `granite-switch-4.1-30b` и `phi-4` (рядом с их `test_*_dense_kernels`).

- [ ] **Step 2: Прогнать полный барьер**

```bash
cmake --build --preset linux-debug 2>&1 | rg -c "warning:" || echo "no warnings"
ctest --preset linux-debug-fast
ctest --preset linux-debug-convert
ctest --preset linux-debug-inference -R "test_(granite_dense|granite_switch|phi|dense_kernels)" --output-on-failure
```

Expected: `100% tests passed` в каждом; быстрый набор — 135/135 (134 + `test_dense_kernels`, если он получил label `fast`; иначе 134), convert — 37/37, inference-набор — все PASS.

- [ ] **Step 3: Снять perf-строки и сравнить с baseline**

```bash
ctest --preset linux-debug-perf -R "test_perf_baseline_(granite_4_2_30b|granite_switch_4_1_30b|phi_4)" --output-on-failure
```

Expected: PASS; t/s в пределах run-to-run шума относительно строк в `tests/perf/perf_baselines.json` (granite 76.79, switch 69.28, phi 71).

- [ ] **Step 4: Записать артефакты фазы**

Создать `docs/open-tasks/2026-11-03-dense-trio-neutral-layer-closed.md` с числами: до/после LOC по трём семействам, вывод `tools/dedup_provenance.py` для обеих проверок, снятые хеши, t/s до/после, время сборки до/после (замерить `time cmake --build --preset linux-debug --target lsxcommon` до Task 3 и после Task 6).

В `AGENTS.md` § Structure добавить нейтральный слой `lsx_dense_inference.{h,cu}` (`lsxcommon::dense`: RMSNorm / RoPE / causal GQA + split-KV / split-QKV / SwiGLU; потребители — granite, granite-switch, phi) и убрать упоминания семейных RoPE/attention как «единственных family-local kernels» там, где они теперь делегируют.

В skill `logestix-repo-map` обновить per-file inventory (новый файл и заголовок; у granite/phi/switch — «delegates to dense»).

В `CHANGELOG.md` добавить строку о консолидации.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_suite_guard.cpp AGENTS.md \
        .opencode/skills/logestix-repo-map/SKILL.md CHANGELOG.md \
        docs/open-tasks/2026-11-03-dense-trio-neutral-layer-closed.md
git commit -m "docs: record the dense-trio consolidation and update the registries"
```

---

## Приёмка фазы

- `tools/dedup_provenance.py` — 7/7 для granite и 0 семейных копий в phi/switch.
- `fast` и `convert` зелёные, 0 предупреждений в наших файлах.
- SelfCheck/FSI/CODE/DET зелёные у трёх моделей, хеши не изменились.
- perf-строки в пределах шума, `perf_baselines.json` не пере-сижен.
- Ветка `refactor/dense-trio-neutral-layer`, мерж — только по явному запросу оператора.
