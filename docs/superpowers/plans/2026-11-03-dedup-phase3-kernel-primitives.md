# Фаза 3: остаточные дубликаты (AttnRes, конверсии) — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. **Subagents are forbidden in this repository** (AGENTS.md § Autonomous Mode): every step is executed inline by the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Убрать последние измеренные дубликаты: `depth_mix_kernel` (AttnRes) из alice/kimi-k3 и шесть копий `f32_to_f16_kernel`/`f16_to_f32_kernel` из moe-семейств.

**Architecture:** `depth_mix_kernel` — побайтово одинаковый код, переносится в нейтральный `lsxcommon::attn` дословно (provenance-проверка обязательна). Конверсионные ядра — **не** побайтово одинаковые (grid-stride против one-shot, `int` против `int64_t`, `__restrict__` различается), поэтому объединяются в один grid-stride `int64_t` вариант в `lsx_kernels_common.cuh` с отдельным доказательством эквивалентности по значениям; это единственная намеренно не-дословная консолидация цикла.

**Tech Stack:** C++20, CUDA 13.x, Catch2 v3.16.1, `lsx_add_cuda_test`.

**Spec:** `docs/superpowers/specs/2026-11-03-code-duplication-consolidation-design.md` §2.3, §4, §5 (Ф3).

## Global Constraints

- Порядок: фаза выполняется **после** фазы 2. Копии `scale_half_kernel`, `scale_float_kernel`, `silu_mul_kernel` существовали только в паре `deepseek_moe_inference_moe.cu` / `deepseek_moe_pro_inference_moe.cu` и исчезают вместе с pro-копией; в фазе 3 их нет.
- `silu_mul_kernel` в `deepseek`-файле **не** сливается с `lsx_kernels_act.cuh`: у deepseek асимметричный clamp (только сверху по gate), это записано в `lsx_kernels_act.cuh:14`. Семантика не меняется.
- Побитовая идентичность моделей: `logits_hash` alice, kimi-k3 и всех семейств, чьи конверсии затронуты (`gigachat`, `glm-5.3-flash`, `mistral-small`, `mistral-large4`, `xing`, `nemotron`, `qwen3.8-flash-next`), не меняется; FSI/CODE/DET не хуже.
- Пере-сид `tests/perf/perf_baselines.json` запрещён.
- Заголовки, достижимые из `.cu`, не включают `<absl/container/*>`; сборка — 0 предупреждений в наших файлах.
- Коммиты: английские, `refactor:`/`test:`/`docs:`. Субагенты запрещены.

---

### Task 1: `depth_mix_kernel` → `lsxcommon::attn::RunAttnResDepthMix`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_attn_inference.h` (новий лаунчер)
- Modify: `libs/lsxcommon/src/lsx_attn_inference.cu` (ядро + лаунчер)
- Modify: `libs/lsxcommon/src/alice_moe_kernels.cu:54-95` (удалить копию), `:192-201` (делегировать)
- Modify: `libs/lsxcommon/src/kimi_moe_k3_kernels.cu:152-193` (удалить копию), `:455-469` (делегировать)

**Interfaces:**
- Produces:

```cpp
namespace lsxcommon {
namespace attn {
// AttnRes depth mixer: softmax over n_src depth sources, then a weighted
// combination projected by proj_w. n_src is capped by the caller (alice 64,
// kimi-k3 16) — the neutral layer keeps the kernel only.
cudaError_t RunAttnResDepthMix(cudaStream_t stream, const __half *sources,
                               int n_src, const __half *norm_w,
                               const __half *proj_w, __half *out, int hidden,
                               float eps);
} // namespace attn
} // namespace lsxcommon
```

- Consumes: ничего.

- [ ] **Step 1: Написать падающий нейтральный тест**

Добавить в `tests/test_attn_kernels.cu` (если файла нет — создать и зарегистрировать через `lsx_add_cuda_test(NAME test_attn_kernels SOURCES test_attn_kernels.cu LINK lsxcommon LABEL inference)`) кейс:

```cpp
#include <catch2/catch_all.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <lsxcommon/lsx_attn_inference.h>
#include <cmath>
#include <vector>

namespace {
__half H(float v) { return __float2half(v); }
}

TEST_CASE("attnres depth mix matches a CPU reference", "[attn][attnres]")
{
    constexpr int kSrc = 4, kHidden = 8;
    std::vector<__half> sources(kSrc * kHidden), norm_w(kHidden),
        proj_w(kSrc * kHidden), out(kHidden);
    for (int i = 0; i < kSrc * kHidden; ++i)
        sources[i] = H(0.1f * static_cast<float>(i) - 0.5f);
    for (int i = 0; i < kHidden; ++i) norm_w[i] = H(1.0f);
    for (int i = 0; i < kSrc * kHidden; ++i)
        proj_w[i] = H(0.05f * static_cast<float>((i * 7) % 11));

    REQUIRE(lsxcommon::attn::RunAttnResDepthMix(
                nullptr, sources.data(), kSrc, norm_w.data(), proj_w.data(),
                out.data(), kHidden, 1e-6f) == cudaSuccess);

    // CPU reference: depth softmax over the per-source L2 statistics, then the
    // projection — computed in double so the comparison is not itself lossy.
    std::vector<double> score(kSrc, 0.0), nrm(kHidden);
    for (int s = 0; s < kSrc; ++s) {
        double ss = 0.0;
        for (int i = 0; i < kHidden; ++i) {
            const double v = __half2float(sources[s * kHidden + i]);
            ss += v * v;
        }
        score[s] = ss / kHidden;
    }
    double mx = score[0];
    for (int s = 1; s < kSrc; ++s) mx = std::max(mx, score[s]);
    double z = 0.0;
    for (int s = 0; s < kSrc; ++s) z += std::exp(score[s] - mx);
    for (int i = 0; i < kHidden; ++i) {
        double acc = 0.0;
        for (int s = 0; s < kSrc; ++s) {
            const double w = std::exp(score[s] - mx) / z;
            acc += w * __half2float(sources[s * kHidden + i]) *
                   __half2float(proj_w[s * kHidden + i]);
        }
        nrm[i] = acc;
    }
    for (int i = 0; i < kHidden; ++i)
        REQUIRE(std::fabs(__half2float(out[i]) - nrm[i]) < 6e-3);
}
```

Эталонная формула скопирована из хостового комментария к ядру в `alice_moe_kernels.cu:54` — перед реализацией сверить с ним и, если комментарий описывает иную нормировку, привести CPU-эталон в соответствие с комментарием (не наоборот).

- [ ] **Step 2: Убедиться, что тест падает**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug --target test_attn_kernels 2>&1 | tail -5
```

Expected: FAIL на линковке — `undefined reference to lsxcommon::attn::RunAttnResDepthMix`.

- [ ] **Step 3: Перенести ядро дословно и добавить лаунчер**

Скопировать `depth_mix_kernel` из `alice_moe_kernels.cu:54-95` в `lsx_attn_inference.cu` (внутри существующего анонимного пространства имён), переименовав в `attn_res_depth_mix_kernel`; добавить в `lsx_attn_inference.h`/`.cu` лаунчер:

```cpp
cudaError_t RunAttnResDepthMix(cudaStream_t stream, const __half *sources,
                               int n_src, const __half *norm_w,
                               const __half *proj_w, __half *out, int hidden,
                               float eps)
{
    attn_res_depth_mix_kernel<<<1, 256, 0, stream>>>(sources, n_src, norm_w,
                                                     proj_w, out, hidden, eps);
    return cudaGetLastError();
}
```

`256` — значение `kBlock` в обоих семействах; guard-и (`n_sources > 64` у alice, `n_src <= 0 || n_src > 16` у kimi) в нейтральный слой **не** переносятся.

- [ ] **Step 4: Доказать, что перенос дословный**

```bash
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/alice_moe_kernels.cu \
    --to   libs/lsxcommon/src/lsx_attn_inference.cu \
    --rename depth_mix_kernel=attn_res_depth_mix_kernel
```

Expected: `missing in target: 0  unexpected in target: 0`. (Целевой файл содержит и чужие ядра; инструмент проверяет только те, что есть в источнике, поэтому «unexpected» будет ненулевым — допустимо; критично `missing in target: 0`.)

- [ ] **Step 5: Переключить оба семейства, сохранив их guard-и**

`alice_moe_kernels.cu`:

```cpp
cudaError_t RunAliceDepthMix(cudaStream_t stream, const __half *sources,
                             int n_sources, const __half *norm_w,
                             const __half *proj_w, __half *out, int hidden,
                             float eps)
{
    if (n_sources > 64) return cudaErrorInvalidValue;
    return attn::RunAttnResDepthMix(stream, sources, n_sources, norm_w, proj_w,
                                    out, hidden, eps);
}
```

`kimi_moe_k3_kernels.cu`:

```cpp
cudaError_t RunKimiK3DepthMix(cudaStream_t stream, const __half *sources,
                              int n_src, const __half *norm_w,
                              const __half *proj_w, __half *out, int hidden,
                              float eps)
{
    if (n_src <= 0 || n_src > 16) return cudaErrorInvalidValue;
    return attn::RunAttnResDepthMix(stream, sources, n_src, norm_w, proj_w, out,
                                    hidden, eps);
}
```

Имя лаунчера kimi уточнить по факту (`rg -n "RunKimiK3.*DepthMix\|depth_mix" libs/lsxcommon/src/kimi_moe_k3_kernels.cu`).

- [ ] **Step 6: Собрать, прогнать тесты и побитовый гейт**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head
ctest --preset linux-debug-inference -R "test_attn_kernels|test_alice_moe_kernels|test_kimi_k3_kernels" --output-on-failure
ctest --preset linux-debug-inference -R "test_alice_moe_selfcheck|test_kimi_k3_selfcheck" --output-on-failure
```

Expected: 0 предупреждений; все PASS; `logits_hash` alice `f900547d25be7914` и kimi-k3 без изменений.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/lsx_attn_inference.h \
        libs/lsxcommon/src/lsx_attn_inference.cu \
        libs/lsxcommon/src/alice_moe_kernels.cu \
        libs/lsxcommon/src/kimi_moe_k3_kernels.cu \
        tests/test_attn_kernels.cu tests/CMakeLists.txt
git commit -m "refactor(attn): share the AttnRes depth mixer between alice and kimi-k3"
```

---

### Task 2: Конверсионные ядра → `lsx_kernels_common.cuh`

**Files:**
- Modify: `libs/lsxcommon/src/lsx_kernels_common.cuh` (два нейтральных ядра)
- Modify: `libs/lsxcommon/src/gigachat_moe_inference_attn.cu`, `gigachat_moe_inference_pipeline.cu`, `glm_moe_flash_inference.cu`, `mistral_moe_inference.cu`, `mistral_moe_large4_kernels.cu`, `xing_moe_inference.cu`
- Modify: `libs/lsxcommon/src/nemotron_moe_inference_pipeline.cu`, `qwen_moe_exp_inference.cu`, `deepseek_moe_load.cuh`

**Interfaces:**
- Produces:

```cpp
namespace lsxcommon {
namespace kernels {
// Elementwise FP32 -> FP16 cast, grid-stride over `n` (n may exceed one launch).
__global__ void F32ToF16Kernel(const float *__restrict__ src,
                               __half *__restrict__ dst, int64_t n);
// Elementwise FP16 -> FP32 cast, grid-stride over `n`.
__global__ void F16ToF32Kernel(const __half *__restrict__ src,
                               float *__restrict__ dst, int64_t n);
} // namespace kernels
} // namespace lsxcommon
```

- Consumes: ничего.

- [ ] **Step 1: Написать падающий тест эквивалентности**

Добавить в существующий host-тест примитивов (найти: `rg -n "lsx_kernels_common" tests/*.cu tests/*.cpp | head`; если файла нет — `tests/test_kernels_common.cu` через `lsx_add_cuda_test(NAME test_kernels_common SOURCES test_kernels_common.cu LINK lsxcommon LABEL inference)`) кейс, который проверяет оба направления на нескольких `n`:

```cpp
TEST_CASE("elementwise f32<->f16 casts cover every index once",
          "[kernels][convert]")
{
    for (int64_t n : {1, 255, 1024, 100000}) {
        std::vector<float> src(n);
        for (int64_t i = 0; i < n; ++i)
            src[i] = 0.001f * static_cast<float>((i * 7919) % 100003) - 50.0f;
        std::vector<__half> h(n);
        std::vector<float> back(n);
        float *d_src = nullptr;
        __half *d_h = nullptr;
        float *d_back = nullptr;
        REQUIRE(cudaMalloc(&d_src, n * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_h, n * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_back, n * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_src, src.data(), n * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);

        const int blocks =
            static_cast<int>((n + 255) / 256) > 1024
                ? 1024
                : static_cast<int>((n + 255) / 256);
        lsxcommon::kernels::F32ToF16Kernel<<<blocks, 256>>>(d_src, d_h, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        lsxcommon::kernels::F16ToF32Kernel<<<blocks, 256>>>(d_h, d_back, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaMemcpy(h.data(), d_h, n * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(back.data(), d_back, n * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(d_src);
        cudaFree(d_h);
        cudaFree(d_back);

        for (int64_t i = 0; i < n; ++i) {
            REQUIRE(__half2float(h[i]) == __half2float(__float2half(src[i])));
            REQUIRE(back[i] == __half2float(__float2half(src[i])));
        }
    }
}
```

`blocks` ограничен 1024 намеренно: это проверяет, что grid-stride покрывает `n > blocks*256` (для `n = 100000`).

- [ ] **Step 2: Убедиться, что тест падает**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug --target test_kernels_common 2>&1 | tail -5
```

Expected: FAIL на компиляции — `F32ToF16Kernel` не объявлен.

- [ ] **Step 3: Добавить ядра в `lsx_kernels_common.cuh`**

```cpp
// Elementwise casts. Grid-stride so one launch covers any n; every index is
// visited exactly once, which is what the previous per-family one-shot
// variants relied on.
__global__ void F32ToF16Kernel(const float *__restrict__ src,
                               __half *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __float2half(src[i]);
}

__global__ void F16ToF32Kernel(const __half *__restrict__ src,
                               float *__restrict__ dst, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        dst[i] = __half2float(src[i]);
}
```

- [ ] **Step 4: Прогнать тест эквивалентности**

```bash
cmake --build --preset linux-debug --target test_kernels_common 2>&1 | tail -5
ctest --preset linux-debug-inference -R test_kernels_common --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Удалить семейные копии и перевести вызовы**

В каждом из девяти файлов удалить локальное определение `f32_to_f16_kernel`/`f16_to_f32_kernel` (или `half_to_float_kernel`/`float_to_half_kernel` в `deepseek_moe_load.cuh`, `nemotron_moe_inference_pipeline.cu`, `qwen_moe_exp_inference.cu`) и заменить вызовы на `kernels::F32ToF16Kernel<<<blocks, threads, 0, stream>>>(...)` с сохранением прежней конфигурации запуска (`blocks`/`threads` уже вычисляются на месте). `__half`/`float` указатели совпадают по типу; `n` приводится к `int64_t`.

Проверка, что копий не осталось:

```bash
rg -n "void (f32_to_f16|f16_to_f32|half_to_float|float_to_half)_kernel" libs/lsxcommon/src | rg -v "lsx_kernels_common.cuh"
```

Expected: пустой вывод.

- [ ] **Step 6: Собрать и прогнать побитовый гейт затронутых семейств**

```bash
cmake --build --preset linux-debug 2>&1 | rg "error:|warning:" | rg -v "_deps|third_party" | head
ctest --preset linux-debug-inference -R "test_gigachat.*selfcheck|test_glm_5_3_flash.*selfcheck|test_mistral.*selfcheck|test_xing.*selfcheck|test_nemotron.*selfcheck" --output-on-failure
```

Expected: 0 предупреждений; SelfCheck всех перечисленных — PASS с прежними хешами (у mistral FSI/CODE/DET — declared debt, SelfCheck обязателен).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/lsx_kernels_common.cuh \
        libs/lsxcommon/src/gigachat_moe_inference_attn.cu \
        libs/lsxcommon/src/gigachat_moe_inference_pipeline.cu \
        libs/lsxcommon/src/glm_moe_flash_inference.cu \
        libs/lsxcommon/src/mistral_moe_inference.cu \
        libs/lsxcommon/src/mistral_moe_large4_kernels.cu \
        libs/lsxcommon/src/xing_moe_inference.cu \
        libs/lsxcommon/src/nemotron_moe_inference_pipeline.cu \
        libs/lsxcommon/src/qwen_moe_exp_inference.cu \
        libs/lsxcommon/src/deepseek_moe_load.cuh \
        tests/test_kernels_common.cu tests/CMakeLists.txt
git commit -m "refactor(kernels): share the elementwise f32<->f16 casts"
```

---

### Task 3: Проверить остаток и закрыть фазу

**Files:**
- Modify: `AGENTS.md` § Structure (AttnRes в нейтральном attn-слое)
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`
- Modify: `CHANGELOG.md`
- Create: `docs/open-tasks/2026-11-03-kernel-primitives-closed.md`

- [ ] **Step 1: Закрепить скан дублирования в репозитории и снять остаток**

Создать `tools/dedup_scan.py` (тот же алгоритм, что дал исходные 40 групп / 3 121 строку; нормализация — идентификаторы → `ID`, комментарии и пробелы удаляются):

```python
#!/usr/bin/env python3
"""Report duplicated __global__/__device__ bodies across the CUDA sources.

Used by the dedup phases to measure the delta and to keep the deliberate
remainder honest. Bodies are compared after stripping comments and
whitespace and collapsing every identifier, so only the arithmetic survives.

    python3 tools/dedup_scan.py            # summary + the 20 biggest groups
    python3 tools/dedup_scan.py --all      # every group
"""
import argparse
import collections
import glob
import hashlib
import re
import sys

KEYWORDS = re.compile(
    r"\b(?:if|for|while|return|else|const|float|int|void|__half|half|"
    r"uint8_t|uint32_t|uint64_t|bool|unsigned|char|double|auto|template|"
    r"typename|size_t|int64_t|int32_t|uint16_t|short|long|sizeof|"
    r"__restrict__|__forceinline__|__device__|__global__|__shared__|"
    r"__syncthreads|threadIdx|blockIdx|blockDim|gridDim|static|inline|"
    r"struct|class|new|delete|true|false|nullptr|extern|constexpr|switch|"
    r"case|break|continue|do|default|namespace|using|static_cast|"
    r"reinterpret_cast|float4|half2|__float2half|__half2float|"
    r"__shfl_down_sync|__shfl_xor_sync|__shfl_sync|min|max)\b")


def definitions(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(("__global__", "__device__", "static __device__",
                                "static __global__", "__forceinline__")):
            start, depth, seen, body = i, 0, False, []
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                seen = seen or "{" in lines[i]
                body.append(lines[i])
                if seen and depth <= 0:
                    break
                i += 1
            out.append((lines[start].strip()[:80], "\n".join(body), len(body)))
        i += 1
    return out


def normalized(body):
    body = KEYWORDS.sub("KW", body)
    body = re.sub(r"\b\w+\b", "ID", body)
    return re.sub(r"\s+", "", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob("libs/lsxcommon/src/*.cu") +
                   glob.glob("libs/lsxcommon/src/*.cuh"))
    groups = collections.defaultdict(list)
    for path in files:
        for signature, body, size in definitions(path):
            if size >= 8:
                groups[hashlib.sha1(normalized(body).encode()).hexdigest()].append(
                    (signature, path, size))
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    total = sum(sum(x[2] for x in v) for v in dups.values())
    print(f"duplicate groups: {len(dups)}  lines in duplicated defs: {total}")
    ranked = sorted(dups.values(), key=lambda v: -sum(x[2] for x in v))
    for group in ranked if args.all else ranked[:20]:
        print(f"  {group[0][2]:4d} lines x{len(group)}  {group[0][0][:66]}")
        print("        " + ", ".join(sorted({g[1] for g in group})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Прогнать и записать результат:

```bash
python3 tools/dedup_scan.py
```

Ожидание после фаз 1–3: групп < 12, строк < 700 (было 40 / 3 121); `scale_half_kernel`/`scale_float_kernel`/`silu_mul_kernel` — по одной копии; `granite_dense_inference.cu` против `phi_dense_inference.cu` — только семейный код. Каждая оставшаяся группа попадает в отчёт с объяснением, почему она остаётся.

- [ ] **Step 2: Полный барьер**

```bash
ctest --preset linux-debug-fast
ctest --preset linux-debug-convert
ctest --preset linux-debug-inference
```

Expected: `fast` и `convert` — 100 %; `inference` — все PASS, кроме документированных SKIP 4 (модели без дерева) и зафиксированных RED-гейтов (gemma-dense FSI-01, deepseek-pro FSI-02, flint FSI/CODE/DET, kimi-k3 FSI/CODE/DET и прочие declared-debt строки из `docs/SELFCHECK_MATRIX.md`).

- [ ] **Step 3: Записать артефакты и обновить документацию**

`docs/open-tasks/2026-11-03-kernel-primitives-closed.md`: таблица «что удалено / что осталось намеренно и почему», вывод provenance, повторный скан дублирования до/после, снятые хеши.

`AGENTS.md`: в описании `lsx_attn_inference.{h,cu}` добавить `AttnRes depth mixer (= RunAttnResDepthMix)`, в описании `lsx_kernels_common.cuh` — конверсионные ядра.

- [ ] **Step 4: Commit**

```bash
git add tools/dedup_scan.py AGENTS.md .opencode/skills/logestix-repo-map/SKILL.md \
        CHANGELOG.md docs/open-tasks/2026-11-03-kernel-primitives-closed.md
git commit -m "docs: record the kernel-primitive consolidation and the deliberate remainder"
```

---

## Приёмка фазы

- `depth_mix_kernel`: одна копия, в `lsxcommon::attn`, provenance 0 missing.
- `f32_to_f16_kernel`/`f16_to_f32_kernel`/`half_to_float_kernel`/`float_to_half_kernel`: одна реализация каждая, в `lsx_kernels_common.cuh`.
- Повторный скан `tools/dedup_scan.py`: группы < 12, строки < 700 (baseline 40 / 3 121), и каждая оставшаяся группа объяснена в отчёте.
- Это единственная фаза с не-дословной консолидацией (конверсии); она доказана тестом эквивалентности на `n ∈ {1, 255, 1024, 100000}` и побитовыми гейтами моделей.
- Ветка `refactor/kernel-primitives`, мерж — только по явному запросу оператора.
