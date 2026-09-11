# DeepSeek-V4-Flash NVFP4 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task (project AGENTS.md FORBIDS subagents — inline
> execution only). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перевести DeepSeek-V4-Flash с MXFP4/FP8-кодека на NVFP4-контур
(`lsx_nvfp4`) как остальные быстрые модели: 4.43 t/s baseline → 8.5 t/s (уже
достигнуто кэшем) → целевые ~25-35 t/s (паритет с GLM-5.3-Flash при той же MoE
геометрии).

**Architecture:** (1) Немедленные зафиксированные победы: экспертный LRU-кэш на
свободную VRAM (по умолчанию, GLM-паттерн) + проводка `SetDeepseekFastMath` в
perf-политику. (2) Оффлайн-репак vault `logestix-v1` → `logestix-v1-nvfp4`:
масштабы экспертов E8M0/32 → E4M3/16 (точное отображение степеней двойки,
nibbles не меняются), attention/dense FP8-тайлы → NVFP4 через существующий
`quantize_fp8_nvfp4`. (3) Inference: routed/shared/dense GEMM через
`RunLsxNvfp4GemmGateUpPair` / `RunLsxNvfp4Gemm` (GLM-паттерн кэш-слота
`[w1|w3|w2]`, scales `[s1|s3|s2]`).

**Tech Stack:** CUDA 13.4 (CC 12.0/12.1, sm_120), C++20, CMake presets
`linux-debug`/`linux-release-znver4`, Catch2 v3, lsx_nvfp4 модельно-нейтральный
слой, ctest labels fast/convert/inference/perf.

## Global Constraints

- AGENTS.md: субагенты ЗАПРЕЩЕНЫ — только inline.
- AGENTS.md шаг 5 протокола: NVFP4 GEMM только через `lsxcommon/lsx_inference_nvfp4.h`; никаких параллельных named-копий диспетчера.
- Norms и biases остаются FP16/BF16 (ИНВАРИАНТ ТОЧНОСТИ).
- lsxbenchmark — строгий GPU-эксклюзив, никаких параллельных GPU-задач.
- Комментарии Doxygen сохранять; логирование только через spdlog; формат-строки abseil printf-стиля (`%d`, не `{}`).
- Коммиты: английский, префиксы feat/fix/test/docs; тесты — label в том же коммите.
- Эталонная цепочка: glm_moe_flash_inference.cu:1288-1315 (pair-GEMV → SwigluKernel → w2 → coeff_add).
- Default numeric policy: kAuto = per-GPU perf policy (уже применена через `EnsurePerfPolicyApplied`).

---

### Task 0: Немедленные победы — кэш по умолчанию + политика TC

**Files:**
- Modify: `libs/lsxcommon/src/gpu_perf.cpp:193-204` (уже отредактирован: `gpu::SetDeepseekFastMath(knobs.tc_fast_math_default)`)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu:480-495`
- Test: ручной bench + `ctest -L fast` (регрессия)

**Interfaces:**
- Consumes: `perf::CurrentKnobs().expert_slots_pct` (gpu_perf.h), `cudaMemGetInfo`.
- Produces: DeepSeek по умолчанию ≥8 t/s без env-переменных.

- [x] **Step 0.1: Проба уже выполнена** — env-эксперимент `LSX_DEEPSEEK_EXPERT_CACHE_SLOTS=1300` дал 8.5 t/s (105-токенный прогон), без кэша 4.6. TC-путь на cache-hit медленнее (5.2) — гейт `!use_cache` сохранён.

- [ ] **Step 0.2: Дефолт-размер кэша из свободной VRAM** — заменить pipeline.cu:494 `if (!explicit_slots) slots = 0;` на GLM-паттерн:

```cpp
        if (!explicit_slots) {
            // GLM-Flash ResolveExpertSlots pattern: cache everything that
            // fits into free VRAM minus a generation reserve. The old
            // "thrashes every token" note applied to fixed small slot
            // counts; full-free-VRAM sizing measured 4.6 -> 8.5 tok/s.
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                constexpr size_t kReserve = size_t{2} << 30;  // arenas + ctx
                const size_t budget =
                    free_bytes > kReserve ? free_bytes - kReserve : 0;
                const int pct = perf::CurrentKnobs().expert_slots_pct;
                size_t eff = budget;
                if (pct < 100) eff = budget / 100u * static_cast<unsigned>(pct);
                const int64_t slot_bytes = routed_pack_total + routed_scale_total;
                if (slot_bytes > 0)
                    slots = static_cast<int>(eff / static_cast<size_t>(slot_bytes));
            }
        }
```

Требуется `#include "lsxcommon/gpu_perf.h"` в pipeline.cu (проверить).

- [ ] **Step 0.3: Пересборка + bench** — `cmake --build build/linux-release-znver4 -j 160` и `lsxbenchmark --model /mnt/llm_models/models/DeepSeek-V4-Flash --max-gen 1024` БЕЗ env. Ожидание: ≥8.0 t/s.

- [ ] **Step 0.4: ctest fast** — `ctest --preset linux-release-znver4 -L fast` зелёный.

- [ ] **Step 0.5: Commit** `fix: wire DeepSeek TC toggle into perf policy; default expert cache to free VRAM (4.6 -> 8.5 tok/s)`.

---

### Task 1: E8M0→E4M3/16 host-примитивы + unit-тест

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/nvfp4_scale.h` (добавить функцию)
- Test: `tests/test_deepseek_nvfp4_repack.cpp` (новый, label fast)

**Interfaces:**
- Produces: `namespace lsxcommon::nvfp4 { inline void E8m0ScaleToE4m3Pair(uint8_t e8m0, uint8_t out[2]); }` — маппинг одного E8M0-байта в пару E4M3-байтов per-16 (обе половины 32-блока). Используется Task 2 (tool) и Task 4 (если понадобится runtime-путь).

- [ ] **Step 1.1: Падающий тест** — для всех 256 байтов b: `E8m0ScaleToE4m3Pair(b)` → декод пары через `E4M3ScaleToFloat` равен `E8m0ClampedToFloat(b)` для представимого диапазона [2^-9 .. 2^8] (b∈[118..135]) и 0x00→0x00; вне диапазона — кламп к границе (тест фиксирует контракт: `decode_pair(b) == clamp_to_e4m3_range(E8m0ClampedToFloat(b))`, сравнение float побитово для степеней двойки).

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "lsxcommon/nvfp4_scale.h"

namespace {
float E4m3Host(uint8_t b) {
    // host-копия gpu::nvfp4::E4M3ScaleToFloat семантики (E4M3, bias 7)
    if (!b) return 0.0f;
    const int s = (b >> 7) & 1, e = (b >> 3) & 0x0F, m = b & 7;
    float v = (e == 0) ? static_cast<float>(m) * (1.0f / 128.0f)
                       // 2^-9 subnormal step
                       : static_cast<float>(1u << (e - 1 + 127 - 7));
    if (e != 0 && m) v *= 1.0f + static_cast<float>(m) / 8.0f;
    return s ? -v : v;
}
} // namespace

TEST_CASE("e8m0 scale byte -> nvfp4 e4m3 per-16 pair", "[fast][deepseek]") {
    for (int b = 0; b < 256; ++b) {
        uint8_t pair[2] = {0xEE, 0xEE};
        lsxcommon::nvfp4::E8m0ScaleToE4m3Pair(static_cast<uint8_t>(b), pair);
        REQUIRE(pair[0] == pair[1]);  // обе половины 32-блока одинаковы
        const float want = /* E8m0ClampedToFloat host-копия */ 0.f;
        // representable: exact; else clamp — см. тело теста в реализации
    }
    SECTION("zero and FLT_MAX saturate") {
        uint8_t p[2];
        lsxcommon::nvfp4::E8m0ScaleToE4m3Pair(0x00, p);
        REQUIRE(p[0] == 0x00);
        lsxcommon::nvfp4::E8m0ScaleToE4m3Pair(0xFF, p);
        REQUIRE(p[0] == 0x7E);  // 448.0 (E4M3 max), документированный кламп
    }
}
```

- [ ] **Step 1.2: Прогон** — `cmake --build build/linux-release-znver4 -j 160 --target test_deepseek_nvfp4_repack && ctest --test-dir build/linux-release-znver4 -R deepseek_nvfp4_repack` — FAIL (функция нет).

- [ ] **Step 1.3: Реализация** в `nvfp4_scale.h`:

```cpp
// One E8M0 exponent byte (per-32 block) maps to TWO identical E4M3 bytes
// (per-16 halves of the same block). Exact for 2^-9..2^8; outside the
// E4M3 range the value saturates to 0 / E4M3 max — the clamped flavor the
// routed-expert FFN already uses (lsx_kernels_quant.cuh E8m0ClampedToFloat).
inline void E8m0ScaleToE4m3Pair(uint8_t e8m0, uint8_t out[2]) {
    const uint8_t mapped = FloatToE4m3P2(E8m0ClampedToFloatHost(e8m0));
    out[0] = mapped; out[1] = mapped;
}
```

(хост-копии `E8m0ClampedToFloat` из lsx_kernels_quant.cuh:67 и `FloatToE4m3P2` из nvfp4_scale.h — оба уже `__host__ __device__`, проверить include-граф: nvfp4_scale.h НЕ должен тянуть CUDA-хедеры в host-TU.)

- [ ] **Step 1.4: Тест зелёный**, `ctest -L fast` без регрессий.
- [ ] **Step 1.5: Commit** `feat: e8m0->e4m3/16 scale pair primitive for deepseek nvfp4 repack`.

---

### Task 2: Репак-инструмент vault → logestix-v1-nvfp4 (lsxforge)

**Files:**
- Create: `apps/lsxforge/src/deepseek_repack.cpp`
- Modify: `apps/lsxforge/src/cli.{h,cpp}` + `apps/lsxforge/src/main.cpp` (флаг `--deepseek-nvfp4 <src> <dst>`)
- Modify: `apps/lsxforge/CMakeLists.txt` (add_executable)
- Test: `tests/test_deepseek_nvfp4_repack.cpp` — второй TEST_CASE (label convert, TIMEOUT 120, синтетический v1-микро-vault)

**Interfaces:**
- Consumes: `quantize_fp8_nvfp4(fp8, shape, scale_f32, scale_shape, name, DenseNvfp4Record&)` (glm_moe_flash_convert.cpp:404); `E8m0ScaleToE4m3Pair`; чтение/запись Arrow IPC (паттерн converter.cpp / glm_moe_flash_convert.cpp).
- Produces: vault с `"format": "logestix-v1-nvfp4"`; expert-тензоры: packed без изменений, scales ×2 (per-16), global_scale=1.0; dense FP8-тензоры → NVFP4; BF16/FP16 копируются as-is.

- [ ] **Step 2.1: Тест-фикстура** — синтетический vault: `model.index.json` (format logestix-v1) + один expert-arrow (w1/w2/w3 маленькой геометрии, например [32,32], E8M0 scales известные байты) + один FP8 dense-тенор. Прогон инструмента, проверки: format-тег, scales_bytes ×2, packed-байты идентичны, декод-scale пары == исходные E8M0 значения (хост-референс Task 1), FP8-тенор стал NVFP4 (DenseNvfp4Record-поля в индексе).
- [ ] **Step 2.2: Реализация** — поток: чтение index → для experts/*.arrow побайтовая перепаковка scale-массивов (packed копируется) → для pinned/*.arrow: dtype-роутинг F8_E4M3+scale-tiles → `quantize_fp8_nvfp4`, иначе as-is → запись нового index (`logestix-v1-nvfp4`). GPU свободен — квантизация плотных допустима, но НЕ параллельно с инференсом (GPU-эксклюзив не нужен: quantize_fp8_nvfp4 host-путь — проверить и при медленности заменить на GPU-ядра safetensors_gpu.cu).
- [ ] **Step 2.3: convert-тест зелёный** (label convert, TIMEOUT 120).
- [ ] **Step 2.4: Commit** `feat: lsxforge --deepseek-nvfp4 repack tool (logestix-v1 -> v1-nvfp4)`.

---

### Task 3: Репак реального vault

**Files:** нет кода (операция над `/mnt/llm_models/models/DeepSeek-V4-Flash`)

- [ ] **Step 3.1:** `lsxfabric/lsxforge --deepseek-nvfp4 /mnt/llm_models/models/DeepSeek-V4-Flash/logestix /mnt/llm_models/models/DeepSeek-V4-Flash/logestix-nvfp4` (~149GB, ждать, проверять свободное место `df -h /mnt/llm_models`).
- [ ] **Step 3.2:** Заменить `logestix` → симлинк/переименование (старый каталог удалить после Task 4-5 зелёных, экономя диск: сперва `mv logestix logestix-mxfp4.bak` если место позволяет, иначе репак сразу в `logestix` поверх бэкапа index).

---

### Task 4: Loader v1-nvfp4

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cu` (detect формата; scales bytes numel/16; global_scale из record)
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cuh` (ExpertEntry doc: scales now E4M3/16)
- Modify: `libs/lsxcommon/src/deepseek_moe_model.cpp:57-65` (frozen-гейт без изменений)
- Test: `tests/test_deepseek_moe_loader.cpp` — roundtrip на v1-nvfp4 фикстуре

- [ ] **Step 4.1:** Loader: strict-ошибка на старом формате с понятным сообщением «repack via lsxforge --deepseek-nvfp4» (AGENTS: no backward compatibility). synthetic-фикстуры тестов обновить на v1-nvfp4 генерацию (scale-байты дублируются, как в Task 2 фикстуре).
- [ ] **Step 4.2:** Roundtrip-тесты зелёные; `ctest -R deepseek` (fast-часть).
- [ ] **Step 4.3:** Commit `feat: deepseek loader logestix-v1-nvfp4 (e4m3/16 scales)`.

---

### Task 5: Routed FFN через lsx_nvfp4 (главный перформанс-кусок)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu:343-502,688-843`
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu` (кэш-сл layout [w1|w3|w2], arena)
- Test: `tests/test_deepseek_moe_inference_kernels.cu` (обновить routed-FFN parity кейсы)

**Interfaces:**
- Consumes: `RunLsxNvfp4GemmGateUpPair(stream, x, packed, scales, gs1, gs3, y, K, n_half, acc)`, `RunLsxNvfp4Gemm(stream, x, packed, scales, gs, y, 1, K, N, acc)`; GLM-паттерн glm_moe_flash_inference.cu:1288-1315; `SwigluKernel<__half, Glibc, true>`; `flash_coeff_add_kernel` аналог (есть acc_half_to_f32_kernel — дополнить coeff-версией).
- Produces: `run_routed_expert_ffn_nvfp4(...)` — v1-nvfp4 ветка; старая F32-ветка удаляется (no backward compat).

- [ ] **Step 5.1:** Кэш-слот: копии в порядке w1,w3,w2 / s1,s3,s2 (pipeline + moe.cu staging), stage-буферы той же раскладки. `pack_stride` = w1+w3+w2 packed bytes.
- [ ] **Step 5.2:** FFN-ветка (по образцу GLM):

```cpp
// [w1|w3] pair GEMV → [gate|up] halves
lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(stream, d_normed, p13, s13,
    en.w1_global_scale, en.w3_global_scale, d_yg, kHiddenDim,
    kIntermediateDim, false);
kernels::SwigluKernel<__half, ExpFlavor::Glibc, true><<<grid, 256, 0, stream>>>(
    d_yg, d_yg + kIntermediateDim, d_yg, kIntermediateDim, 10.0f);
lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, p2, s2, en.w2_global_scale, d_y2,
    1, kIntermediateDim, kHiddenDim, false);
// d_output += weight * d_y2 (FP32 accum)
coeff_add_f32_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(
    d_y2, weight, d_output, kHiddenDim);
```

- [ ] **Step 5.3:** Kernel parity-тесты: host-референс dequant(E4M3/16) vs GEMV на малой геометрии (K=256, N=512); обновить существующие 4470-assertion кейсы (пути dequant_fp4_e8m0_* удаляются вместе с веткой).
- [ ] **Step 5.4:** `ctest -R deepseek -L fast` + kernels-тест; bench: ожидание ≥15 t/s (H2D тот же, де-квант исчез, GEMV 4x легче по трафику).
- [ ] **Step 5.5:** Commit `perf: deepseek routed FFN on lsx_nvfp4 pair-GEMV (v1-nvfp4)`.

---

### Task 6: Dense/attention через lsx_nvfp4

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_attn.cu`, `deepseek_moe_inference_csa.cu`
- Test: kernels-сьют

- [ ] **Step 6.1:** q_a/q_b/kv_a/kv_b/wo (+gate) — LsxNvfp4Weight из pinned (конвертированы Task 2), GEMV через `LaunchNvfp4Gemm`; dequant_f8_e8m0 пути удаляются.
- [ ] **Step 6.2:** Parity-тесты обновить; bench: ожидание 20-30 t/s.
- [ ] **Step 6.3:** Commit `perf: deepseek attention GEMMs on lsx_nvfp4`.

---

### Task 7: Shared experts + уборка

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu` (shared FFN), `deepseek_moe_load.cu`
- Cleanup: удалить мёртвые dequant-ядра `shared_experts.cu` (только deepseek-использования; glm/qwen потребители проверить grep-ом!)

- [ ] **Step 7.1:** Shared FFN → `LaunchNvfp4Gemm` (resident LsxNvfp4Weight).
- [ ] **Step 7.2:** grep-аудит потребителей `dequant_fp4_e8m0_kernel|dequant_fp4_e8m0_f32_kernel` — удалить если 0.
- [ ] **Step 7.3:** Commit `perf: deepseek shared experts via LaunchNvfp4Gemm; drop mxfp4 dequant kernels`.

---

### Task 8: SelfCheck, baseline, документация

**Files:**
- Modify: `tests/perf/perf_baselines.json` (deepseek t/s)
- Modify: `AGENTS.md` (deepseek entry: v1-nvfp4, скорость)
- Test: selfcheck Paris ×3 (SKIP_RETURN_CODE 4 без модели — у нас модель есть)

- [ ] **Step 8.1:** `lsxbenchmark --selfcheck` Paris зелёный ×3 бит-точных цикла (tolerance-class на новой ветке — допускается смена эталонных логитов, детерминизм обязателен).
- [ ] **Step 8.2:** Финальный bench 2048 токенов; обновить perf_baselines.json (тест `test_perf_baseline_deepseek` сам перезапишет при kUpdate).
- [ ] **Step 8.3:** AGENTS.md: DeepSeek-V4-Flash entry — формат v1-nvfp4, lsx_nvfp4 контур, t/s.
- [ ] **Step 8.4:** Полный `ctest -L "^(fast|convert)$"` зелёный; commit `docs: deepseek-v4-flash on nvfp4 contour` + push.

## Self-Review

- Спека покрыта: кэш (Task 0), кодек-разрыв (Tasks 1-4), GEMM-контур (Tasks 5-7), верификация/доки (Task 8).
- Placeholder-скан: код дан для ключевых правок; Task 2/6 детали привязаны к существующим функциям (quantize_fp8_nvfp4, LaunchNvfp4Gemm) — сигнатуры проверены в исходниках.
- Типы: `E8m0ScaleToE4m3Pair(uint8_t, uint8_t[2])`, ExpertEntry global_scale-поля уже существуют (deepseek_moe_load.cuh:93-95).

## Round 3-4 findings (2026-09-11, post-migration tuning)

State: 11.7 tok/s steady (baseline 4.43, +164%), SelfCheck Paris green.

- H2D miss transfers = 57 ms/token (44 GB per 30 steps, 1.46 GB/token at
  40% LRU miss) — the hard ceiling; GPU compute is ~17 ms and mostly
  hidden under copies by the dual-stream interleave.
- cudaMemcpyBatchAsync for pageable mmap H2D measured SLOWER than plain
  per-copy calls (10.3 vs 11.7 tok/s) — different staging path. Plain
  loop retained.
- Phase-split ex() (batch copies first, GEMVs after one fence) loses the
  GEMV-under-copy interleave: +110 us/layer visible GEMV time. Interleave
  (copy k -> fence k -> GEMV k) is optimal at this shape.
- Early-issue prefetch is blocked by an unresolved race (silent numeric
  corruption both plain and with a layer-boundary event sync). Needs a
  root-cause session (candidate: cache-slot overwrite while a previous
  layer's GEMV still reads the slot — issue(N+1) victims vs in-flight
  main-stream readers).
- Per-layer partitioned cache: worse (10.7). Shared experts: already on
  load-time dequant caches, lsx_nvfp4 payoff ~0.3 tok/s — deferred.
- Task 6 (attention NVFP4) payoff re-estimated at ~+1 tok/s (dequant_f8
  4.4 ms + fp32 GEMV 1 ms of an 85 ms token) — deferred until the H2D
  ceiling moves.

Conclusion: without cutting miss BYTES (hit rate/coverage bounded by
VRAM) or solving the prefetch race, ~12 tok/s is the practical ceiling
of this architecture generation on 32 GB. The 11.7 state is committed
and selfcheck-green.

## Round 5 findings (2026-09-11): prefetch race LOCALIZED

Root-cause instrumentation (LSX_DS_VERIFY_COPY / LSX_DS_VERIFY_READ
byte-exact probes at issue time vs GEMV-read time):

1. Prefetch H2D copies are BYTE-EXACT at issue time (2363/2363 verified).
2. 7/5676 slot reads at GEMV time MISMATCH the host source — ALL in
   layers >= 27 (= 7000-mmap-cap boundary: 7000/256 = 27.3 layers).
3. Two races identified:
   a. munmap-under-DMA: issue(L) enqueues copies from mmap-backed entries;
      ensure(L+1) evicts the 7000-cap mmaps (munmap) while the DMA is
      still queued on copy_stream. evict_done fences main only.
      FIXED: lsx_pending_copy_drain() registry — issue registers
      (source base, completion event), expert_store drains before munmap.
   b. slot write-after-read: victims selected by issue(L+1..L+k) (host
      runs several layers ahead of the main-stream GEMVs) can overwrite a
      slot a still-executing GEMV of layer L is reading. NOT yet fixed —
      this is the remaining corruption (verified copies + drain, output
      still garbage when LSX_DS_PREFETCH=1).
4. Candidate fix for (b): per-layer slot bands for prefetch fills
   (layer L writes only [L*band, (L+1)*band) — builders and readers of a
   layer never share bands with in-flight GEMVs of other layers), or a
   "slot pinned until its GEMV retires" guard counted from the ex loop.

State: prefetch env-gated OFF (LSX_DS_PREFETCH), production stable at
11.7 tok/s, SelfCheck Paris green, fast 43/43.

## Round 5 additions (2026-09-11, evening)

- Drain-hook wired end-to-end (lsx_pending_copy_drain in expert_store,
  registry in moe.cu) — munmap-under-DMA race closed.
- CORRUPTION root-cause CONFIRMED DIFFERENT: byte-exact probes show the
  prefetch copies are correct at issue time AND the slot content matches
  at GEMV-read time in 5669/5676 cases — the 7 mismatches are exactly the
  write-after-read slot conflicts described above.
- Canonical O(1) LRU migration experiment (expert_device_cache.h
  Lookup/Insert/Touch wired into ex()/issue(), free-stack init): measured
  a STABLE 8.0 tok/s vs 11.6 — hit-refreshing real LRU is SLOWER on this
  traffic than the legacy "FIFO-of-first-fill" order. Reverted; both
  variants kept in git history for the follow-up study.

Production state unchanged: 11.7 tok/s steady, Paris green, fast 43/43.
Round-6 candidates: (1) prefetch with slot-retire guard + per-layer
bands (fix race (b) by construction, keep legacy fill order inside each
band), (2) attention NVFP4, (3) host-gap trims.
