# Kimi-K2.7-Code — full NVFP4 contour + maximum speed

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Subagents are FORBIDDEN by AGENTS.md policy — execute inline in the main
> session only. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Kimi-K2.7-Code to the full NVFP4 contour (converter/repack/loader), measure decode t/s, push to the maximum of the RTX 5090 32GB hardware, and find and fix the causes of degradation.

**Architecture:** Kimi is already on the `logestix-kimi-r2` format (MLA NVFP4 + NVFP4 routed experts via `PackExpertNVFP4`, round-2, 127/127). The experts are NVFP4 E2M1+E4M3/16 (24.77 MiB/expert × 60×384 = 23040 → ~558 GiB vault). Decode stayed at **1.3 t/s** (round-2): the file mmap source gives ~17 GiB/s effective H2D. The solution is to port the proven DeepSeek package: a **pinned expert store** (a single cudaHostAlloc pool, layout [w1|w3|w2][s1|s3|s2], LSX_EXPERT_PIN=0 opt-out) over the already working full-resident mmaps (max_open=65536 store default) + an LRU cache over free VRAM. Loading into cache slots (6×cudaMemcpyAsync from entry pointers) is unchanged — the pinned source speeds it up automatically.

**Tech Stack:** CUDA 13.4 / Blackwell sm_120, CMake presets linux-release-znver4, Catch2, spdlog, abseil (host-only).

## Global Constraints

- CPU/GPU exclusivity: do not run lsxbenchmark/perf tests concurrently with other GPU work (AGENTS.md).
- RAM: 750 GiB total; pinning ~558 GiB is acceptable (the "we do not skimp on RAM" policy), a fallback to mmap on a cudaHostAlloc failure is mandatory.
- Bit-exact SelfCheck "Paris" ×3 (LSX_TEST_KIMI_SELFCHECK_CYCLES / SKIP_RETURN_CODE 4).
- Windows-only env tests: `setenv` only under an `#ifdef _WIN32` guard (reference `test_qwen_nvfp4_gemv.cu`).
- All logs via spdlog; new tests — a label in the same commit.
- Physical ceiling: miss traffic ~7–12 GiB/token; pinned H2D ~45–50 GiB/s (DeepSeek measurement) ⇒ a ceiling of ~4–6 t/s at zero reuse, higher with a hit rate. Below 80% of the ceiling — find the cause (nsys).

---

### Task 1: Vault conversion + baseline measurement (ops, GPU-exclusive)

**Files:**
- Create: `/mnt/llm_models/models/Kimi-K2.7-Code/logestix/` (vault, outside the repo)
- Create: `/tmp/opencode/kimi_convert.log`

**Interfaces:**
- Consumes: `lsxforge --kimi --input <dir> --output <dir>` (main.cpp:324 → `kimi::ConvertKimiModel`), in-place like DeepSeek.
- Produces: vault `logestix/model.index.json` (format `logestix-kimi-r2`, `mla_nvfp4`), `logestix/pinned/*.arrow`, `logestix/experts/layer_N/exp_E.arrow`; baseline t/s in the task journal.

- [ ] **Step 1: Rebuild the current main (fresh lsxforge)**

```bash
cmake --build build/linux-release-znver4 -j 160 --target lsxforge 2>&1 | tail -2
```
Expected: `ninja: no work to do` or a successful link, 0 errors.

- [ ] **Step 2: Run the in-place conversion (background, 20–60 min)**

```bash
nohup ./build/linux-release-znver4/apps/lsxforge/lsxforge --kimi \
  --input /mnt/llm_models/models/Kimi-K2.7-Code \
  --output /mnt/llm_models/models/Kimi-K2.7-Code \
  > /tmp/opencode/kimi_convert.log 2>&1 &
echo $! > /tmp/opencode/kimi_convert.pid
```
Expected: PID recorded; the log grows. While it runs — the GPU is busy, no CUDA launches.

- [ ] **Step 3: Wait for completion, verify the vault**

```bash
while kill -0 $(cat /tmp/opencode/kimi_convert.pid) 2>/dev/null; do sleep 60; done
tail -3 /tmp/opencode/kimi_convert.log
python3 -c "
import json;d=json.load(open('/mnt/llm_models/models/Kimi-K2.7-Code/logestix/model.index.json'))
print(d['format'], d['layers'], d['experts_per_layer'], d.get('total_experts'), 'mla' , 'mla_nvfp4' in d)"
ls /mnt/llm_models/models/Kimi-K2.7-Code/logestix/experts | wc -l
```
Expected: the log `Done ... experts processed, incomplete 0 / Conversion complete`; format `logestix-kimi-r2`; experts dir = 60 layers (layer_1..layer_60; layer_0 dense); `mla_nvfp4` present.

- [ ] **Step 4: Baseline measurement (GPU exclusive)**

```bash
LSX_TEST_KIMI_MODEL_PATH=/mnt/llm_models/models/Kimi-K2.7-Code \
  timeout 600 ./build/linux-release-znver4/tests/test_blackwell_perf_kimi 2>&1 | grep -E "PERF|tok/s|expert cache|pinned"
```
Expected: tok/s ≈ 1.3 (round-2 reproduction); record the number in the task journal as the baseline.

---

### Task 2: Pinned expert store for Kimi (TDD)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_load.h:148-164` (KimiExpertEntry +3 fields)
- Modify: `libs/lsxcommon/src/kimi_moe_load.cpp` (`load_kimi_expert_index`, the pin pass)
- Test: `tests/test_kimi_moe_loader.cpp` (extend "Kimi expert index resolves routed experts" + a new negative case)

**Interfaces:**
- Consumes: `ExpertStore::pinned_pool/pinned_bytes` (already in expert_store.h, added in DeepSeek round-8), `Free()` already does `cudaFreeHost(pinned_pool)`.
- Produces: `KimiExpertEntry::pinned_blob (const uint8_t*), blob_pack_bytes (int64_t), blob_scale_bytes (int64_t)`; the pin pass inside `load_kimi_expert_index`; env `LSX_EXPERT_PIN=0` disables it.

- [ ] **Step 1: Red test — the pinned pool in a synthetic fixture**

In `tests/test_kimi_moe_loader.cpp`, extend `TEST_CASE("Kimi expert index resolves routed experts", ...)`: (extend it)

```cpp
    // Pinned store (r3): blobs live in one locked host pool; pointers stay
    // valid for async H2D without the file-backed VMA penalty.
    REQUIRE(db.pinned_pool != nullptr);
    REQUIRE(db.pinned_bytes >= static_cast<std::size_t>(e.blob_pack_bytes
                                                     + e.blob_scale_bytes));
    REQUIRE(e.pinned_blob != nullptr);
    REQUIRE(e.pinned_blob >= static_cast<const uint8_t *>(db.pinned_pool));
    REQUIRE(e.pinned_blob < static_cast<const uint8_t *>(db.pinned_pool)
                              + db.pinned_bytes);
    // pair layout [w1|w3|w2][s1|s3|s2] — DeepSeek round-8 convention
    REQUIRE(e.blob_pack_bytes == e.w1_bytes + e.w3_bytes + e.w2_bytes);
    REQUIRE(e.blob_scale_bytes == e.w1_scales_bytes + e.w3_scales_bytes
                                + e.w2_scales_bytes);
    // upload source pointers were re-pointed INTO the pool
    REQUIRE(e.host_w1 >= static_cast<const uint8_t *>(db.pinned_pool));
    REQUIRE(e.host_w1 < static_cast<const uint8_t *>(db.pinned_pool)
                           + db.pinned_bytes);
```

- [ ] **Step 2: Run the test — red**

```bash
cmake --build build/linux-release-znver4 -j 160 --target test_kimi_moe_loader 2>&1 | tail -1
LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4/tests/test_kimi_moe_loader "Kimi expert index resolves routed experts" 2>&1 | tail -5
```
Expected: FAIL — `pinned_pool` is not filled yet (the field probably already exists in the store — then nullptr).

- [ ] **Step 3: Implementation — the entry fields**

`kimi_moe_load.h`, in `struct KimiExpertEntry` after `w3_scales_bytes`:

```cpp
    // Pinned-store view (DeepSeek round-8 parity): when the whole store is
    // pinned into locked host memory, blobs live contiguously per expert as
    // packed [w1|w3|w2] then scales [s1|s3|s2]; host_* pointers are re-pointed
    // into the pool and pinned_blob marks the expert block start.
    const uint8_t *pinned_blob = nullptr;
    int64_t blob_pack_bytes = 0;
    int64_t blob_scale_bytes = 0;
```

- [ ] **Step 4: Implementation — the pin pass in `load_kimi_expert_index`**

After `db.Load(model_path, /*fallback=*/true);`, insert (mirror deepseek_moe_load.cu:280-355, but per-entry tolerant — dense layer 0 has no experts):

```cpp
    // ── Pin the routed-expert store into locked host memory ──────────────
    // File-backed mmap sources carry a driver VMA-lookup penalty on every
    // H2D copy and vanish under munmap while an async copy is in flight.
    // One cudaHostAlloc pool, per expert packed [w1|w3|w2] then scales
    // [s1|s3|s2] — the order the slot upload streams. Entries without
    // weights (dense layer 0 of the 61-layer passport) are skipped.
    // Disable with LSX_EXPERT_PIN=0 (falls back to mmap views).
    if (std::getenv("LSX_EXPERT_PIN") == nullptr
        || std::getenv("LSX_EXPERT_PIN")[0] != '0') {
        int64_t pool_bytes = 0;
        for (const auto &en : db.entries) {
            if (!en.host_w1 || !en.host_w2 || !en.host_w3
                || !en.host_w1_scales || !en.host_w2_scales
                || !en.host_w3_scales)
                continue;   // dense layer or partial expert
            pool_bytes += ((en.w1_bytes + en.w3_bytes + en.w2_bytes + 255)
                           / 256) * 256;
            pool_bytes += ((en.w1_scales_bytes + en.w3_scales_bytes
                            + en.w2_scales_bytes + 255) / 256) * 256;
        }
        void *pool = nullptr;
        if (pool_bytes > 0
            && cudaHostAlloc(&pool, static_cast<size_t>(pool_bytes),
                             cudaHostAllocDefault) == cudaSuccess) {
            db.pinned_pool = pool;
            db.pinned_bytes = static_cast<std::size_t>(pool_bytes);
            uintptr_t cur = reinterpret_cast<uintptr_t>(pool);
            auto aln = [](uintptr_t v) { return (v + 255) / 256 * 256; };
            int pinned_cnt = 0;
            for (auto &en : db.entries) {
                if (!en.host_w1 || !en.host_w2 || !en.host_w3
                    || !en.host_w1_scales || !en.host_w2_scales
                    || !en.host_w3_scales)
                    continue;
                en.pinned_blob = reinterpret_cast<const uint8_t *>(cur);
                en.blob_pack_bytes = en.w1_bytes + en.w3_bytes + en.w2_bytes;
                en.blob_scale_bytes = en.w1_scales_bytes + en.w3_scales_bytes
                                    + en.w2_scales_bytes;
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w1,
                            static_cast<std::size_t>(en.w1_bytes));
                en.host_w1 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w1_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w3,
                            static_cast<std::size_t>(en.w3_bytes));
                en.host_w3 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w3_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w2,
                            static_cast<std::size_t>(en.w2_bytes));
                en.host_w2 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w2_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w1_scales,
                            static_cast<std::size_t>(en.w1_scales_bytes));
                en.host_w1_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w1_scales_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w3_scales,
                            static_cast<std::size_t>(en.w3_scales_bytes));
                en.host_w3_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w3_scales_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w2_scales,
                            static_cast<std::size_t>(en.w2_scales_bytes));
                en.host_w2_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w2_scales_bytes);
                ++pinned_cnt;
            }
            log::info("[kimi_load] pinned routed-expert store: "
                      + std::to_string(pinned_cnt) + " experts, "
                      + std::to_string(pool_bytes / (1024 * 1024))
                      + " MiB locked host memory");
        } else {
            log::warn("[kimi_load] expert pinning unavailable — "
                      "keeping file-backed mmap sources");
        }
    }
```

- [ ] **Step 5: Green + negative-case test**

A new TEST_CASE nearby:

```cpp
TEST_CASE("kimi expert pinning: LSX_EXPERT_PIN=0 falls back to mmap",
          "[kimi][loader][pin]")
{
    auto model = BuildConvertedFixture();
    int overridden = 0;
#ifdef _WIN32
    overridden += _putenv_s("LSX_EXPERT_PIN", "0") == 0;
#else
    setenv("LSX_EXPERT_PIN", "0", 1);
    ++overridden;
#endif
    auto db = load_kimi_expert_index(model, 3, 2);
    REQUIRE(db.pinned_pool == nullptr);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    ensure_kimi_expert_resident(db, 1, 0, stream);
    REQUIRE(db.entries[1 * 2 + 0].host_w1 != nullptr);   // mmap view works
    free_kimi_expert_index(db);
    cudaStreamDestroy(stream);
#ifdef _WIN32
    _putenv_s("LSX_EXPERT_PIN", "");
#else
    unsetenv("LSX_EXPERT_PIN");
#endif
    (void)overridden;
}
```

```bash
cmake --build build/linux-release-znver4 -j 160 --target test_kimi_moe_loader 2>&1 | tail -1
LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4/tests/test_kimi_moe_loader 2>&1 | tail -3
```
Expected: all assertions green, the pinned-store log `pinned routed-expert store: ... experts`.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/kimi_moe_load.h libs/lsxcommon/src/kimi_moe_load.cpp tests/test_kimi_moe_loader.cpp
git commit -m "perf: kimi pinned routed-expert store (LSX_EXPERT_PIN=0 opt-out)"
```

---

### Task 3: Measurement with the pinned store + degradation diagnostics

**Files:**
- Create: `/tmp/opencode/kimi_r3_*.log`, possibly `/tmp/opencode/kimi_r3.nsys-rep`

**Interfaces:**
- Consumes: Task 1 vault, Task 2 pin pass; `test_blackwell_perf_kimi` / `lsxbenchmark --model`.
- Produces: the final t/s, a verdict "ceiling/cause", a list of the levers extracted.

- [ ] **Step 1: Full rebuild + no-env measurement**

```bash
cmake --build build/linux-release-znver4 -j 160 2>&1 | grep -cE "error|FAILED" || true
LSX_TEST_KIMI_MODEL_PATH=/mnt/llm_models/models/Kimi-K2.7-Code \
  timeout 900 ./build/linux-release-znver4/tests/test_blackwell_perf_kimi 2>&1 | grep -E "PERF|pinned|expert cache" 
```
Expected: the log `pinned routed-expert store: 23040 experts, ~558000 MiB`; tok/s 2–4× above the 1.3 baseline. Record the number.

- [ ] **Step 2: Correctness verification on real output**

```bash
./build/linux-release-znver4/apps/lsxbenchmark/lsxbenchmark \
  --model /mnt/llm_models/models/Kimi-K2.7-Code --max-gen 256 2>&1 | tail -5
```
Expected: coherent text (the "Paris..." canon), not garbage; the t/s is consistent with step 1.

- [ ] **Step 3: "Degradation vs ceiling" analysis**

Ceiling = effective pinned H2D (~45–50 GiB/s, DeepSeek measurement) ÷ miss bytes/token (from the cache log: misses × 24.77 MiB). If t/s < 80% of the ceiling:
1. `nsys profile -o /tmp/opencode/kimi_r3 --trace=cuda ./build/.../test_blackwell_perf_kimi` — look for serialization of copies (copy_stream vs main stream), stalls.
2. Check the slot budget: the log `expert cache: N slots` — if slot ≈ 48 MiB (2× an expert) — find the double reservation (`KimiMoeExpertDeviceCache` alloc), raise slots.
3. Env experiments: `LSX_KIMI_EXPERT_CACHE_SLOTS` (0.5×/1×/2× of the default).
Each hypothesis — a before/after measurement; useless ones are reverted (the DeepSeek-round discipline).

- [ ] **Step 4: Record the findings in the plan (this file), an intermediate commit on code changes**

---

### Task 4: SelfCheck ×3 + fast suite

- [ ] **Step 1: Real Paris selfcheck, 3 bit-exact cycles**

```bash
LSX_TEST_KIMI_MODEL_PATH=/mnt/llm_models/models/Kimi-K2.7-Code \
  ./build/linux-release-znver4/tests/test_kimi_moe_selfcheck 2>&1 | tail -4
```
Expected: PASS ×3, `r.selfcheck_passed`, exit 0 (SKIP 4 is not allowed when a vault is present).

- [ ] **Step 2: The whole fast suite**

```bash
ctest --test-dir build/linux-release-znver4 -L fast 2>&1 | tail -3
```
Expected: 100% tests passed, 0 failed (skips only where other models' vaults are absent).

- [ ] **Step 3: Guards**

```bash
ctest --test-dir build/linux-release-znver4 -R "test_model_independence_guard|test_code_naming_guard|test_absl_format_guard" --output-on-failure 2>&1 | tail -3
```
Expected: green without touching the registries (the kimi consumer was registered earlier).

---

### Task 5: Finalization (baseline JSON + AGENTS.md + docs + push)

- [ ] **Step 1: perf baseline kimi (kAdd/kUpdate)**

Add a kimi entry to `tests/perf/perf_baselines.json` (gemma format: model_type `kimi-k2.7-code`, gpu_key `rtx 5090`, os `linux`, tokens_per_sec = the final measurement, max_gen 2048, the canonical prompt "The capital of France is", date, commit `-`), then:

```bash
ctest --test-dir build/linux-release-znver4 -R test_perf_baseline --output-on-failure 2>&1 | tail -3
```
Expected: kUpdate/kAdd, the test green.

- [ ] **Step 2: AGENTS.md kimi entry** — update the `kimi-k2.7-code` bullet (the new t/s, pinned store, r2 NVFP4-expert format; verify the `deepseek_inference_*.cu`-style paths against the actual files).

- [ ] **Step 3: Findings into this plan doc** (the final stage table, the ceiling, the levers extracted, what remains) + commit + push:

```bash
git add tests/perf/perf_baselines.json AGENTS.md docs/superpowers/plans/2026-09-12-kimi-nvfp4-full-contour.md
git commit -m "perf: kimi-k2.7-code pinned routed-expert store — <T> tok/s (final)"
git push
```

## Final results (2026-09-12)

| Metric | Was (round-2) | Became | Gain |
|---|---|---|---|
| Decode t/s (bit-exact, default) | 1.3 | **3.1** | 2.4× |
| Decode t/s (TC opt-in, tolerance) | — | 5.5 | 4.2× |
| Expert store load (542 GiB) | — | 302 s | — |
| SelfCheck Paris ×3 | ✅ | ✅ | — |
| VRAM resident (dense+MLA) | 8.9 GiB | 8.9 GiB | — |
| Expert cache slots | 595 | 809 | +36% |

### Levers extracted (session commits)
1. **Pinned expert store** 542 GiB in 34×16 GiB locked chunks — the driver does not
   allow a single cudaHostAlloc >~300 GiB (a 12-minute reclaim storm, probe:
   mlock passes, chunked cudaHostAlloc passes). `LSX_EXPERT_PIN=0`
   disables it (927c4fe, 0d84844).
2. **TC-batch was not enabled** — the legacy `cfg_.fast_math` bool is always false on the
   CLI; the tri-state was ignored. A resolution fix (751aa44) gave 3.0→5.1 t/s.
3. **The TC path is broken on the real checkpoint** — tolerance-class act-quant
   drives the greedy top-1 into garbage ('碧芙碧芙' instead of Paris), while the synthetic
   tolerance test is green. Moved to opt-in `LSX_KIMI_TC=1`
   until root-cause (14:23 fix). The default is the bit-exact legacy GEMV chain.
4. **kv_b dequant thrash** — a static-memo slot for ALL 61 layers re-dequantized
   kv_b 60×/token; a per-layer cache in KimiLayerWeights (30% of the GPU window).
5. **Split-fence upload** — layout [w1|w3][s1|s3][w2][s2], ev13/ev2:
   w13 GEMM starts without waiting for the down upload (7c2058b).
6. **VRAM reserve 8→2 GiB** (measured: runtime allocations <1 GiB) →
   549→809 slots (7c2058b).
7. **Fast load**: 635→~280 s. Parallel parse with 32 threads
   (450→8 s, c332d53); the cudaHostAlloc zerofill (332 s, serialized by the
   driver, evicting the page cache → copying from disk) replaced with
   mmap → fault-at-write copy from the hot cache → register last
   (b76c76e + register-after-copy).

### Measured token anatomy (nsys, TC on, 2.6 t/s snapshot)
- H2D 53% (PCIe ~50-57 GiB/s — saturation, experts 200 MiB/token cold)
- Nvfp4Dequant 30% → ~0 after the kv_b fix (decode)
- Expert GEMV: cuBLASLt nvjet_**tmaAB** (TMA already engaged)
- gaps 7% (route split-phase sync)

### Ceiling
Miss traffic ~7-11 GiB/token at 809/23424 slots; the physical PCIe
limit of the RTX 5090 (~57 GiB/s) ≈ 4-6 t/s cold. 3.1 bit-exact = 65% of the ceiling
(tail: route sync, GEMV compute). Qualitative growth further — only by
hitting the cache more precisely (hit rate) or a wider bus.

### Root cause TC-batch divergence (2026-09-12, test_kimi_moe_tc_diag)

Differential diagnostics on a real MoE-block input vector
(a dump via LSX_KIMI_DUMP_L1):
- The TC kernels are correct on quantized data: w13 rel 0.16-0.19%,
  down rel 0.19%, reduce rel 0.17%.
- Quantizing x (post-norm hidden): rel 12% — acceptable.
- **Quantizing the swiglu output (hq): rel 120%** — E2M1 per-16 pow2
  turns the tensor into noise: the swiglu output has a heavy tail + a mass
  of values near zero, and the blocks are uneven.
- The hq-quant contribution to the down output: **69%** — matches the total
  divergence of legacy vs TC (67%).
- Conclusion: the mxf4nvf4 format (per-16 E2M1 activations) is fundamentally
  unsuitable for Kimi's swiglu tensors. TC-batch stays opt-in;
  the default is the bit-exact legacy GEMV (3.1 t/s). The path to 5.5 t/s is a
  higher-precision activation format for the down-GEMM (the HW per-16
  ue4m3 format is already full E4M3 — but the E2M1 data nibbles remain the
  bottleneck; an hw path with a BF16 A operand is needed, which mxf4nvf4 lacks).

### Backlog
- Pin the expert store by default for ALL models (768 GB RAM)
- Unify the DeepSeek/Kimi pin pass into ExpertStore::PinAll
- The Kimi TC path: raise the precision of the down-GEMM act-quant (see root cause)
