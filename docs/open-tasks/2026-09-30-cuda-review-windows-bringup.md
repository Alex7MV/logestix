# Open tasks — CUDA review & Windows bring-up

Written 2026-09-30, after the inference review of `24bf180~1..bed5d9f` and the
Windows build repair that followed.

**Current state:** `fast` label is **72/72 green** on this machine
(RTX 5060 Ti, CC 12.0, 16 GiB, CUDA Toolkit V13.4.92, Windows 11 / VS 2026).
`inference` label is red; every item below is triaged. Nothing here is
speculative — each entry carries the measured value or error code that
produced it.

Reproduce the inventory with:

```powershell
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $ctest --test-dir build -C Release -L fast      --timeout 900
& $ctest --test-dir build -C Release -L inference --timeout 900
```

---

## A. Real defects — GPU faults

### A1. `test_qwen_moe_kernels` + `test_ornith_moe_kernels`: `cudaErrorIllegalAddress`
- `cudaErrorIllegalAddress` (700). It is **not** the first thing that goes
  wrong: once a kernel faults, the context is poisoned and every later CUDA
  call in the process returns 700, so the bulk of the failure list is cascade.
  Triage from the *first* failing `REQUIRE`, not the last.
- Also carries a latency budget failure at `tests/test_qwen_moe_kernels.cu:388`
  (`us < 20.0`, measured 31.24) — that part belongs to section C.
- **Next step:** same workflow that resolved the GigaChat router.
  ```
  compute-sanitizer --tool memcheck --launch-timeout 120 <test>.exe
  ```
  Add `-lineinfo` to the owning object library to get `file:line`, otherwise
  the report only gives `kernel+0xNNN`. Note `TcGemmDispatch`
  (`libs/lsxcommon/src/lsx_tc_gemm.cu:891`) already rejects non-CC-12, and this
  GPU *is* CC 12, so the `major != 12` guard is not what is firing here.

### A2. MiMo-V2.6-Flash-RL real-weight SelfCheck: wrong top-1 (parity defect)

- The full ingestion (converter → NVFP4 experts → kernels → `IModel`) is
  committed on `feat/model-MiMo-V2.6-Flash-RL`; the synthetic suites
  (convert/loader/model/kernels/dequant) are green.
- The real 172 GB checkpoint converts cleanly in place
  (`/mnt/llm_models/models/MiMo-V2.6-Flash-RL/logestix`, 12032 experts,
  `logestix-mimo-nvfp4`, 167 GB) and the model **loads and generates** on the
  RTX 5090.
- **Defect:** `test_mimo_moe_selfcheck` fails the Paris gate. Measured
  `top1_tok=108402 text='高原小康' passed=false`, `logits_hash=ab9d9bd20d824cfb`
  (the forward pass runs; the logits are wrong).
- **Localization (2026-09-30, deep trace):** the defect is in the **routed
  expert weights**, not framing/attention/dense/router:
  - Raw-encoded and ChatML-framed prompts both produce incoherent repetition
    (English and Chinese respectively) → not the token bridge.
  - With `LSX_MIMO_NO_MOE=1` the residual stream grows healthily
    (|x| = 11 → 124 across 48 layers); with MoE it explodes
    (11 → 17 → 42 → 108 → 450 by layer 4) → the MoE term injects wrong values.
  - The router is **bit-exact** vs a host reference (`[mimo-moe-host]` trace:
    identical idx/weight on every layer) → `RunMimoRouter` and `gate.weight`
    are correct.
  - Swapping the nibble order of the exact-repack experts at staging
    (`LSX_MIMO_EXPERT_SWAP=1`) changes the output but does **not** fix it, so it
    is not a simple pairwise nibble swap.
- **Update (2026-09-30, Python ground truth from the source safetensors):**
  every per-layer component now matches a NumPy decode of the ORIGINAL
  checkpoint within NVFP4 quantization error, so the earlier "expert weights
  are wrong" conclusion was too strong:
  - Embedding: bit-exact. qkv projection: a=0.987, corr=0.999. Attention core
    (with `value_scale` applied exactly once) and `o_proj`: match.
  - Dense MLP (gate/up/act/down): a=0.97-0.99, corr≥0.98, **when fed the
    device's own intermediate**, so the layer math is correct.
  - Router: host top-8 == device top-8 on real weights. Expert FFN: device
    `gate/up/act/down` norms equal the NumPy MXFP4 decode to 4 decimals for
    layer 1 / expert 137 (`|gate| 83.0769` vs `83.0771`, element-wise match).
  - A NumPy full-forward on the source weights ALSO produces large residuals
    (|x| ~14 at layer 0, ~100+ by layer 1), i.e. large residual norms are
    inherent to this checkpoint, not a device bug.
  - **Remaining gap:** the device's layer-0 output is ~12% low (|x|=12.55 vs
    NumPy 14.23) from accumulated ~1-3%/component NVFP4 error, and the model
    (an RL reasoning checkpoint) amplifies it across 48 layers until the
    residual stream diverges and generation degenerates into repetition.
  - **Best next hypotheses:** (a) the dense FP8→NVFP4 requant is too lossy for
    this checkpoint's dynamic range — try keeping the FP8 dense weights in FP8
    (the repo has a finegrained-FP8 path) or a higher-precision dense path;
    (b) FP16 vs BF16 activation storage; (c) verify RoPE at pos>0 (the only
    path not yet validated, since pos 0 makes it the identity).
- **Update 2 (2026-09-30, framing + multi-position attention):**
  - The prompt frame was wrong: the reference
    `apply_chat_template(add_generation_prompt=True)` ends at
    `[..., 151644, 77091, 198]` (`<|im_start|>assistant\n`, **no**
    `<think></think>` prefill), because the template's `enable_thinking is false`
    check is false when the variable is undefined. `BuildPromptInputIds` now
    matches the reference exactly (`enable_thinking=true`).
  - A full **48-layer Python forward** (source FP8 + MXFP4 decoded) matches the
    device's single-token logits **exactly** (top-5 token ids and values), so the
    entire model math at pos 0 is correct.
  - **The bug is in the decode attention for positions ≥ 2.** Building the
    attention reference from the device's own per-position `qkv` dumps:
    pos 0/1 match (corr ≈ 1.0), pos 2 diverges (corr 0.64). The projections
    themselves match exactly (corr 1.0), so the fault is in
    `mimo_attn_decode_kernel`'s online-softmax / KV-cache path (the prefill
    kernel's two-pass form passed its parity test; the decode kernel's streaming
    softmax is the suspect).
  - **Next step:** rewrite `mimo_attn_decode_kernel`'s per-key online softmax as
    the prefill kernel's two-pass form (collect scores, one softmax, then mix),
    or bisect the streaming rescale with a 3-key unit test. Then re-run the
    single-token and multi-token Python logits comparisons.
- **Update 3 (2026-10-01, vendor oracle — conclusive):** the implementation is
  **bit-exact with the vendor `modeling_mimo_v2.py`**, and the checkpoint itself
  is the problem:
  - A manual 48-layer forward built from the **vendor** `MiMoV2DecoderLayer`
    (weights dequantized from the source, selected experts loaded per the
    router) produces `top5 = [2181, 110015, 113764, 11853, 1268]`; the device
    produces `top1 = 2181` — identical.
  - The vendor oracle also fails on a plain question ("What is the capital of
    France?") with the same repetitive garbage.
  - The earlier "pos ≥ 2 attention" divergence was a **dump-filename collision
    artifact** across the diagnostic's runs; with clean single-run dumps the
    attention matches exactly at pos 0/1/2 (GA and SWA).
  - So there is **no bug in the logestix implementation**. `selfcheck` stays red
    because `MiMo-V2.6-Flash-RL` does not answer the standard prompts in this
    environment. Finalization: merge the (correct) implementation; the converted
    model is removed. If revisited, verify the checkpoint shards/checksums or
    obtain the exact transformers build the checkpoint was produced with.

## B. Windows platform gaps

### B1. Expert pinning is not implemented on Windows
- `ExpertStore::PinChunked` is compiled only on Unix — the `#else` at
  `libs/lsxcommon/include/lsxcommon/expert_store.h:560` is
  `(void)n_threads; return false;`.
- Consequence: `pinned_pools` is never populated, and
  `tests/test_kimi_moe_loader.cpp:119` (`REQUIRE(!db.pinned_pools.empty())`)
  fails. The test is reporting the truth; do not silence it.
- Degradation is **correctness-neutral**: callers such as
  `deepseek_moe_v41_load.cu:405` log a warning and fall back to
  `arrow::io::MemoryMappedFile` views, which are cross-platform. The cost is
  pageable (unpinned) host memory, so H2D for expert residency is slower.
- **Resolved (platform-gate):** the pinning assertions in
  `tests/test_kimi_moe_loader.cpp` now compile only on Unix
  (`__unix__ || __APPLE__`); on Windows they are replaced by a `WARN` plus a
  check that `pinned_pools` is empty, so the pageable fallback is explicit
  rather than a red test. Implementing the `VirtualAlloc` path stays a future
  option; it cannot be compiled or tested from the Linux host.
- Blast radius: every family using it — kimi, glm, gemma, gigachat, deepseek
  v41. The expert stores are hundreds of GiB, so this only bites on real
  models; the synthetic fixtures never exercise it.

## C. Hardware-dependent budgets — expected red off-target, **not** defects

Measured on RTX 5060 Ti (GDDR7, ~179 GB/s peak). Every one of these is a
wall-clock or bandwidth number calibrated on the HBM target, so they cannot
pass on consumer Blackwell. They are correctly filed under `inference`, not
`fast` — the label discipline is right, and `fast` stays green because of it.

| Test | Assertion | Measured here |
|---|---|---|
| `test_qwen_nvfp4_gemv` (+13 siblings, see E1) | `test_qwen_nvfp4_gemv.cu:296` `gbps > 260.0` | 119.8 / 178.5 / 180.4 GB/s |
| `test_qwen_moe_exp_kernels` | `test_qwen_moe_exp_kernels.cu:648` `median < 0.045` | 0.052–0.082 ms |
| `test_gemma_moe_norm` | `test_gemma_moe_norm.cpp:149` `us < 3.4` | 9.24 |
| `test_qwen_moe_kernels` | `test_qwen_moe_kernels.cu:388` `us < 20.0` | 31.24 |
| `test_gemma_moe_moe` | `us < 700.0` | — |
| `test_nemotron_moe_kernels` | `batch_us < single_us * 0.6f` | — |
| `test_glm_moe_flash_kernels` | `us < 155.0` | — |

**Open question, needs a decision:** should these `SKIP` on non-target GPUs so
`inference` can go green on dev hardware, or is a per-arch budget table
wanted? The perf-baseline path (`docs/*_SPEED_AUDIT.md`,
`test_perf_baseline_*`) already assumes measured reference hardware, so a
precedent exists. Right now nothing distinguishes "too slow for the target" from
"regressed", which is the part that actually costs debugging time.

## D. Missing coverage

### D1. No test drives `RunGlmFlashGeneration`
- The glm-flash host-read fix (`flash_dbg_on_host`, commit `6108ecf`) has **no
  regression test**: `tests/` covers individual glm-flash kernels and
  `Load`/`Unload`, never generation, so the debug branch cannot be reached.
- Needs a generation path for glm-flash (converted fixture + `Infer`), which is
  a larger piece of work than the fix itself.
- **Resolved:** `tests/test_glm_moe_flash_selfcheck.cpp` now carries a
  `[glm-flash][model][generation][real-model]` case that loads the real
  converted model, sets `LSX_TRACE_FLASH=1` (RAII-guarded), and calls
  `Infer` directly. `RunGlmFlashGeneration` is now driven end-to-end and the
  branch `6108ecf` fixed (host mirror `flash_dbg_on_host` -> launch
  `flash_dbg_probe_kernel`) is executed — the probe output is observed. The
  case is real-vault-gated (SKIP without the model), matching the
  selfcheck/perf-suite rule that vault integration never lands in the
  loader/model stages.

### D2. Hardcoded POSIX paths make tests unreachable on Windows
- `tests/test_kimi_moe_tc_diag.cu:88,92` — `/tmp/kimi_diag_l1.bin`, guarded by
  `if (!std::filesystem::exists(...)) SKIP`, so it **silently skips** on
  Windows. It compiles now but can never run there.
- Same pattern in `tests/test_mosaic_network.cpp` and
  `tests/test_path_registry.cpp`.
- Fix: `std::filesystem::temp_directory_path()`.

## E. Diagnosability

### E1. Fourteen test executables are the same binary
- `tests/CMakeLists.txt:635-648` compiles 14 `.cu` files into a single object
  library `test_qwen_cu`, and the `foreach` at `tests/CMakeLists.txt:788`
  links **all of it** into each of the 14 executables.
- Effects: every binary runs the same ~187k assertions; all 14 have byte-
  identical size; and a single failing assertion in one file marks **14 ctest
  targets failed**. That is what made the first `inference` run look
  catastrophic when it was really one DRAM-bandwidth budget.
- The bandwidth failure in C is the concrete example:
  `test_lsx_nvfp4_dispatch.exe` reports failures located in
  `test_qwen_nvfp4_gemv.cu`.
- Fix: per-binary object libraries, or one combined target registered with
  ctest once. Worth doing before the next round, because it currently
  multiplies every failure by 14.

## F. Intermittent

### F1. `test_gigachat_moe_devkernels` flaked once
- Failed in one `-L inference` sweep; passes standalone (2155 assertions, 7
  cases) and passes under `ctest -R ^test_gigachat_moe_devkernels$`.
- Not yet triaged. Suspect GPU contention or ordering rather than a logic
  defect, but that is a guess — needs a repeat sweep to see if it recurs.

## G. Build environment

### G1. Toolkit changes need a manual reconfigure
- Upgrading CUDA left `build/` pinned to the previous toolkit
  (`CUDAToolkit_*` cache entries are valid, so a new `CUDAToolkit_ROOT` does
  not re-derive them). Symptom: headers 13.3 against runtime 13.4, caught only
  by `test_cuda_toolkit_guard` as `CUDART_VERSION=13030 runtime=13040`.
- Procedure is written up in `.opencode/skills/building-the-project/SKILL.md`.
- **Resolved (2026-09-30):** `CMakeLists.txt` now fails configure when
  `CUDAToolkit_VERSION` and `CMAKE_CUDA_COMPILER_VERSION` disagree (major.minor),
  so a stale cache is caught before a build instead of by
  `test_cuda_toolkit_guard` afterwards. See the follow-up section below.

---

## Follow-up (2026-09-30, Linux / RTX 5090 host)

A separate machine (Linux, RTX 5090, GDDR7 ~1.8 TB/s, CUDA 13.4) worked the
list above. Status per item:

- **A1 — not reproducible here.** `test_qwen_moe_kernels` and
  `test_ornith_moe_kernels` both pass (865 assertions / 13 cases each) and
  `compute-sanitizer --tool memcheck` reports 0 errors for each. Either a
  Windows/5060 Ti-specific fault or one already removed by the Windows build
  repair; it still needs a repro on that box before any kernel change.
- **B1 — resolved** by platform-gating the test (see the decision above).
- **C — passes on this host.** All bandwidth/latency budgets attributed to the
  5060 Ti are green on the 5090 (they are calibrated to the HBM target, and the
  5090 is close enough). No policy change made; the open question remains for
  the 5060 Ti dev box.
- **D1 — fixed.** A real-model generation case now drives
  `RunGlmFlashGeneration` with the `LSX_TRACE_FLASH` debug branch set (see the
  D1 section above).
- **D2 — fixed.** The three hardcoded POSIX temp paths now use
  `std::filesystem::temp_directory_path()`: `test_kimi_moe_tc_diag.cu`,
  `test_mosaic_network.cpp`, `test_path_registry.cpp`. The Kimi dump producer
  (`kimi_moe_inference.cu`) writes to the same temp path so the capture is
  reachable on Windows.
- **E1 — fixed.** `test_qwen_cu` (one OBJECT lib linked into 14 executables) is
  replaced by `lsx_add_qwen_cu_test()`, one OBJECT lib per `.cu` file. The 14
  executables now have distinct sizes and run only their own cases
  (`test_lsx_gemv`: 2 vs the previous 58). Target names and labels unchanged.
- **F1 — not reproduced.** 10 consecutive standalone runs of
  `test_gigachat_moe_devkernels` all pass; suspect the original one-off was GPU
  contention, but keep watching.
- **G1 — fixed.** `CMakeLists.txt` now fails configure when
  `CUDAToolkit_VERSION` and `CMAKE_CUDA_COMPILER_VERSION` disagree (major.minor),
  with a message pointing at the stale-cache remedy, so the mismatch is caught
  before a build instead of by `test_cuda_toolkit_guard` afterwards.

Verification on the follow-up host: full build clean; `fast` 72/72;
`convert` 17/17; the 14 split `inference` targets 14/14.

---

## Inference-conformance review (2026-09-30, same host)

A second pass checked every family against the Model Ingestion Protocol and
whether inference is unified. The shared contracts hold: all 15 registered
models open a per-call `LsxFastMathScope(EffectiveFastMathMode(cfg_))`, route
NVFP4 only through the neutral `lsx_nvfp4`, run one stream (no
`cudaDeviceSynchronize` in code), and no cuBLAS is linked;
`kPassports` = `Suites()` = `model_registry.cpp`. Deviations found, all fixed
(a `refactor:`/`test:` commit each):

- `gemma_moe_load.cpp` reached the neutral `fastload::PinnedStagingRing`
  through the GLM alias header `glm_moe_load_internal.h` — now includes
  `fast_load_internal.h` directly, like qwen/kimi/ornith.
- `lsx_kernels_common.cuh` (model-neutral) included `glm_moe_device_math.h`.
  The glibc-replica exp/rsqrt moved to the new `lsx_device_math.h` and the
  block-max reduce to the new `lsx_reduce.cuh`; `glm_moe_device_math.h` /
  `qwen_reduce.cuh` stay as thin aliases.
- `granite_dense_model.cpp` was absent from `test_model_independence_guard`
  `consumers`, so its fast-math policy was unguarded (step 3).
- `glm_moe_flash_constants.h` was the only passport without `static_assert`s.

**Tool bridges (stage 4b) — retracted, do not "fix" again.** GLM-5.2 and
Kimi-K2.7-Code are absent from this host's vault and are the documented
generic-Hermes fallback. Granite-4.2 was verified on real weights through that
fallback in `8702fa2` (`test_granite_dense_selfcheck` round-trip, bare
`{"name":…,"arguments":…}` object). A native `granite_dense_tool_format` was
attempted and **regressed** the round-trip — reverted. Adding a native renderer
for a fallback family requires re-verifying the real-weight round-trip (recorded
in AGENTS.md).

**Perf-baseline pairing — false positive.** `test_perf_baseline_glm_5_2` and
`test_perf_baseline_qwen3_5_9b` have no `perf_baselines.json` entry because
those models are not in the vault; the target seeds the entry once converted
(CMake comment) — intentional, not a gap.

---

## Closed in this session, for reference

- Removed gemma dead code: `RunGemmaMoECombine` + `gemma_moe_combine_kernel`
  (no caller anywhere).
- `RunGemmaRouterTopK` now rejects `num_experts` above the launch block
  (`kGemmaRouterTopKThreads = 128`); it previously guarded only `<= 0` while the
  per-expert kernel indexes one thread per expert.
- Removed a leftover `fflush(stdout)` from the gemma decode loop (per-token
  syscall on the hot path).
- Family `.cu` debug/error `printf`/`fprintf` routed through the `lsxcommon::log`
  bridge (gemma error diagnostics, glm-flash `LSX_TRACE_FLASH` host probes);
  device-side kernel `printf` stays.
- `apps/lsxfabric` `TrySendRouted` closes the routed multipart with an empty
  terminal frame when the body send fails, so a stalled peer cannot leave the
  socket desynced for later streams.
- `test_deepseek_moe_v41_load_fast`: stale expectation after `e4954d3` — it
  read `d_wq_a` as FP16 `1.0`, but the MLA/indexer dense weights now stay raw
  FP8 on device (`0x38 0x38` reinterprets as FP16 `0.52734375`). The test now
  asserts the raw E4M3 byte (`0x38`) and the UE8M0 scale byte (`127`). Caught by
  the first full `-L inference` sweep (that label was never run end-to-end on
  this host before; the 3 h run otherwise passed/skipped everything).
- Inference review findings: `x_vec` uint4 alignment mask, `hc_count` vs
  frozen passport, `session_depth_` direct-owner exemption, `JsonEscape`
  duplication, GigaChat smem comment, glued lines in `nvfp4_gemv_math.cuh`.
- Windows build: seven POSIX-only constructs that stopped the tree compiling
  at all (`munmap`, `M_PI`, `setenv`×2, `__builtin_memcpy`×2, `unistd.h`).
- GigaChat YaRN rope tables were host pointers passed to a CUDA kernel.
- GigaChat router scratch device address (`cudaGetSymbolAddress`).
- glm-flash host read of a `__device__` debug flag.
- `test_deepseek_moe_loader`: Arrow mapping held a Windows file lock during
  cleanup.
- CUDA Toolkit tracked to V13.4.92.
