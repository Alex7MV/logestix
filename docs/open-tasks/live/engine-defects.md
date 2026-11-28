# Engine defects — runtime faults any family can hit

### A1. `test_qwen_moe_kernels` + `test_ornith_moe_kernels`: `cudaErrorIllegalAddress`
- **Status: open, needs the 5060 Ti box.** Not reproducible on the Linux
  RTX 5090 host: both tests pass (865 assertions / 13 cases each) and
  `compute-sanitizer --tool memcheck` reports 0 errors for each. Either a
  Windows/5060 Ti-specific fault or one already removed by the Windows build
  repair. **Do not change a kernel until it reproduces on that box.**
- `cudaErrorIllegalAddress` (700). It is **not** the first thing that goes
  wrong: once a kernel faults, the context is poisoned and every later CUDA
  call in the process returns 700, so the bulk of the failure list is cascade.
  Triage from the *first* failing `REQUIRE`, not the last.
- Also carries a latency budget failure at `tests/test_qwen_moe_kernels.cu:388`
  (`us < 20.0`, measured 31.24) — that part is the GPU-keyed-budget question
  settled in the 2026-10-23 archive (`open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md`
  § B): the number is machine-specific, so it needs that card, not a patch.
- **Next step:** same workflow that resolved the GigaChat router.
  ```
  compute-sanitizer --tool memcheck --launch-timeout 120 <test>.exe
  ```
  Add `-lineinfo` to the owning object library to get `file:line`, otherwise
  the report only gives `kernel+0xNNN`. Note `TcGemmDispatch`
  (`libs/lsxcommon/src/lsx_tc_gemm.cu:891`) already rejects non-CC-12, and this
  GPU *is* CC 12, so the `major != 12` guard is not what is firing here.

## Memory-leak audit (2026-11-20, branch `audit/memory-leaks`)

Planned-hygiene audit of the raw-ownership surfaces; design
`docs/superpowers/specs/2026-11-20-memory-leak-audit-design.md`. Every fix
landed red-test-first. Stage 1 (expert caches & loaders) closed; stages
2–4 (inference error/cancel paths, pinned-host parity, network/apps) in
progress on the branch.

### A3.1. Inkling `attn_query_offset` never freed — **fixed on the branch**
- `libs/lsxcommon/src/inkling_moe_load.cpp` allocated
  `pw.attn_query_offset = new int[num_layers]` but
  `FreeInklingPinnedWeights` had no `delete[]` — 192 B leaked per
  Load/Unload cycle on the real 48-layer model, process-lifetime
  accumulation on reload.
- Red: `test_inkling_moe_loader` "inkling pinned load releases the
  query-offset array (A3.1)" — array-new/delete interposition counts live
  `int[6]` allocations; delta 1 per cycle before the fix, 0 after.
- Fix: `delete[] pw.attn_query_offset; pw.attn_query_offset = nullptr;`
  in `FreeInklingPinnedWeights`.

### A3.2. Kimi `LoadGlobal` error paths leak uploaded globals — **fixed on the branch**
- `libs/lsxcommon/src/kimi_moe_load.cpp` `LoadGlobal`: every error return
  after `cudaMalloc(d_embed)` (head/norm alloc+upload failures, missing
  norm) returned without `pw.Free()`; `LoadKimiPinnedWeights` propagated
  the error without freeing either. The model path was covered by
  `AbortLoad`, but the loader's own contract (partial allocation released
  on error) was violated for direct callers.
- Red: `test_kimi_moe_loader` "LoadKimiPinnedWeights releases partial
  globals on failure (A3.2)" — norm.weight removed from global.arrow;
  second failed cycle on a fresh struct left `d_embed`/`d_head` live.
- Fix: `pw.Free()` on every post-allocation error return in `LoadGlobal`
  and on its failure in `LoadKimiPinnedWeights`.

### A3.3. DeepSeek V4-Flash / V4-Pro `load_pinned_weights` same defect — **fixed on the branch**
- `deepseek_moe_load.cu` / `deepseek_moe_pro_load.cu`: embed/head/norm
  alloc+upload failures and the hc-head presence/upload loop returned
  without `pw.Free()` (first release was at the layer-file loop). The
  `LayerWeights::Free` comment states the contract these paths broke.
- Red: "DeepSeek loader releases partial globals on a failed load (A3.3)"
  and "deepseek-v4-pro loader releases partial globals on failure (A3.3)"
  — hc_head_scale removed from global.arrow (batch stays valid, so five
  globals are allocated before the strict check fails); second failed
  cycle left `d_embed` live (e.g. `0x7a4a7a000000`).
- Fix: `pw.Free()` on every post-allocation error return in the globals
  section of both loaders.

### Stage 1 dismissed suspicions (documented, not fixed)
- `glm_moe_flash_load.cpp:868` pinned slab and `qwen_moe_exp_load.cpp:995`
  pinned arena: both register ownership in the store immediately after
  allocation; every later failure return leaves them owned, and the
  store `Free()` releases them. Clean.
- Expert-cache `new int32_t[]` sequences (mimo/nemotron/nemotron-ultra/
  gigachat/minimax/qwen4exp/`AllocateMoEExpertCache`): throwing `new`
  leaks earlier arrays + `d_base` only if host `operator new` throws
  mid-sequence (host OOM). Theoretical; no test can drive it without
  interposition; left as-is.
- minimax loader error paths, qwen4exp hc-cache builders, deepseek infer
  arena (`goto fail` → `Free()`), expert_store chunked pinning (failure
  munmaps all chunks): checked, clean.

## Memory-leak audit — stage 2 (inference error/cancel paths)

### A3.4. `RunKimiGeneration` leaked its per-call scratch on EVERY exit — **fixed on the branch**
- `libs/lsxcommon/src/kimi_moe_inference.cu`: `d_states`
  (61 × `sizeof(KimiMlaState)` ≈ 70 MiB), `d_bias_all`, `d_staging`,
  `d_ids`, nine workspace buffers and the pinned router pair were allocated
  per call and never freed — not even on success, because generation stages
  write into them (`d_states` carries the MLA KV). ~180 MiB leaked per
  `RunKimiGeneration` call, i.e. per prefill.
- Red: `test_kimi_moe_experts` "RunKimiGeneration error path releases its
  scratch (A3.4)" — 3 failed generations left **564,133,888 B** (538 MiB)
  of free VRAM missing; after the fix < 8 MiB.
- Fix: an RAII `KimiGenScratch` guard owns every per-call pointer
  (stream-drain then free on destruction); the hand-written frees at the
  two error tails removed.

### A3.5. `XingSession` had `Free()` but no destructor — **fixed on the branch**
- `libs/lsxcommon/include/lsxcommon/xing_moe_inference.h`: the model owns
  the session as `std::unique_ptr<XingSession>` and drops it with
  `session_.reset()` (TeardownWeights, and every non-continuation Infer),
  so the whole session — KV caches, ~30 scratch planes, the lazy FP16
  embed table (≈ 900 MiB on the real geometry) — leaked per reset.
  `MistralSession` (`~MistralSession() { Free(); }`) was the existing
  pattern; Xing was the only session struct missing it.
- Red: `test_xing_moe_model` "XingSession destructor releases the session
  (A3.5)" — 6,291,456 B over 3 unique_ptr cycles at capacity 8; after the
  fix < 1 MiB.
- Fix: `~XingSession() { Free(); }` **plus** replacing
  `FreeXingSession`'s terminal `s = XingSession{}` with in-place nulling —
  the temporary's destructor re-entered `Free` (infinite recursion; the
  first cut segfaulted every xing test, 16k-frame backtrace). Same
  recursion note MistralSession documents.

### A3.6. `EnsureWorkspace` left the partial workspace live on a failed chain — **fixed on the branch**
- `libs/lsxcommon/src/gemma_moe_inference.cu:1445`: the 47-buffer
  allocation chain short-circuits on the first failure and returned
  without `ws.Free()`, while `ws_ready` stayed false. The caller holds the
  context across Infers and has no handle on the individual pointers.
- Red: `test_gemma_moe_model` "gemma failed generation leaves no workspace
  behind (A3.6)" — after a failed generation `ctx->ws.d_x` was non-null
  (0x7e6ada022000). The VRAM-delta probe alone read 0 because
  `~GenWorkspace` frees at context destruction, which is exactly why the
  contract has to be checked at pointer level.
- Fix: `ws.Free()` on the chain's error return.
- Observed while writing the probe (not fixed, not a leak):
  `EnsureGemmaExpertCache` leaves `ctx.ec.slots > 0` with `ec.d_base ==
  nullptr` when its shrink-retry exhausts, and its entry guard is
  `if (ctx.ec.slots > 0) return cudaSuccess;` — a later call would report
  success with no cache. State defect, not an ownership defect; needs its
  own item if it ever bites.

### A3.7. Gemma NVFP4 embed gather reads out of bounds (pre-existing) — **open, not a leak**
- Found while collecting A3.6 evidence, reproduced **on a clean tree**
  (all audit changes stashed, full rebuild): `test_gemma_moe_model`
  "factory creates registered gemma model on synthetic fixture" fails with
  `an illegal memory access was encountered` and poisons the context, so
  the later cases in the same binary cascade.
  `compute-sanitizer --tool memcheck`: **446 errors**, all
  `Invalid __global__ read of size 1` in
  `gemma_embed_nvfp4_kernel` — access 32 B before the nearest allocation
  (2,048 B = the V=64/H=64 packed embed plane), i.e. the gather indexes
  just outside the NVFP4 embed planes on the synthetic vocabulary.
- The real checkpoints use vocab 262144 (tile-aligned), which is why no
  gate has caught it; the synthetic vocab is 64. Not caused by this audit
  and not an ownership defect — filed so it stops masquerading as a
  regression the next time this suite runs red.
- **Next step:** confirm the row/tile index the kernel derives
  (`row_packed = token*(hidden>>1)`, `row_scales = token*(hidden>>4)`)
  against the converter's packed/scales sizes for a non-16-multiple row
  count, then either pad the converter output or bound the kernel.

### Stage 2 dismissed suspicions (documented, not fixed)
- `InklingScratch::Allocate` partial failure: caller `AbortLoad` →
  `TeardownWeights` → `scratch_.Free()` covers every field. Clean.
- `hy`/`qwen`/`ornith`/`alice` pipelines: teardown is a `fail()` lambda or
  a `WorkspaceGuard` that frees the full set on every return. Clean.
- `mistral_moe`: `~MistralSession` releases partial `AllocateMistralSession`
  state through the owning `unique_ptr` on the error path. Clean.
- `FlashKdaStateAlloc` (frees d_state when the second alloc fails),
  `FlashIndexerStateAlloc` (single block, `Free` on failure): clean.

### A3.8. GLM-5.3-Flash loader/model suites red on `main` (pre-existing) — **open, not a leak**
- `test_glm_moe_flash_loader` and `test_glm_moe_flash_model` fail on a
  clean tree (`c4c4cd17`, audit changes stashed, full rebuild):
  `LoadGlmFlashDenseWeights: strict roundtrip (synthetic)` →
  `load error: layer_003 missing router bias`. The loader test's own
  synthetic fixture — which the converter still writes as `OK` — no longer
  satisfies the loader's strict tensor set, so the family's loader/model
  gates are red regardless of this branch.
- Not touched by the audit (no glm-flash file is in the diff) and not an
  ownership defect. Filed because a red inference label on `main` makes
  every later "the suite is green" claim false unless it is named.
- **Next step:** diff the fixture's `LayerRecords`-equivalent tensor list
  against `LoadGlmFlashExpertStore`'s strict set (`router bias` /
  `e_score_correction_bias`) and add the missing tensor to the fixture.

### A3.9. `GlmRealCtx::Free` double-`cudaFreeHost`s the pinned staging block — **open, no leak**
- `libs/lsxcommon/src/glm_moe_inference.cu:1183` allocates one pinned block
  and carves it into `c.h_routing = hbase` and
  `c.h_argmax = hbase + h_route`. `Free()` (:1076-1077) calls
  `cudaFreeHost` on **both**: the second call runs on an offset pointer
  inside a block that is already released, returns `cudaErrorInvalidValue`
  and leaves it sticky, so the caller's next `cudaGetLastError()` reports a
  failure that belongs to the teardown.
- Same class of defect the `FreeMistralSession` comment documents
  ("`h_slots` is page-locked, so it must not go through `cudaFree`").
- Not a leak (the block is released once) and not fixed here: `GlmRealCtx`
  lives in an anonymous namespace and the public entry point
  (`RunGlmGeneration`) needs the real GLM-5.2 vault, so no red-first test
  exists for it.
- **Next step / fix:** keep `if (h_routing) cudaFreeHost(h_routing);`, drop
  the `h_argmax` free (set it to null only), and add a generation-teardown
  case to `test_glm_moe_full_compute` asserting a clean `cudaGetLastError()`
  afterwards. Full pinned-host parity table:
  `docs/superpowers/...` — see the audit session record (stage 3).

## Memory-leak audit — stage 4 (network & apps)

**Result: no leaks found (documented, nothing to fix).** Two regression
pins added to `tests/test_http_server.cpp` (label `fast`, host-only), both
green on first run — they pin the contract rather than fixing a defect:

- "start/stop cycles leak no descriptors (A3.10)" — five
  build→serve→`Stop()` cycles must leave `/proc/self/fd` at the warm-up
  level (a leaked listener, detached worker or unclosed connection ratchets
  it by ≥1 per cycle).
- "the concrete port is reusable after Stop (A3.10)" — the same concrete
  port must bind again after `Stop()`, which proves the listening socket was
  actually released rather than left open.

Checked and clean by inspection:
- `ZmqContext` / `ZmqSocket` (`libs/lsxnetwork/src/zmq_context.cpp`): both
  close in their destructors, and every move-assignment closes the previous
  handle before taking the new one (the classic leak in this pattern). The
  zero-copy `SendZeroCopy` token is freed on all three failure paths and by
  `FrameReleaseCb` on the success path.
- `IoWorker`: `Start()` refuses a second start instead of reassigning a
  joinable thread; `Stop()` (and the destructor) joins. `std::promise`
  handoff cannot strand a thread.
- `lsxfabric --serve`: the cancel listener owns its PULL socket **inside the
  thread** (ZMQ sockets must not migrate threads), loops on a 200 ms
  `ZMQ_RCVTIMEO`, and is joined before `Serve()` returns; message payloads
  are closed on every branch (`zmq_msg_close` before `continue`/`break`).
- `taskmosaic` chat watcher: joined at the end of the scope that spawns it.

### A3.12. `test_kimi_moe_loader` red on `main` (pre-existing)

- `test_kimi_moe_loader` fails at its own fixture conversion:
  `REQUIRE(lsxcommon::kimi::ConvertKimiModel(in.string(), out).ok())` →
  `false`, immediately after `Reading: model-00001-of-00001.safetensors`. The
  synthetic checkpoint the case builds is no longer accepted by the converter it
  is meant to exercise, so two cases in that target are red.
- Reproduced 2026-11-28 by stashing an unrelated branch's changes and rebuilding
  at `207cac43`: identical failures, same two lines. So it is not caused by that
  branch — filed for the same reason as `A3.7`/`A3.8`,
  a red `inference` label on `main` makes a later "the sweep is green" claim
  false unless it is named.
- **Next step:** the converter rejects the fixture without a message that names
  the tensor or the key, so first give `ConvertKimiModel` a reason for this path
  (it currently returns a bare failure status), then decide whether the fixture
  or the converter is wrong.

### A3.11. `test_qwen_dense_converter_integrity` red on `main` (pre-existing)
- `Qwen: tokenizer without bos special still converts` fails with
  `ConvertQwenModel error: Invalid: tokenizer: no model object` — the test
  deletes a property from a copy of `tokenizer.json` and the qwen converter's
  parser then rejects the whole document instead of treating the missing `model`
  object as "no bos special". Reproduced 2026-11-22 in the untouched
  `/mnt/fast_storage/projects/cuda/logestix` checkout at `3bdff5f6`, so it is not
  caused by any branch under review.
- Filed for the same reason as `A3.7`/`A3.8`: a red `convert` label on `main`
  makes any later "the converter sweep is green" claim false unless it is named.
- **Next step:** decide which side is wrong — the fixture's edit (it removes more
  than the bos entry) or the converter's failure mode (a tokenizer without a
  `model` object should convert, since `bos` is optional by construction).
