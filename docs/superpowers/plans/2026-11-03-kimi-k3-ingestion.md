# Kimi-K3 ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline — AGENTS.md forbids subagents). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ingest Kimi-K3 (2.8T, KimiLinear: KDA + gated NoPE MLA + AttnRes + Stable LatentMoE + SiTU) into logestix under registry `kimi-k3`.

**Architecture:** New frozen passport + dedicated converter/loader/kernels/model for the `kimi` family variant `k3`; attention pinned, latent+shared MoE plane mmap-streamed, routed experts through the neutral `ExpertStore`.

**Tech Stack:** C++20, CUDA (Blackwell, sm_120), CMake presets, Catch2, Arrow, neutral `lsx_inference_nvfp4` / `lsx_moe_inference` / `lsx_attn_inference`.

Design: `docs/superpowers/specs/2026-11-03-kimi-k3-ingestion-design.md`.

## Global Constraints

- TDD: failing test first, then minimal code; each test gets its ctest label in the same commit.
- Frozen geometry in `kimi_moe_k3_constants.h` + `static_assert`; runtime config reads are forbidden.
- NVFP4 only via the neutral layer; one stream; no per-token `cudaDeviceSynchronize`; no `atomicAdd` into `final_logits`.
- No abseil containers in headers reachable from `.cu`.
- Host routed-expert storage is the neutral `gpu::expert_store::ExpertStore`; Arrow payloads are mmap views.
- Every family TU follows `<family>_<type>_<variant>_<role>`: `kimi_moe_k3_*`; add `k3` to `SchemeRegex()`'s variant set in the same commit as the files.
- Lifecycle contract in the generation TU: `RunUnderResidencyLock`, `GenerationStream` + abort, `TeardownContext`, `AcquirePermanentWeights`/`AcquireExpertCacheLease`, filled `InferenceResult`.
- Build/test: `cmake --preset linux-debug && cmake --build --preset linux-debug`, `ctest --preset linux-debug -L fast`, `... -L convert`, `... -L inference`.

## Geometry constants (verbatim, used by every task)

```
kHidden=7168 kLayers=93 kDenseLayers=1 kAttnResBlock=12
MLA(24): kMlaHeads=96 kQLora=1536 kKvLora=512 kQkNope=128 kQkRope=64 kVHead=128
         kMlaScale=1/sqrt(192) kMlaLayerMask = layer%4==3 || layer==92
KDA(69): kKdaHeads=96 kKdaHeadDim=128 kConvKernel=4 kGateLowerBound=-5.0
MoE: kExperts=896 kTopK=16 kMoeInter=3072 kLatent=3584 kSharedInter=6144
     kSharedExperts=2 kRenorm=true kRouteScale=1.0
SiTU: kSituBeta=4.0 kSituLinearBeta=25.0
vocab 163840, bos 163584, eos 163586 (+163585), pad 163839, eps 1e-5
format tag logestix-kimi-k3-nvfp4, tokenizer_kind 2
```

---

### Task 1: Frozen passport + guards

**Files:** Create `libs/lsxcommon/src/kimi_moe_k3_constants.h`; modify `tests/test_code_naming_guard.cpp` (variant set + `kimi_moe_k3_tool_format` allowlist), `tests/test_model_independence_guard.cpp` (`kPassports` row `{"kimi-k3","kimi_moe_k3_constants.h","lsxcommon::kimi_k3::frozen"}`), `tests/test_model_suite_guard.cpp` (`Suites()` row + tool verdict).

- [ ] Add the `kimi-k3` passport row and the `k3` variant token; run `ctest -L fast -R "naming|independence|suite"` → RED (missing header / missing stages).
- [ ] Write `kimi_moe_k3_constants.h`: geometry above, `kMlaLayers`/`kKdaLayers` tables, `static_assert`s (`kLayers==93`, 69 KDA + 24 MLA, `kHidden % 128 == 0`, `kTopK<=kExperts`).
- [ ] Re-run the guards → GREEN for passport existence; suite guard still RED until stages 2–8 add targets (row marked with the final targets at the end of the ingestion).
- [ ] Commit `feat(kimi-k3): frozen geometry passport`.

### Task 2: Converter + synthetic E2E (stage 1)

**Files:** Create `libs/lsxcommon/src/kimi_moe_k3_convert.{h,cpp}`, `libs/lsxcommon/src/kimi_moe_k3_dequant.cu`; modify `apps/lsxforge/main.cpp`, `libs/lsxcommon/CMakeLists.txt`; test `tests/test_kimi_k3_convert.cpp` (`convert` label, TIMEOUT 120, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`).

- [ ] RED: synthetic fixture with `language_model.model.layers.{0,1,3}.…`, MXFP4 expert pairs (`U8 [rows,cols/2]` + `U8 [rows,cols/32]`), globals, `vision_tower.*` + `mm_projector.*` ballast; assert index tag, `skipped_visual_tensors`, per-layer `kda`/`mla` tables, `.res`-free records, and expert dequant reconstructing the source (relL2 < 0.09 vs base).
- [ ] Implement: parse `config.json` (`text_config`), classify tensors, strip+count ballast, BF16→NVFP4 via `gpu::quantize_tensor_gpu`, MXFP4→BF16 (GPU, `DequantMxfp4E8m0Group32`) → NVFP4 per expert with its own `global_scale`, write `<out>/logestix/{pinned,experts,model.index.json}`.
- [ ] GREEN, commit `feat(kimi-k3): converter and synthetic E2E`.

### Task 3: Tokenizer bridge + XTML tool codec (stages 4/4b)

**Files:** `libs/lsxcommon/src/tokenizer_convert.cpp` (only if kind 2 needs the K3 pattern), `libs/lsxcommon/src/kimi_moe_k3_tool_format.{h,cpp}`, `tests/test_kimi_k3_tool_framing.cpp` (`fast`), `tests/test_kimi_k3_tokenizer_parity.cpp` (`inference`, SKIP 4).

- [ ] RED: parity test vs Python `tiktoken` on `tiktoken.model` (30 mixed prompts incl. Han, digits, contractions) → watch fail if the kind-2 pattern differs; fix the pretokenizer/kind only if it fails.
- [ ] RED: framing test — system/tool/user/assistant with `tool_calls`, generation prompt (`<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`), parse a generated XTML tool call back to `{"name":…,"arguments":…}`.
- [ ] Implement `BuildConversationInputIds` (XTML segments exactly as `encoding_k3.py`) and `ParseToolCalls` (scan for `<|open|>call tool="…"`, `argument` blocks, `json` blocks); control markers injected as ids, free text BPE-encoded.
- [ ] GREEN, commit `feat(kimi-k3): XTML chat and tool bridge`.

### Task 4: Registry + loader (stage 3)

**Files:** `libs/lsxcommon/src/kimi_moe_k3_load.{h,cpp}`, `libs/lsxcommon/src/model_registry.cpp`, `tests/test_kimi_k3_loader.cpp` (`load`).

- [ ] RED: synthetic converted fixture loads: passport cross-check rejects a wrong geometry, pinned tensors uploaded, streamed plane kept as Arrow mmap views, `ExpertStore` geometry from the index (8×4 fixture), `Free()` releases every plane.
- [ ] Implement: `KimiK3Weights` (pinned `LsxNvfp4Weight` graph + streamed plane views + `KimiK3ExpertDb = ExpertStore<KimiK3ExpertEntry>`), env knobs `LSX_KIMI_K3_EXPERT_CACHE_SLOTS`, `LSX_KIMI_K3_MAX_OPEN`, `LSX_KIMI_K3_PIN_DENSE`.
- [ ] GREEN, commit `feat(kimi-k3): loader with streamed latent/shared plane`.

### Task 5: Kernels (stages 5/6)

**Files:** `libs/lsxcommon/src/kimi_moe_k3_kernels.{h,cu}`, `kimi_moe_k3_inference_attn.{h,cu}`, `tests/test_kimi_k3_kernels.cpp` (`kernels`).

- [ ] RED (host references): SiTU-GLU; router top-16 (bias selects, raw sigmoid renormalized); AttnRes 9-source softmax mix; KDA one-step + 4-step recurrence (per-channel A_log, lower-bound gate, conv-4 silu); gated NoPE MLA attention (latent cache, scale 1/sqrt(192), output gate).
- [ ] Implement kernels as specified in the design §5; deterministic reductions; no atomicAdd into logits.
- [ ] GREEN, commit `feat(kimi-k3): KDA/MLA/AttnRes/SiTU kernels`.

### Task 6: Model + lifecycle (stage 8)

**Files:** `libs/lsxcommon/src/kimi_moe_k3_inference_pipeline.{h,cu}`, `kimi_moe_k3_model.{h,cpp}`, `tests/test_kimi_k3_model.cpp` (`model`), `tests/test_kimi_k3_model_lifecycle.cpp`, `tests/test_kimi_k3_selfcheck.cpp` (`inference`, SKIP 4).

- [ ] RED: synthetic model test (streaming: every token emitted through `req.token_callback`; pre-set `abort_flag` yields no tokens; session state published under the lock); selfcheck target registered with `SKIP_RETURN_CODE 4`.
- [ ] Implement the 93-layer driver (AttnRes block bookkeeping, streamed plane H2D per layer, expert residency, `kOff`/auto fast math, `TeardownContext`), `kType = "kimi-k3"`.
- [ ] GREEN, commit `feat(kimi-k3): model, pipeline and lifecycle`.

### Task 7: Parity + perf gates (stage 8 continued)

**Files:** `tests/test_kimi_k3_fsi_selfcheck.cpp`, `..._code_selfcheck.cpp`, `..._det_selfcheck.cpp`, `tests/fixtures/{fsi,code,det}/kimi-k3.json`, `tests/perf/perf_baselines.json`, `tests/test_perf_baseline.cpp`, `tests/CMakeLists.txt`.

- [ ] Capture template-derived goldens with the exact XTML framing; write the three gates (`inference`, SKIP 4) and the `kGoldenDebt` entry if a gate is red for a measured reason.
- [ ] `add_test(NAME test_perf_baseline_kimi_k3 COMMAND test_perf_baseline "perf baseline: kimi-k3")`, label `perf`, SKIP 4; seed rows through the harness only.
- [ ] Commit `test(kimi-k3): parity gates and perf baseline`.

### Task 8: Real-weight conversion + gates

- [ ] `LSX_MODEL_ROOT=/mnt/fast_storage/huggingface lsxforge --kimi_k3 Kimi-K3` (writes `Kimi-K3/logestix/`), verify tree size ≈1.56 TB and index counts (93 layers, 896×92 experts, ballast counted).
- [ ] SelfCheck Paris ×3 bit-exact; FSI/CODE/DET; tool round-trip; perf rows via the harness. Record what cannot run and why.

### Task 9: Docs/report (stage 9)

- [ ] `docs/model-reports/2026-11-03-Kimi-K3.md` (publish-ready post + evidence), `AGENTS.md` structure bullet, `docs/OPEN_TASKS.md` section, `docs/SELFCHECK_MATRIX.md` row; final `ctest -L fast` green.
- [ ] Commit `docs(kimi-k3): ingestion report and open items`.
