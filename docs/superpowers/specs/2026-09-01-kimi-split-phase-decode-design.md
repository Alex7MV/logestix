# Kimi Router Split-Phase + Device-Side Decode — Design Spec

**Date:** 2026-09-01
**Status:** Approved (design)
**Scope:** `libs/lsxcommon` Kimi-K2.7-Code inference handshake sites. Resolves audit finding **F-11** (`2026-08-23-blackwell-cuda-audit-findings.md` — full-drain `cudaStreamSynchronize` per layer per token) and the per-step greedy argmax drain.
**Branch:** `feat/deepseek-expert-prestage` (both F-11 work items in this branch; user-approved scope: router handshake **+ device-side decode**).

## 1. Problem Statement

Kimi-K2.7-Code drains the GPU completely at two hot-loop sites:

1. **Router ids readback** (`kimi_moe_inference.cu:1015`): after
   `KimiRouterTopK`, the host does `cudaMemcpyAsync` D2H (top-k ids +
   weights) followed by a **blocking `cudaStreamSynchronize`** — per token,
   per MoE layer. The GPU idles until every in-flight kernel retires, and
   expert staging cannot even be *enqueued* until the sync returns. This is
   the pre-H-01 pattern that DeepSeek already replaced with a split-phase
   event fence.
2. **Greedy decode readback** (`:1300`): per generation step, argmax output
   (`d_best`/`d_bestv`) is copied D2H with a blocking sync to feed the next
   embed gather from host.

The handoff queue (Round 22) prescribes: "Kimi router split-phase handshake
(H-01 pattern) + TDD bitwise-equivalence test for the split API."

## 2. Decisions (user-approved)

| Decision | Choice |
|---|---|
| Router handshake | Split API mirroring DeepSeek `route_token_begin/finish`: `KimiRouteBegin` (top-k kernel + async D2H + event record) / `KimiRouteFinish` (busy-poll event, no blocking sync). |
| Decode feed | Device-side: next embed-gather reads `*d_best` on device; one D2H per generation (tokens batched at the end); ttft via events; `first_maxv` from a device slot. |
| EOS handling | Host-truncate after the final bulk D2H: the loop runs all `max_gen_tokens` steps on device; the host then cuts `gen.tokens` at the first EOS and sets `eos_terminated`. Removes the per-step fence entirely; wasted post-EOS compute is bounded by `max_gen_tokens`. The hardcoded `kKimiEosId = 163586` in the .cu is replaced by an `eos_id` parameter threaded from `tok_db_->eos_id` (Ingestion Protocol step 4). |

## 3. Architecture

### 3.1 Split-phase router handshake

New public pair (defined in `kimi_moe_inference.cu`, declared in
`kimi_moe_kernels.h`, namespace `lsxcommon::kimi`):

```cpp
// Begin: launch KimiRouterTopK on the compute stream, issue the two async
// D2H copies of the top-k ids/weights into pinned host staging, and record
// @p ev on the stream. Returns immediately; the GPU keeps running whatever
// was already queued (and the caller may enqueue more work between phases).
cudaError_t KimiRouteBegin(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_top_ids, float *d_top_w,
                           int *h_ids, float *h_w,
                           cudaEvent_t ev, cudaStream_t stream);

// Finish: busy-poll @p ev (cudaEventQuery loop — event-scoped, never waits
// for unrelated stream work, unlike the old cudaStreamSynchronize), then
// return. The pinned h_ids/h_w staging is valid after this call.
cudaError_t KimiRouteFinish(cudaEvent_t ev);
```

- Pinned staging (`h_ids`/`h_w`) lives in the generation context next to
  the existing `d_top_ids`/`d_top_w` device slots (pinned host block).
- Event is caller-owned (create per call site or reuse a persistent one
  created with `cudaEventDisableTiming`, primed once — mirror of the
  `ready` fence in `kimi_moe_expert_cache.cu`).
- `KimiLayerStack` call site (`:1004-1015`): fused
  `KimiRouterTopK + memcpyAsync + cudaStreamSynchronize` becomes
  `KimiRouteBegin` → (host: resolve batch, enqueue upload) →
  `KimiRouteFinish` before reading `h_ids`/`h_w`. The cache resolve/upload
  logic itself is untouched; only the fence changes from stream-drain to
  event-scoped busy-poll.
- Bit-exactness: the same top-k kernel runs; the host consumes identical
  bytes. No numeric change whatsoever.

### 3.2 Device-side greedy decode feed

`RunKimiGeneration` decode loop (`:1240-1330`):

- Steps ≥ 1: after `kimi_argmax_kernel`, replace the
  `cudaMemcpyAsync(next_id) + cudaStreamSynchronize` pair with a
  **device-side feed**: the next iteration's embed-gather consumes
  `d_best` directly (device pointer — either pass `d_best` as the ids
  pointer for the 1-token gather, or a one-liner copy kernel
  `d_ids[prefill_seq + step] = *d_best` on the stream). No host round-trip.
- `gen.tokens`: collected by ONE bulk D2H after the loop finishes
  (contiguous `d_ids` tail), instead of per-step sync.
- `ttft_ms`: `cudaEventRecord(t_start/t_ttft)` around the first token —
  host wall-clock replaced by event timing (host no longer observes
  per-step boundaries).
- `first_maxv`: read from `d_bestv` in the same final bulk D2H.
- Loop control: the current loop has a per-step host EOS check
  (`kKimiEosId = 163586` hardcoded at `:1323` — an Ingestion Protocol step 4
  violation). With device-side feed the host no longer sees per-step ids, so
  EOS becomes **host-truncate**: the loop runs all `max_gen_tokens` steps;
  after the final bulk D2H the host scans `gen.tokens`, cuts at the first
  EOS token, sets `gen.eos_terminated` accordingly and resizes
  `kv_slots_used`. Token stream before the cut is identical to the fused
  path (same kernels, same argmax); behavior differs only in that
  post-EOS steps compute but are discarded — bounded waste, zero syncs.
  `RunKimiGeneration` gains an `int eos_id` parameter (from
  `tok_db_->eos_id` at the two model call sites); the hardcoded constant
  is removed.
- Bit-exactness: identical kernels, identical argmax, identical id
  sequence — the token stream is unchanged by construction; the selfcheck
  contract (`"Paris"`, 3-cycle bit-identical) is the binding test.

### 3.3 What stays untouched

- `KimiMoeResolveBatch` / `UploadKimiMoeExpertBatch` / copy-stream + `ready`
  event machinery (already event-scoped).
- Legacy per-call staging path (cache disabled).
- The fused `KimiRouterTopK` entry (tests + legacy callers keep it).
- All kernel math, router math, expert FFN accumulation order.

## 4. Components & files

| Unit | File | Change |
|---|---|---|
| Split API | `libs/lsxcommon/src/kimi_moe_inference.cu` | `KimiRouteBegin`/`KimiRouteFinish` + call-site swap |
| API decl | `libs/lsxcommon/src/kimi_moe_kernels.h` | declarations next to `KimiRouterTopK` |
| Decode feed | `libs/lsxcommon/src/kimi_moe_inference.cu` (`RunKimiGeneration`) | device-side id feed, single bulk D2H, event ttft |
| Tests | `tests/test_kimi_moe_router.cu` | begin/finish vs fused bitwise equivalence |
| Tests | `tests/test_kimi_moe_selfcheck.cpp` | existing 3-cycle contract re-run |

No new files (naming guard stays green); all edits inside existing kimi
units.

## 5. Acceptance gate

1. **Split-API TDD (fails first):** new test in
   `test_kimi_moe_router.cu` — `KimiRouteBegin/Finish` vs fused
   `KimiRouterTopK + sync` produce bitwise-identical ids and weights
   (n=384, top_k=8, adversarial duplicate scores included).
2. **EOS TDD (fails first):** new test covering the host-truncate helper —
   token list with an embedded EOS cuts at EOS, sets `eos_terminated`,
   `kv_slots_used` shrinks; no-EOS list passes through; eos_id `-1`
   disables truncation. Plus: `RunKimiGeneration` EOS path exercised
   end-to-end in `test_kimi_moe_inference.cu` with a forced-EOS prompt.
3. **Bit-exact e2e:** `test_kimi_moe_selfcheck` — "Paris", 3 cycles
   byte-identical (real weights present at the vault; runs on this box).
4. **Suites:** `ctest -R kimi` green (12 executables).
5. **Perf guard:** tok/s must not regress; expected large gain from
   removing 384 × steps full-drain syncs — measured before/after via the
   kimi perf/selfcheck harness and recorded in the findings registry
   (Round 25).

## 6. Non-goals

- No sampling changes (top-k math untouched).
- No change to the expert cache LRU policy.
- No changes to DeepSeek/Qwen/GLM paths.
- No `kind::mxf8f6f4` / kernel-level work (separate items).
