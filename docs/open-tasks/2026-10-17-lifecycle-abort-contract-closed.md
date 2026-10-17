# Lifecycle audit: E1a / E1b / E1c closed (2026-10-17)

Items **§E1a**, **§E1b** and **§E1c** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md),
the deferred-decision cluster of the 2026-10-02 lifecycle review plus the
2026-10-16 audit that re-opened two of them. All three closed on 2026-10-17 on
branch `audit/lifecycle-abort-contract`. Kept verbatim, resolutions and
measurement trail included, per the archive convention.

Two real defects were found by the work rather than merely catalogued, and a
third was found and deliberately NOT fixed:

- **§E1a found:** a cancel poisoned the *next* request in qwen3.8-flash-next (a
  sticky `cudaEventElapsedTime` error on an unrecorded event). Fixed.
- **§E1c** then became a real change: per-step emit plus early EOS stop.
- **§E1d** was discovered while gating §E1c and is **still open** — it stays in
  the live file. Its trail is reproduced at the end of this snapshot, because it
  was measured here.

The §E1b resolution is documentation, not code: the lock's semantics were left
alone on purpose.

---

## §E. Design decisions deferred (from the 2026-10-02 lifecycle review)

### E1a. Cooperative-cancel is pinned end-to-end for every family
- **2026-10-16 audit — this item was understated, and it was not only a missing
  test.** `kLifecycle`'s abort marker was the literal string `abort_flag`, which
  every family satisfies by merely *assigning* the field
  (`gs.abort_flag = req.abort_flag;`). It could not tell "the flag was handed
  over" from "the flag is read", so the marker proved nothing. Separately, 12 of
  the 22 rows named only the `Infer` TU while their Emit/abort polls live in an
  unlisted device pipeline, so the check ran against code that makes no such
  decision. Both are fixed: the marker now requires a poll
  (`gs.Aborted()`/`gs->Aborted()`) and every row names the TU owning its
  generation loop (`869cb10`).
- That stricter marker immediately caught the case E1a did not describe:
  **qwen3.8-flash-next never polled the flag at all.** `RunExpGeneration`
  (`qwen_moe_exp_inference.cu`) took the `GenerationStream`, emitted every token
  through it, and called `Aborted()` zero times in the whole driver, so a
  cancelled request ran to `max_gen_tokens`. Fixed in the same commit; it is the
  only one of 22 families that did not route the call.
- **2026-10-16 (cont.) — closed.** `GenerationStream::Aborted()` had unit
  coverage (`tests/test_generation_stream.cpp`, `tests/test_model_engine.cpp`),
  but only five families asserted that a raised flag stops a real decode (alice,
  mimo, kimi, qwen3.8-27b, qwen3.5-35b-a3b), plus two with a pre-set-only check
  (gemma-dense, nemotron-nvfp4). Every one of the 22 `kLifecycle` families now
  calls the shared gate:
  - `tests/test_streaming_assert.h` grew `ExpectCooperativeCancelStops`: the
    pre-set poll, the mid-generation poll raised from the token callback (the
    real `--serve` path), and the control, in one call. It forces `ignore_eos`
    on all three runs so an EOS hit on random weights cannot make a short run
    pass for the wrong reason, and the control run (same budget, no flag) has to
    produce the full token count.
  - Wired into each family's model-stage target — the synthetic tests
    (qwen3.5-9b, qwen3.6-35b-a3b, gemma-moe, granite, phi, nemotron, gigachat,
    mistral) and the real-weight selfchecks (glm-5.2, glm-5.3-flash,
    deepseek-v4/v4.1, ornith, xing, qwen3.8-flash-next). The families that had
    only a pre-set check (alice, mimo, kimi, qwen3.8-27b, gemma-dense,
    nemotron-nvfp4) were upgraded to the full gate.
  - **qwen3.8-flash-next pins the pre-set half only**: its generation still
    emits in one bulk drain at the end (§E1c), so a flag raised from the token
    callback cannot be observed mid-generation until that drain is made
    incremental. Its gate is the pre-set run plus the control.
- **The new gate immediately found a second real defect in the same family.** An
  aborted `RunExpGeneration` produces zero tokens, so `t_ttft_ev` was never
  recorded; `cudaEventElapsedTime` on an unrecorded event sets a **sticky**
  `cudaErrorInvalidResourceHandle` that the *next* request's first checked
  launch reports as its own failure — a cancelled request poisoned the following
  one. Fixed by tracking whether TTFT was recorded and skipping the read
  otherwise (`qwen_moe_exp_inference.cu`).
- Verified 2026-10-16 (RTX 5090/Linux, `linux-debug`): fast suite 94/94; the
  synthetic cancel targets green; `test_qwen_moe_exp_selfcheck`,
  `test_ornith_moe_selfcheck` and `test_xing_moe_selfcheck` green on the vault
  weights. The real-weight gates for glm-5.2, glm-5.3-flash and DeepSeek-V4/V4.1
  are wired but SKIP here — those vault trees are not converted on this machine.

### E1b. The residency lock is a `shared_lock`, so it does not serialise Infer — resolved by documentation 2026-10-17
- Found 2026-10-16 (lifecycle audit). `ModelBase::RunUnderResidencyLock`
  (`model_base.h:132`) takes `std::shared_lock`. A shared lock admits any number
  of holders at once, so it orders `Infer` against `Unload()`'s unique lock and
  nothing else.
- The comments claimed more. Eight family comments and the guard's own rationale
  asserted that moving session mutation inside the lock stops a concurrent
  `Infer` from freeing another's KV cache. It does not: two concurrent `Infer`
  calls both hold the shared lock simultaneously. The ordering check the guard
  performs is real but proved less than its comment claimed.
- Latent, not live: every dispatch path is sequential (orchestrator: one
  consumer thread; `lsxfabric --serve`: one request at a time).
- **Resolved by taking the second option** — state the precondition and correct
  the claims, rather than change the lock's semantics:
  - `RunUnderResidencyLock`'s own comment now says what it does (order against
    `Unload()`'s unique lock) and what it does not (not `Infer` vs `Infer`),
    names the one-`Infer`-per-instance precondition explicitly, and cites this
    item.
  - All eight family comments (qwen_base, qwen_moe, qwen_moe_exp, qwen_moe_36
    via its base, ornith, granite, phi, gemma_dense, xing) carry the same
    wording.
  - The guard's rationale and its violation message no longer claim mutual
    exclusion; the check is documented as a scope/ordering check, which is what
    it has always actually been.
- **Still open if the fleet ever wants concurrent dispatch:** make the generation
  scope exclusive (serialises generation — measure before choosing) or plumb
  per-request state. Neither was done here.

### E1c. qwen3.8-flash-next streams in one bulk drain at the end — closed 2026-10-17
- Found 2026-10-16. `RunExpGeneration` generated the whole `max_gen_tokens`
  budget and only then emitted, in a single pass ("Bulk token drain"). A
  streaming consumer saw nothing until the request finished, and EOS did not
  stop the loop early either — truncation happened after the drain.
- **Landed.** Both halves, in the step where the token is already read back from
  the device for the PLE n-gram history (so it costs no extra transfer):
  - the token is published through `gs->Emit` per step, and EOS is not published
    (the contract the drain had, now applied per step);
  - EOS terminates the decode loop instead of being applied to the result
    afterwards.
- **The published token list is unchanged, deliberately.** Greedy decoding makes
  the early stop an observation, not a behaviour change: every token before the
  EOS depends only on the tokens before it, and the post-loop
  `MoETruncateTokensAtEos` is a no-op on the early-stop path because the EOS is
  the last recorded token. Confirmed by the goldens — DET, FSI and CODE all pass
  unchanged (they run 512/256-token budgets with EOS enabled, so they do hit the
  early stop). `final_logits` does move when EOS lands mid-budget (it becomes the
  EOS step's row instead of the last budget step's); the bench protocol sets
  `ignore_eos`, so the perf baselines are untouched, and the Paris selfcheck
  compares `final_logits` only against itself.
- The E1a gate for this family is now the full
  `ExpectCooperativeCancelStops`: its callback half is the E1c gate, and it was
  watched failing first (`24 < 24`) against the old drain.
- The early-EOS half is a separate target, `test_qwen_moe_exp_eos_selfcheck`, and
  it pins `LSX_QWEN38_EXPERT_CACHE_SLOTS=1024`. Two things it took to get right:
  - **`gen_tokens.size() < max_gen_tokens` cannot show the loop stopped early.**
    The post-loop truncation makes the list short in both cases, so that
    assertion also passes with the early break disabled. The gate reads KV
    growth instead (one row per executed step): `kv_slots_used` is ~10 with the
    early stop and 69 without it. Verified by mutation — disabling the early
    break turns the target red on exactly that assertion.
  - **It needs a fresh process**, because of §E1d.

### E1d. qwen3.8-flash-next's answer depends on what ran before it in the process
- Found 2026-10-17 while gating E1c. Pre-existing, unrelated to the E1c change,
  and more serious than E1c: **the same request with the same weights returns a
  different answer depending on the request that preceded it in the process.**
  Measured on the real weights (RTX 5090), prompt "Say nothing at all.",
  `max_gen_tokens=64`:
  - first request in the process → EOS on the 5th generated token;
  - after a 2-token request (the Paris selfcheck) in the same process → a 64-token
    explanation of the idiom, no EOS.
- Not explained by expert-cache residency: pinning
  `LSX_QWEN38_EXPERT_CACHE_SLOTS` (1024) does not remove it, and neither does a
  warm-up request at the same budget. Zeroing the whole generation arena changes
  nothing either, so it is not stale arena contents.
- **Narrowed to the model instance, not the request.** Within ONE instance the
  driver is stable: five consecutive requests, with an unrelated short request
  interleaved, all returned the identical result. The trigger is a *completed
  `Infer` on a previous model instance*: Load/Unload with no Infer leaves the
  next instance correct, Load → one Infer → Unload makes the next instance
  answer differently (EOS on token 5 instead of 8).
- **Suspect identified, not yet fixed.** Disabling the cross-call lookup in the
  process-global fused-shared-expert cache (`sh_cache`, keyed by the *device
  pointer* `sw1.d_packed`, in `RunMoESharedCombine`) makes the second instance
  correct again, reproducibly. Its comment — and the sibling
  `LsxTcFusedOutputScratch` in `lsx_tc_gemm.cu` — assume the key is stable for
  the process lifetime ("the LsxTcBuildFusedGemmWeights cache holds the pointers
  for the process lifetime, so the key is stable"). That assumption is what
  breaks: CUDA recycles device addresses across model lifetimes, so a new model
  can land on a freed key and inherit entries built from the previous model.
- Separately, the same class of pressure shows up in the other direction: the
  cache is sized from free VRAM at `Load`, and at `LSX_QWEN38_EXPERT_CACHE_SLOTS=8192`
  the Paris selfcheck itself fails on real weights (256, 1024 and 5350 pass).
- **Why the DET gate does not see it:** DET runs its stress cycles against one
  loaded model with one residency setting, so it is deterministic within a
  configuration and blind across configurations.
- **Next step:** bisect further — the trigger is narrowed to "a completed
  `Infer` on a previous model instance", not to a request shape. Then either key
  those caches on an owner-scoped identity (the pinned-weights object owns the
  fused buffers and frees them) or remove them and re-measure; both touch the
  shared-expert hot path of every MoE family, so neither can be landed without
  re-running the DET/FSI/CODE goldens of the families that use it. Until then,
  the E1c EOS gate lives in its own binary, which is a workaround for the test,
  not a fix.


---

## Trail of the §E1d discovery (measured 2026-10-17, still open)

§E1c's EOS gate had to be made order-independent before it could be trusted, and
that work is what surfaced this. Kept here because the live §E1d entry is a
summary and this is the measurement trail.

The symptom: the same request with the same weights returns a different answer
depending on which request ran before it in the process. Real weights (RTX 5090),
prompt "Say nothing at all.", `max_gen_tokens=64`:

| position in the process | result |
|---|---|
| first request | EOS on the 5th generated token |
| after a 2-token request (the Paris selfcheck) | a 64-token explanation, no EOS |

Ruled out, each by measurement:

| hypothesis | test | result |
|---|---|---|
| expert-cache residency | `LSX_QWEN38_EXPERT_CACHE_SLOTS` pinned to 256 / 1024 / 8192 | does not remove it (and is not monotonic: 256 and 1024 pass, 8192 fails) |
| arena/thread-local scratch growth | warm-up request at the same budget | does not remove it |
| stale arena contents | zero the whole generation arena each call | no change at all |
| request shape / ordering within one instance | 5 consecutive requests, an unrelated short one interleaved | all identical — the driver is stable within one instance |

Narrowed to the model instance: Load/Unload with **no** Infer leaves the next
instance correct; Load → one Infer → Unload makes the next instance answer
differently. The trigger is a completed `Infer` on a *previous* model instance.

Suspect: the process-global fused-shared-expert cache `sh_cache` in
`RunMoESharedCombine` (`lsx_moe_inference.cu`), keyed by the **device pointer**
`sw1.d_packed`, holding entries for the process lifetime. Disabling its
cross-call lookup makes the second instance correct again, reproducibly. Its
comment, and the sibling `LsxTcFusedOutputScratch` in `lsx_tc_gemm.cu`, assume
the key is stable ("the LsxTcBuildFusedGemmWeights cache holds the pointers for
the process lifetime, so the key is stable") - that assumption is what breaks,
because CUDA recycles device addresses across model lifetimes.

**Not landed, deliberately.** The fix touches the shared-expert hot path of every
MoE family and could not be validated against the DET/FSI/CODE goldens of all of
them in that session; shipping it unverified would risk silently changing
generations for 8+ families. The §E1c EOS gate lives in its own executable as a
workaround for the test, which is not a fix.

Why DET cannot see it: its stress cycles share one loaded model and one residency
setting, so it is deterministic *within* a configuration and blind *across*
configurations.
