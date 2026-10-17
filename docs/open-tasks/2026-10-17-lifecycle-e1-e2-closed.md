# Lifecycle audit: E1 and E2 closed (2026-10-17)

Items **§E1** and **§E2** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), closed on
branch `audit/lifecycle-abort-contract`. Both by decision rather than by patch:
E1 was resolved by making the existing single-dispatch assumption machine-checked
(no parallel dispatch is planned), E2 by aligning alice with the rest of the fleet.
Kept verbatim with the resolutions, because the E2 record also corrects a wrong
assumption about the goldens that is worth preserving.

---

### E1. `LsxFastMathScope` assumes sequential cross-model dispatch — closed 2026-10-17 by a guard test
- The kernel-routing toggles are process-global (`std::atomic<bool>`, so no UB
  — `libs/lsxcommon/src/lsx_inference_nvfp4.cu:18`), and every model asserts
  its own policy per `Infer`. Two models inferring **concurrently** would flip
  each other's kernel routing mid-decode: silently wrong numerics, no crash.
- Today every dispatch path is sequential (orchestrator: one consumer thread;
  `lsxfabric --serve`: one request at a time), and the assumption is written
  down in `lsx_inference_nvfp4.h:57`. So this is latent, not live.
- **Resolved by the decision not to plan parallel dispatch, so the assumption is
  made machine-checked rather than left as a comment.** `LsxFastMathScope`
  maintains a process-global depth counter (`Detail::ActiveFastMathScopeDepth`,
  inline in the header so the guard needs no lsxcommon link). Two scopes open at
  once logs a loud `VIOLATION` naming E1 instead of silently producing wrong
  numerics. Same-thread nesting counts as depth and is NOT flagged — nesting is
  legitimate sequential dispatch; only a second *thread* is a violation.
- Pinned by two guard cases in `tests/test_lsx_nvfp4_dispatch.cu`:
  - depth tracks live scopes and returns to zero, and `kAuto` correctly does not
    open a scope while `kOn`/`kOff` do;
  - a second thread opening a scope while the main thread holds one is observed
    at depth 2 — so the counter is proven to be shared state, not a constant or a
    per-thread value, and the guard is not vacuous.

### E2. `alice` publishes the FIRST logits row; every other family the LAST
- `alice_moe_inference_pipeline.cu:492` fills `final_logits` when
  `first_token < 0`, so `InferenceResult::final_logits` is the row of the
  first generated token. All other families publish the last row
  (`qwen_inference.cu:2263`, `granite_dense_inference.cu:987`,
  `nemotron_moe_model.cpp:226` "Keep the last step's logits row").
- Within alice this is self-consistent (its DET gate compares it to itself),
  but it makes any cross-model logits/hashes comparison apples-to-oranges.
- **Not changed because it would move alice's DET/FSI goldens**, which are
  captured from real weights. Decide the fleet-wide semantics first, then
  re-capture the affected goldens in one commit.


---

## What these two closures rest on

**E1 — a guard, not a redesign.** The kernel-routing toggles stay process-global
because CUDA selects kernels per launch and there is no per-model device context;
that is only safe while dispatch is sequential. With no parallel dispatch planned,
the cost/benefit of moving the state per-stream is not there, so the invariant is
now enforced instead of trusted: `LsxFastMathScope` keeps a process-global depth
counter and logs a loud violation on overlap, with two guard cases pinning that
it works — including one that spawns a second thread to prove the counter is
really shared state and the guard is not vacuous. Same-thread nesting is
deliberately not flagged; only a second thread is the violation.

**E2 — the recorded risk was wrong.** The entry said changing alice would "move
its DET/FSI goldens". Reading the harness first showed it would not: the
DET/FSI/CODE gates read `gen_tokens`, never `final_logits`, and the one test that
reads alice's `final_logits` compares two runs against each other. So alice now
overwrites the row every decode step and keeps the last one — the fleet contract
— with no golden re-captured. All four real-weight alice gates stayed green.
