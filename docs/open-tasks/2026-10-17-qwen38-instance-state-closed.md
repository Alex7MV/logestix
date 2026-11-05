# qwen3.8-flash-next: a second instance decoded from the first one's fused weights (E1d, closed 2026-10-17)

Item **§E1d** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), discovered while gating
§E1c and closed the same day on branch `audit/lifecycle-abort-contract`. It was
catalogued as "suspect identified, not yet fixed"; the fix landed in the same
session once the affected hot path could be validated. Kept verbatim, with the
resolution appended.

This is the second defect this session's lifecycle work found. Like §E1a, it is
not a missing test: the family produced wrong answers, silently, and the test
that should have noticed could not see it.

---

### E1d. qwen3.8-flash-next's answer depends on what ran before it in the process
- Found 2026-10-17 while gating §E1c. Pre-existing, unrelated to the §E1c
  change, and more serious than it: **the same request with the same weights returns a
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
  the §E1c EOS gate lives in its own binary, which is a workaround for the
  test, not a fix. The measurement trail is in the
  [2026-10-17 snapshot](2026-10-17-lifecycle-abort-contract-closed.md).


---

## Resolution (2026-10-17)

Four process-lifetime caches in the shared neutral layers are keyed by a weight's
**device address** and never invalidated. CUDA recycles device addresses across
model lifetimes, so an instance created after another one was unloaded can land
on a freed key and inherit entries built from the previous model's weights. The
symptom is silence: no crash, no log line, just a different answer for the same
request and the same weights.

| cache | file | key | worst case |
|---|---|---|---|
| `sh_cache` | `lsx_moe_inference.cu` (`RunMoESharedCombine`) | `sw1.d_packed` | decodes the shared expert from another model's weights |
| `g_fw_cache` | `lsx_tc_gemm.cu` (`LsxTcBuildFusedGemmWeights`) | the four source pointers | backs the attention `qkv\|z\|a\|b` fusion |
| `g_tma_cache` | `lsx_tc_gemm.cu` (`GetWeightTmaMap`) | `w_packed` | a `cuTensorMap` encodes the global address, so a stale hit makes TMA load **through a freed address** |
| `LsxTcFusedOutputScratch` | `lsx_tc_gemm.cu` | the fused pointer | serves another model's GEMM output buffer |

Their comments asserted the assumption that breaks: "the LsxTcBuildFusedGemmWeights
cache holds the pointers for the process lifetime, so the key is stable".

**Fix.** A monotonic weight epoch, `LsxTcWeightEpoch()` in the neutral TC header,
bumped by `ModelBase` in `FinishLoad()` and in `Unload()`/`AbortLoad()` — exactly
when weights become resident or are released. Every cache records the epoch its
entries were built at and treats a mismatch as a miss. Rebuilding produces values
*identical* to the ones it replaces, because the fused weights are a pure function
of the source weights: the cost is a rebuild, never a numeric change. That is why
this could be landed without re-capturing a single golden.

Deliberately **not** free-on-bump: releasing another instance's buffers could pull
memory out from under its in-flight kernels. The leak is pre-existing (these
entries were never freed) and bounded by models x layers, not by steps.

**The gate.** `tests/test_qwen_moe_exp_eos_selfcheck.cpp` gained a
cross-instance case pinning the invariant directly — a fresh instance must answer
identically whether or not the previous one generated anything. Watched failing
first: with the epoch bumps removed, `replay` produced 32 tokens beginning
"Simply reply with \"Hello\"." against a 5-token reference.

**Why nothing caught it.** DET runs its stress cycles against one loaded model
with one residency setting, so it is deterministic *within* a configuration and
blind *across* configurations. The same blind spot appears at the top of this
file: the expert cache is sized from free VRAM at `Load`, and at
`LSX_QWEN38_EXPERT_CACHE_SLOTS=8192` the Paris selfcheck itself fails (256, 1024
and 5350 pass). A gate that varies one dimension at a time cannot catch a defect
that lives in the *interaction* between two.

**Verified 2026-10-17** (RTX 5090/Linux, `linux-debug`): fast 94/94; all nine
`qwen_moe_exp` targets green including DET/FSI/CODE with **goldens unchanged**;
ornith and xing real-weight selfchecks green; gemma / qwen-dense / granite / phi /
nemotron synthetic model tests green; `lsxbenchmark` 43.8 t/s at 2048 tokens in
Debug against a 44.85 t/s recorded Release baseline — no throughput regression.

**Also found while verifying, and still open (§E1e):** gemma-4-26b-a4b-it's
synthetic decode fails on its first step with a swallowed
`cudaErrorInvalidValue` (`[gemma] nohs3: invalid argument`) and reports
`ok=true` with one token. The §E1a cancel gate could not be added to
`test_gemma_moe_model` because of it, so that family keeps no cancel gate until
the swallowed error is propagated and the `nohs3` fault is fixed.
