# gemma-4-26b-a4b-it: the synthetic decode failed at once and reported `ok=true` (E1e, closed 2026-10-17)

Item **§E1e** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), found and closed the
same day on branch `audit/lifecycle-abort-contract`. It was catalogued while
wiring the §E1a cooperative-cancel gate for gemma-MoE and blocked that gate; the
gate is now in place. Kept verbatim, with the correction and the resolution
appended.

This is the third defect this session's lifecycle work found, and the one that
hid the longest: the model's synthetic MoE decode had almost certainly never
run.

---

### E1e. gemma-4-26b-a4b-it synthetic decode fails at once and reports `ok=true`
- Found 2026-10-17 while wiring the §E1a cancel gate for gemma-MoE. Pre-existing,
  not caused by the lifecycle work (reproduced with all of it reverted).
- On the synthetic converted fixture, `GemmaBaseModel::Infer` returns
  **exactly one token with `ok=true`, `error_msg` empty and `eos_terminated`
  false, for any `max_gen_tokens`**. The log shows `[gemma] nohs3: invalid
  argument` (a `cudaErrorInvalidValue` from the routed-expert path) and the
  decode loop breaks on it **without propagating it** — so the caller is told the
  request succeeded while the model produced a single token from the prefill.
- Why nothing caught it: every gemma-MoE synthetic assertion is satisfied by one
  token. `test_gemma_moe_model` asserts `gen_tokens.size() >= 1 && <= 2` with
  `max_gen_tokens = 2`, and the determinism check compares two runs of the same
  broken path. A swallowed error that produces a stable, short answer is
  indistinguishable from a working decode through those assertions.
- **Consequence for §E1a:** the cooperative-cancel gate could NOT be added to
  `test_gemma_moe_model`. The gate's control run (same budget, no flag, must
  produce the full token count) is what makes the two cancel assertions
  meaningful, and it fails here for a reason unrelated to cancellation — the
  decode never starts. Adding the gate without the control would have been a
  test that passes for the wrong reason. The gate was removed from that file;
  gemma-4-31b-it (dense) does carry the full gate and passes.
- **Next step:** make the routed-expert failure propagate (set `error_msg` and
  return the status) instead of breaking silently, then fix the `nohs3`
  geometry error itself. Only after the synthetic decode runs can the §E1a gate
  come back. Both are worth doing on their own merits: the swallowed error
  affects every synthetic and real gemma-MoE request, since `ok=true` with a
  silently truncated answer is the worst shape this API can take.
  - **Corrected 2026-10-17:** the last sentence above was wrong. The fault is
    SYNTHETIC-GEOMETRY-only. On the real 26B weights gemma-MoE decodes normally —
    `test_gemma_moe_selfcheck` and `test_gemma_moe_det_selfcheck` pass, and the
    red `test_gemma_moe_{fsi,code}_selfcheck` are the known §J text divergences
    (e.g. `m=l+(h-l)/2` against a golden of `m=l+((h-l)/2)`), not this fault.
    §E1e never touched a real request.


---

## Resolution (2026-10-17)

Two independent defects, one general and one in the fixture.

### 1. The trailing copy masked any decode-loop failure (general)

`RunGemmaGeneration` ended with the last-step logits read-back and assigned its
status straight back to `e`:

```cpp
e = cudaMemcpyAsync(out.final_logits.data(), ...);   // overwrites `e`
if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
out.ok = (e == cudaSuccess);
```

So any error raised inside the decode loop was **overwritten by a successful
copy**: `out.ok` came back true, `error_msg` stayed empty, and the caller got an
answer truncated at whatever step the loop died on. The decode loop itself is
written correctly — it breaks on `e != cudaSuccess` — so the bug was entirely in
how the status was reported. Fixed by saving the loop's status, treating the copy
as best-effort, and preferring the loop's status:

```cpp
const cudaError_t decode_e = e;
out.final_logits.resize(static_cast<size_t>(V));
cudaError_t fl = cudaMemcpyAsync(...);
if (fl == cudaSuccess) fl = cudaStreamSynchronize(stream);
e = (decode_e != cudaSuccess) ? decode_e : fl;
```

This one is **not independently gated**: the fixture fix below removes the only
known trigger, and a synthetic fixture that deliberately fails would just make the
suite red. Its value is that the next shape violation is loud instead of silent.

### 2. The synthetic MoE intermediate violated the kernel's shape gate

`[gemma] nohs3: invalid argument` was a real `cudaErrorInvalidValue`, not a
launch failure. `RunRoutedGemvBatchSeg` rejects a K that is not a multiple of 32
(`(k_gate_up & 31) != 0 || (k_down & 31) != 0`), and in the no-handshake call
`k_down` is the **MoE intermediate**. The fixture declared:

```cpp
int intermediate_size = 32;       // dense MLP (32-multiple: NVFP4 GEMV)
int moe_intermediate_size = 16;   // <-- not a 32-multiple
```

The dense intermediate carried the comment explaining the requirement; the MoE
one silently did not satisfy it. Fixed to 32, with the reason recorded, and the
two tests that pinned the old value updated (`test_gemma_moe_convert`,
`test_gemma_moe_loader` — its byte-count checks were `16 * 64 / 2`).

### Why nothing caught it

Every gemma-MoE synthetic assertion is satisfied by one token: the model test
asserted `gen_tokens.size() >= 1 && <= 2` at `max_gen_tokens = 2`, and the
determinism check compared two runs of the same broken path. A swallowed error
that produces a stable, short answer is indistinguishable from a working decode
through those assertions. The cooperative-cancel gate is what exposed it — and
only because §E1a requires a **control** run that must produce the full budget.

### Verified

`test_gemma_moe_model` now carries the full `ExpectCooperativeCancelStops` gate
(pre-set + callback + control) and passes; the synthetic MoE decode runs for the
first time. fast 94/94; every `test_gemma_moe*` target green except the two
known §J text divergences; `test_gemma_dense_fsi_selfcheck` confirmed red with
all of this work reverted, so it is §R debt, not a regression here.
