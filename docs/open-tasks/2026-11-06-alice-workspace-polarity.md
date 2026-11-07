# AliceAI-Foundation-80B-A3B: an inverted allocator predicate disabled generation (2026-11-06)

Alice's four real-weight gates were re-run after the model-manifest migration,
which is where this surfaced: every one of them was red with a single message,
`alice workspace alloc failed` — FSI/CODE/DET returned `raw=''` for all three
cases and SelfCheck failed on the Paris prompt. It was **not** the migration.

## Symptom, and the first wrong hypothesis

The obvious suspects were the migration's own surface (the new single gate
driver, the manifest-sourced vault path, the registration rewrite). The cheapest
decisive experiment refuted all three: the **pre-migration binary**
(`build/linux-release-znver4-aocc`, whose per-family `test_alice_moe_fsi_selfcheck`
still carries the old body) failed identically — same empty `raw`, same
`workspace alloc failed`. So the defect predates the manifest work.

The next tempting hypothesis was VRAM exhaustion: alice keeps ~27.4 GB of an
80B-A3B device-resident (`governor ... vram=27431/32202`) and streams the routed
experts, so an allocation failure reads plausible. It is also wrong, and the way
it was refuted is the useful part.

## Root cause — the failure message discarded the failure

`RunAliceGeneration` allocated ~40 workspace buffers through a macro and failed
the whole set on the first non-success, reporting a fixed string. The
`cudaError_t` was captured in `e` but never printed, so "out of memory", "bad
API call" and "the check itself is wrong" were indistinguishable. Recording the
error and the buffer identity turned it into:

```
[alice] workspace alloc failed: ? needs 0 MiB, cudaMallocAsync -> cudaSuccess (0), free 27399 of 32202 MiB
```

`?` and `cudaSuccess` mean **no allocation failed at all** — the guard was
inverted. The macro's polarity had been:

```cuda
// before 397ab361: arms are cudaError_t, so success is 0 == falsy
? (ws_free.track(ptr), static_cast<cudaError_t>(cudaSuccess))
: cudaErrorMemoryAllocation)
```

and `if (A(x,HID) || A(mixed,HID) || ...)` reads "any allocation **failed**"
only because `cudaSuccess` is 0 and therefore falsy. Commit **397ab361** ("fix:
clear the warnings surfaced by the build-noise work", 2026-11-03 19:33) rewrote
those arms as `true`/`false` to clear 42 `-Wint-in-bool-context` sites and
**inverted the meaning while leaving the chain untouched**: `true` now meant
success, so the very first successful `cudaMallocAsync` satisfied the `||` and
aborted the workspace. Alice could not generate a single token from 2026-11-03
19:33 onward, while the message said "alloc failed".

## Why it stayed invisible for three days

Three independent blindnesses had to line up:

1. **The message named nothing.** A discarded `cudaError_t` makes an inverted
   guard look like an environment problem; the number that would have refuted VRAM
   exhaustion (`free 27399 MiB`) was never printed either.
2. **The driver printed no verdict.** `tests/test_parity_gate.cpp` attached a
   per-case `INFO` inside the loop, so it left scope before the aggregate
   `REQUIRE(result.all_passed)` — Catch2 printed `false` and nothing else. A wrong
   answer, an empty generation and a CUDA error all looked the same. Fixed
   (`ExplainCases()` collects the per-case verdicts and attaches them at the
   assertion); it is what made the CODE branch readable in this session.
3. **Nothing runs the gates by default.** FSI/CODE/DET/SelfCheck carry label
   `inference`; `ctest --preset` runs `fast` (and `convert`). The breakage was
   introduced in a commit whose own verification ran "fast 134/134, convert 37/37,
   kernel/inference subset green" — a subset that did not include Alice's gates,
   even though the tree was in the vault and `kGoldenDebt` did not exempt it.

## Fix

- `lsxcommon/alice/WsAllocFailed(cudaError_t)` — a named predicate in
  `alice_moe_inference.h` (`true` only for a real failure), used by the macro
  instead of inline ternary arms. The contract is now written down in one place
  rather than implied by the truthiness of `cudaError_t`.
- The allocation failure now logs the buffer, its size in MiB, the
  `cudaGetErrorName`/code and `cudaMemGetInfo` free/total.
- `tests/test_parity_gate.cpp` prints per-case expected/extracted/error at the
  assertion (above).
- Pin: `test_alice_moe_kernels.cu` → case *"alice workspace allocator polarity"*
  asserts `WsAllocFailed(cudaSuccess) == false` plus two real errors `== true`.
  Proven non-vacuous by inverting the predicate (`return e == cudaSuccess;`): the
  case reports `!true` and fails 1/1.

## Real-weight verdicts after the fix (linux-debug, RTX 5090, one test at a time)

| gate | result |
|---|---|
| SelfCheck ("Paris") | PASS ×3, bit-identical, `logits_hash 0bf456024cc05255`, top-1 id 828 |
| FSI | 3/3 in 230 s |
| CODE | 3/3 in 221 s |
| DET | 3/3 in 183 s |
| native tool round-trip | green (`{"city": "Paris"}` after the `[COT_END]` strip) |

## One measurement moved: the logits hash

The 2026-11-03 record pinned `logits_hash f900547d25be7914`; today the same
prompt, same top-1 id (828), same bit-exact repetition across cycles gives
`0bf456024cc05255`. Two candidate causes were not separated:

- **397ab361** also dropped `__restrict__` from 39 in-place elementwise call
  sites (formal UB), which is exactly the class of change that permits FP
  reordering;
- the **CUDA 13.5 toolkit** landed 2026-11-05 (OPEN_TASKS §U), and U1 measured that
  it changes the SASS of NVFP4 decode paths.

Not bisected: each arm needs an 80B-model rebuild plus a 3-cycle bit-exact gate,
i.e. hours, and nothing machine-checked depends on the old value — the parity
goldens are text-level and SelfCheck asserts *determinism across cycles*, not a
pinned hash. The `f900547d25be7914` in the 2026-11-03 report and in this archive's
sibling is a dated measurement, not a contract.

## Residual

Alice's gates are still label `inference`. This defect needed nothing more than
one run of them, so the systemic gap is not this entry — it is that no default
sweep runs the parity gates (see `docs/OPEN_TASKS.md`, "Model manifest
migration", and `docs/OPEN_ENDEDNESS.md`).

One neighbouring question was left alone on purpose, because it is a different
defect and this fix should not bundle it: `WorkspaceGuard` frees the workspace
with `cudaFree()`, while the buffers come from `cudaMallocAsync()`. The CUDA
ownership rules for that pair are not obviously satisfied and the guard ignores
the return code, so a silent per-call leak of `~155 MiB` cannot be ruled out by
inspection. Measured cost of not chasing it now: nothing in this session (the
gates that leaked most, FSI's three 2048-token cases, still had 4.6 GB free and
the process releases everything at exit); the reason to chase it is a long-lived
server, not a test run.
