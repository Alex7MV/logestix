# Dense trio → neutral `lsxcommon::dense` — phase 1 closed (2026-11-03)

Branch `refactor/dense-trio-neutral-layer` (from `main` @ `f9118e03`).
Design: `docs/superpowers/specs/2026-11-03-code-duplication-consolidation-design.md`
§2.2, §5 (phase 1). Plan: `docs/superpowers/plans/2026-11-03-dedup-phase1-dense-trio.md`.

## What was done

`granite_dense_inference.cu` held seven kernels (RoPE, causal GQA attention,
split-KV attention, combine, fused split-QKV, fused SwiGLU, vectorized
RMSNorm) and six launchers, duplicated byte-for-byte in
`phi_dense_inference.cu` and `granite_dense_switch_inference.cu` under
different names. The kernels and launchers were moved into the new neutral
module `lsxcommon::dense` (`include/lsxcommon/lsx_dense_inference.h` +
`src/lsx_dense_inference.cu`, a separate OBJECT library
`lsxcommon_dense_cu`); the family `RunGranite*` / `RunGraniteSwitch*` /
`RunPhi*` entry points became inline delegates.

The move is verbatim, and that was verified by an instrument, not by eye:

```
python3 tools/dedup_provenance.py \
    --from libs/lsxcommon/src/granite_dense_inference.cu \
    --to   libs/lsxcommon/src/lsx_dense_inference.cu --rename granite_=dense_
source kernels: 7  target kernels: 7
missing in target: 0  unexpected in target: 0
```

For phi the same check was run against the **pre-refactor** state
(`git show b2c45cb9:libs/lsxcommon/src/phi_dense_inference.cu`): 7/7, 0
discrepancies; i.e. the code that executed before the phase and the code
that executes after it are the same text.

`granite-switch` kept only its own aLoRA kernels
(`switch_scale`, `switch_scale_add`, `switch_lora_shrink`,
`switch_lora_expand_add`) — the check shows exactly those four "missing"
and zero "extra".

## Measured delta

| file | before | after |
|---|---|---|
| `granite_dense_inference.cu` | 1 014 | 611 |
| `phi_dense_inference.cu` | 1 066 | 661 |
| `granite_dense_switch_inference.cu` | 1 223 | 824 |
| `lsx_dense_inference.cu` (new) | — | 478 |
| **total `libs/`** | **3 303** | **2 574** (−729)** |

| test | before | after |
|---|---|---|
| `test_granite_dense_kernels.cu` | 433 | 164 |
| `test_phi_dense_kernels.cu` | 446 | 169 |
| `test_dense_kernels.cu` (new) | — | 340 |
| **total `tests/`** | **879** | **673** (−206)** |

The five shared parity cases moved into the neutral `test_dense_kernels.cu`;
the family tests kept the NVFP4-budget case and a smoke case that keeps
`RunGranite*`/`RunPhi*` on the tested path. The plan estimated −1 400 LOC in
`libs/`; the actual result is −729, because the neutral module (478 lines) is
our code too: the real saving is "removed copies minus the new module", not
"the size of the copies".

## Barrier

| check | result |
|---|---|
| `ctest --preset linux-debug-fast` | 134/134 |
| `ctest --preset linux-debug-convert` | 37/37 |
| `test_dense_kernels` | PASS (5 cases) |
| `test_granite_dense_kernels/_loader/_model/_selfcheck/_fsi/_code/_det` | 7/7 PASS |
| `test_phi_*` | 7/7 PASS |
| `test_granite_switch_*` | PASS, except the documented DET-02 |
| build warnings in our files | 0 |
| `test_model_suite_guard` | 14/14 |
| `test_model_independence_guard`, `test_infrastructure_guard`, `test_absl_container_guard` | PASS |

Bit-exact references after the phase:

- granite: `logits_hash=c165e80fde6401cf`, top-1 12366 `Paris`, ×3 — matches
  the value recorded in `docs/model-reports/2026-11-03-granite-4.2-30b.md`;
- granite-switch: `logits_hash=fdc632042c30919c` (reference:
  `docs/model-reports/2026-10-31-granite-switch-4.1-30b.md`), FSI 3/3,
  CODE 3/3, DET 2/3 — DET-02 (`CRITICAL` instead of `CATASTROPHIC`) is a
  recorded defect of the source model (`docs/SELFCHECK_MATRIX.md` §44,
  `docs/OPEN_TASKS.md` §R10), not a regression of this phase;
- phi: `logits_hash=cf51b9bbd5c913dd` — **this is the first measurement**:
  the phi-4 hash was not recorded in the documentation. Bit-exactness is
  proven not by it but by the provenance check against the pre-refactor file
  (see above); the hash is recorded here as the baseline for the next phases.

## What remains open

- **The perf rows are not measured.** `test_perf_baseline` requires a Release
  build (`NDEBUG`) — in Debug it SKIPs, and the existing `build/linux-release*`
  trees had been configured against the pre-move tree path and were unusable
  after the move to `/mnt/fast_storage`; those trees were deleted 2026-11-11.
  The command to close this (from the current tree directory):
  `cmake --preset linux-release && cmake --build --preset linux-release --target test_perf_baseline && ctest --preset linux-release-perf -R "test_perf_baseline_(granite_4_2_30b|granite_switch_4_1_30b|phi_4)"`.
  The code is provably identical (level 0), so the expectation is t/s within
  noise; but that is an expectation, not a measurement, and it is recorded
  here as exactly that.
- `granite-switch` DET-02 and gemma-dense FSI-01 remain red for reasons
  unrelated to this phase (`docs/SELFCHECK_MATRIX.md`).

## References

- Phase commits: `b2c45cb9` (test and skeleton, RED), `bc7e3e5f` (kernel move),
  `c4dea989` (granite), `aa788759` (family-test trim), `40c144ea` (phi),
  `5652b868` (granite-switch).
- Instruments: `tools/dedup_provenance.py` (level 0: "this is a move"),
  `tools/dedup_scan.py` (phase 3: the absolute duplication residue).