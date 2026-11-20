# Closed 2026-11-20 — records that moved out of the live list

**A rejected golden set (2026-10-25), kept here as evidence.** Three
`nemotron-3.5-lightning-30b-a3b-nvfp4` goldens were committed and reverted the
same day. They were not a source capture: the `input_ids` were
Qwen3.8-Flash-Next ids (up to **248069** against this model's **131072**-entry
vocabulary — `convert_ids_to_tokens(248045)` returns `None`, and decoding the
row with the Qwen tokenizer reproduces the prompt exactly), the `expected`
values equalled that family's goldens verbatim in all nine cases, and `source`
was a descriptive sentence rather than the model path both capture tools write.
The three gates therefore failed with an empty `raw` (out-of-vocabulary ids →
immediate turn close), and the commit also left `test_model_suite_guard` red
("debt row whose golden now exists" ×3). No fix is available locally: the
checkpoint is `modelopt 0.44.0rc5` `MIXED_PRECISION` with `W4A16_NVFP4` experts
plus an FP8 KV cache, and the installed transformers ships **no** modelopt
quantizer, so the source cannot be executed for an oracle; the bf16 passport is
not in the vault. The debt rows stay. The lesson matches the gemma-4-26b-a4b-it
case (archived 2026-10-24): a golden that is
not the source's own generation is not a parity oracle, and a golden with the
wrong tokenizer cannot even be a framing test.

## U. CUDA 13.5 adoption — landed, one measured negative (2026-11-05)

Toolkit 13.4.92 → **13.5.37** (driver stays 13.2). Design and the full
measurement: `docs/superpowers/specs/2026-11-05-cuda-13.5-adoption-design.md`,
plan `docs/superpowers/plans/2026-11-05-cuda-13.5-adoption.md`.

**Landed.** (a) A build-toolkit **stamp** (`lsx_cuda_toolkit_stamp.txt`) — the
tree really was half-converted: 1250 of 1460 objects were 13.4 headers linked
against a 13.5 runtime and every build step succeeded, because CUDA's headers
carry *package* mtimes older than the objects. Configure now refuses the swap and
`test_cuda_toolkit_guard` (moved to label `fast` — it was the only instrument that
saw it, and `ctest --preset linux-debug` never ran it) catches a mixed tree.
(b) Device codegen arch lives in ONE expanded place: 30 test targets had their own
`--generate-code=arch=compute_120,code=sm_120` literal, so the arch was in 31
places and would have stayed non-family on any bump. (c) The AOCC preset pinned
only the C++ compiler, so a fresh release configure died in uriparser with
`-Qunused-arguments`; `CMAKE_C_COMPILER` is now AOCC's clang.

**U1 — REJECTED on measurement: family-specific targets are neutral here.**
`sm_120f` really does change the SASS — the plain build compiles the NVFP4 decode
to `F2FP.BF16.F32` (FP32 emulation), the family build to
`F2FP.BF16.E2M1.UNPACK_B` (384 in the gemma TU alone) — and the sm_120f cubin
runs on the 13.2 driver. It does not move the wall clock: qwen3.5-9b, 2048-token
protocol, GPU exclusive, three runs per arm gave control 171.7/171.7/171.79,
family-arch 171.71, family-arch + hardware decode 171.70/171.70/171.70. The
spread inside the control arm (168.99 on its first cold run) exceeds any
difference between arms. Cause: every NVFP4 decode path in this repository is
weight-streaming bound — the GEMV waits on HBM/page cache, not on the converter,
which is §R17 P6 seen from the other side. The numeric half was removed rather
than kept with tests behind it; `-DLSX_CUDA_ARCH=120f;121f` remains a
one-variable experiment for a compute-bound case (a long prefill is the only
plausible one). **Not worth re-deriving** unless the decode stops being
byte-bound.

**U2 — free win, attributed.** qwen3.5-9b `r3` 147.00 → **170.00 t/s** (harness
row) on the same weights. This is the 13.5 toolkit, not our code: the control arm
shows it. Other families' `r3` rows predate 13.5 and will re-seed upward on their
next harness run — do not read those as regressions.

**U3 — driver re-probed after the 2026-11-05 update: nothing opened that helps.**
The driver moved `595.99.02` (API 13.2) → **`615.71.09` (API 13.4)**; the toolkit
did not change (13.5.37). The runtime is therefore one minor ABOVE the driver,
which is supported by CUDA's minor-version compatibility but means 13.5-only
APIs are absent at runtime. Measured delta of `libcuda`'s exported `cu*` surface:

| item | 13.2 driver | 13.4 driver |
|---|---|---|
| `cuMemGetLocationInfo` | absent | **present** |
| `cuGraphAddNode_v3` | absent | **present** |
| `cuDeviceGetCliqueInfo` / `GetCliqueCount` | absent | **present** |
| `cuTensorMapSetAllocationClass` (TMA large class) | absent | still absent |
| `cuGraphAddMemcpyBatchNode` / `...NodeGetParams` | absent | still absent |
| `cuMemsetBatchAsync`, `cuFabricCpyBatchAsync` | absent | still absent |
| `cuMemcpyBatchAsync` (`_v2`), `cuCheckpointProcess*` | present | present |

None of the three that opened is a lever here: the clique API is multi-GPU
fabric, `cuGraphAddNode_v3` adds graph node kinds we do not need, and
`cuMemGetLocationInfo` answers "which device backs this UVa page" for a store we
map through the page cache. **Correction to what this entry said before:** the
memcpy-batch **graph node** was described as "the shape §R17 P6 needs" — that was
wrong. P6 needs batched copies overlapped with compute, and the stream API
`cuMemcpyBatchAsync` has been there all along; only the graph-node convenience is
driver-gated. P6 is **not** blocked on a driver update.

Two things that used to be inferences are now measurements on a driver that can
answer:

- **Oversized shared memory buys nothing on this card, now measured.**
  `MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK` resolves and equals
  `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (101 376) — the attribute is queryable on
  this driver, and there is no headroom above the ordinary opt-in limit.
- **Locality domains are useless here.** `LOCALITY_DOMAIN_COUNT = 1`
  (`..._MULTIPROCESSOR_COUNT = 170`): one domain, nothing to partition across.

So §U remains a record, not a work item, and the only real lever for the
host-store-bound families is still §R17 P6.

## G5. The manifest migration dropped the cooperative-cancel gate for 8 models and flattened the selfcheck probe (2026-11-07)

**Status: closed — restored as a declared hook.** Recorded because it is the
second coverage regression this refactor caused (the first is §G2), and because
nothing in the suite could see it.

Eight per-family selfchecks called `ExpectCooperativeCancelStops` — deepseek
v4-flash / v4.1-flash / v4-pro, glm-5.2, glm-5.3-flash, ornith, qwen3.8-flash-next,
xing — and v4.1 additionally called `ExpectStreamMatchesTokens`. The single
shared driver (`tests/test_selfcheck_driver.cpp`) reproduced only the Paris case
plus the two hooks it declared (`tool_roundtrip`, `auto_path_determinism`), so for
those eight models **the cancel gate stopped running** while every target stayed
green. A gate that no longer runs is indistinguishable from a gate that passes:
the target still exists, still has its label, still reports Passed.

This is what §A2 had been blocked on. §A2 records a real defect
(`deepseek-v4.1-flash`: a cancelled request poisons the next one) whose *only*
witness was that cancel case — and after the migration the failure could no longer
be observed at all, which would have read as "fixed".

**Fixed:** a third declared hook, `cooperative_cancel`, in the manifest and the
driver. Enabled for the eight models that lost it plus both granite-switch
members (whose decode loop polls the flag). A model that does not declare it gets
`SUCCEED("cooperative_cancel hook not declared …")`, the same shape as the other
two hooks, so opting out is explicit rather than silent.

**The same migration flattened the probe itself (found next, same day).** The
cancel hook was the visible loss; underneath it the driver also dropped four knobs
each file had chosen for itself: the prompt framing (17 files fed
`BuildPromptInputIds`, 18 fed raw BPE), the generation budget (2 for most,
16/24/32/64/192/256 for the rest), the math mode (14 pinned `FastMathMode::kOff`,
21 left the kAuto default), and the extra "raw greedy continuation" assertion
(12 files had it). The first three feed the recorded `logits_hash`, so the uniform
driver changed the hash for nearly every family — phi-4 went `cf51b9bbd5c913dd` →
`c0955d0e9967e4af`, ornith `c3fe24e9dbc55b1e` → `8ea28e0ee80bc323` — and two
families' *gates* went red (phi answered "The capital" instead of "Paris" under a
framed probe; the driver's added continuation assertion is simply wrong for a
framed family, ornith answers "The capital" and should). All four are now manifest
fields — `selfcheck_frame_prompt`, `selfcheck_max_gen`, `selfcheck_bit_exact` —
transcribed from the deleted files, and the continuation assertion was dropped
because the canonical contract is `SelfCheck`.

**The oracle that settles it:** the release tree's binaries are dated
2026-11-06 17:08–17:09 and contain none of the new driver's strings, so they are
pre-refactor. Replaying four of them against the restored driver reproduces the old
hashes bit-exactly — phi-4 `cf51b9bbd5c913dd`, ornith `c3fe24e9dbc55b1e`,
qwen3.5-9b `a2af669548e59c73`, xing `53e18f6dd77a64a2`, same top-1 and `n`. Two of
the four were RED under the uniform driver. A hash mismatch is only actionable
against such an oracle; without one, "the hash changed" and "the weights changed"
are indistinguishable.

**What this class teaches, again:** the migration's own tests check that a
*declared* target exists and runs; they cannot see coverage that was never
declared, and they cannot see a *parameter* that was never declared. The only
defence is transcribing the old files line by line — assertions **and** their
constants — which is what should have happened at the time. §G2 has the same shape
one level up.

## G6. A ctest target can report **Passed** for a gate that never loaded a weight (2026-11-07)

**Status: fixed in the selfcheck driver; the general rule is worth keeping.**

Catch2 returns `AllTestsSkippedExitCode` (4) only when **every** case in the binary
skipped, and `SKIP_RETURN_CODE 4` is what turns that into ctest's `***Skipped`. The
selfcheck driver's three "hook not declared" branches were `SUCCEED`, so for a model
with no converted tree the binary reported:

    test cases: 4 | 1 passed | 3 skipped

The single "pass" was an undeclared hook — no weights, no vault, no model. The
binary exited 0 and ctest printed **Passed**. A sweep over the 37 SelfCheck targets
therefore reported 37 green when only 27 had executed the Paris case; the other 10
(the models whose converted tree is absent — `beam`, `deepseek-v4-pro-0813`,
`flint`, `gigachat`, `glm-5.2`, `ling`, `mimo`, `mistral-large-4-1t`,
`nemotron-3.5-lightning-30b-a3b`, `step-5-preview`) had skipped everything that
matters.

**Why it is not visible from the guard side:** `test_ctest_labels_guard` enforces
that a skip-capable target *sets* `SKIP_RETURN_CODE 4`; it cannot see that the
binary's own bookkeeping keeps the skip from reaching the exit code. The property
is per-BINARY, not per-test: one always-green case anywhere in the file defeats it.

**Fix:** the three branches `SKIP` instead of `SUCCEED`, so a weight-less model
exits 4 and ctest says `Skipped`. An undeclared hook was never a pass — it is "not
applicable to this model", which is what a skip means. A model that DOES load
weights still reports `Passed` with its hook cases skipped, which is correct: the
Paris case ran.

**The transferable rule:** in a Catch2 binary, "green" only means "nothing failed",
and a case that opts out with `SUCCEED` is indistinguishable from a case that
passed. Any driver with declared optional cases must use `SKIP` for them, or count
its own cases explicitly.

## G7. A relative fixture-dir fallback makes a gate SKIP while its golden sits in the tree (2026-11-07)

**Status: fixed; guarded.**

`tests/test_granite_switch_model.cpp` resolves its tool-round-trip golden with the
macro `LSX_GRANITE_SWITCH_FIXTURE_DIR`, whose `#ifndef` fallback is the **relative**
path `"tests/fixtures/granite_switch"`. `test_granite_switch_model` and
`test_granite_switch_8b_model` never received that macro from CMake (the other
granite-switch targets did), so under ctest — which runs the binary with CWD =
`build/linux-debug` — the path resolved to a directory that does not exist there and
the case reported

    SKIPPED: granite-switch tool golden not captured

while `tests/fixtures/granite_switch/tool_roundtrip.json` sat in the source tree.
The lifecycle gate requires "a native tool round-trip", so for both members of this
family that requirement was **not being verified at all** — and nothing was red. The
8B's stage-9 report quotes the real-weight line (`pinned weights loaded: 40 layers
(6088178432 bytes resident)`), which was captured by running the binary **by hand
from the repository root**, where the relative path happens to work. Reproducible by
accident of CWD is not reproducible.

**Fix:** the absolute fixture dir is now passed to both targets. Verified by
re-running through ctest: the case loads 40 layers (6 088 178 432 bytes) for the 8B
and 64 layers (17 416 338 176 bytes) for the 30B, and both binaries report 6/6 cases
with 136 assertions where they previously reported 5 passed + 1 skipped, 126.

**The class, and why the guard is the fix rather than the two defines.** 13
directory macros in the tree carried a relative fallback; 11 were correctly defined
by CMake and 2 were not, and the failure mode is silent in every case. Two guards
now make a repeat loud:

- `infra guard: a fixture dir macro is never satisfied by its relative fallback`
  scans every `#ifndef X / #define X "<relative>"` and requires each target that
  compiles that source to pass `X`. It has non-vacuity assertions (>= 5 macros and
  >= 100 `add_executable` blocks) because an empty scan passes everything — proven
  by removing one define and watching it name `test_granite_switch_model compiles
  test_granite_switch_model.cpp without defining LSX_GRANITE_SWITCH_FIXTURE_DIR`.
- `tests/test_parity_gate.cpp` no longer has per-kind fallbacks at all: the three
  `LSX_{FSI,CODE,DET}_GOLDEN_DIR` macros collapsed into one `LSX_GOLDEN_DIR`, with
  `#error` when absent. The kind is a string and cannot be compared in `#if`, so
  three per-kind macros forced three "define it or fall back" branches — the
  fallback was the design's fault, not a reviewer's. The runtime env overrides keep
  their documented per-kind names.

**Why this is the third G-class item with the same shape:** G5 (a declared hook that
stopped running), G6 (a case that skipped while the target said Passed) and G7 (a
path that resolved differently under ctest) are all "the instrument reported absence
as success". None of them was caught by a red target; all three were caught by
asking what a green number actually measured.

## G8. A DEGRADED `perf` verdict hid the `r3` verdict entirely (2026-11-07)

**Status: fixed.** Found while re-checking the mistral-small-4 report, and it is the
fourth item in this file with the same shape: an instrument reporting *absence* as
if it were a result.

`tests/test_perf_baseline.cpp` measures two numeric modes per model in one loop —
`perf` (kAuto) and `r3` (kOff, bit-exact) — and judges each with
`CheckAgainstBaseline`. The DEGRADED verdict used `FAIL(...)`. In Catch2 v3 `FAIL`
is a **fatal** assertion (`ResultDisposition::Normal`), unlike `FAIL_CHECK`
(`ContinueOnFailure`), so the first degraded mode aborted the test case and the
second mode was **never measured**. Verified: `mistral-small-4-119b` printed one
`weights loaded` line and one verdict; after the change it prints two of each and
reports both (`perf` 95.9%, `r3` 96.3% of their references).

The consequence while it lasted: any model whose `perf` mode regressed had an
*unverified* `r3` row, and the report/catalogue quote both numbers as if measured.
The test still fails after the fix — `FAIL_CHECK` records the failure and the run
exits non-zero — it just finishes the pair first.

**Why the guards did not catch it:** `test_perf_baseline_logic` tests `JudgePerf`
and `JudgePerfLen`, i.e. the decision, not how many modes a failing decision lets
reach the judge.

## R18. Dory1.0-4B (4B, looped nGPT) — ingestion landed, one engine lesson on
direction-carrying residuals (2026-11-10)

Branch `feat/dory1-4b`. New family (`model_type: dory`, kType `dory-1.0-4b`):
a **looped latent-recurrent normalized-GPT** decoder, not a transformer with
pre-norms. All nine lifecycle stages ran on the real 7.95 GB checkpoint and are
green: SelfCheck Paris ×3 bit-exact (`f100088895adb12d`, top-1 6993), FSI 3/3,
CODE 3/3, DET 3/3, native tool round-trip, perf seeded through the harness at
**94.2 t/s** (`r3`) / 93.1 (`perf`) 2048 tokens, 4.83 GiB resident. Report:
`docs/model-reports/2026-11-10-NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16.md`.

### R18a. The auto (TC fast-math) policy silently produces non-source tokens

**Status: fixed by pinning `FastMathMode::kOff` in `DoryDenseModel::Infer`**, the
same remedy granite-4.2 and granite-switch already carry.

Symptom: the converted model answered `# Prompt\n\nWrite a short story about a
man who is a professional photographer…` to "The capital of France is" framed as
a chat, while the strict path answers `The user is asking a simple factual
question … Paris` — fluent, plausible, and unlike anything the source produces.
The SelfCheck does not see it because it runs at `kOff`; the FSI/CODE/DET gates
do not see it because they also pin `kOff`. Only a `RunCliGeneration` /
`lsxfabric --testprompt` invocation at the production policy exposes it, which is
the same blind spot `granite-4.2-30b` had.

Root cause, measured with a scratch GEMV harness (deleted after use): the
Blackwell W4A4 `mxf4nvf4` path is **tolerance-class per GEMM** — relL2 against
the scalar reference is 0.062 at `seq=1`, 0.113 at 3, 0.182 at 8, 0.192 at 16,
0.199 at 32 (K=2560, N=6144). For a magnitude-carrying transformer that is
invisible. For Dory it is not, because **every mixer output is L2-normalized
onto the unit sphere**: a few percent of relative error in a projection becomes
a *direction* error of the residual stream at the very next layer, and the
2 × 40 recurrent passes compound it. The residual NVFP4 plane plus the strict
scalar route is what makes the 4-bit body track the source (first 64 tokens of a
long-form prompt agree with the source oracle for 43, then one near-tie forks).

The general lesson, worth applying before the next ingestion: "auto policy" is
safe for a family whose residual stream carries magnitude, and unsafe for one
whose residual stream carries direction. Check what the architecture normalizes
before accepting a tolerance-class kernel route.

Pinned by `test_dory_dense_model.cpp::"dory pins the strict routing (TC
fast-math cannot change the output)"`, which drives the process-global routing to
the TC path before `Infer` and requires the strict token stream bit-for-bit.

### R18b. A reasoning-always model needs the mode named per gate

The checkpoint's template defaults to `enable_thinking=true` and both vendor
serving recipes enable a reasoning parser. Two of the three parity probes are
mode-sensitive:

- DET-TEST-02 (letter match) answers `CRITICAL` with reasoning off and
  `CATASTROPHIC` with it on — the latter is the correct answer, so the DET golden
  is captured in reasoning mode (2048-token budget, which is what lets all three
  probes close their `<think>` block);
- FSI and CODE are captured with reasoning off: this model's trace on those
  prompts is long and self-repeating and does not close the block inside 2048
  tokens, so the extraction would fall back inside the reasoning text.

The engine frames with reasoning ON by default (the served mode). Both capture
modes are named in the capture scripts' `OVERRIDES` comments and in
`docs/SELFCHECK_MATRIX.md`; this is a recorded deviation, not a silent one.
