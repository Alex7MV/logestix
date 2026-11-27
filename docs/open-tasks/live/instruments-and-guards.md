# Instruments and guards — when a gate reports the wrong thing

### G1. `RESOURCE_LOCK "gpu"` does not serialise two ctest processes (2026-11-07)

**Status: open — a documentation/process gap, not a code defect.**

Every GPU target carries `RESOURCE_LOCK "gpu"`, and `AGENTS.md` states the rule
"one GPU workload at a time". Both are true *inside one ctest invocation*: CTest
implements a resource lock as a scheduler constraint among the tests of a single
`ctest` run. Two concurrent `ctest` processes do not coordinate at all — there is
no lock file, so both schedules believe they own the GPU.

Measured on 2026-11-07: a background parity sweep (`glm_moe_flash`, `gigachat`,
`deepseek_v41`, `minimax_moe_flash`, `mistral`, `deepseek`) was holding
**30 272 MiB of 32 607** while a second ctest ran `test_granite_switch_*`. The
DET gate then failed on DET-03 with `deterministic=false`, `extracted=''` — an
inference error under allocation pressure, not a numeric divergence. Re-run with
the card otherwise idle, DET-03 passed (`deterministic=true`,
`extracted='BIT_ZERO'`).

The failure mode is worse than a red test: the reporter said
`deterministic=false`, which points at a numerics/instability bug, and the actual
cause was VRAM exhaustion in a *different process*. That half is fixed — the DET
reporter now prints `inference-error error='…'` when `Infer` returns not-ok,
instead of folding it into the determinism verdict
(`libs/lsxcommon/src/det_selfcheck.cpp`; FSI/CODE already carried the per-case
error, and the shared gate driver now attaches per-case verdicts to its assertion
too).

What is **not** fixed, and is a decision rather than a patch:

- A cross-process lock would have to be a file (`flock` on a path in
  `lsx_cache_dir()`), taken by every GPU target before it allocates. That is a
  real mechanism, but it also means a crashed test leaves a stale lock, and every
  target would need to opt in — worth doing only if unattended parallel runs
  become normal.
- The cheap alternative is the one now written into `AGENTS.md`: a GPU-labelled
  run starts only after `nvidia-smi --query-compute-apps` is empty. That needs no
  code and no lock lifecycle, and it is what the human operator already does.

Do not "fix" this by removing `RESOURCE_LOCK`: within one invocation it is
load-bearing (the perf sweep would otherwise run concurrently with the gates in
the same `ctest -j`).

**Fixed on the same day, and the numbers that forced it.** The lock was not
merely per-process, it was also mostly *absent*. `AGENTS.md` claimed "every perf
target takes `RESOURCE_LOCK "gpu"`"; the configured tree said otherwise:

| label | locked before | locked after | total |
|---|---|---|---|
| `inference` | 54 | **289** | 297 |
| `perf` | 8 | **37** | 44 |
| `convert` | 8 | 36 | 38 |
| `fast` | 10 | 10 | 140 |

`STAGE_LOCKS` in `tools/gen_model_manifest.py` was empty for every stage except
where the hand-written CMake happened to set a property, so 263 of 279
GPU-labelled targets ran unprotected. Every model stage allocates on the GPU
(convert quantises there, load uploads, model/kernels/selfcheck/fsi/code/det/perf
decode), so the table is now `"gpu"` for all nine stages, with one exception:
`fast` means host-only by definition and keeps the lock off — otherwise a
140-target fast sweep would serialise on a lock it never needed.

Reproducing the change against the pre-migration oracle:
`--diff-cmake` reports **292 differences, and every one of them is this field**
(284 `resource_lock: null -> "gpu"` plus the 8 new `granite-switch-4.1-8b`
targets that the 2026-11-06 snapshot predates). No label, timeout, skip code or
environment moved. That is the check that this was a targeted change and not a
registration drift.

The 17 hand-written GPU targets that were also unprotected are now locked too
(8 `perf`, 8 `inference`, 1 `convert`), so the configured tree reads:

| label | locked | total |
|---|---|---|
| `inference` | 297 | 297 |
| `perf` | 44 | 44 |
| `convert` | 37 | 38 |
| `fast` | 10 | 140 |

The one unlocked `convert` target is `test_arrow_tree_integrity`, allowlisted as
host-only **with a checkable reason** (0 occurrences of `cuda` in the file: it is
an mmap walk over the Arrow vault). The guard requires a reason for every
allowlist entry rather than a bare name, because "we think it is cheap" is how a
device call gets added later without anyone revisiting the exception.

**What the guard cannot see (open).** `test_ctest_labels_guard` parses literal
`set_tests_properties(<name> ...)` and `lsx_add_*_test(...)` calls. Names that
appear only inside a *group* list — `set_tests_properties(a b c PROPERTIES
LABELS inference)` — or inside a `foreach(qt ...)` registration are invisible to
that scan. That is exactly how the 8 `inference` targets above stayed both
unlabelled-by-the-scan and unlocked: they are registered through
`foreach(qt test_lsx_kernels)` / a group properties call. They are fixed, but the
*detection* still has the hole, and a new target added the same way would be
neither labelled nor locked with the guard's blessing. Closing it means reading
the configured tree (`ctest --show-only=json-v1`) or generating a lock-audit
manifest at configure time — a mechanism decision, not a patch.

### G2. A sibling model can be "covered" by a test that verifies its parent (2026-11-07)

**Status: open — one instance found and recorded, one instance fixed.**

Found while ingesting `granite-switch-4.1-8b`, which is the first family member
added *after* its sibling in the manifest era. A `models/*.json` row claims
targets by name, and `test_model_suite_guard` checks that a claimed target
EXISTS — not that it tests the model that claims it. So a source with a
hard-coded model name can sit under two rows and verify only one of them, while
every guard stays green. That is the shape of the migration's own recurring
defect (a SKIP naming a neighbour), one level up.

Measured: 25 test sources are claimed by more than one manifest. Four contain a
literal `model_type` string:

| source | claimed by | verdict |
|---|---|---|
| `test_granite_switch_adapter_parity.cpp` | 30b, 8b | **was a defect** — fixed: parameterized via `LSX_GSW_MODEL`, with the 30B as the `#ifndef` default and a separate 8B target |
| `test_granite_switch_model.cpp` | 30b, 8b | **was a defect** — same fix (it also `dynamic_cast` to `GraniteSwitch41_30BModel`, which is how the 8B would have failed at runtime instead of at review) |
| `test_granite_switch_tokenizer_parity.cpp` | 30b, 8b | **was a defect** — same fix; the golden cases stay shared because the two tokenizers are md5-identical |
| `test_kimi_k3_constants.cpp` | kimi-k3, kimi-k3.1-pro | **open** — see below |

Also in the capture tools, which is worse because a golden carries the mistake
forward: `granite_switch_adapter_capture.py` and `granite_switch_tool_capture.py`
wrote `"model_type": "granite-switch-4.1-30b"` as a literal. Both now derive it
from the checkpoint directory (the same rule the C++ converter uses), and the
parity test rejects a golden whose `model_type` is not its own — so a
mis-attributed capture fails loudly instead of being compared as if it were
right.

#### G2a. `kimi-k3.1-pro` claims a target named after its sibling — name, not coverage

`kimi-k3.1-pro` lists `test_kimi_k3_constants` in its `model` stage. The name says
`k3`, and `test_kimi_k3_constants.cpp` does assert `kModelType == "kimi-k3"`, so
the first reading — recorded here earlier the same day — was "the Pro is covered by
its parent's test". **That reading was wrong**, and refuting it is cheap:

- `tests/CMakeLists.txt` compiles that binary from **two** sources,
  `test_kimi_k3_constants.cpp` *and* `test_kimi_pro_constants.cpp`;
- `--list-tests` shows 12 cases, six of them `kimi-k3.1-pro passport: …`
  (registry/format identity, text tower geometry, Lightning Indexer, derived
  scales, the DSA table = MLA table);
- all 12 pass (615 assertions), and the Pro cases assert
  `kModelType == "kimi-k3.1-pro"` / `kFormatTag == "logestix-kimi-k31-pro-nvfp4"`
  from `kimi_moe_pro_constants.h`.

So the coverage is real. What is genuinely wrong is the **name**: a target called
`test_kimi_k3_constants` sitting in the Pro's row is exactly the shape that made
three granite tests verify their parent, and the next reader will draw the same
wrong conclusion this entry did — from the name alone, without `--list-tests`.

**Not renamed here on purpose.** The honest split is one target per passport: the
k3 cases belong to `test_kimi_k3_constants.cpp`, the Pro cases to
`test_kimi_pro_constants.cpp`, each registered under its own model. That is a small
refactor touching both manifests, `tests/CMakeLists.txt` and the generated
fragment — mechanical, but it should be done deliberately rather than renamed into
a second false impression.

The lesson kept from this entry: **a target's name is not evidence about what it
asserts.** Check `--list-tests` and the source list before recording a coverage gap.

#### G2b. The guard, written — and why it checks only the vault env

`test_model_suite_guard` gained *"a shared test source cannot name a sibling as
itself"* plus a non-vacuity case (`the sibling-identity rule bites`). It builds
`source -> claiming models` from the generated registry (25 sources are claimed by
more than one manifest) and flags a source that names a sibling's **vault
environment variable**, or hard-codes a sibling as `cfg.model_type` /
`kModelType`. A literal is accepted when it is the default arm of an
`#ifndef`-guarded macro, i.e. the parameterised form a shared source is supposed to
use (`test_granite_switch_model.cpp`).

Two candidate signals were **rejected after measuring them**, and the reasons
matter for whoever extends this:

- **the `model_type` string**: a sibling test legitimately compares two passports —
  `test_granite_switch_prepare.cpp` asserts that the 8B passport *rejects* a 30B
  topology, so it names both on purpose;
- **the vault directory name**: the same string is a natural name for a test
  fixture directory. The first version of this guard flagged its own fixtures
  (`test_granite_switch_convert.cpp` builds `/tmp/…/granite-switch-4.1-8b`), which
  is how the signal was retired — a check that cannot tell identity from data
  produces false positives and gets ignored.

G2a is the case this guard deliberately does **not** catch: the target's *name*
is misleading while the assertions are correct. That is a naming discipline, not a
mechanical rule — the only mechanical check available would be "a source may only
assert its own `kModelType`", which would forbid the two-source binary that is
actually right.

### G8a. `mistral-small-4-119b` reads 95.9% / 96.3% of its recorded throughput (open)

The other half of the same run: re-measuring the committed tree eleven days later
gives 43.89-44.31 t/s (`perf`) and 47.99 t/s (`r3`) against references of 45.77 and
49.82 — stable to under 1% over six runs and across both modes, so a systematic
~4%, not noise. **Thermal and power throttling are ruled out by measurement**: a
cold-card (45 °C start) run and an immediately-following hot-card run give the same
verdict with 46/51 °C, 141-145 W mean against a 600 W cap, and the harness's own
DRAM oracle reads 99.4% (1524 vs 1533 GB/s). Nothing in `mistral_moe_*` or the
shared kernels changed since the reference was written. `granite-4.2-30b` shows the
same shape (96.7%); `phi-4` passes.

**The reference has deliberately NOT been re-seeded.** Two candidate causes, in the
order they should be chased:

1. **The reference is a ratcheted best-of sample.** The harness re-seeds on any
   improvement, so a recorded row is the best observed run, not a median, and the
   pass band is only 3%. This row replaced 45.04, and `r3` replaced 41.57 — a 20%
   jump on the same tree, which is itself evidence that this measurement is
   unstable. If that is the cause, the fix is in the harness's statistics (record a
   median of N, or widen the band for a family whose spread is measured), not in
   the model.
2. **The capacity oracle is disabled for this entry.** `expert_cache_slots` is
   present in 2 of 67 entries; `mistral-small-4-119b` is not one of them, so
   `baseline_slots` is 0 and `AttributeDegrade` cannot return `kCapacity` for the
   family most sensitive to it — 4 608 routed experts streamed through 1 398 device
   slots, with the capacity derived from ambient free VRAM at load time. Backfilling
   the field on a conforming run restores the oracle.

Do not re-seed until one of those is settled; a wrong re-seed would erase the only
evidence if the ~4% is real.

### G9. `test_gpu_perf`'"'"'s bandwidth-probe case is flaky under a PARALLEL fast
sweep (2026-11-10)

**Status: open. Pre-existing, not caused by any model branch** — reproduced with
every `dory` target excluded from the run.

`tests/test_gpu_perf.cpp` case *"memory-bandwidth probe releases its device
buffers"* (added in `d56305d3`) asserts

```
free_before >= free_after        // after MeasureMemoryBandwidthGbs()
```

to prove the probe does not leak its 2 × 512 MiB working set. The assertion is a
*cross-process* comparison: `cudaMemGetInfo` reports the whole device, so any
other process that releases memory during the probe window makes `free_after`
larger than `free_before` and the case fails while proving nothing. Measured on
`linux-debug`, `-j 24`, three consecutive sweeps of the 140 non-dory fast
targets: **fail, pass, fail**. Isolated (5 consecutive single-process runs):
5/5 pass, and the underlying probe is fine — the failure message shows
`free before 29543 MiB, after 29843 MiB`, i.e. free memory went UP by 300 MiB.

Two candidate fixes, in order of preference:

1. **Assert the leak, not the delta.** The probe's contract is "no 512 MiB
   buffer survives". Measuring that needs the process's own held bytes, not the
   device's free bytes — e.g. `cudaDeviceGetLimit`/a pool snapshot, or a
   `cudaMemGetInfo` pair taken around alloc/free of a same-sized sentinel buffer
   inside the case. Attribute the slack to the sentinel, not to the world.
2. Give the case `RESOURCE_LOCK "gpu"` so it never shares the card with another
   target of the same ctest. This hides the cross-process window without making
   it measurable — and it costs a lock on a label that deliberately stays
   parallel (`test_ctest_labels_guard` exempts `fast` for exactly that reason),
   so it would just move the problem to whatever runs next.

Do not "fix" it by widening `kSlack`: 300 MiB of foreign churn would still fail,
and a slack large enough to absorb it also hides the 1 GiB leak the case exists
to catch.

### I1. The CODE gate's regex anchor is order-sensitive (2026-11-22, open)

**Status: open — an instruments gap that produces a false red, no defect in any
model.**

`deepseek-v4.1-flash-1017` CODE-TEST-03 asks for a PCRE pattern and the model
answers `\b[A-F0-9]{4}\b`; the prompt's Expected Template is
`\b[0-9A-F]{4}\b`. Those are the *same expression* — the member order inside a
character class carries no meaning — but `ExtractRegex`
(`libs/lsxcommon/src/code_selfcheck.cpp`) locates its answer by the **literal
anchor** `[0-9A-F]`, finds nothing, and returns an empty string, which the gate
reports as `passed=false extracted=''`. Measured 2026-11-22 on the AOCC Release
tree: CODE 2/3 for that row (01 and 02 pass), while its parent
`deepseek-v4.1-flash` fails CODE-03 differently (the anchor matches and the
model appends a stray `?`), so the two rows look alike for unrelated reasons.

Why it is not fixed here: the anchor is the *shared* extractor for all 43 rows,
and making it canonical (normalize a bracketed class by sorting its members, or
match the anchor as a set rather than a substring) changes what every row is
measured against. That needs its own commit, a re-run of `ctest -L inference -R
"_code_selfcheck"` across the corpus, and a diff of the verdict grid — not a
side effect of ingesting one model, and not something to smuggle in under a
model row. Until then the row records its measured verdict (2/3) with this
mechanism, which is the honest form: a red that names its cause beats a green
that hides one.

The one thing not to do is to loosen the *comparison* (accept any textual
near-miss): the extractor is deliberately a form check, and FSI-02 on the Pro row
— a genuinely wrong symbolic form that a tolerant comparator would accept — is
the case it exists for.

### I2. Global-`cudaMemGetInfo` leak probes are red inside a parallel sweep (2026-11-22)

**Status: open — the same defect class as `G9`, now with two more victims.**

Three `fast` targets assert a leak by comparing the **device-wide** free-VRAM
figures taken around their own alloc/free:

| target | assertion |
|---|---|
| `test_gpu_perf` (bandwidth probe) | free-before minus free-after, bounded |
| `test_deepseek_moe_loader` | "a failed second load leaks nothing": `delta < 64 KiB` |
| `test_deepseek_moe_pro_loader` | the same check |

Measured 2026-11-22 during ingestion verification of
`deepseek-v4.1-flash-1017`: consecutive `ctest --preset linux-debug -j 24` runs
of an unchanged tree reported **green / `test_gpu_perf` only / all three**, and
every one of them passes in isolation (`-j 1` or `-R '^name$'`). The card was
otherwise idle — one process was measured against another's allocator churn,
which is exactly what `G9` describes for the perf probe.

Why it matters more than a flake: a reader sees "loader leak" in a red test
name, and the actual cause is a neighbour process. It also means a real 128 KiB
leak would be indistinguishable from this noise whenever the sweep is busy.

The fix direction is the one `G9` already argues for: attribute the delta to the
process (a pool snapshot, or an alloc/free of a same-sized sentinel *inside* the
case, so the slack belongs to the sentinel). Widening `kSlack` is explicitly not
the answer.

### I3. The DET/FSI extractors resolve a reasoning model's *preamble*, not its answer (2026-11-27, open)

**Status: open — the `R21` class, now reproduced on a second row; no defect in
the model.**

`gigachat4-708b-a41b` DET-TEST-03 asks for the token `BIT_ZERO` and the model's
raw text says `BIT_ZERO` as its final line — but `ExtractToken`
(`libs/lsxcommon/src/det_selfcheck.cpp`) returns the **first** 2+ character
uppercase token that is not part of a word, and the model's own reasoning
preamble paraphrases the prompt ("*Context: **DET**-Validation-Core-03 // …*"),
so the gate extracts `DET` and reports a red. `CutPromptEcho` is supposed to
remove that echo first, but it searches for the prompt body's first 32 bytes
**verbatim**; a model that paraphrases instead of echoing (this one writes
`Context:` bullets) leaves it nothing to cut. Measured 2026-11-27 on the AOCC
Release tree: DET 2/3, and the failing case is `deterministic=true` (3 repeats),
so this is not sampling noise.

The same row's FSI-TEST-01 is red for a sibling reason: `ExtractFsiFormula`
selects **one line** (`SelectAnswerLine`) and the model rendered
`PV=CF*(1-exp(-r*n))/(1-exp(-r))` as two lines. The answer is arithmetically
identical to the golden and passes on the next process run, which is what makes
it flaky rather than wrong.

`R21` (`docs/open-tasks/live/models/tml-d24-varloop-research.md`) filed the same
DET-03 shape on 2026-11-22 and closed it as "the extractor resolves the prompt
echo, identically on the source and the engine". This entry exists because the
class now has a second, independent victim and a *second* mechanism (paraphrase
instead of echo; a wrapped expression instead of a wrong one).

Fix direction, in order of blast radius: (a) `CutPromptEcho` should cut on a
token-level overlap rather than 32 verbatim bytes; (b) `ExtractToken` should
accept a *trailing* answer line when the reasoning preamble is present
(`</think>` / `[COT_END]`, or a `Final …:` label); (c) `ExtractFsiFormula`
should join a trailing `*`, `+`, `/`, `-` or `=` continuation line. None of the
three is safe to land inside a single model ingestion: all three are shared by
every row, so each needs its own before/after sweep over the 43 manifests.
