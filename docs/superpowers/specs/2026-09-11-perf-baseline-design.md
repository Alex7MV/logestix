# Perf Baseline Reference (справочник замеров) — Design

Date: 2026-09-11
Status: approved

## Problem / Goal

Inference speed of each model on each GPU type must be tracked in a committed
reference. After every perf test run the actual value is checked against the
reference:

- entry absent **or** actual > recorded → record (update the reference);
- actual within a run-noise tolerance below the record → pass;
- actual below tolerance → FAIL: the degradation cause must be found and
  fixed, restoring the recorded performance.

## Storage

`tests/perf/perf_baselines.json`, committed to the repo. Key:
`(model_type, gpu_key)`. Schema (hand-rolled stable-order writer, simdjson
parser — same pattern as `gpu_perf` perf-cache):

```json
{
  "format_version": 1,
  "entries": [
    {
      "model_type": "gemma-4-26b-a4b-it",
      "gpu_key": "rtx 5090",
      "os": "linux",
      "tokens_per_sec": 98.1,
      "max_gen": 2048,
      "prompt": "Write a detailed long essay ...",
      "date": "2026-09-11",
      "commit": "8c3a20e"
    }
  ]
}
```

`gpu_key` = `perf::GpuProfileKey(ProbeGpuPerfProfile())` (lowercased, vendor
prefix stripped) — the same normalization the knob policy uses.
`os` = "windows" / "linux", fixed at compile time (`_WIN32`): driver latency
differs materially between the OS profiles (WDDM vs Linux), so baselines and
degradation checks are per-OS. Build config (debug/release) is deliberately
NOT part of the key: measured delta on this project is ~1.7% (gemma 96.4
debug vs 98.1 release), inside the 3% run-noise tolerance.

## Components

1. **`libs/lsxcommon/include/lsxcommon/perf_baseline.h` +
   `src/perf_baseline.cpp`** (host-only, model-neutral infrastructure — no
   family prefix per the naming convention):
   - `struct PerfBaselineEntry` (fields as in the schema).
   - `struct PerfBaselineTable { int format_version; std::vector<Entry>; }`.
   - `std::optional<PerfBaselineTable> ParsePerfBaselines(std::string_view)`.
   - `std::string SerializePerfBaselines(const PerfBaselineTable&)`.
   - `enum class PerfVerdict { kUpdate, kPass, kDegrade };`
   - `PerfVerdict JudgePerf(const std::optional<double> &baseline,
     double actual, double tolerance_pct = 3.0)`:
     - baseline absent → `kUpdate`;
     - actual > *baseline → `kUpdate` (improvement recorded);
     - actual >= *baseline * (1 - tolerance_pct/100) → `kPass`;
     - else → `kDegrade`.
   - Pure functions: no I/O, no globals — fully unit-testable.
2. **`tests/test_perf_baseline_logic.cpp`** — label `fast`: JudgePerf branch
   coverage (absent / better / within tolerance boundary / degrade, exact
   −3% boundary) + JSON round-trip + foreign-format-version rejection.
3. **`tests/test_perf_baseline.cpp`** — label `perf` (manual command, same
   discipline as all perf tests). Static model registry in the test:
   `{model_type, vault_subdir, prompt, max_gen=2048}` — all vault models
   present on the machine (gemma-4, GLM-5.3-Flash, DeepSeek-V4-Flash,
   Qwen3.5/3.6 MoE). Per model:
   - model dir / `logestix/model.index.json` missing → SKIP (that entry);
   - no CUDA device → SKIP;
   - run `LsxModelFactory::Create` + `Load` + `Infer` (wall-clock around
     Infer, t/s = num_tokens / total_ms — the lsxbenchmark formula, ttft
     included);
   - verdict via `JudgePerf`:
     - `kUpdate` → rewrite the JSON in the repo tree, INFO the change
       (agent commits the reference afterwards);
     - `kPass` → INFO (actual vs baseline);
     - `kDegrade` → **FAIL** with baseline/actual/delta diagnostics. The
       reference is never rewritten on degradation.
   - One measurement per model per run (perf label = manual command;
     statistical smoothing is deliberately out of scope — YAGNI).
4. **AGENTS.md** — Test section: perf workflow sentence (run perf preset →
   reference check → degradation means root-cause analysis before any
   commit of perf-affecting changes).

## Invariants

- The measurement protocol (prompt, max_gen) is stored inside each entry —
  comparisons stay honest if the protocol ever changes (entry key stays,
  mismatched protocol → treated as absent → re-seeded).
- The test writes the working tree only on absence/improvement; the repo
  copy is committed by the agent, so every speed change is reviewable in
  git history.
- Degradation never updates the reference — only fails the test.
- Baselines are per (gpu_key, os): a different GPU or OS re-seeds its
  own entries.

## Testing (TDD)

- Logic: RED → GREEN in `test_perf_baseline_logic` (fast).
- Perf binary: build-verified without GPU models (registry SKIP paths unit
  covered by the logic suite where host-only); real runs seed the reference
  on this machine (RTX 5090): gemma ≈ 98 t/s expected from the current
  state, others re-measured.
