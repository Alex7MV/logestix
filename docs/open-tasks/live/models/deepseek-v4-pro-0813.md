# DeepSeek-V4-Pro-0813 — open items

### R3. DeepSeek-V4-Pro-0813 — ingestion landed, parity/perf pending (2026-10-26)

The family is registered end to end (convert -> load -> model -> kernels ->
selfcheck -> fsi -> code -> det -> perf), and the frozen passport is
61 layers / 7168 hidden / 384x3072 experts / 129280 vocab with the
`128,128,4,128,4,...` compression grid. Two items remain open:

- **P1 — SelfCheck is green on real weights (2026-10-26).** The 892.7 GB
  source converted cleanly in place (838 GB tree: 27 GB pinned + 811 GB
  experts, `deferred.mtp = 7009`; `incomplete 0`). `test_..._pro_selfcheck`:
  Paris x3 with an identical `final_logits` hash, streaming order exact, and
  both abort contracts green; the real-weight tool round-trip returned
  `get_weather {"city":"Paris"}`. Getting there exposed two non-obvious
  defects: the HC residual exceeds the FP16 range at depth (the source runs
  BF16/FP8), so `hc_pre`/`hc_head` now publish an **FP32** combined vector
  (`d_single_f32`) before the scale-invariant RMSNorm; and `LayerWeights::Free`
  skipped the separately-allocated shared-expert blobs (~4 GiB) on the
  arena-backed path, so a second `Load()` in one process OOM'd — both fixed.
- **P2 — parity goldens are template-derived and the gates now run (2026-10-26).**
  `tools/{fsi,code,det}_capture.py --template-golden/--template-only` frame
  each prompt with Pro's own `encoding/encoding_dsv4.py` (chat mode) and record
  the prompt's authored Expected Template as `expected`; `source` says the TP-4
  FP8 runner cannot execute here, so these are **not** source-parity goldens
  (same status as DeepSeek-V4.1). Verdicts: **FSI 2/3**, **CODE 3/3**,
  **DET 3/3** (3 runs each, deterministic). FSI-02 is RED for a real
  instruction-following reason: the prompt asks for the symbolic form and the
  model inlines the numeric constants
  (`((100000000*1.083541-0.0005)*83.152402-...)`); FSI-01/03 pass. That is an
  open model-behavior defect, not a kernel defect, and it is not "fixed" by
  loosening the golden.
- **P3 — perf is seeded through the harness (2026-10-26).** Running
  `test_perf_baseline "perf baseline: deepseek-v4-pro-0813"` on
  `linux-release-znver4-aocc` (RTX 5090, 2048-token protocol, one process per
  mode) wrote both rows: **perf 0.77 t/s** (TTFT 8987 ms, 114 W avg) and
  **r3 0.79 t/s** (TTFT 9023 ms, 116 W avg), 28.0 GiB resident. The number is
  host-store bound, as expected: every token streams 61 x 6 experts (~13.5 GB)
  from the 811 GB mmap tree; the 5-slot device cache (177 MiB, all the free
  VRAM the 26.7 GiB resident set leaves) contributes almost nothing. A pinned
  arena is not an option (822 GiB vs 750 GB RAM). This is the honest ceiling
  for this card, not a tuneable regression.

Residency note for the next operator: do NOT turn on
`LSX_DEEPSEEK_PRO_PIN=1` on this box. The expert tree is 822 GiB against
750 GB of RAM; the loader caps an opt-in pool at 64 GiB for exactly that
reason, and the default (mmap + `LSX_DEEPSEEK_PRO_MAX_OPEN`, 4096) is the
bounded path.
