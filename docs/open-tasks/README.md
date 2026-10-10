# Open-tasks archive

Snapshots of `docs/OPEN_TASKS.md` taken when a work session closed its items.
One file per session, named `<YYYY-MM-DD>-<slug>.md` — the date the work
happened, not the date it was archived.

Each archived file is a **verbatim snapshot**, resolutions included: the point
is to keep the investigation trail (what was measured, which hypotheses were
wrong and how they were ruled out) that a one-line "fixed" would destroy. Do
not rewrite an archived file when new facts arrive — write the new fact in the
live `docs/OPEN_TASKS.md` and, if it changes the conclusion, in the newest
archive entry.

The live file keeps only genuinely open work. When an item is resolved, move it
here and add a row to the Archive table at the bottom of
[`../OPEN_TASKS.md`](../OPEN_TASKS.md).

| Date | File | Session |
|---|---|---|
| 2026-10-10 | [`2026-10-10-neutral-moe-and-tc-mtile-closed.md`](2026-10-10-neutral-moe-and-tc-mtile-closed.md) | Shared TC defect closed: `TcGemmKernel` emitted only rows `[0,16)` of its 64-row M tile (row-group index `wm` computed and never used) - the recorded "partial-M store" diagnosis was wrong; fixed, and the TC shape sweep now covers seq 17/18/33/64. Ornith de-duplicated onto the new neutral `lsx_moe_inference` layer (-2776 lines, real-weight gates pass). Coverage gaps closed for the Ornith fused-expert split and gemma `final_logits` |
| 2026-10-07 | [`2026-10-07-ornith-framing-closed.md`](2026-10-07-ornith-framing-closed.md) | Ornith-1.5-35B-A3B-MLX conversion + a real prompt-framing defect: the hand-rolled `BuildPromptInputIds` omitted the leading system turn the reasoning fine-tune requires (repetition collapse), now rendered through the native Qwen renderer (mirrors Nemotron) with a regression assertion; Paris selfcheck passes. FSI/CODE/DET goldens remain declared debt (MLX `language_model.*` prefix not HF-loadable) |
| 2026-10-07 | [`2026-10-07-nemotron-nvfp4-and-codec-closed.md`](2026-10-07-nemotron-nvfp4-and-codec-closed.md) | NVFP4 block-scale codec brought to the complete E4M3FN table (mantissa + subnormals, bit-identical to `cuda_fp8`/`torch.float8_e4m3fn`); NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 ingestion finished (stages 3-8, guards, conversion, real-weight SelfCheck passes after fixing the router-load F32-vs-BF16 dtype); granite-4.2-30b tool round-trip recovered |
| 2026-10-05 | [`2026-10-05-xing-kv-layout-closed.md`](2026-10-05-xing-kv-layout-closed.md) | KV cache layout `[head][pos][dim]`: implemented, measured, **rejected** (2-6% slower end to end); reusable layer + parity gate shipped; `kAttnSplits=4` rejected on bit-identity; 2 pre-existing defects fixed |
| 2026-10-02 | [`2026-10-02-mimo-closed.md`](2026-10-02-mimo-closed.md) | MiMo-V2.6-Flash-RL closed as a vendor-model defect: the implementation is bit-exact with the vendor's own `modeling_mimo_v2.py`, so the checkpoint — not logestix — is what fails the Paris gate. Converted model removed, perf target withdrawn, selfcheck header now states the permanent failure. The eight synthetic MiMo suites stay green and remain the family's real signal. |
| 2026-10-02 | C1 update (no new file) | F1/C1 `test_gigachat_moe_devkernels` flake investigated in the live `docs/OPEN_TASKS.md`: the flake was the Release sweep (2155 assertions identifies the build), the contention hypothesis is measured and rejected (+0.3% under load), and the concealment it exposed is fixed — the two latency-budget cases compiled to nothing in Debug and now SKIP. |
| 2026-09-30 | [`2026-09-30-cuda-review-windows-bringup.md`](2026-09-30-cuda-review-windows-bringup.md) | CUDA inference review of `24bf180~1..bed5d9f` + the Windows build repair. Archived 2026-10-02 with all items closed: A2 (MiMo — vendor-oracle match, not a bug), B1, D1, D2, E1, G1, plus a 9-item inference-conformance pass and a 15-item "closed in this session" list. |