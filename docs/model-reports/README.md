# Model reports (ingestion artifacts)

Every completed model ingestion leaves one **public-facing report** here,
named with the date the real-weight gates were measured followed by the model
card / vault directory name:

```
docs/model-reports/YYYY-MM-DD-<model-card-name>.md
```

A later re-ingestion of the same model adds a new dated file; it never
overwrites the previous report, so the history stays readable.

The report is an artifact, not a status file:

1. it must be **publishable as-is** (the `Publish-ready post` section), and
2. it must carry the **internal evidence** that backs every claim (gates run,
   measured numbers, declared debt), so a later session can re-read it without
   replaying the whole ingestion.

What goes in (mandatory):

- one-line scope: model, format tag, what was converted, on what hardware;
- what the checkpoint forced into the engine (semantics that are easy to get
  wrong), with the reference used (`modeling_*.py` / vendor encoder);
- **artifact-level defects and questions for the model authors** (config /
  tokenizer / generation-config traps), each with the observed failure mode;
- measured gates and numbers (SelfCheck, determinism, tool round-trip, perf) —
  and explicitly what was **not** measured and why (declared debt);
- a "Publish-ready post" section in English, no internal paths or codenames;
- an "Evidence" section: internal docs, tests and the perf rows.

Keep the publishable section free of repository-internal file paths; move those
to Evidence. Do not rewrite a gate result: a SKIP is a gap, never a pass.

| model | report |
|---|---|
| GigaChat3.5-432B-A28B-Reasoning | `2026-10-31-GigaChat3.5-432B-A28B-Reasoning.md` |
| MiniMax-M3.1-Flash | `2026-10-31-MiniMax-M3.1-Flash.md` |
| Beam-1.0-501B-A23B | `2026-10-31-Beam-1.0-501B-A23B.md` |
| Granite-Switch-4.1-30B | `2026-10-31-granite-switch-4.1-30b.md` |
| Hy-SimulMT-30B-A3B | `2026-10-31-hy-simulmt-30b-a3b.md` |
| Step-5-Preview | `2026-10-31-Step-5-Preview.md` |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | `2026-11-01-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.md` |
| Mistral-Large-4-1T-A52B | `2026-11-02-Mistral-Large-4-1T-A52B.md` |
| Qwen3.8-27B | `2026-11-03-Qwen3.8-27B.md` |
| AliceAI-Foundation-80B-A3B-Base | `2026-11-03-AliceAI-Foundation-80B-A3B-Base.md` |
| Kimi-K3 | `2026-11-03-Kimi-K3.md` |
| granite-4.2-30b | `2026-11-03-granite-4.2-30b.md` |
| Qwen3.8-Flash-Next | `2026-11-04-Qwen3.8-Flash-Next.md` |
| Mistral-Small-4-119B-2603-NVFP4 | `2026-11-06-Mistral-Small-4-119B-2603-NVFP4.md` |
| Kimi-K3.1-Pro | `2026-11-05-Kimi-K3.1-Pro.md` |
| NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | `2026-11-10-NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16.md` |
| NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | `2026-11-11-NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.md` |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | `2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md` |
