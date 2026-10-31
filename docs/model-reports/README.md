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
| Beam-1.0-501B-A23B | `2026-10-31-Beam-1.0-501B-A23B.md` |
| Granite-Switch-4.1-30B | `2026-10-31-granite-switch-4.1-30b.md` |
| Step-5-Preview | `2026-10-31-Step-5-Preview.md` |
