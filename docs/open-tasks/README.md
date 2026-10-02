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
| 2026-09-30 | [`2026-09-30-cuda-review-windows-bringup.md`](2026-09-30-cuda-review-windows-bringup.md) | CUDA inference review of `24bf180~1..bed5d9f` + the Windows build repair. Archived 2026-10-02 with all items closed: A2 (MiMo — vendor-oracle match, not a bug), B1, D1, D2, E1, G1, plus a 9-item inference-conformance pass and a 15-item "closed in this session" list. |