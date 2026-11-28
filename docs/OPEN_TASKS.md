# Open tasks

**Rule:** only genuinely open work lives here — this file routes, it does not
explain. Detail is one hop away in `open-tasks/live/`. When an item is resolved
it moves to `open-tasks/<date>-<slug>.md` and leaves a row in
[`open-tasks/README.md`](open-tasks/README.md), which is the single index of
closed work (and of retired IDs).

Nothing here is speculative — each live item carries the measured value or error
code that produced it.

**IDs are stable.** `A1`, `G2a`, `R17`, `S4` … keep their numbers across moves:
`tests/CMakeLists.txt` and a dozen `tests/*.cpp` comments cite them by ID. A new
item takes the next free number in its area. An archived ID still resolves
through the `Retired IDs` table in `open-tasks/README.md`.

## Areas

| file | state |
|---|---|
| [`live/engine-defects.md`](open-tasks/live/engine-defects.md) | `A1` `cudaErrorIllegalAddress` in the qwen/ornith MoE kernel tests — 0 errors under `compute-sanitizer` on this host, so it needs the Windows/5060 Ti box; `A3.7`/`A3.8` two pre-existing reds on `main` (gemma NVFP4 embed gather OOB; GLM-flash fixture missing the router bias); `A3.9` `GlmRealCtx::Free` double-`cudaFreeHost`; `A3.11` `test_qwen_dense_converter_integrity` red on `main` (converter rejects a tokenizer.json whose `model` object the test removed); `A3.12` `test_kimi_moe_loader` red on `main` (its own fixture conversion returns false); `A3.13` CLOSED 2026-11-28 (plan Task 15a): a `bf16`-profile artifact passed the codec gate then loaded empty — the dense loaders were NVFP4-only. One shared dispatch (`RunWeightGemm`) + `gemv::DenseWeight` on the layer structs wire phi, qwen and dory; each bf16 tree now passes the canonical SelfCheck x3 on real weights with the same top-1 as its NVFP4 profile |
| [`live/instruments-and-guards.md`](open-tasks/live/instruments-and-guards.md) | `G1` the GPU `RESOURCE_LOCK` is per ctest process, so two runs share the card; `G2`/`G2a` a test can be named after a sibling; `G8a` mistral-small-4 reads 95.9 % / 96.3 % of its rows; `G9` the bandwidth probe fails only inside a parallel sweep; `I1` the CODE gate's regex anchor is order-sensitive, so `[A-F0-9]` scores as empty against `[0-9A-F]`; `I2` the three global-`cudaMemGetInfo` leak probes (`test_gpu_perf`, both deepseek loaders) are red only inside a parallel sweep |
| [`live/parity-goldens.md`](open-tasks/live/parity-goldens.md) | `K` the declared-debt mechanism — enforced in both directions so a missing golden cannot be invisible and a stale debt row cannot rot (the rejected golden set is kept as evidence in the 2026-11-20 archive). `L` (the MCP pilot rollout) is CLOSED: `mcp` is a required lifecycle stage with 48 targets, 39 goldens and 9 debt rows — [`2026-11-28-mcp-gate-rollout-closed`](open-tasks/2026-11-28-mcp-gate-rollout-closed.md) |
| [`live/infra-migrations.md`](open-tasks/live/infra-migrations.md) | `M` ~390 hand-written test registrations remain; `S1`–`S7` Arrow IPC residuals; no stage-9 report for 20 models; the ctest oracle is a dated snapshot, not a target list; `V` CLOSED 2026-11-28 (converters produce a second profile; selection by target and `--explain-artifact` shipped; the runtime half `A3.13` is closed by plan Task 15a for phi/qwen/dory, the other families still to port); `W` `baidu/ERNIE-5.5-Flash-Base-BF16` is on disk with no registry row (23 org dirs on disk, 22 in the registry) |

## Models with at least one open item

| model | file | state |
|---|---|---|
| Beam-1.0-501B-A23B | [`beam-1.0-501b-a23b`](open-tasks/live/models/beam-1.0-501b-a23b.md) | `R8` FSI/CODE/DET are declared debt — the 934 GB source exceeds 750 GB of RAM; 1.61 t/s |
| DeepSeek-V4-Pro-0813 | [`deepseek-v4-pro-0813`](open-tasks/live/models/deepseek-v4-pro-0813.md) | `R3` FSI-02 red: the model inlines the constants instead of the symbolic form; 0.77 t/s, host-store bound |
| DeepSeek-V4.1-Flash | [`deepseek-v4.1-flash`](open-tasks/live/models/deepseek-v4.1-flash.md) | `A2` a cancelled request used to poison the next one (fix in, verification pending); `Q` FSI 1/3, CODE 1/3, DET 2/3 on the E4M3FN tree |
| DeepSeek-V4.1-Flash-1017 | [`deepseek-v4.1-flash-1017`](open-tasks/live/models/deepseek-v4.1-flash-1017.md) | `S1` CODE-03 is a false red (the model answers `[A-F0-9]`, the extractor's anchor is literal `[0-9A-F]`) → instruments item `I1`; FSI 3/3, DET 3/3, SelfCheck Paris ×3; 16.27 t/s |
| Recursive Flint | [`flint`](open-tasks/live/models/flint.md) | `R6` routed 4-bit experts flip top-1 at prefix ~64; the 8-bit plane needs +21 GB and does not fit; goldens and a perf row owed |
| gemma-4-31b-it | [`gemma-4-31b-it`](open-tasks/live/models/gemma-4-31b-it.md) | `R` FSI-01 stays red — ~8 bits on every linear is 33.7 GiB, i.e. a 48 GiB card; 47.3 t/s. The residual plane is now a per-model manifest knob (`convert_residual_plane: false` here, `true` on the 12B), so re-converting this row stays 4-bit/16.5 GiB and loadable |
| Gemma-4-12B-IT | [`gemma-4-12b-it`](open-tasks/live/models/gemma-4-12b-it.md) | `R22` FSI-02 red: the SOURCE is a 0.556-logit coin flip at the fork (`1` 21.61 vs `Rate` 21.05), so the engine's 8-bit body can flip it; every other gate green, **59.85 t/s** auto / 48.77 bit-exact (the second number is attributed to the head's K=3840 sitting under the shared `gemv_large` K>=4096 floor) |
| GigaChat4-37B-A4B | [`gigachat4-37b-a4b`](open-tasks/live/models/gigachat4-37b-a4b.md) | COMPLETE (2026-11-26): hybrid GDN + DSA MLA + MoE, all nine stages green on real weights; `G1` source-side agreement is a per-layer probe, the FSI/CODE/DET goldens are template-derived; **59.1 t/s** |
| GigaChat4-708B-A41B | [`gigachat4-708b-a41b`](open-tasks/live/models/gigachat4-708b-a41b.md) | `G5` ingested 2026-11-27: the 37B's 64-layer sibling (61 MoE layers, 256 experts, 16 DSA layers), SelfCheck Paris ×3 `4c91393c68d0b3e8`, CODE 3/3, **3.71 t/s**; FSI 2/3 and DET 2/3 are the `I3` extractor class (preamble line / wrapped formula), goldens template-derived; 361 GB of experts stream from host RAM (488 touches/token) |
| GigaChat3.5-432B-A28B | [`gigachat3.5-432b-a28b`](open-tasks/live/models/gigachat3.5-432b-a28b.md) | `G4` the converted tree was built in `/tmp` and is gone (three gates SKIP); re-conversion needs disk headroom; `R12` decode is expert-H2D bound |
| Granite-Switch 4.1 30B | [`granite-switch-4.1-30b`](open-tasks/live/models/granite-switch-4.1-30b.md) | `R10` DET-02 red because the source answers `CRITICAL` where the template says `CATASTROPHIC`; 131k context is card-bound |
| Hy-Memory-30B-A3B | [`hy-memory-30b-a3b`](open-tasks/live/models/hy-memory-30b-a3b.md) | `R5` 87 t/s against a ~750 t/s roofline — the weight-streaming GEMV; prefill is token-by-token |
| Hy-SimulMT-30B-A3B | [`hy-simulmt-30b-a3b`](open-tasks/live/models/hy-simulmt-30b-a3b.md) | `R11` the speech encoder is not implemented (text translator only); the Hy perf debt applies unchanged |
| Hy4-flash | [`hy4-flash`](open-tasks/live/models/hy4-flash.md) | SelfCheck/CODE/DET green after `H1` (the iHC post term multiplied the **pre-norm** MoE sum on 41 of 42 layers — `H1` records the self-consistency check that found it, plus the tool-call decode-buffer and slot-cache-leak defects it exposed); `H2` FSI 2/3 (FSI-02 numericises where the golden is symbolic); Engram (126.7 GB) cut and counted |
| Inkling-2-Mini | [`inkling-2-mini`](open-tasks/live/models/inkling-2-mini.md) | COMPLETE (2026-11-21): looped `inkling2_loop`, all 9 stages green on real weights; goldens source-derived (first in the family); **34.1 t/s** (2026-11-22, experts device-resident) |
| TML-d24-varloop-research | [`tml-d24-varloop-research`](open-tasks/live/models/tml-d24-varloop-research.md) | COMPLETE (2026-11-22): the `is_varloop` sibling — same tower, pass count chosen at load time from the trained range 1..5; SelfCheck + FSI + CODE green on real weights, perf seeded, **DET-02 measured red** (`R21`: the DET gate extractor resolves the prompt echo, identically on the source and the engine) |
| Kimi-K2.7-Code | [`kimi-k2.7-code`](open-tasks/live/models/kimi-k2.7-code.md) | `R2` goldens infeasible here (~2.0 TB decompressed against 750 GB of RAM); KV capped at 1024 slots |
| Kimi-K3 | [`kimi-k3`](open-tasks/live/models/kimi-k3.md) | `R15` FSI/CODE/DET not run (multi-hour at 2.13 s/token); `A_log` read per channel; 4096 context; load flaky under VRAM pressure |
| Kimi-K3.1-Pro | [`kimi-k3.1-pro`](open-tasks/live/models/kimi-k3.1-pro.md) | `R17` the DSA indexer's arithmetic is reconstructed, not verified; the parity sweep is ~7 h of exclusive GPU, unrun |
| Ling-3.1-flash | [`ling-3.1-flash`](open-tasks/live/models/ling-3.1-flash.md) | `R7` goldens are declared debt (560B + GPU-only FLA kernels); 2.77 t/s |
| MiniMax-M3 | [`minimax-m3`](open-tasks/live/models/minimax-m3.md) | `R4` FSI-01 red (annuity-due form); decode is PCIe-bound at ~0.31 s/step; MSA past 2048 keys reads the whole KV |
| Mistral-Large-4-1T-A52B | [`mistral-large-4-1t-a52b`](open-tasks/live/models/mistral-large-4-1t-a52b.md) | `R14` DSA past `index_topk` has no real-weight verdict; 1.04 t/s through a 120-slot expert cache |
| Mistral-Small-4-119B | [`mistral-small-4-119b`](open-tasks/live/models/mistral-small-4-119b.md) | `T` goldens uncommitted — the source cannot be loaded (50 missing / 3118 unexpected keys); throughput reads 95.9 % / 96.3 % of its rows |
| Nemotron-3-Ultra-550B-A55B | [`nemotron-3-ultra-550b-a55b`](open-tasks/live/models/nemotron-3-ultra-550b-a55b.md) | `R19` FSI-01 appends `*exp(r)` — a real generation error, excluded in writing; no full-model source reference |
| Nemotron-3.5-Lightning-30B-A3B (BF16) | [`nemotron-3.5-lightning-30b-a3b`](open-tasks/live/models/nemotron-3.5-lightning-30b-a3b.md) | `R20` FSI 2/3, CODE 2/3, DET 1/3 — every case runs to its budget; the residual plane is the lever, not taken |
| Nemotron-3.5-Lightning-30B-A3B (NVFP4) | [`nemotron-3.5-lightning-30b-a3b-nvfp4`](open-tasks/live/models/nemotron-3.5-lightning-30b-a3b-nvfp4.md) | `R13` FSI red on the 4-bit body; the mamba expert store probes non-MoE layers at load |
| Step-5-Preview | [`step-5-preview`](open-tasks/live/models/step-5-preview.md) | `R9` the DSA indexer only runs as the dense case (≤4096 tokens); goldens are declared debt; 1.82 t/s |

A model with no open item has no row here — `Dory1.0-4B` is the current case
(`R18` closed on 2026-11-10, both halves pinned).

## Where a new item goes

| item is about | file |
|---|---|
| a kernel/runtime fault any family can hit | `live/engine-defects.md` |
| a gate, guard, harness or instrument reporting the wrong thing | `live/instruments-and-guards.md` |
| golden provenance, declared debt, capture tools | `live/parity-goldens.md` |
| build wiring, ctest registration, Arrow IPC, manifest gaps | `live/infra-migrations.md` |
| one model's gates, precision, perf, tokenizer, residency | `live/models/<model>.md` |

Closed work: [`open-tasks/README.md`](open-tasks/README.md) — one file per work
session, plus the `Retired IDs` table that keeps an archived `§ID` resolvable.
