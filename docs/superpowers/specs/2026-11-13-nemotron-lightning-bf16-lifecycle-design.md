# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 — lifecycle completion design

Date: 2026-11-13
Status: approved design (approach A — sibling completion, no new lineage), pending implementation plan
Branch: `feat/nemotron-lightning-bf16-lifecycle`
Temp workspace: `/mnt/fast_storage/tmp/nemotron_bf16/`
NVIDIA CUDA reference mirror: `/home/alexmv2025/projects/cuda/docs` (offline copy of the
`nvidia-cuda-docs` MCP; used for any kernel-facing decision — none is expected here)

## Intent

`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (`NemotronHForCausalLM`,
`model_type = nemotron_h`) already has a manifest row, a registered `kType`
(`nemotron-3.5-lightning-30b-a3b`), a leaf model class (`NemotronHModel`), a
converter (`lsxforge --nemotron`), a loader, kernels and eleven ctest targets —
all shared with the NVFP4 sibling through `NemotronModelBase`. What it does
**not** have, on this host, is any evidence:

| missing | evidence |
|---|---|
| converted tree | vault `<vault>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/logestix/` absent (folder re-downloaded 2026-11-13) |
| every real-weight gate | `docs/SELFCHECK_MATRIX.md` §4: "no tree, no goldens — never converted on this host" |
| parity goldens | `tests/fixtures/{fsi,code,det}/nemotron-3.5-lightning-30b-a3b.json` do not exist |
| real-weight tool round-trip | manifest `tool_roundtrip: false` (the NVFP4 sibling runs it: `true`) |
| perf rows | 2026-09-16 rows only (`98.24 t/s`, `r3 24.07` — a degraded r3 sample), `commit: "-"` |
| stage-9 report | manifest `report: ""` |
| manifest hygiene | `format_tag: ""` while the loader requires `logestix-nemotron-nvfp4`; no `aliases` |

So this is **not** a new ingestion. It is the lifecycle-completion session that
turns a declared-debt stub into measured reality, and the first time the BF16
distribution (rather than the vendor's ModelOpt tree) is converted and gated on
this host.

## Stage 0 triage (measured 2026-11-13, before anything is built)

**0.1 Classification — sibling variant.** Same `model_type` (`nemotron_h`), same
architecture, same frozen geometry as the ingested member
`nemotron-3.5-lightning-30b-a3b-nvfp4`; both converters emit the same
`logestix-nemotron-nvfp4` layout and the leaf models differ only in which source
encoding they accept (the NVFP4 leaf additionally requires the tree's `model`
field to name its own distribution, which is what keeps two trees under one
format tag distinguishable). Per the ingestion protocol § 0.1 this is a
**sibling**: reuse the parent's converter family, loader, kernel set, expert
cache, tool codec and base lifecycle class. No new constants header, no new
kernels. Documented precedent for the split: `NemotronHModel` vs
`NemotronNvfp4Model` already exist and `models/*.json` already carries both rows.

**0.2 Geometry** — read from `config.json` + the safetensors index (6513 tensor
names classified, not sampled):

| quantity | value | source |
|---|---|---|
| hidden | 2688 | `hidden_size` |
| layers | 52 | `num_hidden_layers` / `len(layers_block_type)` |
| hybrid pattern | 23 `mamba` + 23 `moe` + 6 `attention` at 5/12/19/26/33/42 | `layers_block_type` |
| full attention | 32 Q / 2 KV, head_dim 128, **NoPE** (no rotary module in the reference class; `rope_theta`/`partial_rotary_factor` are vestigial) | `transformers` 5.15.1 `NemotronHAttention`; measured 2026-11-01 §R13 |
| Mamba-2 | 64 heads × 64 ⇒ d_inner 4096; conv_dim 6144; in_proj 10304; state 128; 8 groups; conv 4; dts 0.001…0.1, floor 1e-4 | config + frozen passport |
| MoE | 128 routed, top-6, `moe_intermediate` 1856, relu2, `routed_scaling_factor` 2.5, 1 shared expert at 3712, `norm_topk_prob` true, `n_group=1`/`topk_group=1` | config |
| vocab / eps | 131072 (untied) / 1e-5 | config |
| checkpoint | 14 shards, 65 842 365 568 B, 31 577 937 344 params | index metadata |
| ballast to strip+count | **270** MTP tensors (`mtp.*`: 128 experts × up/down, 1 attention layer, 1 shared expert, norms) | index inventory |

**0.3 Tokenizer** — byte-level BPE via `tokenizer.json` (17 MB), `bos 1`
(`<s>`), `eos 2` with generation also stopping at 11, `pad 0`, `added_tokens`
carrying the ChatML sentinels; `tokenizer_kind = 9` (already the family's kind,
proven 22/22 by `test_nemotron_moe_tokenizer_parity` on the sibling).
`chat_template.jinja` is ChatML (`<|im_start|>…<|im_end|>`) with a Qwen-style
XML tool block and `<tool_call><function=NAME><parameter=K>V…` calls; the
generation prompt opens `<|im_start|>assistant\n<think>\n` (thinking **on** by
default), and an assistant turn with no content renders `<think></think>`.
`NemotronModelBase::RenderNemotronConversation` already reproduces exactly this
(verified in source 2026-11-01).

**0.4 Ballast** — vision/audio: none. MTP: 270 tensors, cut and counted into the
index (`deferred.mtp = "cut"`, `mtp_tensors`). Everything else in the index must
classify or the converter errors; the inventory above accounts for 6243 backbone
+ 270 mtp = 6513.

**0.5 Feasibility on this box** — RTX 5090 (32 607 MiB, CC 12.0), 750 GB RAM,
2.4 TB free on the vault disk. Converted tree ≈ 18 GB on disk and 17.5 GiB
device-resident (measured on the NVFP4 sibling's tree, same layout and
quantization), so the BF16 source's only extra cost is the one-time 65.8 GB
read. No expert pinning, no `ulimit -l` work. Fits with margin.

**0.6 Semantics to verify before trusting output** — all previously established
for this architecture and unchanged by the source precision: NoPE on the six
attention layers; relu2 on experts; top-6 with `norm_topk_prob` renormalisation
and 2.5 routed scale; shared expert added outside the routed sum; Mamba-2 state
in FP32, `time_step_floor` clamp; MTP not applied. The one thing the BF16 source
changes is **quantization provenance**: the NVFP4 tree quantized *after undoing*
ModelOpt encodings (9.5 % relL2 on the attention projections, 5.3 % on experts,
0.19 % on the head), whereas here every matrix is quantized straight from BF16 —
so the per-tensor errors get re-measured rather than inherited.

**Stage 0 gate: passed.** Rows to extend: `models/nemotron-3.5-lightning-30b-a3b.json`
(manifest fields only — no guard registry is hand-edited). Geometry, tokenizer
verdict, ballast count and feasibility are in the table above.

## Goals

- A converted tree at `<vault>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/logestix/`
  that loads through `NemotronHModel` and whose index names its own distribution.
- Every lifecycle stage **run on real weights**, each verdict recorded as a
  measured green or a measured debt — never a SKIP read as a pass:
  convert → load → kernels/session → SelfCheck (Paris ×3 bit-exact) → tokenizer
  parity → FSI/CODE/DET → tool round-trip → perf (canonical 2048-token protocol).
- Parity goldens whose provenance is the **BF16 source itself** (the checkpoint
  is runnable here with `transformers` 5.15.1 — it is its own oracle, unlike the
  ModelOpt tree that needed `tools/nemotron_nvfp4_oracle.py`), or an explicit,
  measured statement of why a gate has no source-faithful golden.
- Perf rows re-seeded through `test_perf_baseline_nemotron` (never hand-edited),
  with TTFT and NVML power evidence.
- Stage-9 report `docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md`.
- Manifest, `docs/MODELS.md`, `docs/SELFCHECK_MATRIX.md` §1/§4 and
  `docs/OPEN_TASKS.md` in step with the measurements; guards green.

## Non-goals

- **The residual plane.** Operator decision 2026-11-13: measure first. If FSI is
  red for the documented 4-bit-body reason, the number is recorded and the lever
  (attn+Mamba residual planes, ~+1.1 GiB resident, doubled attn+Mamba read, all
  perf rows re-seeded, BF16/NVFP4 trees no longer comparable) becomes its own
  task — not a change smuggled into a lifecycle run.
- Any perf work beyond re-seeding: no new kernels, no tuning, no layout change.
- Editing the NVFP4 sibling's TUs, goldens, tree or perf rows.
- Context beyond the canonical 4096 `max_seq` / 2048-token perf protocol.
- MTP/speculative decoding (stored and counted, never applied).

## Anti-goals

- A gate that SKIPs being written up as a pass, or a golden rewritten to pass.
- Converting the real checkpoint inside a ctest target (converter tests stay
  synthetic, `convert` label, bounded TIMEOUT).
- Recording the sibling's NVFP4 measurements as if they were this model's: every
  number in the report must come from this session's BF16 run.
- Declaring "all stages verified" on the strength of the fast suite alone.

## Approaches considered

**A — sibling completion (chosen).** Convert with the existing `--nemotron`
pipeline, run the existing targets, add only what the evidence gap needs (parity
goldens, capture override, manifest fields, report). Trades almost no code risk
for a hard dependency on the sibling's correctness: if a semantic defect is
found here, it is fixed in the shared base and both trees inherit it — which is
what the model-independence guard expects (one passport per geometry, no
cross-family borrowing).

**B — separate lineage (rejected).** Ultra's shape: own passport, converter,
loader, kernels, pipeline. Rejected because the geometry is *identical*, and
`test_model_independence_guard` enforces unique frozen geometry per passport —
a second passport for the same widths is exactly what it forbids. The Ultra
split was forced by different geometry, not different precision.

**C — fold the BF16 distribution into the NVFP4 `kType` (rejected).** One model
row, one tree, source chosen at conversion. Rejected on provenance: the parity
gates are keyed by `model_type`, so one `kType` would mean one golden set for two
source distributions, and the tree's `model` field check — added precisely so
"the sibling's tree would load here and then verify against the wrong
FSI/CODE/DET goldens" — would have nothing to distinguish.

## Design

Order follows the ingestion protocol; each stage lands with the test (or the
measurement record) that pins it, in the same commit.

**1. Branch and workspace.** `feat/nemotron-lightning-bf16-lifecycle` in the main
checkout (not a worktree: a second tree would re-fetch and rebuild Arrow/CCEL
from scratch for zero isolation value — the checkout is clean and single-operator).
Build: `linux-release-znver4-aocc` for every measured number (the preset the
sibling's baselines came from) and `linux-debug` for the fast/guard sweep.
Temp: `/mnt/fast_storage/tmp/nemotron_bf16/` for oracle dumps and probes.

**2. Conversion (stage 1/2).** `lsxforge --nemotron --input <vault> --output <vault>`
writes `<vault>/logestix/` (+ `pinned/`, `experts/`). Verification, all recorded
in the report: index `model` field, 2944 routed experts (23 × 128), pinned tensor
count, `mtp_tensors = 270`, and a **per-tensor relL2 vs the BF16 source** table
for one attention projection, one Mamba projection, one routed expert, the
shared expert and `lm_head` — measured by decoding the written `pinned/*.arrow`
records back to BF16 (the codec the sibling used for the same purpose). The
existing synthetic E2E `test_nemotron_moe_convert` must stay green and unchanged.
Nothing in the converter is expected to change; if the BF16 source trips a
classification branch the sibling never hit, the fix is a failing converter test
first (TDD), then the branch.

**3. Load / kernels / session (stages 3, 5, 6).** No code expected.
`test_nemotron_moe_loader`, `test_nemotron_moe_expert_cache`,
`test_nemotron_moe_model`, `test_nemotron_moe_math` (fast, synthetic) plus
`test_nemotron_moe_kernels` and `test_nemotron_session` on the GPU. If the tree
from a BF16 source exposes a loader assumption (e.g. a dtype or a missing
`weight_scale`), the failing test is written first.

**4. Tokenizer (stage 4).** `test_nemotron_moe_tokenizer_parity` against the real
vault (kind 9, 22/22 expected); `tools/nemotron_tokenizer_capture.py` already
owns the corpus. `tokenizer_kind` in the manifest must match what the converter
actually writes.

**5. Tool-calling bridge (stage 4b).** The native codec exists
(`RenderNemotronConversation` + `ParseQwenTokenCalls`) and the suite guard names
`test_nemotron_moe_model` as its carrier, but the **real-weight** round-trip has
never run for this distribution. Work: flip the manifest to `tool_roundtrip:
true`, run the selfcheck driver's tool hook on real weights (expect
`get_weather {"city":"Paris"}`), and pin the render/parse shape against
`chat_template.jinja` in a host-only `fast` test. If a gap between the rendered
frame and the template appears (e.g. the `<think></think>` empty-content branch),
the test goes red first and the shared renderer is fixed — which also re-verifies
the NVFP4 sibling.

**6. SelfCheck (stage 8).** `test_nemotron_moe_selfcheck` on real weights: Paris,
≥ 3 cycles, bit-exact `final_logits`, `logits_hash` recorded for the matrix row.
The manifest's `selfcheck_max_gen = 2`, `selfcheck_bit_exact = false`,
`selfcheck_frame_prompt = false` are transcribed from the sibling's measured
probe and are **not** normalised; if the BF16 tree's probe disagrees, the
difference is recorded before any of the three is touched (§G5 rule).

**7. FSI / CODE / DET (stage 8, gates).** Capture from the **BF16 source** with
`tools/fsi_capture.py` / `code_capture.py` / `det_capture.py` (torch CPU, 65.8 GB
BF16 resident, 750 GB host), writing `tests/fixtures/{fsi,code,det}/nemotron-3.5-lightning-30b-a3b.json`.
Capture must reproduce the engine's frame, which for this checkpoint means
thinking **on**: the capture override gets `enable_thinking: True` (the default
`False` would emit `<think></think>` and capture a different prompt than the
engine consumes). Verified mechanically: the golden's `input_ids` are what the
gate feeds the engine, so the alignment test is "the captured ids equal
`BuildConversationInputIds` on the same conversation" — asserted in the run
transcript and in the report, not assumed. Then run
`test_nemotron_moe_{fsi,code,det}_selfcheck` on the converted tree. Expectation,
stated up front so a green is not mistaken for a design: FSI may be red on the
4-bit body exactly as measured for the sibling (§R13 N1); CODE/DET may have no
source-faithful golden because the source echoes the prompt body. Either outcome
is written into `docs/SELFCHECK_MATRIX.md` + `docs/OPEN_TASKS.md` with its
measurement, and the manifest's `golden_debt` reasons are replaced by those
measurements.

**8. Perf (stage 4 of the lifecycle skill).** `test_perf_baseline_nemotron`
(Release, `LSX_PERF_VAULT_SUBDIR` already pinned by the manifest, label `perf`,
`SKIP_RETURN_CODE 4`) runs the canonical 2048-token protocol twice (kAuto `perf`
and kOff `r3`) and writes/validates the rows including TTFT and NVML power. The
2026-09-16 rows are replaced by this run — through the harness, never by hand.
One GPU workload at a time: check `nvidia-smi --query-compute-apps` before each
GPU-labelled target.

**9. Documentation and report (stage 9).**
`docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md`
(publish-ready post + evidence section, every number traceable to a gate file),
manifest `report`/`notes`/`aliases`/`format_tag`/`tokenizer_kind` finalised,
`docs/MODELS.md` regenerated with `tools/gen_model_manifest.py --docs`, the
matrix row moved out of §4 with its measured verdicts, and a new
`docs/OPEN_TASKS.md` section recording the two source-level facts this run
establishes (BF16-source quantization error table; parity provenance) plus the
residual-plane decision as an open item with its measured trigger.

**Guard sweep before the report is written (not after):**
`test_manifest_sync --strict`, `test_model_independence_guard`,
`test_model_suite_guard`, `test_code_naming_guard`, `test_ctest_labels_guard`,
`test_expert_store_guard`, plus `ctest --preset linux-debug` (fast),
`-convert` and `-inference`.

## Verification (definition of done)

- Tree present; index names its own distribution; expert/pinned/MTP counts match
  the inventory.
- `test_nemotron_moe_convert` green (synthetic, bounded TIMEOUT) — unchanged.
- Loader / kernels / session / math targets green on the BF16 tree.
- Tokenizer parity 22/22 on real weights.
- SelfCheck Paris ×3 bit-exact, hash recorded.
- FSI/CODE/DET each **run**; green, or red/debt with the measured reason and a
  matrix + OPEN_TASKS entry.
- Real-weight tool round-trip green; manifest `tool_roundtrip: true` backed by it.
- Perf rows re-seeded with TTFT + power, dated to this run.
- Report written after the gates, evidence section pointing at them.
- All guards green; `docs/MODELS.md` current.

## Risks and their handling

| risk | handling |
|---|---|
| FSI red on 4-bit body (sibling precedent) | measure, record, escalate as the residual-plane task; do **not** touch the tree mid-run |
| Codec/`<think>` frame mismatch between capture and engine | mechanical check on `input_ids` equality before any gate run |
| Converter assumption that never met a BF16 source | failing converter test first, then the branch |
| Perf rows moved by a stale/absent tree | Release preset + harness re-seed only |
| A GPU-labelled sweep started while another holds the card | `nvidia-smi --query-compute-apps` gate before every run |

## Left to the builder

Task decomposition, commit boundaries, the exact capture flags per prompt set,
and the measurement formats inside the report. Nothing in this document
pre-approves a design change: a tree-affecting change (residual plane, new
passport, layout change) comes back to the operator.
