# Inkling-2-Mini — ingestion report (2026-11-21)

## Publish-ready post

**What was converted.** *Inkling-2-Mini* (architecture `inkling2_loop`) is the
**looped** sibling of Inkling-2-Small: 24 physical MoE decoder layers, of which
layers `[6, 18)` are run **three times per token**, so one token is 48 layer
applications drawn from 24 layers of weights. The source ships as a single
18.19 GiB BF16 safetensors shard (plus a 0.59 GiB `mtp.safetensors` that the
index does not list) and converts to a **17.7 GiB** tree with 645 pinned tensors
and 3 072 routed-expert records. The dense plane is device-resident (3.0 GiB
minimum, 4.2 GiB at the planner's preferred layout); the experts stay
memory-mapped on the host.

**The loop is the architecture, not an optimization.** A pass is not a residual
stack that runs again: passes 1 and 2 **restart from the pre-loop hidden state
and block stack** snapshots, and each loop layer receives four *messages*
retrieved by a single-head attention over the PREVIOUS pass's block states
(skipping the one state that pass re-stated) plus its final hidden state. Those
messages are blended into the layer's q/k/v inputs and its MLP input through
four rank-64 **MessageGates** — `(1 − tanh(forget(gate)))·h + tanh(gain(gate))·msg`
— applied *after* the depth mixture and *before* the input norm. Every pass owns
its own KV rows and its own four short-convolution rings: 48 cache slots, one per
(layer, pass). A pass that continues from the previous pass's output instead of
its snapshot, or that reuses pass 0's KV slots, still produces fluent text — it
just stops being this model. Both are pinned (see "Measured numbers").

**Nothing is requantized.** The release is mixed-precision and says so only in
`hf_quant_config.json`: routed experts of layers 0–19 are **NVFP4** (E2M1 nibbles,
per-16 E4M3 block scales, per-expert global scale `amax/2688`), layers 20–23 are
**MXFP8** (E4M3 with a `ue8m0` exponent per 32), and everything else is BF16
except the AttnRes queries, the merged cross-pass message queries and the router
correction bias, which are FP32. All of it is carried byte-for-byte; the FP8
layers are decoded by the family's dedicated 8-bit GEMV. Dense BF16 weights are
widened to FP16 on the device.

**Six semantics break a generic decoder**, and each was verified against the
release's own modelling code (`transformers` 5.24.0,
`models/inkling2_loop/modeling_inkling2_loop.py`) before a line of the pipeline
was written:

1. **No RoPE.** Position enters through a learned bank of `d_rel` = 16 profiles
   over 1024 backward distances; q and k are **L2-clipped** per head to 22.627
   (`q` through a learned per-head scale, `k` weight-free) and `v` is
   RMS-normalized without a learned scale.
2. **Log scaling starts at position 8192**, not 128 000 as in Small — copying the
   sibling's floor would leave the whole 8k engine context unscaled.
3. **GQA is 3** (24 query heads over 8 kv heads), not Small's 6.
4. **Depth mixing (AttnRes) over block prefixes**: block size 6, a boundary layer
   appends its incoming hidden state to the stack **instead of** keeping a local
   residual, the four full layers ask for **three** mixtures and everything else
   for one.
5. **Hybrid attention**: 20 sliding layers (window 1024) and 4 full layers at
   `{5, 11, 17, 23}`; the full layers' k/v convolutions take **dynamic** taps
   (a per-token projection summed with a static table) and swap the normalizer
   from `sqrt(k+1)` to `sqrt(2k+1)`.
6. **MoE**: 128 routed experts, top-8, no shared expert, sigmoid scoring with a
   **selection-only** FP32 bias, weights renormalized to 1 then scaled by 8.0;
   experts are softcapped SwiGLU, `tanh(g/4)·4 · sigmoid(g) · tanh(u/25)·25`,
   accumulated in FP32.

**The checkpoint's own traps** (all silent, all pinned by a test now):

- *The fused expert slab is row-interleaved.* `w13_weight` stores gate and up rows
  alternating (`[gate₀, up₀, gate₁, up₁, …]`). Taking the first half as gate and
  the second as up keeps every byte, plane size and per-expert `amax`, and every
  artifact-reading reference agrees — what it destroys is knowledge. The
  converter de-interleaves; the knowledge probe below is the gate that catches a
  regression.
- *The codec changes mid-stack* (NVFP4 → MXFP8 at layer 20) and only
  `hf_quant_config.json` knows; `config.json` describes neither.
- *Two cross-pass query tensors are merged into one* (`message_queries`, shape
  `[12][4][2304]`, order `[q, k, v, mlp]` per loop layer), and `attn_res.
  attn_queries.i` belongs to layer `i+1`, not `i`.
- *`loop_config` lives only in `text_config`* and is not reflected anywhere else;
  a decoder that reads `num_hidden_layers` alone runs 24 applications instead of
  48 and answers confidently.
- *`mtp.safetensors` is 0.59 GiB that the index does not list.* Correctly ignored;
  a converter that enumerates the directory instead of the index would ingest it.

**Measured numbers.** Real 17.7 GiB tree, one GPU target at a time, RTX 5090
(32 GiB), CUDA 13.5 on a 13.4 driver:

| measurement | result |
|---|---|
| SelfCheck ("The capital of France is" → "Paris") | PASS, 3 bit-exact cycles, `logits_hash=12e684ecd27e40d7` |
| … vs the SOURCE's own greedy stream | **identical ids**: prompt length 20 (the source's own `prompt_ids`) and the continuation opens `200001 200008 371 150409 171960 25 196980 371 151451 149879 151194 149884`, exactly the release's first 12 tokens; " Paris" lands at token 27 of 32 |
| knowledge probe (one prefill per position) | mean NLL 3.458 → perplexity **31.8**, top-1 `152717` (" Paris") |
| FSI parity gate | **3/3, source-captured** — the release's own answers, produced by running its code (see below) |
| CODE parity gate | **3/3, source-captured** |
| DET determinism gate | see the matrix row; byte-identical repeat runs |
| native tool round-trip | 56 framing assertions, byte-exact against the checkpoint's `chat_template.jinja` rendered by jinja2 |
| scheduling/state suite | 46 assertions: per-pass rows differ at every loop layer, pass 1's KV slot is neither pass 0's nor untouched, two identical requests reproduce their logits bit-for-bit |
| kernel numeric suite | 14 014 assertions over 8 cases (MessageGate and message retrieval vs host references, GQA-3 attention with sinks/window/bias, MXFP8 GEMV at K = 1152, router 128/top-8, SwiGLU 1152) |
| artifact / conversion | 17.7 GiB, 645 pinned tensors, 3 072 expert records, 422 vision tensors stripped; **16 s** wall for the whole conversion |
| decode throughput, 2048 tokens, auto policy | **6.10 tok/s** (TTFT 3.41 s, 135.3 W average, 138.8 W peak) |
| decode throughput, 2048 tokens, strict/bit-exact | **6.05 tok/s** (TTFT 3.38 s, 139.5 W average, 142.0 W peak) |

**The parity goldens are source-derived, which is new for this family.** The
sibling's goldens are template-derived because its far larger source cannot run
on one card. This checkpoint's 18 GiB BF16 source fits the host, so the capture
path loads the release's own `inkling2_loop` text tower on the CPU (bf16, 64
threads, ~0.9 s/token). The FSI, CODE and DET goldens are therefore the model's
own answers, and the converted engine reproduces all nine.

**One shared-harness fix came out of this ingestion, and it is disclosed.** The
FSI comparison extractor used to bail on the FIRST `anchor` occurrence on the
chosen line, while the capture-side reference (which produced every committed
`expected`) searches past candidates that cannot satisfy the
`anchor operator identifier` bound. A reasoning stream restates the prompt, so
the echo `Drawdown = 14.999991` — an *assignment*, not a comparison — is exactly
such a candidate: the extractor reported an EMPTY answer for a model that
answered correctly. The line chooser now also requires a bounded candidate. This is
provably non-regressive (a line with a bounded candidate also contains
`anchor` + operator, so it was the line the old scan picked first) and it is
pinned by two new unit cases plus the existing "arithmetic keeps the historical
first-occurrence behaviour" case.

## Throughput

**6.10 tok/s** auto / **6.05 tok/s** bit-exact at the canonical protocol (2048
tokens, prompt "The capital of France is"), TTFT 3.4 s, recorded in
`tests/perf/perf_baselines.json`.

The number is 4× the sibling's 1.47 tok/s and the reason is memory residency, not
arithmetic: the whole 17.7 GiB artifact fits in 32 GiB of VRAM, so the expert LRU
pins all 3 072 records and a token is a cache hit rather than a first-touch read
of a 366 GB tree. What remains is compute: the loop makes one token cost **48
layer applications** — twice a 24-layer model of this width — which is why the
rate sits far below what a 3B-active dense model of the same size would reach.

Two corroborations that the decode is compute-bound and not latency-bound: the
auto and bit-exact modes are within **0.8 %** of each other (there is almost no
fast-math-sensitive arithmetic left to change), and average power is 135 W on a
575 W card with the SMs busy — the opposite signature of the sibling's 95 W
read-bound idle.

The measurement ran on the `linux-release` preset (NDEBUG, generic `-O3`), not
the `linux-release-znver4-aocc` reference build used for the earlier catalogue
rows: that preset's build directory no longer re-configures under the installed
CMake 4.2.3 (Arrow's own `CMakeLists` rejects it), so re-seeding it would have
meant repairing an unrelated dependency. The baseline is a floor with a -3 %
tolerance and self-updates upward, so a later znver4 measurement can only raise
it.

**A second shared-harness defect, found by the tree verifier.** The routed-expert
scale check assumed every expert plane is NVFP4 (two elements per packed byte).
This checkpoint's last four layers are MXFP8 — one element per byte — so
`lsxforge --verify` rejected them as "165888 scales for 2654208 packed bytes,
expected 331776". The sibling has been shipping in exactly that state without
anyone noticing: its MXFP8 files never failed because the deep-sample budget
(256 MiB of a 366 GB tree) never reached one, and raising the budget on its own
tree reproduces the same rejection. The check now takes the packing from the
converted index's per-layer codec table instead of guessing it, and maps each
`experts/layer_<n>/` file onto its layer. Both trees verify clean afterwards
(3099 files / 50 deep and 18483 files / 100 deep respectively), and a scale
plane matching neither convention is still rejected — the defect class the check
exists for.

## What was not measured, and why

- **Context beyond 2048 tokens.** The engine bound is 8192 positions
  (`kLoopMaxContext`), the KV block is allocated full-context for all 48 cache
  slots, and the perf protocol is fixed at 2048. The checkpoint's own context is
  larger; the log-scaling path that activates from position 8192 is exercised
  only by kernel-level reference tests at synthetic positions, because the
  selfcheck prompt is 20 ids long.
- **The MTP head** (0.59 GiB, `mtp.safetensors`, absent from the index). Counted,
  never applied; speculative decoding is not implemented for this family.
- **The vision tower** (422 tensors). Stripped and counted, never applied — the
  ingestion is text-tower only, and the tool/tokenizer contract is the one the
  sibling already verified (the tokenizers are byte-identical).
- **Multi-request concurrency.** One `Infer` per instance at a time is a
  precondition of the residency helper (OPEN_TASKS E1b), as for every family
  here; the loop's per-pass state is per-sequence and is reset at each request
  start, which is what the bit-identical repeat case pins.

## Evidence

- Frozen passport: `libs/lsxcommon/src/inkling_moe_loop_constants.h` (+ its
  constants test).
- Reference runner: `tools/inkling_loop_transformers.py` (stage-0 verification)
  and `tools/inkling_loop_oracle.py` (golden captures).
- Stage-0 trail with the verified semantics, the probe ids and the merge order:
  `docs/open-tasks/live/models/inkling-2-mini.md`.
- Manifest: `models/inkling-2-mini.json`; matrix row:
  `docs/SELFCHECK_MATRIX.md`.
- Full-lifecycle sweeps (2026-11-22): fast 158/158 serial; convert 44 targets;
  inference 343 targets, in which all eight of this family's targets passed
  (loader 0.5 s, model 0.4 s, pipeline 0.5 s, kernels 0.3 s, SelfCheck 118 s,
  FSI 354 s, CODE 339 s, DET 909 s). Every guard green. The inference sweep's
  other reds are pre-existing and belong to families this work does not touch
  (five documented in `docs/OPEN_TASKS.md`: DeepSeek-V4.1-Flash's `Q`, Gemma's
  FSI-01, both Granite-Switch DET-02 rows, and Nemotron's `R20`).
