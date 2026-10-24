# phi-4: the kAuto perf-mode row, the continuation gap, and two instrument defects (closed 2026-10-24)

Session: full phi-4 lifecycle re-audit at the operator's request, then the open
`perf`-mode item ("the 134.67 t/s row of 2026-10-01 is not reproducible"). The
vault had no converted artifact at the start — the `logestix/` tree used for the
10-07/10-23 measurements was gone — so the model was re-converted in place
(`lsxforge --phi`, 7.7 GiB, 59 s) and every stage re-run.

Result: two model/harness defects fixed, one codec cost recovered by a
bit-identical rewrite, and the perf rows re-seeded by the harness with real
evidence. Remaining: the DET-02 near-tie (live §O, unchanged) and the
harness-vs-app measurement offset (recorded below as a lead, bounded).

---

## 1. Continuation never received the audit-X2 fix (P1, fixed)

`phi_dense_inference.cu` is granite's generation body with phi's fused qkv /
gate_up layouts, but it did not get granite's X2 continuation fix. Its prefill
hard-coded `pos[i] = i` and `kv_base = 0`, then reset `kv_rows = seq`, so a
continuation `Infer` (whose `input_ids` equal the previous call's `gen_tokens`,
the session rule `phi_dense_model.cpp` implements) wrote its K/V over cache row
0 and attended only key 0. The session context was dropped and the answer came
from a one-row attention; the request still returned `ok = true`.

Ported granite's contract: `pos0 = cache->kv_rows`, positions `pos0 + i`, the
attention base `pos0`, capacity `pos0 + P + max_new`, `kv_rows = pos0 + seq`.
Fresh sessions are byte-identical (`pos0 = 0`), so no golden moved.

Pinned by the new `test_phi_dense_model` case "phi continuation matches a fresh
prefill": RED before the fix (`kv_slots_used` 3 for the continuation vs 10 for a
fresh prefill of the same context), GREEN after; the case also compares
`gen_tokens` and `final_logits` against the fresh run.

## 2. A Debug perf run rewrote the tracked baseline oracle (P2, fixed)

The `#ifndef NDEBUG` skip lived only in `MeasureModel()`, i.e. after
`RunPerfModel()` had already parked the GPU and run the 512 MiB DRAM probe — and
`UpdateProbeReference()` writes the `__gpu_probe__` row into
`tests/perf/perf_baselines.json`. Two `ctest --preset linux-debug -R
test_perf_baseline_phi_4` runs rewrote that row (1524.37 -> 1528.54 -> 1526.45)
before SKIPping, so a Debug sweep — which measures a debug-compiled probe that is
never compared against the Release reference — moved the committed environment
oracle. The gate now sits at the top of `RunPerfModel`, before the cooldown and
the probe: a Debug run SKIPs in 0.01 s with the file untouched. Pinned by a
source-scanning case in `test_perf_baseline_logic`.

## 3. The kAuto perf-mode row: attribution and resolution

### The measurements

| path | perf (kAuto) | vs 134.67 | r3 (kOff) | vs 128.68 |
|---|---|---|---|---|
| `lsxbenchmark --bench`, before any change | 132.1 / 131.5 | 98.1% / 97.7% | 130.9 | 101.7% |
| `test_perf_baseline`, before any change | 129.89 / 129.54 | 96.5% / 96.2% (DEGRADED) | — | — |
| `lsxbenchmark --bench`, final (decode rewritten) | 133.7 / 133.2 | 99.3% / 98.9% | — | — |
| `test_perf_baseline`, final | 130.72 | 97.1% (PASS) | 129.68 -> pass | — |

Environment oracle nominal throughout (DRAM probe 1524-1534 GB/s against the
1524 reference). qwen3.5-9b reads 100.4% of its 2026-10-23 re-seed.

### What the deficit was

`pow2` below means the pre-2026-10-06 codec (the decode state the reference was
measured with), reproduced locally by reverting only the decode body:

| build | phi perf, 2048 tokens |
|---|---|
| E4M3FN decode, early-return form (shipped until this session) | 131.5-132.1 |
| E4M3FN decode, single-return selects | 132.4-132.7 |
| E4M3FN decode, single-return + FMA subnormal (final) | **133.2-133.7** |
| pow2 decode (pre-codec) | 133.1-133.6 |

So the whole measurable part of the gap was the **E4M3FN block-scale decode in
the seq==1 streaming-GEMV inner loop**, added 2026-10-06 (`b30eb71`, subnormals
in `c037ed7`) — five days *after* the reference row. The cost is real and not
cosmetic: the current phi artifact contains **3 636 148 subnormal scale bytes
(3.40%)** across the sampled layers and 0x7F is absent, so the subnormal path is
hot and the pre-codec decode would have zeroed those blocks.

The final form keeps the codec semantics byte-identical (exhaustive host table +
exhaustive device parity in `test_lsx_kernels` both pass) and removes the
int-to-float convert:
`bits = ((e + 120) << 23) | (m << 20)`, `v = bitcast(bits)`,
`sub = 2*v - 2^-6 == m * 2^-9` (Sterbenz-exact), `0x7F` (bits == 0x43F00000)
-> NaN, `e == 0` -> sub. That is one return path, no branch on the hot path, and
it lands at pow2 parity.

### Hypotheses ruled out (each measured)

- **Compiler/build tree.** The reference predates the AOCC preset (added
  2026-10-06), so it was recorded on the GCC `linux-release-znver4` tree. HEAD
  rebuilt on that tree reads 133.2 / 132.7 — identical to AOCC (132.4-132.7).
- **First-Infer cost / cold DVFS.** A 32-token untimed warm-up before the
  measured run did not move the harness reading (129.65 vs 129.5-130.4).
- **NVML power-sampler contention.** Disabling `GpuPowerSampler::Start()` did
  not move it (130.0 / 129.8).
- **Cooldown state.** `LSX_PERF_COOLDOWN_TIMEOUT_S=0` on a hot card reads 128.6
  (worse), so the cooldown is doing its job rather than causing the deficit.
- **The artifact.** A fresh conversion has the same byte counts and the same
  resident footprint (7865 MiB); weight bytes do not move kernel time.
- **Environment drift.** phi's own r3 row (same streaming-GEMV route) moved
  128.68 -> 132.12 -> 129.68 across consecutive harness runs today, i.e. the
  harness's own run-to-run spread on this model is ~2%. A three-week-old row
  cannot be held to a 3% band with more confidence than that.

### Resolution

Decode rewrite landed (above); the perf rows were then re-seeded **by the
harness**, with its own TTFT/power evidence. The first pass wrote `perf`
134.67 -> 130.72 (ttft 30.17 ms, 565.5 W avg / 602.2 W peak, 781 samples),
then the next sweep tightened both rows as kUpdate is designed to: **perf
130.91**, **r3 130.31**. `test_perf_baseline_phi_4` passes both modes.

## 4. New instruments

- **`phi.dense_gemm.bandwidth`** budget row (min 1013.71, observed 1267.14
  GB/s). The case existed but had **no row for this GPU**, so it SKIPped — the
  silent-skip failure mode. The shape set also gained the real `qkv` shape
  (5120x7680), which the four-shape set never measured (22 MB/token, 18% of the
  decode weights). Isolated rates 2026-10-24: qkv 831, q_o 722->758, gate_up
  1267, down 1112, head 1395 GB/s. The 2026-09-22 four-shape 1327 reference is
  not comparable to the five-shape aggregate.
- **`tests/test_kernel_budget.h` parser.** The loader was strictly
  one-object-per-line and read only the fields on the first line of an entry, so
  the hand-formatted multi-line rows (`xing.kv_read.decode_gbps`,
  `mem.stream_read_ceiling_gbps`) came back with empty `note/date/samples` —
  and any `LSX_KERNEL_BUDGET_WRITE=1` run then rewrote the file from that
  truncated view, destroying their notes (observed). It now joins continuation
  lines (brace-balanced, string-aware) and throws on an unterminated or
  unnamed entry instead of dropping it into a false "no record for this gpu"
  SKIP. Pinned by "kernel budgets: a hand-formatted multi-line entry is parsed",
  which also re-reads the committed file and checks every entry survives whole.

## 5. Still open / leads

- **DET-02** — the C-vs-CR near-tie stays red as source parity (live §O,
  re-verified today: source top-8 `C` 30.75, `CR` 29.625, `CAT` 22.75).
- **Harness-vs-app offset on fast models.** `test_perf_baseline` reads
  130.3-130.7 where `lsxbenchmark --bench` reads 133.2-133.7 on the same binary,
  artifact, prompt protocol and fast-math mode (~1.5-2%, reproducible). NVML,
  warm-up, cooldown state, compiler and artifact are eliminated; the remaining
  candidates are process/VRAM-layout differences (the harness holds an
  `InferenceEngine::Session` and allocates through a different sequence) or a
  Catch2-process effect. Worth one instrument audit, because it sets the floor
  for every fast model's row (granite, 2x slower per token, shows <1%).
- **Reroute lead (not adopted).** Serving the mid-N shapes on the warp-per-row
  kernel measured qkv +9.6% (831 -> 910) and q_o +6.5% (728 -> 775) in
  isolation, but `down` (deep K) loses 9.6% (1112 -> 1005) and granite's N=4096
  shapes already preferred the sliced route (453 vs 514). A rule that fits all
  six measured points is `N >= 5120 && K <= 8192`, which is a fill-ratio
  heuristic with only this family's data behind it — a neutral-dispatch change
  for a different session, with every affected family re-measured.
- **Launch gaps.** The decode step spends 0.28 ms/token (3.7%) between kernels
  across ~360 launches; the CUDA-graph route Xing uses (`LSX_XING_CUDA_GRAPH`)
  has not been ported to phi.

---

## 6. Resolution of §O (same day): DET-TEST-02 green — 4-bit body + residual plane + FP16 edges

§O above (and the §O section of the live file, now removed) recorded the DET-02
near-tie as a deliberate, permanently-red source-parity gate. That conclusion is
**withdrawn**: the gate is green, and the trail of how the boundary was found is
the interesting part.

### 6.1 The first suspicion — "our quantizer is bad" — was measured, and refuted

An offline bench (the converted tree dequantized back to an fp16 checkpoint)
reproduces the engine's logits to ~0.02, so the bench is trustworthy for
counterfactuals. On the same prompt:

| variant | source margin `C−CR` | greedy |
|---|---|---|
| source bf16 | +1.125 | `CATASTROPHIC` |
| our NVFP4 (all 4-bit) | −0.688 | `CRITICAL` |
| NVFP4 body + BF16 LM head | −0.344 | `CRITICAL` |
| NVFP4 body + BF16 embed & head | −0.484 | `CRITICAL` |
| NF4 (QLoRA table, block 64, bf16 embed) | **+2.562** | `CATASTROPHIC` |

NF4 keeps the prompt, so "4 bits in general" was not yet an answer. Two more
measurements settled it:

* **Weight reconstruction**: relL2 vs the source is 9.52% (NVFP4) against 9.11%
  (NF4) globally, and NVFP4 is *better* on most tensors (qkv 0.093 vs 0.105). The
  stored weights are not the problem.
* **Distribution, not one prompt**: on a 35-prompt battery (9 with a source
  top-2 margin < 2 logits) and a 2048-token perplexity probe:

  | variant | PPL (source 13.4811) | top-1 | agreement | near-ties |
  |---|---|---|---|---|
  | NVFP4, everything 4-bit | 14.0024 (+3.9%) | 0.5100 | 31/35 | 5/9 |
  | NVFP4 + FP16 embed | 14.0109 | 0.5095 | 30/35 | 5/9 |
  | NVFP4 + FP16 embed & head | **13.8845 (+3.0%)** | **0.5129** | **32/35** | **7/9** |
  | NF4 (embed bf16) | 13.8519 (+2.8%) | 0.5110 | 33/35 | 7/9 |

  At parity with NF4 once the edges are high precision, and near-tie flips go in
  both directions (NF4 loses DET-01, which NVFP4 keeps). A single prompt ranks
  nothing.

So the honest statement is: **the 4-bit body is at parity with the standard
4-bit recipe; the remaining failure is the arithmetic boundary, reached by a
source margin of 1.1 logits.**

### 6.2 How much precision is needed

* **Per-layer ablation** (each layer's tensors replaced by the source, one at a
  time): every layer moves the margin by 0.1-0.5 logits in either direction and
  none recovers it — the error is accumulated over all 40 layers, so there is no
  cheap high-precision subset.
* **FP16 edges only**: still `CR` (−0.48).
* **FP8 E4M3 body (per-tensor) + FP16 edges**: `C` by **+1.59** → passes, but
  needs a new 8-bit runtime kernel.
* **NVFP4 body + a second NVFP4 plane carrying the residual + FP16 edges**:
  `C` by **+0.81** → passes, and reuses everything that already exists.

### 6.3 What shipped

* `quantize_residual_tensor_gpu` (`safetensors_gpu.{h,cu}`): converts the source
  to f32, subtracts the dequantized base plane on device, and quantizes the
  residual through the ordinary codec. The converter stores it as
  `<name>.res` in the same `layer_L_nvfp4.arrow` batch (same schema, no new
  format), and the index records `"residual": true`.
* Loader: four optional residual weights per layer (`qkv_res`, `o_res`,
  `mlp_gate_up_res`, `mlp_down_res`); a legacy tree without them loads exactly
  as before.
* Decode: one extra `RunLsxNvfp4Gemm(..., acc=true)` per projection.
* Edges: `embed_tokens` and `lm_head` are FP16 records in `global.arrow`
  (`"edges": "fp16"` in the index). The embedding is used in place as the gather
  table, which also removes a 1 GiB dequant from **every** `Infer` call; the head
  runs through the neutral device-independent `RunLxGemmVec`.
* `VramTensorPolicy::has_residual` so the conversion-time VRAM profile counts the
  second plane (it read 9 GiB while the model needed 16.6 GiB).

Measured on the real model: **all four parity gates green** (Paris ×3 bit-exact,
FSI 3/3, CODE 3/3, DET 3/3 — DET-02 now answers `CATASTROPHIC`), 16.6 GiB
resident, 17 GiB on disk, 41 ms TTFT and **71 t/s** at 2048 tokens (from 133
t/s for the 4-bit-only tree: the residual doubles the weight read, which is the
whole decode).

Perf rows re-seeded by the harness with its own evidence: `perf` 130.91 ->
**71.06** (ttft 42.91 ms, 584.5 W avg), `r3` 130.31 -> **71.16** (ttft 167.87 ms).
The four-shape/two-plane trade is deliberate: the operator's requirement is a
model that passes every golden.

TDD trail: the converter case "phi residual plane halves the 4-bit
reconstruction error" (asserts the `.res` record and a >=2x lower reconstruction
error, host-decoded with the canonical codec), the loader case (both planes
uploaded per projection), and the pre-existing synthetic model/determinism case
now exercises the `acc=true` path.
