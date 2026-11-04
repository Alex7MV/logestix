# Qwen3.8-Flash-Next — full lifecycle re-verification, two perf-harness defects (2026-11-04)

Session branch `feat/qwen38-flash-next-lifecycle`. The family already existed
(`qwen_moe_exp_*`, `logestix-qwen38flashnext`); this session re-ran **every**
Model Ingestion Protocol stage against the real checkpoint, fixed what the run
exposed, and re-seeded the perf rows. Nothing here is a re-conversion of the
architecture — the geometry is unchanged.

## 0. Stage 0 triage facts (measured from the checkpoint)

| | |
|---|---|
| Source | `Qwen/Qwen3.8-Flash-Next`, `Qwen4ExpForConditionalGeneration`, text `qwen4_exp_text`, 131 BF16 shards, 336 GB of tensor data (499 GB directory) |
| Text geometry | 48 layers = 12 × [3 GatedDeltaNet + 1 QSA full attention]; 2560H; 24Q / 2KV / 256HD; partial RoPE on 64 dims, θ 1e7; GDN 16K·48V·128HD conv-4; QSA indexer 4 × 128, block 4, budget 2048, top-512; hyper-connections (4 streams, lowrank 320) |
| MoE | 512 routed experts top-10, I = 640, one shared expert I = 640 |
| PLE | n-gram embedding on `ple_layer_ids = [2]` (HF `layer_idx` 1), 3-gram, 8 heads/gram → 16 heads × 160 dims, 128 shards |
| Edges | vocab 248 320; eos 248 046 / 248 044; 1 MTP layer (`mtp.hybrid`, never applied) |
| Ballast stripped + counted | 333 `model.visual.*` tensors (vision tower), `mtp.*` stored |
| Feasibility | NVFP4 dense ~3.5 GiB pinned + 64 GB routed-expert arena on host + 96 GiB PLE table mmap'd; does not fit device-resident on a 32 GB card |

Checked and recorded as *not* applicable to the text path: the config carries
`rope_parameters.mrope_interleaved = true` with `mrope_section [11,11,10]` —
that is the multimodal M-RoPE branch. Folding it into the text path would
corrupt positions; the text decoder uses plain 64-dim partial rotary.

## 1. Stage 4b — the 3.8 chat template differs from 3.5/3.6 (corrected below)

The shared `qwen_tool_format.{h,cpp}` codec was written against the Qwen3.5/3.6
`chat_template.jinja`. Rendering the **3.8** template (transformers 5.15.1,
`apply_chat_template(..., enable_thinking=False)`) for the same conversation
showed three deltas from what the engine emitted:

1. **Every assistant history turn carries the closed think block**
   (`assistant\n<think>\n{reasoning}\n</think>\n\n{content}`), including when
   thinking is disabled and the reasoning trace is empty. The codec rendered
   `assistant\n{content}` — a shape the checkpoint never saw in training.
2. **Tool definitions serialise like jinja `tojson`**
   (`json.dumps(obj, ensure_ascii=False)`: `", "` / `": "` separators), not the
   compact form.
3. **Message bodies are trimmed** — the template renders every message through
   `render_content(...)|trim` (lines 70/78/92/103).

Fix as shipped here: a `QwenRenderPolicy` struct (`enable_thinking`,
`history_think_block`, `jinja_tool_json`, `trim_content`) with the
3.5/3.6-reproducing defaults, so the older members keep their exact wire;
`Qwen38FlashNextModel` renders with the exported `kQwen38RenderPolicy`, which
switches on all three. Pinned TDD-first by two new cases in
`tests/test_qwen_tool_framing.cpp` ("qwen4_exp template fidelity: …"), whose
expected text **is** the template output rendered by transformers, not our
renderer (86 assertions / 11 cases), plus a case in
`tests/test_qwen_moe_exp_model.cpp` that pins the whole policy the model renders
with (10 assertions / 3 cases) — the `trim_content` flag was covered by a
renderer test yet left off for the model, which is exactly the class of defect a
per-flag test cannot catch. Re-run after the change: SelfCheck
`logits_hash 2e3bcd090441ad65` unchanged and the tool round-trip green, i.e. the
9 gate prompts (3 + 3 + 3, none with surrounding whitespace) frame
byte-identically.

### Correction (2026-11-04, later the same day)

Deltas (2) and (3) are **not** 3.8-only, and this session's handling of them was
wrong. `diff` of `Qwen3.6-35B-A3B/chat_template.jinja` against
`Qwen3.8-Flash-Next/chat_template.jinja` shows the `|trim` sites and the
`{{- tool | tojson }}` line **identical**; a sweep of all six checkpoints finds 4
trim sites and 1 tojson site in each. 3.8's real deltas are `reasoning_effort`,
the `preserve_thinking` default and an empty-system-turn guard. Consequences:

- the shared codec had been emitting the compact tool JSON and the untrimmed
  content for **every** qwen member — the "older members keep their exact wire"
  plan would have frozen that deviation, not preserved fidelity;
- 3.8 applied `trim_content` at every `esc()` call site, so tool names and
  descriptions were trimmed too, while the oracle keeps
  `"  Get the weather  \n"` byte for byte (`tool | tojson`, no trim).

Resolved the same day on branch `fix/qwen-template-fidelity` (commit
`b3eaa066`): trim and tojson became unconditional (both flags deleted), trim was
scoped to message content, `suppress_empty_system` became the third genuine
per-family delta, four new oracle-pinned cases were added, and the two members
that had **no** real-weight tool round-trip (qwen3.5-9b, qwen3.5-35b-a3b) got
one. All six round-trips green on the new wire, every `logits_hash` unchanged,
`ctest -L fast` 134/134 — [`2026-11-04-qwen-template-fidelity.md`](2026-11-04-qwen-template-fidelity.md).

## 2. Stage 8 — parity gates on real weights (all green)

| gate | result |
|---|---|
| Convert (`lsxforge --qwen38flashnext`, in place) | OK — 6 min 26 s wall, 336 GB → 164 GB tree (64 G experts + 96 G PLE + 3.3 G pinned + 1.4 G MTP), 1067 pinned tensors, 333 visual skipped, 128 PLE shards, `deferred` MTP |
| `test_qwen_moe_exp_convert` | 158 assertions / 12 cases |
| `test_qwen_moe_exp_loader` | 94 assertions / 10 cases |
| `test_qwen_moe_exp_kernels` | 468 assertions / 13 cases |
| `test_qwen_moe_exp_model` | 10 assertions / 3 cases |
| `test_qwen_tool_framing` | 86 assertions / 11 cases |
| `test_qwen_moe_exp_selfcheck` | PASS — Paris ×3 bit-exact, `logits_hash 2e3bcd090441ad65`, top-1 11751, ~128 s |
| `test_qwen_moe_exp_fsi_selfcheck` | PASS 3/3 (83 s) |
| `test_qwen_moe_exp_code_selfcheck` | PASS 3/3 (78 s) |
| `test_qwen_moe_exp_det_selfcheck` | PASS 3/3 (94 s) |
| `test_qwen_moe_exp_eos_selfcheck` | PASS — EOS stop + cross-instance identity (193 s) |
| native tool round-trip | PASS — `<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>`, 26 tokens, parses to `{"city":"Paris"}` |
| guard suite | 9/9 (`code_naming`, `absl_format`, `absl_container`, `logging`, `expert_store`, `infrastructure`, `model_independence`, `ctest_labels`, `model_suite`) |
| `ctest -L fast` | 134/134 |

The converted tree stays on the **pinned arena** (`LoadQwen4ExpExpertStore`,
allowlisted in `test_expert_store_guard`) — the 2026-10-14 negative result
still holds: this family's decode working set is the whole expert set, so the
neutral mmap store with a bounded window measures 9.2 t/s against the arena's
32 t/s.

## 3. Perf — the harness was under-reporting this family, twice

The strict row was stale (26.47 t/s, 2026-10-23) and the fresh run read 42.97
against the 44.85 baseline. Two **instrument** defects were found by
attribution, both fixed test-first, and neither is a model regression.

### 3.1 The bandwidth probe leaked 1 GiB of VRAM for the process lifetime

`perf::MeasureMemoryBandwidthGbs` held its two 512 MiB device buffers in
`thread_local` statics. `test_perf_baseline` runs the probe **before** the
model in the same process, so from 2026-10-23 (when the probe landed) every
capacity-bound family measured with 1024 MiB less free VRAM than the card
actually had — fewer expert-cache slots, lower t/s, recorded as the reference.

- RED first: `TEST_CASE("memory-bandwidth probe releases its device buffers",
  "[gpu_perf]")` in `tests/test_gpu_perf.cpp` — 86 assertions / 15 cases after
  the fix; the probe now allocates and releases per call.
- Effect on this model: 4962 → **5350** slots (free 24 985 → 26 009 MiB).

### 3.2 The generation context was charged twice on the second load

`RunExpGeneration`'s arena + fp16 embed table are process-lifetime statics
(`ExpGenScratch`). The harness runs `perf` (auto) and `r3` (strict) as **two
`Load()`s in one process**, so the second load counted the first leg's
5581 MiB as free *and* subtracted the same reserve again: 4319 slots instead of
the ~5350 the card state allows.

- Fix: `SizeExpertCacheSlots(..., already_resident_bytes)` plus a new
  `GenContextResidentBytes()`; the loader logs
  `[qwen38-model] free=… resident … of gen context, N/M experts resident`.
- RED first: `TEST_CASE("cache slots sizing does not charge resident scratch
  twice", "[qwen-exp-loader]")` (compile-red: too many arguments), then green —
  `test_qwen_moe_exp_loader` 94 assertions / 10 cases.

### 3.3 Capacity is the model's throughput variable (measured, not assumed)

Forced-slot sweep on the real weights (`LSX_QWEN38_EXPERT_CACHE_SLOTS`), one
run each, card idle between runs:

| slots | t/s | hit rate |
|---|---|---|
| 4962 | 41.40 | 0.826 |
| 5350 | 43.00 | 0.862 |
| 5750 | 44.15 | 0.887 |
| 6000 | OOM (`gdn l26 lin20 kb0 mt2053: out of memory`) | — |
| 7000 | OOM | — |
| 9000 | `QSA indexer state alloc failed` | — |

The 2026-10-14 reference of 44.85 t/s needs ≈5960 slots, which this card state
cannot hold (6000 fails to allocate). So the residual is capacity, not code —
and the perf-baseline logic now says so instead of calling it a regression:
`PerfCause::kCapacity`, `AttributeDegrade(model_ratio, probe_ratio,
actual_slots, baseline_slots)`, `expert_cache_slots` on
`PerfBaselineEntry`/`MeasureResult`/`gpu::InferenceResult`/`GenResultView`, and
a `CAPACITY (not a regression)` WARN path in `CheckAgainstBaseline`. Pinned by
`test_perf_baseline_logic` (169 assertions / 19 cases), including "the tracked
`perf_baselines.json` is exactly what the harness writes" (byte round-trip
through the serializer, first-divergence message).

Ruled out by measurement, in this order: `LSX_MOE_TC_HOIST` (43.0 vs 42.9 —
neutral), a pageable zero-copy regression (none: the expert arena is pinned and
the PLE table is mmap'd, not copied), and environment (probe nominal,
1524/1531 GB/s). A `WaitEventChecked` spin-loop change (yield once per 1024
iterations) measured 43.0 t/s — no gain, reverted.

### 3.4 Re-seeded through the harness (never by hand)

Old rows deleted first so the harness measured instead of judging;
`ctest -R test_perf_baseline_qwen3_8_flash_next` (261.85 s), then a second
confirmation run (245.25 s) which upserted the small improvement:

| row | t/s | TTFT | power avg/peak | samples | slots |
|---|---|---|---|---|---|
| perf | **42.98** | 193.11 ms | 217.32 / 227.06 W | 2375 | 5350 |
| r3 | **33.88** | 198.00 ms | 211.39 / 223.69 W | 3013 | 4319 |

The r3 row is still ~21 % below perf because the second leg holds the first
leg's 5581 MiB of scratch, i.e. it runs with 1031 fewer slots. That is now
*visible* (the row carries `expert_cache_slots`) rather than mysterious.

## 4. What is left open

nsys over 64 decode steps (`--max-gen 64`) attributes the remaining step:

| component | per token |
|---|---|
| step window | 25.23 ms |
| kernels | 17.78 ms |
| H2D (expert miss staging) | 5.58 ms, 302.5 MB, 17 267 copies |
| kernels + copies | 23.36 ms of 25.23 ms → essentially **zero overlap** |

All 17 267 H2D copies ride **stream 13** (the compute stream); the driver's
dedicated `copy_stream` carries 0.2 MB over the entire trace. Cause: the family
stages residency for **all K = 10 experts before any FFN compute**
(`EnsureQwen4ExpExpertResident(..., stream, ...)` with no `h2d_stream`), so
there is no window to overlap; `moe::RunMoERoutedFFN` is then handed
`copy_stream`/`ev_uploads`/`ev_front` and finds every expert already resident.
Making the staging overlap needs the miss for expert *k+1* to be issued while
expert *k* computes (as the neutral `moe::EnsureMoERoutedFFN` path does) —
a real change to the driver's handshake, not a flag. Recorded as open work,
not attempted here: the ordering change touches the router readback that the
numeric gates depend on, so it needs its own TDD cycle and a hash re-check.

Kernel attribution for whoever picks it up (top of the 64-step window):
`hc_epilogue_kernel` 2.66 ms, `hc_down_gemv_kernel` 2.16, `hc_norm_kernel`
1.57, `ple_inject_kernel` 1.39, `TcGemmKernel` variants ~2.9 combined,
`moe_router_dots` 0.83, `exp_head_gemv` 0.75 — the hyper-connection block is
the single largest host of serial work, not attention.

## 5. Rows affected by the probe leak (self-healing)

Every perf row dated **2026-10-23 … 2026-11-03** was measured with the leaky
probe (49 model rows across 25 families, plus the probe's own row). Only
capacity-bound families are affected — a device-resident model
loses nothing from 1 GiB of unusable VRAM unless it fails to load — and the
effect is one-directional: the recorded reference is *low*, which can only make
a later run pass more easily. `UpsertPerfBaseline` writes on an increase, so
each row re-seeds itself the next time its target runs. No manual re-seed, and
no "regression" verdict from the old numbers, is warranted.

Affected families whose reference is a streamed-expert design: glm-5.3-flash
(12.79/12.27), kimi-k2.7-code (4.86/5.04), deepseek-v4-pro-0813 (0.77/0.79),
minimax-m3 (2.98/2.98), minimax-m3.1-flash (6.23/6.18), ling (2.77/2.41),
beam (1.61/1.60), step-5-preview (1.82/1.83), mistral-large-4-1t (1.04/1.03),
gigachat3.5 (11.32/11.10).
