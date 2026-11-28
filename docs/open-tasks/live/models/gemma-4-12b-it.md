# gemma-4-12b-it — stage-0 triage and ingestion plan

Source: `google/gemma-4-12B-it` (single `model.safetensors`, 23.9 GB BF16, 677
tensors, `architectures: ["Gemma4UnifiedForConditionalGeneration"]`,
`model_type: gemma4_unified`). Branch `feat/gemma-4-12b-it`, 2026-11-28.

## Stage 0 — triage

### 0.1 Classification

**Sibling variant of the existing `gemma-4-31b-it` dense member**, not a new
architecture. The text tower is tensor-for-tensor the same layout as the 31B
(`model.language_model.layers.N.{input_layernorm, post_attention_layernorm,
pre_ffeedforward_layernorm, post_feedforward_layernorm, layer_scalar,
self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm, mlp.{gate,up,down}_proj}`,
`model.language_model.{embed_tokens,norm}`), including the
`attention_k_eq_v` tie (a `full_attention` layer ships no `v_proj`).

It is NOT a sibling *reuse of geometry*: every width/depth/head field differs,
so it owns its own frozen passport (Model Ingestion Protocol step 3) and its
own `kType`. Following the `granite-switch-4.1` precedent (one converter, one
kernel set, one lifecycle body; the frozen numbers passed as data), the shared
`gemma_dense` runtime serves both checkpoints and each class validates the
converted tree against its own passport.

### 0.2 Geometry (from `config.json` `text_config`)

| field | 12B | 31B (sibling) |
|---|---|---|
| hidden_size | 3840 | 5376 |
| num_hidden_layers | 48 | 60 |
| num_attention_heads | 16 | 32 |
| num_key_value_heads (sliding) | 8 | 16 |
| num_global_key_value_heads | 1 | 4 |
| head_dim (sliding) | 256 | 256 |
| global_head_dim | 512 | 512 |
| intermediate_size | 15360 | 21504 |
| vocab_size | 262144 | 262144 |
| sliding_window | 1024 | 1024 |
| layer pattern | 5 sliding + 1 full, (l+1)%6==0 -> 8 full / 40 sliding | same, 10/50 |
| rope sliding / full theta | 1e4 / 1e6 | same |
| partial_rotary_factor (full) | 0.25 -> 64 rotary angles of 512 | same |
| final_logit_softcapping | 30.0 | same |
| tie_word_embeddings | true (no `lm_head` tensor) | same |
| rms_norm_eps | 1e-6 | same |
| embed scale | bf16(sqrt(3840)) = **62.0** | bf16(sqrt(5376)) = 73.5 |

`attention_k_eq_v: true`, `enable_moe_block: false`, `num_experts: null` ->
dense. `hidden_size_per_layer_input: 0`. `use_bidirectional_attention:
"vision"` applies to the stripped vision tower only.

### 0.3 Tokenizer

Raw-text BPE (`tokenizer.json` vocab+merges, U+2581 space normalization, byte
fallback) — the existing gemma-4 kind 11 / pretokenizer 9 bridge, unchanged.
`bos` 2, `pad` 0, `eos` {1, 106} from `config.json`; `generation_config.json`
adds 50 -> the merged eos set {1, 106, 50}. `tie_word_embeddings: true`.
`chat_template.jinja` is **byte-identical to the 31B's** (verified), so the
stage-4b tool bridge is the existing `gemma_tool_format` codec, no new codec.

### 0.4 Ballast to strip (stored and counted, never applied)

`model.vision_embedder.*` (9 tensors: patch_dense, patch_ln1/2, pos_embedding,
pos_norm), `model.embed_vision.embedding_projection.weight` (3840x3840),
`model.embed_audio.embedding_projection.weight` (3840x640) — 11 tensors total.
The `audio_config` block describes a tower that this checkpoint does not ship;
the audio path is a single 640->3840 projection. The 31B classifier only knows
`model.vision_tower.*` / `model.embed_vision.*`, so the `vision_embedder` and
`embed_audio` names must be added to it (a leftover name errors, never skips).

### 0.5 Feasibility (RTX 5090, 32 GB, 768 GB host)

NVFP4 dense: ~5.9 GB of projection/MLP nibbles + per-block scales, plus the
tied embed/lm_head table at NVFP4 (262144x3840/2 = 503 MB + 32 MB scales)
=> ~7.0 GiB resident. An 8-bit (base + residual) upgrade of every linear is
~13.5 GiB — i.e. the residual-plane escape hatch that the 31B could not fit
(33.7 GiB) **does fit here**. KV at 8k context: sliding 8x256x2x2 B = 8 KB/token
x40 layers, global 512x2x2 B = 2 KB/token x8 layers => ~0.36 MB/token,
~2.9 GB at 8192.

### 0.6 Semantics to verify before trusting output

- attention scale 1.0 (q/k per-head RMS-norm, no 1/sqrt(d)).
- per-layer-type RoPE theta (1e4 sliding / 1e6 full) and the *proportional*
  partial rotary on the full layers (64 of 256 pairs).
- K tied to V on the full layers (no `v_proj`), plain (unweighted) v-norm.
- `layer_scalar` is BF16 [1] here, not F32 (the dtype-guard trap from
  2026-08-24).
- tied embed/lm_head with the bf16-rounded `sqrt(hidden)` embed scale.
- softcapped logits (tanh, cap 30).

### Stage-0 gate (registry rows this work extends)

`models/google/gemma-4-12B-it.json` (new), `tests/test_model_suite_guard.cpp`
tool verdict table, `tests/test_perf_baseline.cpp` + `tests/perf/perf_baselines.json`,
`docs/MODELS.md` (generated), `docs/SELFCHECK_MATRIX.md` row.

## Plan (TDD, in lifecycle order)

1. **Converter**: extend the ballast classifier for `vision_embedder` /
   `embed_audio`; stop hardcoding the model name in `model.index.json`.
   RED: `tests/test_gemma_dense_12b_convert.cpp` on a synthetic *unified*
   checkpoint (audio + vision_embedder tensors) — today it fails with
   "unclassified tensor(s)".
2. **Passport**: `gemma_dense_12b_constants.h` (`lsxcommon::gemma_dense_12b::
   frozen`) + a `GemmaDenseFrozen` data struct; refactor the 31B model TU into
   a shared `GemmaDenseModel` base (granite-switch shape) with two thin
   concrete classes. RED: a test that the 12B class rejects the 31B tree and
   the 31B class rejects the 12B tree.
3. **Loader/kernels**: the embed scale and attention scale become data filled
   from the topology, so the shared `.cu` is passport-free and the 12B cannot
   silently decode with 73.5.
4. **Manifest + generated targets**, `--strict` and `--docs` clean.
5. **Real weights**: `lsxforge --gemma4dense --input .../gemma-4-12B-it
   --codec-profile nvfp4` into `LSX_ARTIFACT_ROOT/google/gemma-4-12B-it/nvfp4`.
6. **Gates**: SelfCheck (Paris, >= 3 bit-exact cycles), FSI / CODE / DET
   goldens captured from the source checkpoint, tool round-trip.
7. **Perf** row at the canonical 2048-token protocol, through the harness.
8. **Report**: `docs/model-reports/2026-11-28-Gemma-4-12B-IT.md`.

## R22. FSI-02 red — the SOURCE's own fork is a 0.556-logit coin flip

Measured 2026-11-28 on the real 12B tree. FSI-02 asks for a symbolic formula; the
engine emits the same structure with the given values substituted
(`((((100000000*1.083541)-0.0005)*83.152402)-0.0005)*0.011104-0.0005` instead of
`((((100000000*Rate_A)-Slip_A)*Rate_B)-Slip_B)*Rate_C-Slip_C`).

Teacher-forced on the SOURCE (bf16, transformers 5.15.1) at the fork
`((((100000000*`: `1` = 21.61 vs `Rate` = 21.05, i.e. **0.556 logits**. The
source's greedy run happens to take `Rate`; an 8-bit body's accumulated error is
larger than that margin, so the gate is reading a coin flip.

Not a kernel defect, and the two checks that would expose one are green:

- host-side decode of the written records vs the source bytes: relL2 0.0926 /
  cos 0.9964 for 4-bit, 0.0049 with the residual plane (the expected pair);
- the logits dot applies the table's per-tensor `global_scale` — pinned by
  `tests/test_gemma_dense_kernels.cpp` "logits apply the table's global_scale
  (gs != 1)", which a `gs = 1` fixture cannot see (gemma-4-26b shipped 7.01x-hot
  logits that way).

**Levers, both measured:**

| variant | artifact | FSI |
|---|---|---|
| 4-bit only | 6.3 GB | 1/3 |
| + residual on attn+MLP (**shipped**) | 12.0 GB | **2/3** |
| + residual on the tied head as well | 13.0 GB | 2/3 (byte-identical → reverted) |

**Next step, if this gate must go green:** the remaining error is not in the
body. Either (a) the goldens for this row are re-captured at a temperature that
makes the answer stable (a golden is source parity, so this needs the operator's
call and must be recorded as a protocol change, not a rewrite), or (b) the family
needs >8 bits, which the 31B sibling documents as unreachable in 32 GB — for the
12B it would be ~25 GiB of weights, i.e. it fits, but it halves throughput again.
Both are decisions, not fixes.

## Residual plane: declared per model (`convert_residual_plane`)

One converter serves both dense Gemma-4 sizes, so the plane cannot be a converter
constant: its cost is the size of the projection set (+5.7 GB here -> 12.0 GB,
which fits the card; ~+17 GB on the 31B -> 33.7 GiB, which does not). This row
declares `"convert_residual_plane": true`; `models/google/gemma-4-31B-it.json`
declares `false`. `lsxforge` reads the value through `ModelIdentityFor` into
`GemmaDenseConvertOptions`, and the converter RECORDS what it wrote in
`model.index.json` (`"residual": "attn+mlp"` / `"none"`), so a tree cannot claim a
plane it does not hold.

Pinned by `tests/test_gemma_dense_convert.cpp` (a `residual_plane=false` tree has
no `.res` records, an index saying `"none"`, and still passes the artifact scan)
and by `tests/test_gemma_dense_12b_model.cpp` "the residual plane is declared per
model, not per converter" (12B true / 31B false, both directions).

## Perf: 59.85 t/s auto vs 48.77 bit-exact, attributed

12 830 211 776 B resident, 2048-token protocol, TTFT 283 ms, ~757 GB/s
effective, 440 W avg. The bit-exact (`kOff`) route reads 18% slower for a
measured reason, not noise: the shared dispatcher's `gemv_large` floor is
`K >= 4096 && N >= 4096`, and this model's widest rows are 3840 wide — so
`q`/`k`/`v`/`gate`/`up`/`down`/head all miss it, and in the bit-exact route the
per-tensor `gemv_small` 32 MiB bound excludes the 540 MB head, which then falls
to the scalar tile kernel. The 31B (5376 wide) clears the floor and shows
identical numbers in both modes (47.62 / 47.63). Raising the floor is a change
to a model-neutral dispatcher shared by ~45 models — recorded here as a lever,
not taken: it needs its own A/B across families.

## Status log

- 2026-11-28 stage 0 complete; branch `feat/gemma-4-12b-it`.
- 2026-11-28 stages 1-8 + 9: converter/loader/kernels/model/passport, manifest,
  real-weight gates, perf seeded, report written. FSI 2/3 (R22); everything else
  green.
- 2026-11-28 **MCP payload gate enabled and PASSED 3/3** (byte-exact to the
  protocol template): stage + `kind: mcp` override in the manifest,
  `lsx_add_parity_gate(... KIND mcp)`, golden captured tokenizer-only by
  `tools/mcp_capture.py`.
- 2026-11-28 **MCP became a required lifecycle stage** (operator directive), not
  a pilot: `mcp` moved into `test_model_suite_guard`'s `kRequired` and out of
  `OPTIONAL_STAGES` for all 48 rows; 39 goldens captured, 9 rows declare
  `golden_debt.mcp` (empty source-store placeholders). This row needed no change
  for that — it was already the third row wired. See
  [`../../2026-11-28-mcp-gate-rollout-closed.md`](../../2026-11-28-mcp-gate-rollout-closed.md).
- **Not run, deliberately:** the `convert`/`inference` label sweeps. This is a
  model-local change verified with `ctest -R gemma_dense -j 1` (13/13) plus
  `ctest --preset linux-debug -j 16` for the guards.

## Ingestion log

### Converter (stage 1/2) — 2026-11-28

Two defects found and fixed while converting the real checkpoint, each pinned by
a RED test first:

1. **`gemma4_unified` ballast was unclassified.** The checkpoint ships
   `model.vision_embedder.*` (9 tensors: patch projection + position table, no
   transformer) and one `model.embed_audio.embedding_projection` (the
   `audio_config` describes a tower this checkpoint does not contain), which the
   26B/31B classifier did not know. A strict-coverage converter therefore
   refused the whole checkpoint ("unclassified tensor(s)"). Both namings are now
   claimed as ballast, stored-and-counted.
2. **The converted tree failed its own artifact scan** (verifier rule D1): the
   NVFP4 triple was written as dtype `U8`, so `lsxforge --codec-profile nvfp4`
   quantised 6.3 GB on the GPU and then threw the tree away with "the converted
   tree is not a valid artifact (nothing was published)". The triple now names
   its codec (`NVFP4` nibbles + `F8_E4M3` block scales + `F32` global scale).

Also fixed in passing (pre-existing, not model work): `test_ctest_labels_guard`
linked only Catch2 and ODR-used absl's inline `StrJoin` failure path, so it
LINKED under g++ (`linux-debug`) and failed to link under clang++
(`linux-release-znver4-aocc`) — the guard had never run on the Release tree.
It now links `absl::strings absl::log_internal_log_sink_set`.
