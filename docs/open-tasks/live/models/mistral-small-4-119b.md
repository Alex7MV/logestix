# Mistral-Small-4-119B — open items

### T. Mistral-Small-4-119B-2603-NVFP4 ingestion — later stages (2026-10-11)

#### T.1 Re-ingestion and full lifecycle verification (2026-11-06)

Branch `feat/mistral-small-4-reingest`. The converted vault tree had been lost
(`<model>/logestix/` was absent while 24 other families still had theirs), so
every real-weight claim about this family was unverifiable. Re-ingested from the
intact 66 GB source and ran the whole lifecycle.

- **Re-converted** (`lsxforge --mistral`, 76 s): `logestix-mistral-nvfp4`,
  36 layers × 128 experts = 4608 expert arrows, 222 vision tensors stripped AND
  counted, tokenizer arrows written (131072 vocab / 269443 merges).
- **Defect found and fixed: the index carried no `vram` block.** Every other
  family writes one, and an absent block does not fail - `ParseVramBlock`
  returns false and the consumers (the placement plan via
  `IModel::VramRequirements`, `lsxcli --info`) silently saw an all-zero
  profile, which is indistinguishable from "this model needs no VRAM". The
  family never overrode `VramRequirements` at all, so even a tree with a block
  would have reported nothing. Fixed in `mistral_moe_convert`/`_model` with
  tests written first; re-converted tree now reports min 4 GiB / optimal 50 GiB
  / max 111 GiB at ctx 1048576. Two conversions of detail are load-bearing: the
  inventory is the tensors the *artifact* holds (stripped ballast and the
  activation-only `input_global_scale` companions excluded), and `is_nvfp4`
  stays null because the source is already NVFP4 - re-deriving a 4-bit size
  would halve every tensor a second time.
- **Gates, all on real weights** (AOCC Release tree, 2026-11-06):
  `test_mistral_moe_selfcheck` PASS, Paris ×3 with byte-identical
  full-vocabulary logits (`logits_hash cdfd7309bb141772`, top-1 6993);
  native tool round-trip green (`get_weather{"city": "Paris"}` → parsed
  `get_weather {"city": "Paris"}`); `lsxfabric --validate` loads in 1053 ms
  (re-measured 2026-11-07: 1161 ms — ±10% run to run, so read it as "about a
  second") with 4 715 087 872 B resident and 1398/4608 expert slots — both
  re-measured identically; convert/loader/model/kernels/tool_framing all green
  (`_model` is a synthetic fixture by design, so it pins wiring, not the
  checkpoint); fast suite 138/138 (written as 137/137 here until 2026-11-07 —
  `744f2c4c` added a guard target and updated the report's verdict but not this
  line. The count is stable across the migration: the pre-migration ctest oracle
  has 140 `fast` targets and today's tree has 140, with one swap — the naming
  guard out, `test_manifest_generator_selftest` in).
- **Perf re-seeded through the harness**: perf **45.77 t/s** (was 45.04),
  r3 **49.82 t/s** (was 41.57), TTFT 525/509 ms, DRAM probe 1532.73 GB/s
  recorded in the same run. `r3` moving up 20% is the strict path now
  benefiting from the same tree/commit; it is a measured row, not a claim.
- **The three gates were RUN on 2026-11-06 with template-derived goldens, and
  measured.** They do not pass, and the measurement is now a fact rather than a
  SKIP. Capture used `tools/{fsi,code,det}_capture.py --template-*` with the
  engine-exact frame (`BOS + [INST]…[/INST]`, no vendor system turn):
  **FSI 1/3, CODE 0/3, DET 0/3** (DET `deterministic=true tokens_ok=true` on all
  three prompts — three byte-identical runs each). The model is coherent and
  on-topic but off-format: FSI-03 returns exactly `Drawdown>Threshold`, FSI-01
  answers `PV=CF*(1-exp(-r*n))` (the annuity-due form, missing the geometric
  denominator), FSI-02 opens with prose. CODE opens every answer with an
  intro line ("Here's a valid JSON schema…") and CODE-03 degenerates into a
  repeated phrase. DET-02 reproduces the behaviour recorded for the ORIGINAL
  ingestion — it reasons *to* the answer ("The word that starts with the
  character 'C' and ends with the character 'C'") without emitting the token —
  which is a reproducibility signal across both conversions.
  **The goldens are still NOT committed**, for the reason the earlier decision
  gave: with a template-derived (not source-captured) expected value they assert
  nothing about parity that the Paris SelfCheck does not already cover, and
  three permanently-red targets train people to ignore red. The rows stay in
  `kGoldenDebt`; what changed is that the debt is now measured.
- **Two capture-tool defects were found and fixed while running them.**
  `tools/fsi_capture.py` had no `chat_template` override for this model (code
  and det had it) AND never applied `opts["chat_template"]` at all — the first
  FSI run therefore used the checkpoint's own template, which injects
  `default_system_message` when no system turn is present, and all three prompts
  answered "It seems like you're asking about a specific dates, you can use
  tools to fetch…" (0/3, a pure framing artefact). With the override applied and
  honoured the same three prompts answer on-topic, so the 0/3 was the harness,
  not the model.
- **Why no source-faithful golden: the reference stack cannot consume this
  checkpoint.** When the debt was declared the source was "unrunnable"
  (`params.json`, no `config.json`, no `model_type`). That part is now stale —
  `transformers` 5.15.1 ships `Mistral4ForCausalLM`, its defaults match this
  checkpoint exactly, and a synthesized `config.json`
  (`tools/mistral_small4_oracle.py::mistral_small4_config_dict`) makes
  `AutoConfig` construct and `from_pretrained` read all 13 shards. Two rope keys
  are load-bearing in that synthesis and each fails differently: without
  `partial_rotary_factor` the rotary embedding is built at the full head width
  (128) against a 64-wide `q_rot` (`size of tensor a (32) must match b (64)`),
  and without `llama_4_scaling_beta` the Llama-4 logit temperature multiplies
  `None` (`unsupported operand type(s) for *: 'NoneType' and 'Tensor'`).
  **The real blocker is the tensor naming and expert layout, not the config.**
  The consolidated checkpoint names are `layers.N.attention.wq_a`,
  `layers.N.attention_norm`, `tok_embeddings`, `output`, `norm` and **one index
  per expert** (`layers.N.experts.E.w{1,2,3}`), while the class expects
  `model.layers.N.self_attn.q_a_proj`, `model.layers.N.input_layernorm`,
  `model.embed_tokens`, `lm_head`, `model.norm` and the **fused stacked**
  `model.layers.N.mlp.experts.{gate_up_proj,down_proj}`. A plain
  `from_pretrained` therefore reports **50 missing / 3118 unexpected** keys and
  silently runs a randomly-initialized 119B model; its forward pass then emits
  NaN logits (first NaN at `model.layers.0.mlp.shared_experts.down_proj` with a
  zero input — garbage weights, not a numeric defect). `load()` in the new tool
  REFUSES that model (`allow_missing=False` by default) precisely so nobody
  captures a golden from random weights. Closing the debt needs a
  name+repack mapping (the vendor `mistral-inference` loader is the reference
  implementation, and the expert unfusing is the hard half), or a
  de-consolidated bf16 HF export, at which point the capture scripts need no
  change beyond `loader: "mistral_small4"` and the deletion of the
  `unrunnable` override in `tools/{fsi,code,det}_capture.py`.
- **The checkpoint's own data was cleared while investigating that.** A host
  dequant of `layers.0.shared_experts.w2` gives min/max ±0.277 with
  `weight_global_scale` as a DIVISOR (independently confirming the converter
  rule fixed on 2026-10-12), and all 202 899 456 E4M3 block-scale elements of
  layer 0 contain **zero** NaN. So the NaN was the reference stack, never the
  weights — which is a positive statement about the converted tree.

- **Branch** `feat/mistral-small-4-119b`; design
  `docs/superpowers/specs/2026-10-11-mistral-small-4-119b-ingestion-design.md`.
- **Landed (verified, green):** stages 0–5 + 4b. Stage 1 converter
  (`mistral_moe_convert.{h,cpp}`, `lsxforge --mistral`, synthetic `convert`
  test); stage 3 loader (`mistral_moe_load.{h,cpp}`, synthetic loader test);
  stage 4b tool codec (`mistral_moe_tool_format.{h,cpp}`, framing test); stage 5
  MLA+MoE decode (`mistral_moe_inference.{h,cu}` + `mistral_moe_model.{h,cpp}`,
  registered `mistral-small-4-119b`), kernels + model tests green. Frozen
  passport `mistral_moe_constants.h`. Four guard registries extended
  (`test_model_independence_guard` `kPassports`/`consumers`/`kLifecycle`,
  `test_model_suite_guard` `Suites()` + tool verdict + golden debt,
  `test_perf_baseline_logic` mid list, `test_code_naming_guard` family token).
  Fast suite 93/93; `test_mistral_*` (convert/loader/tool/kernels/model) green.
- **Real-weight conversion: done.** `lsxforge --mistral` converted the vault
  tree in place (`<model>/logestix/`, 66 GiB, 4608 experts, 222 vision tensors
  stripped, tokenizer arrows written) in ~75 s. The loader then loads it
  (70 s: dense BF16 graph 4.7 GiB + 56 GiB pinned expert store). Two real bugs
  were fixed on the way:
  1. `weight_global_scale` is a **divisor** (`w = nibble*scale/gs`), not the
     repo's multiply — the converter now stores `1/weight_global_scale`.
  2. A NaN logit on finite real inputs first looked like the neutral
     `RunMoERouterTopK` seq==1 dots path. It was not: the NaN was that FP16 SwiGLU overflow
     arriving in the *next* layer's `d_post`, which is the router's input. With that fixed the neutral device router runs this family
     at its real geometry (E=128/H=4096/top-4) in both numeric modes and the
     host workaround is gone. The investigation did surface a genuine coverage
     gap — the seq==1 decode router had no reference test at all — now pinned
     across six geometries in `test_qwen_moe_kernels`. Details:
     [`open-tasks/2026-10-12-mistral-small-4-decode-defects.md`](open-tasks/2026-10-12-mistral-small-4-decode-defects.md).
- **The decode defects found on real weights, five of them plus three
  follow-ups (2026-10-12) — archived.** Each was isolated against a host NVFP4
  reference built from the source safetensors, and each is pinned by a
  mutation-checked regression case. The headline: the shared expert was run
  through the BF16 GEMV (44x under-contribution, `rms 0.0080` vs `0.3495`), the
  real NaN was an FP16 SwiGLU overflow at layer 32 (`|silu(g)*u| = 2.5e5`
  against the 65504 ceiling), the residual itself reaches ~2.4e5 so `d_h` is
  FP32, `~MistralSession` leaked the whole dense graph per request, and the
  converter's `"model"` field normalised to a name the registry could not
  resolve by path. Two attributions were wrong before they were right: the
  "neutral router NaN" was that FP16 overflow downstream, not the router (the
  investigation did expose a real coverage gap — the seq==1 decode router had no
  reference test at all — now pinned across six geometries), and the layer-4
  "outlier" was a genuine massive-activation expert, confirmed against the host.
  Full trail:
  [`open-tasks/2026-10-12-mistral-small-4-decode-defects.md`](open-tasks/2026-10-12-mistral-small-4-decode-defects.md).
- **Still open — the three FSI/CODE/DET golden gates, with the capture now
  understood.** (The old heading said "the four S* golden gates"; there is no
  S1..S4 scheme anywhere in this file, the body and
  `test_model_suite_guard.cpp` both enumerate exactly these three, and the body
  already said "three permanently-red gates" -- the count in the heading was
  the only stale part.)
  Capture tooling improved first: `tools/fsi_capture.py` gained `--template-only`
  (the source is unrunnable), and `det`/`code` gained a `chat_template` override
  in `OVERRIDES`. Findings:
  1. **The source cannot run.** `AutoConfig.from_pretrained` fails — the
     checkpoint has `params.json`, no `config.json`, no `model_type`, and
     NVFP4-packed experts. So a template-derived golden is the only option, and
     it cannot establish *parity*; it can only assert the converted path's own
     answers.
  2. **The vendor default system prompt suppresses every answer gate.** It
     instructs "when the user's request requires up-to-date or specific data, you
     must use the available tools". With it, all nine prompts answer "It seems
     like you're asking about a specific date, you can use tools to fetch ...".
     The model is *complying*; the engine's own `RenderText` does not inject
     that turn, so capturing with a `chat_template` override (BOS + `[INST]`...,
     no system turn) is what matches what we run.
  3. **Re-framed, the converted model is coherent but off-format, and the two
     numeric modes disagree on these prompts.** DET determinism PASSES
     (`runs=3 deterministic=true`, byte-identical) and answers are on-topic —
     DET-02 correctly reasons "The word that starts with the character 'C' and
     ends with the character 'C'" toward `CATASTROPHIC` — but no prompt emits the
     canonical token/formula, several collapse into repetition loops, and the
     auto and bit-exact modes diverge (auto returns `.`, bit-exact returns a
     full sentence) on DET-01.
  **Decision: the goldens are NOT committed.** Shipping three permanently-red
  gates trains people to ignore red, and with the source unrunnable a
  template-derived golden asserts nothing about parity that the Paris SelfCheck
  does not already cover more cheaply. FSI/CODE/DET remain declared debt in
  `test_model_suite_guard`.

- **Unblocked 2026-10-23: the operator will supply a runnable source.** The
  decision "template-derived or nothing" is retired; the goldens are to be
  captured from a model that actually runs. Two artefact forms are acceptable,
  and the amount of code each needs is known:

  1. **An HF-format checkpoint** (`config.json` with `model_type`, bf16
     safetensors, tokenizer) - e.g. a de-consolidated bf16 export. **Zero code
     changes**: `tools/{fsi,code,det}_capture.py` already fall through to
     `AutoModelForCausalLM`, and the `mistral-small-4-119b` override exists only
     to mark the *consolidated* layout as unrunnable.
  2. **The consolidated NVFP4 checkpoint plus the vendor stack**
     (`mistral-inference` + `mistral_common`, neither installed here). Needs a
     `loader: "mistral"` branch in each capture script - the `qwen4_exp` and
     `qwen3_5` branches are the precedent (load, `eval()`, expose
     `device`/`generate`), roughly 15 lines each. The golden's `input_ids` must
     still come from this repo's tokenizer arrows (`tokenizer.json` +
     `chat_template.jinja` are present in the vault), so a vendor tokenizer
     difference cannot silently move the prompt.
  3. **Not acceptable: the converted model's own answers.** That is
     self-certification (`docs/OPEN_ENDEDNESS.md`) and the reason this gate was
     left SKIPping rather than shipped red.

  Acceptance, all three stages: the golden's `source` field names a real model
  run (not "prompt-template"), the `"unrunnable"` override is deleted from
  `tools/fsi_capture.py` in the same commit, and
  `test_mistral_moe_{fsi,code,det}_selfcheck` run (not SKIP) - red or green is
  then a real finding either way. `code_capture.py`/`det_capture.py` carry the
  same override block, so the deletion is three one-line edits.
