# MiniMax-M3.1-Flash ingestion (2026-10-31)

Session record for the sibling ingestion of `MiniMaxAI/MiniMax-M3.1-Flash`
into the `minimax` family. Everything below was measured on the reference host
(RTX 5090 32 GB, 750 GB host RAM, CUDA 13.4), one GPU workload at a time.

## Stage 0 — triage

Source: `/mnt/fast_storage/lsx_model/MiniMaxAI/MiniMax-M3.1-Flash`, 236 GB in 48
safetensors shards (89 632 indexed tensors once the vision ballast is counted).

| item | M3 | M3.1-Flash |
|---|---|---|
| architecture | `MiniMaxM3SparseForCausalLM` | same |
| geometry | 60L / 6144H / 64Q·4KV·128HD / 128 experts top-4 / shared 3072 | identical |
| dense MLP prefix | layers 0-2 | identical (dense MLP, but...) |
| attention dispatch | dense attention on layers 0-2, MSA from 3 | **MSA on every layer** (`sparse_attention_freq` all ones) |
| RoPE | theta 5e6 | **theta 1e7** |
| MTP | declared but unpopulated | **absent** |
| source encoding | BF16 dense + BF16 experts | **MXFP8 dense (F8_E4M3 + per-32 E8M0 `weight_scale_inv`), ModelOpt NVFP4 experts** (`weight_packed` / `weight_scale` F8_E4M3 per 16 / `weight_global_scale` F32 divisor / `input_global_scale`) |
| tokenizer | Llama-3/o200k byte BPE, vocab 200064 | `tokenizer.json`/`vocab.json`/`merges.txt`/`added_tokens.json` **byte-identical** |
| chat template | thinking disabled, think-END generation prefix | **fixed adaptive** (no prefix), system prompt names M3.1 + `<model_id>8bb8149fe</model_id>`, `developer` blocks for mid-conversation system turns, consecutive tool results share one `]~b]tool` block |
| ballast | vision tower + projector (523 tensors) | same 523 counted; plus a separate `dspark/` (2.2 GB) draft head that is not in the root weight map (stored, unapplied) |

Feasibility: ~217 GiB routed experts streamed through the neutral mmap
`ExpertStore`; 10.5 GiB pinned graph; fits the 32 GB card with the slot LRU.

Verdict: **sibling variant**, not a new family. Geometry is shared, so the
emitted artifact layout is identical (same `logestix-minimax-nvfp4` tag) and
the parent loader/kernels/pipeline are reused; only the passport, the
converter encoding path and the chat-template policy are new.

## Stage 1-2 — converter

`lsxforge --minimax_flash` → `minimax::ConvertMinimaxFlashModel`, the shared
`ConvertMinimaxMoeModel(input, output, variant)` with the sibling identity.

- **MXFP8 dense → NVFP4**: the base `.weight` (F8_E4M3) and its
  `.weight_scale_inv` companion (U8 E8M0, one exponent per 32 columns) are
  consumed together by the existing `quantize_tensor_gpu(..., e8m0_scales, ...)`
  tile path. Verified against the real checkpoint: the dequantized q_proj has
  std 0.0305, matching the BF16 sibling's scale class.
- **Routed experts**: verbatim ModelOpt passthrough (packed nibbles, low first;
  per-16 E4M3 scales) with the compressed-tensors DIVISOR `weight_global_scale`
  stored as its reciprocal — the same rule Mistral's NVFP4 ingestion already
  pins. The block scales/sizes are validated (`[rows, cols/2]`,
  `[rows, cols/16]`).
- **Variant gate**: `ClassifyMinimaxVariant` (theta + first-sparse-layer)
  rejects an M3 checkpoint on the M3.1 entry point and vice versa. The index
  writes `"model": "MiniMax-M3.1-Flash"` so `detect_model_type` resolves
  `minimax-m3.1-flash`.
- Test: `test_minimax_moe_flash_convert` (label `convert`, 120 s timeout,
  synthetic MXFP8+NVFP4 checkpoint; asserts bytes-equal expert passthrough,
  reciprocal global scale, MXFP8 reconstruction within the 4-bit error band,
  indexer on layer 0, determinism).

Real conversion: 236 GB source → **227 GB** tree in ~6 minutes (experts are
read and rewritten verbatim; the dense side is dequantized/quantized on the
GPU). `skipped_visual_tensors = 523`, `total_experts = 7296`.

## Stage 3-8 — load, model, tool bridge, gates

- `LoadMinimaxTopology` now derives `first_moe_layer`/`first_sparse_layer` from
  the dispatch arrays; each registered member cross-checks them (plus the RoPE
  theta) against its own frozen passport — M3 (3, 5e6) and M3.1 (0, 1e7) can no
  longer be loaded with the wrong tree.
- `MiniMaxM31FlashModel` (`minimax-m3.1-flash`) reuses the parent pipeline and
  the neutral expert store; it owns only the passport and the render policy.
- Tool bridge: `MinimaxRenderPolicy` parameterizes the shared renderer;
  `MinimaxFlashRenderPolicy()` reproduces the M3.1 template (adaptive line,
  model-id suffix, no generation prefix, developer blocks, batched tool
  results). `test_minimax_moe_flash_tool_framing` pins every delta against the
  M3 policy.

Real-weight verdicts (serial runs):

- **SelfCheck PASS ×3**, bit-exact `logits_hash 2daad7a5ba024549`:
  `<mm:think>… answer: Paris.</mm:think>Paris.` (adaptive thinking puts the
  answer past token ~60, so the probe window is 192 tokens).
- **FSI 3/3, CODE 3/3, DET 3/3** (DET: 3 runs per prompt, byte-identical).
  Goldens are **template-derived** (`tools/*_capture.py --template-only` /
  `--template-golden`): the source is a 428B QAT checkpoint with no 32 GB-card
  execution path, so the oracle is the prompt's Expected Template framed with
  the checkpoint's own tokenizer/template. Recorded in each golden's `source`.
- **Tool round-trip** on real weights: `get_weather {"city":"Paris"}`.
- **Perf** (harness, 2048 tokens, AOCC release tree): **6.23 t/s (perf) /
  6.18 t/s (r3)**, TTFT 48.8 / 49.2 s, avg 152.2 / 152.0 W, peak 185.7 /
  179.8 W; 28 GiB-scale expert streaming dominates. (M3's rows are 2.98/2.98;
  the M3.1 source is quantization-aware-trained and the generated continuation
  differs, so the expert-cache hit rate — and with it the streamed-expert t/s —
  is a different measurement, not a code delta.)

## Verification run

- `ctest --preset linux-debug` → 122/122 fast tests green (guards included).
- `ctest --preset linux-debug-convert` → 34/34 green.
- The five real-weight targets were run **serially** (`-j1`, one GPU workload
  at a time): model 2 s, selfcheck 310 s, FSI 486 s, CODE 294 s, DET 830 s —
  all green.

## Notes / follow-ups

- `pad_token_id` stays -1 in the index (the checkpoint declares only bos/eos);
  the tokenizer bridge derives the pad row from `tokenizer_config.json`, same as
  M3.
- The `dspark/` draft head is stored-but-unapplied (no speculative decoding in
  the engine). If speculative decoding is ever added, this checkpoint ships the
  draft in that subdirectory, outside the root weight map.
- The MXFP8 fixture must keep `cols % 32 == 0` in every quantized projection
  (the E8M0 scale tile is per 32 columns); the first synthetic geometry used a
  16-wide shared expert and produced an empty scale record — pinned by the
  fixture's comment.
