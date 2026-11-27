# GigaChat3.5-432B-A28B-Reasoning — open items

### R12. GigaChat3.5-432B-A28B-Reasoning — real-weight lifecycle complete (2026-10-31)

Branch `feat/model-GigaChat3.5-432B-A28B-Reasoning`. The code landed in the
2026-09/10 sessions; this session executed the **real-weight half** of the
lifecycle at the current vault path (`/mnt/fast_storage/huggingface`) and closed
two stage-4 defects. Report:
`docs/model-reports/2026-10-31-GigaChat3.5-432B-A28B-Reasoning.md`.

**Stage-4 defects found and fixed (same commit as the tests).**
- `ConvertGigaChatModel` never built the tokenizer arrows (the only family that
  did not) — it left `"tokenizer": "fixture"` in the index and pushed stage 4
  to a separate operator step. It now calls `BuildAndVerifyTokenizer` and writes
  `"tokenizer": "arrow-bpe"`; the synthetic fixture writes a real
  `tokenizer.json` + `tokenizer_config.json` and the converter test pins
  `tokenizer_kind == 3`, `bos/eos/pad`, and a byte-exact encode/decode.
- The `--tokenizer` policy defaulted to the legacy kind-0 split, which
  mis-splits mixed-case identifiers (`getUserName`: `get|UserName` vs the
  checkpoint's `get|User|Name`). GigaChat is the Llama-3/o200k 7-alternative
  case-run BPE, so both the converter and `lsxforge --tokenizer --gigachat`
  now select `pretokenizer = 1` (kind 3) and a 30-case golden captured from the
  checkpoint's own `tokenizer.json` passes 30/30
  (`tests/fixtures/gigachat/tokenizer_parity.jsonl`,
  `tools/gigachat_tokenizer_capture.py`, `test_gigachat_moe_tokenizer_parity`,
  claimed by the gigachat row's `load` stage).

**Conversion (real, in place).** 416 GB FP8-block-128 source → **241 GB**
tree: 9.7 GB pinned dense + 219 GB routed experts (9472 ExpertSchema files,
14 × 16 GiB pinned chunks in ~7.2 s warm) + 13 GB MTP stored-not-applied
(144 tensors). `model.index.json` reports 40 layers, 9472 experts, 1009 pinned
tensors.

**Gates as measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**
- SelfCheck **PASS**: "The capital of France is" → **Paris** (top-1 19904),
  Paris ×3 bit-exact at `logits_hash 2743ee1c26f859d5`, TTFT budget green after
  the cold-store page-cache eviction.
- Native tool round-trip (real weights): `get_weather {"city":"Paris"}` through
  the chatml-v5 GCML codec.
- FSI **3/3**, CODE **3/3**, DET **3/3** (`deterministic=true`, 3 runs/prompt)
  against **template-derived** goldens. They are honest template goldens
  (`source` = prompt-template + the unrunnable reason, `raw` empty in code/det)
  captured with the checkpoint's `chat_template.jinja` and a **1536-token**
  budget: at 512 the FSI-02 reasoning trace is still open when the budget runs
  out, and extraction then latches onto a prompt-echo line
  (`Balance_Alpha=100000000`) — a truncation artifact, not a wrong answer. The
  two capture tools gained an `add_special_tokens` override because the
  GigaChat template spells `<s>` literally (otherwise the ids carry a double
  BOS).
- Tokenizer parity 30/30 (kind 3, bos 1, eos 2, pad 2) against the tree's own
  arrows.
- Perf rows re-seeded through the harness at the canonical 2048-token protocol:
  **perf 11.32 t/s / r3 11.10 t/s**, TTFT 3905 / 3975 ms, 189 / 192 W,
  26.2 GiB resident. The previous `r3 3.47` row (2026-10-06) does not reproduce
  on this tree — the strict class now lands ~2% under `perf`, the expected shape
  for a store-bound model. Both rows were re-seeded through the harness twice
  (417 s each) with identical values, so the new number is measured, not
  inferred; the old row is history, not an explanation.

**Open work, explicitly.**
- L1. Decode is dominated by expert H2D (~3.7 GB/token at the PCIe Gen5
  ceiling; cache hit rate ~49-51% at the auto slot count). The attribution and
  the measured ceiling are in `docs/GIGACHAT_SPEED_AUDIT.md`; nothing is
  filed as a defect because the model fits only as a streamed expert set.
- L2. FSI/CODE/DET goldens are template-derived, not source parity. The source
  is the vendor's multi-GPU FP8 stack and the BF16 body is 864 GB > 750 GB host
  RAM, so no oracle can run here. A future host with enough RAM could replace
  them with source captures; the goldens are labelled so the provenance is
  never mistaken for source parity.
- L3. Session ceiling is 4096 tokens (`max_seq_`); the 262k context in
  `config.json` needs ~34 GiB of MLA KV alone.
- L4. MTP layers 40-42 are stored and counted, never applied.

### G4. A converted tree can be built in `/tmp` and then vanish, leaving a green matrix row (2026-11-07)

**Status: open — vault hygiene, found while re-running parity gates.**

`gigachat3.5-432b-a28b-reasoning` is listed in `docs/SELFCHECK_MATRIX.md` §1 with
SelfCheck/FSI/CODE/DET all green (measured 2026-10-31) and
`docs/model-reports/2026-10-31-GigaChat3.5-432B-A28B-Reasoning.md` exists. On
2026-11-07 its three parity gates SKIP, and the SKIP is *correct*:
`/mnt/fast_storage/lsx_model/ai-sage/GigaChat3.5-432B-A28B-Reasoning/` contains the
source checkpoint and a `lsxforge_runtime.1.log` whose writes all point at
`/mnt/fast_storage/tmp/gigachat_reconvert/logestix/...` — the tree was converted
**outside the vault**, and the tmp directory has since been cleaned. No
`logestix/model.index.json` exists anywhere for it.

So the row is not reproducible from the vault, and nothing in the tree records
that. Two lessons, both cheap to act on:

- **Convert into the vault, or record where the tree lives.** A conversion whose
  output root is a temp directory produces a tree that passes gates today and
  disappears tomorrow; the gates then SKIP, which reads as "not applicable"
  rather than "the artifact is gone".
- **The manifest cannot express this.** `golden_debt` is about goldens, and the
  goldens *are* present for gigachat; what is missing is the converted tree, which
  is vault state. `docs/OPEN_TASKS.md` is the only place that can say it, which is
  why it is here.

Re-converting GigaChat costs a 432B/28B-active pass over ~850 GB of source and
needs the disk headroom the host currently does not have (94% full, 878 GB free).
Treat it as a scheduling decision, not a bug fix.
