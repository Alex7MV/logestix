# DeepSeek-V4.1-Flash-1017 — ingestion plan (sibling variant)

Date: 2026-11-22
Branch: `feat/model-DeepSeek-V4.1-Flash-1017`
Protocol: **`ingesting-a-model`** (Stage 0 triage → stages 1–9), TDD per stage.

## Stage 0 — triage (the gate before any code)

**Classification: SIBLING VARIANT of `deepseek-v4.1-flash`.** Not a new family
and not a new architecture. Evidence collected 2026-11-22 from the two vault
trees side by side:

| file | verdict |
|---|---|
| `config.json` | **byte-identical** (`diff` clean) |
| `inference/config.json` | **byte-identical** |
| `tokenizer.json` | **byte-identical** (`md5 8a8245dc7f6c6bfcb0684a4be4e17217`) |
| `tokenizer_config.json` | **byte-identical** |
| `model.safetensors.index.json` | **byte-identical** (`md5 d6c52eae774e3bd5959a5d64f2004e7b`) |
| shard tensor names / shapes / dtypes | identical on every shard header inspected |
| shard payload bytes | **different** (10 MB head of shards 1/3/47 differ → re-trained weights, not a re-pack) |
| `inference/model.py` | **3 semantic deltas** (below) |
| `chat_template.jinja` | new file in the 1017 tree; matches the parent's encoding.py chat-mode rendering |
| `README.md` | "minor update … reasoning, coding, agentic, visual; lower output length" |

Geometry is therefore the parent's, verbatim: 40 layers, hidden 5120, 64 heads
× 512, q_lora 1280, o_lora 1024 / 8 groups, 384 routed + 1 shared expert,
top-6, vocab 129280, `rms_eps` 1e-20, CED `compress_ratios`
`[0,0,2×18,1×20]`, kv sources {2,8,14,20}, index sources
{2,8,14,20,24,28,32,36}, candidate source 20, hc_mult 4.

**Tokenizer verdict:** byte-level BPE, `tokenizer_kind 0`, identical bytes →
the parent's tokenizer arrows are the same artifact; bos/eos/pad = 0/1/2 from
config; no new bridge work.

**Ballast to strip (unchanged, counted):** vision tower + aligner (266
tensors), engram tables (12 tensors, ~98 GiB each — still not converted),
DSpark MTP (2401 tensors, stored-not-applied).

**Reference deltas in `inference/model.py` (source of truth for semantics):**

1. `Compressor`: index keys are appended when a compression group completes
   (`if self.owns_k:` guard moved) — the parent already publishes only on group
   completion, so the observable sequence is the same.
2. `model.py:785` + `:1070`: the attention output `o` is run through
   `act_quant(o, 32, "ue8m0", E8M0, True)` before the grouped `wo_a` einsum.
   This is a **runtime activation rounding**, not a weight/state change. The
   parent's pipeline feeds BF16 `o` into a dequantized FP16 GEMM. Recorded as a
   documented deviation (numerics-only); it does not touch weights, the KV
   cache or the compressor state, so it cannot change what the checkpoint *is*.
3. `encoding/encoding.py`: tool-result merge/turn-start refinements and
   content-block validation (multimodal). Text-only chat-mode rendering is
   unchanged — verified against the new `chat_template.jinja`.

**Feasibility on this box** (RTX 5090 32 GiB, 768 GiB RAM, 1.3 TiB free on
`/mnt/fast_storage`): the parent's converted tree is **294 GiB**
(`vram.optimal_bytes 289.8e9`), the expert store is host-mmap + LRU, and the
parent's seeded row is 7.7 t/s. Conversion needs ~294 GiB of the 1.3 TiB free —
fits.

## Design (decided by the sibling precedents)

Precedents followed: `minimax-m3.1-flash` (identical geometry, re-declared
passport, parent's loader/kernels/pipeline) and `granite-switch-4.1-8b` (two
classes, one TU, one implementation body).

1. **Passport** `libs/lsxcommon/src/deepseek_moe_v41_1017_constants.h`, namespace
   `lsxcommon::deepseek_v41_flash1017`, `kType = "deepseek-v4.1-flash-1017"`,
   `kModelName = "DeepSeek-V4.1-Flash-1017"`, geometry re-declared (model
   independence: one frozen namespace per registered model) and pinned against
   the parent's passport with `static_assert`s in the model header.
2. **Model class** `DeepSeekV41Flash1017Model` in `deepseek_moe_v41_model.h`.
   The current implementation body becomes the shared non-final base
   `DeepSeekV41BackboneModel`; the parent class and the sibling are two thin
   `final` classes that pass their own `kType`. No duplicated `Load`/`Infer`.
3. **Converter**: no new pipeline. `ConvertDeepSeekV41Model` stamps the index
   `model` field from the **input directory's card name** instead of a hardcoded
   string, so `detect_model_type` resolves each tree to its own `kType`.
4. **Manifest row** `models/deepseek-v4.1-flash-1017.json` claims the parent's
   convert/load/model/kernels targets (one converter, one kernel set) and its
   own selfcheck/fsi/code/det/perf targets.
5. **Gates**: selfcheck (Paris ≥3 bit-exact cycles), FSI/CODE/DET parity
   (template goldens: the source cannot run locally, method already recorded for
   the parent), tool round-trip, perf baseline at the canonical protocol.

## Stages and their tests (TDD: red first, then green, same commit)

| # | change | test that fails first |
|---|---|---|
| A1 | converter stamps the input card name | `test_deepseek_moe_v41_convert` new case: synthetic dir `DeepSeek-V4.1-Flash-1017` → index `"model": "DeepSeek-V4.1-Flash-1017"` |
| A2 | sibling passport + class + registry | `test_deepseek_moe_v41_model` new cases: passport is identical field-by-field to the parent's; factory resolves the new `kType` to a model whose `Type()` is that `kType` |
| A3 | `Load` rejects a wrong `kType`/format for the sibling | `test_deepseek_moe_v41_loader` new case: sibling `Load()` on a foreign-format index fails, on the parent fixture succeeds |
| A4 | manifest + CMake wiring (selfcheck/parity/perf targets) | `test_manifest_sync`, `test_model_suite_guard`, `test_ctest_labels_guard`, `test_code_naming_guard` |
| A5 | perf row + `TEST_CASE` | `test_perf_baseline_logic` |
| B1 | convert the real checkpoint | — (operator step) |
| B2 | real-weight gates | selfcheck, fsi, code, det, tool round-trip, perf |
| C | catalogue, matrix, open-tasks, stage-9 report | `test_manifest_sync --check` |

## Verification scope (per the protocol)

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -j 24                      # fast guards
ctest --test-dir build/linux-debug -R "deepseek" -j 1  # this row + siblings
```

Real-weight gates run on the Release/AOCC tree, not Debug.
