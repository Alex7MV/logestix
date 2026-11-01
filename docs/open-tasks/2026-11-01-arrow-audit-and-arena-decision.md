# Arrow IPC audit + glm-5.3-flash arena decision — 2026-11-01

Session record. Branch `audit/arrow-usage-2026-11-01`. Host: Linux, RTX 5090
(32 GiB), CUDA 13.4. Vault `/mnt/fast_storage/huggingface` (29 models).
Companion report: [`../ARROW_AUDIT_2026-11-01.md`](../ARROW_AUDIT_2026-11-01.md).

## 1. Arrow IPC audit — closed findings

The audit covered every converted model's Arrow read/write path (central
`schema.cpp` layer, `ExpertStore`, `shared_experts.cu`, `tokenizer.cpp`, 24
view loaders, 6 owning-`UnpackBatch` paths, 24 converters).

Fixed (TDD, each with a regression test):

- **A-01** owning `UnpackBatch` paths (`Expert`, `SharedExpert`, `DenseNvfp4`,
  `Pinned`) and `ExpertStore::DecodeEntry` did not validate `num_rows`, list
  column types or offset ranges; `SharedExpertSchema` was a SIGSEGV on a
  foreign scale list. All now go through `lsxcommon/arrow_check`
  (`arrow_validate.h`).
- **A-02** view loaders did not cross-check the payload against `shape`/dtype
  (Pinned) or the NVFP4 invariant (DenseNvfp4).
- **A-03** `ShapePtr` did not range-check the shape list.
- **A-04** `shared_experts.cu` accepted reversed scale offsets.
- **A-05** artifact writes were not atomic: a failure could leave a partial
  `.arrow` at the final path. Now temp + rename, with a fault-injection seam.
- **A-06** `ReadRecordBatchFromFile` now rejects a zero-row file and an
  unexpected column count.
- **A-07** gemma MoE/dense dense-triple is read through views, not a copy.
- **Exact expert size** (follow-up): routed experts are uniformly per-16
  (measured over all vault trees); `CheckExpertScaleCount` runs in
  `ExpertStore::DecodeEntry` and `ExpertSchema::UnpackBatch(batch, block)`.
  **Shared** experts are excluded: the full audit found DeepSeek-V4-Pro/V4-Flash
  store them as raw F8_E4M3 with E8M0 128x128 tile scales (32 768 elements per
  scale), so the schema carries no block size.
- **Write path**: `ExpertSchema::MakeBatch` no longer stages a redundant host
  copy, all four `MakeBatch` builders reserve their values buffers (tracked-pool
  budget: peak 2x -> 1x payload, 0 growing reallocations), and
  `ExpertSchema::MakeBatchView` wraps the source vectors (the writer is the only
  remaining copy). `MemoryMappedFile::WillNeed` replaced the Linux-only
  `posix_fadvise`.

Tooling: `test_arrow_validation` (fast), `test_arrow_usage_guard` (fast),
`test_arrow_tree_integrity` (convert, env-gated) and the shared
`lsxcommon::arrow_check::VerifyArrowTree` / `FindConvertedTrees`, exposed as
`lsxforge --verify --input <model>`.

## 2. Full-vault validation and the three missing trees

`LSX_ARROW_AUDIT_FULL=1` over 29 trees: **310 158 `.arrow` files, 0 problems,
783 s** (`/mnt/fast_storage/tmp/arrow_tree_full29.log`). The full run is what
caught the shared-expert FP8/E8M0 encoding above — the sampled pass had skipped
the 4 GB file under its budget.

Converted in this session (the three models that had no tree):

| model | source | flag | time | tree | files |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 67 GB | `--qwen35moe` | 217 s | 19 GB | 10 446 |
| DeepSeek-V4.1-Flash | 476 GB | `--deepseek_v41` | 228 s | 294 GB | 15 403 |
| DeepSeek-V4-Flash | 155 GB | generic | 124 s | 157 GB | 11 055 |

All three pass `lsxforge --verify` and the full audit; the generic converter
already emits per-16 routed experts, so `--deepseek_nvfp4` was not needed.

## 3. glm-5.3-flash arena — tried, measured, reverted (decision)

The 2026-10-14 archive left glm-5.3-flash on the legacy pinned arena because
"there is no converted tree on this machine to verify against" and its layout
deviates (dense prefix 3, compact slots, zero-padded names). The tree now
exists (171 GB) and the neutral store already supports `first_moe_layer` /
`zero_pad_expert_names`; two small neutral additions were made for the trial
(`layer_dir_offset`, `layer_name_width`).

Trial (real weights, canonical 2048-token protocol, AOCC release build):

| configuration | RSS | t/s |
|---|---|---|
| pinned arena (current) | 171 GB | **12.79** (baseline) |
| neutral store, `max_open=4096` | 57 GB | **4.10** |
| neutral store, `max_open=12096` (all) | 156 GB | **7.23** |

Correctness was green (SelfCheck Paris, FSI/CODE/DET), but the neutral store
copies H2D from a **pageable mmap** while the arena's slabs are pinned, and
GLM-Flash streams ~360 experts x 14 MB per token. The migration is therefore
**rejected on measurement** and reverted; the arena stays, with the numbers
recorded here. qwen MoE / Ornith / Mistral were already migrated on 2026-10-14;
qwen3.8-flash-next keeps its arena deliberately (2026-10-14 negative result).

## 4. Decisions recorded as open work

- **Multi-expert IPC packing** (one file, N record batches): only helps
  metadata/file count, not payload bandwidth. Before any format change,
  measure the `ExpertStore::ParseFile` share of the per-token cost on a
  high-miss model (DeepSeek-V4-Pro / Kimi); migrate only if it is material.
- **Wall-time A/B** of the write-path copy elimination: the deterministic
  tracked-pool budget is the evidence; a time A/B needs a worktree build of the
  pre-change commit.
- **Windows**: atomic artifact writes use remove-then-rename under `_WIN32`;
  never exercised on the Windows preset here.
