# Hardware-Native Codec Contract, `libs/lsxstorage`, and NVIDIA Ampere AOT — Design

Date: 2026-11-24, revised 2026-11-27 (three times: re-measurement, cold start,
ownership + layout)
Stage: 0 (research + design specification; no product code)
Status: proposed — awaiting operator approval
Revision note (2026-11-27b, **cold start**): the operator deleted every converted
tree, so the design drops all legacy compatibility instead of accommodating it —
no dual-read, no per-tree audit, no `kLegacy` resolve status, no ballast records,
no migration runbook, no disk-headroom constraint. §1 is rewritten around an empty
artifact store; §1.2 states what that costs; §1.3 turns the old nine-row `dtype`
mapping into five writer-side rules; §6 becomes a cold-start cookbook, and the
bootstrap conversion is promoted to a Branch-1 exit criterion because with zero
artifacts the real-weight suite is dark.
Revision note (2026-11-27c, **deletion verified complete**): the last leftover tree
(`Inkling-2-Mini/logestix/`, 18 GB) is gone. Measured: 0 `logestix/` dirs and 0
`.arrow` files in the vault (§1.1.1). No control tree exists for any model, so the
contract rests entirely on our own fixtures plus the bootstrap conversions; the
last open question about keeping a reference tree is closed (§8).
Revision note (2026-11-27f, **source store moved and org-nested**): checkpoints
live at `/mnt/fast_storage/lsx_model/<org>/<model>` (39 over 18 orgs); the old
flat vault is gone. §2.4.1 gains `FindSourceModel`/`ListSourceModels`, drops
`LegacyDir`, and records that the checkpoint predicate must accept Mistral's
`params.json` layout as well as HF's `config.json`.
Revision note (2026-11-27d/e, **ownership and layout**): answering "is
`artifact.fb` per model or per artifact?" found a second source of truth — codec
declarations still lived in `model.index.json` — closed by rule 5 and
`Artifact::layer_codec[]` (§2.4.3, §9.3). Then "why a `logestix/` folder in the
logestix store?" withdrew the intermediate path segment entirely: the artifact IS
the profile dir (§2.4.2, §9.4).
Supersedes: the implicit "Blackwell-only, NVFP4-only" assumption in `AGENTS.md`
and `docs/NVIDIA_DOCS_AMPERE.md` §"Repo status"

---

## 0. Intent and the why

The engine today is a Blackwell-only NVFP4 engine that happens to work. It
hard-codes CC 12.0/12.1 in the build, gates every tensor-core path on
`prop.major == 12`, and derives what a converted tree *contains* from file
names and 26 `format` tags. Two things break that:

1. **New silicon.** RTX 3090 (sm_86, 24 GB) and Jetson Orin NX/Nano (sm_87,
   16/8 GB unified, JetPack 6.2.1 / CUDA 12.6) must run models natively.
   Neither has NVFP4 or FP8 tensor cores. A model whose planes are NVFP4
   cannot run there, and must not be *pretended* to run by software decode.
2. **New storage.** The converted trees lived inside the read-only Hugging Face
   source vault (`<source>/<Model>/logestix/`), with no declaration of what they
   contained and no way to hold more than one codec profile of the same model.
   **They have been deleted**; the artifact store now starts empty and is built to
   the contract in this document from the first byte.

The five invariants below are the contract. They are normative: where this
document and code disagree, this document wins until the code is changed.

### 0.1 The five invariants (normative)

| # | invariant | enforceable statement |
|---|---|---|
| I1 | **No software emulation** | A conversion profile may only use `PlaneCodec` values that the target `HardwareTarget` *executes in hardware*. A codec that is merely decodable in software is never in the mask. |
| I2 | **Multi-vendor capability registry** | One abstraction `HardwareTarget {vendor, arch}` and one table `SupportedCodecs(HardwareTarget) -> Bitmask<PlaneCodec>` in `libs/lsxcommon`. Vendors: `nvidia`, `cpu`, reserved `amd`. |
| I3 | **Hard admission gate** | `used_codecs_mask ⊆ SupportedCodecs(active_target)`, evaluated once, in `ModelBase::BeginLoad`. A model runs only if 100 % of the codecs its artifact used are natively supported by the active accelerator. |
| I4 | **Multi-artifact + autoselect** | One source model converts to several native trees, one per codec profile (`lsxforge --codec-profile`). `lsxfabric` selects the best *compatible* artifact for the current GPU/CPU. |
| I5 | **Split storage** | `libs/lsxstorage` (namespace `lsx::storage`) separates read-only HF sources (`LSX_SOURCE_ROOT`) from converted artifacts (`LSX_ARTIFACT_ROOT`). Every artifact is self-sufficient (tokenizer + `chat_template.jinja` inside it) and lives at `<ROOT>/<org>/<model_dir>/<profile>/` with no extra path segment (§2.4.2). **Amended 2026-11-27:** the fallback to `<LSX_MODEL_ROOT>/<Model>/logestix/` is retired — there are no legacy trees, so an unset `LSX_ARTIFACT_ROOT` is a configuration error that names itself, not a compatibility route. |

### 0.2 Operator decisions that constrain all of this

**2026-11-24.** The operator **rejected dual-read and legacy codec inference**.
Concretely:

- `artifact.fb` is **mandatory**. An artifact without it, or with a layout that
  carries no codec mask, is **rejected at `ModelBase::BeginLoad`** with an
  explicit message naming `lsxforge` as the remedy.
- The runtime **never** infers codecs from Arrow columns, file suffixes, or the
  26 legacy `format` tags. `artifact.fb` is the single source of truth for
  artifact *content*.

**2026-11-27 (supersedes the migration plan).** The operator **deleted every
converted tree** and chose to **start the store from scratch, with no rework**.
Concretely:

- There is no legacy set to stay compatible with. The backward-compatibility
  obligation of the original brief is **withdrawn**, not deferred.
- Constraints that existed only to accommodate those trees are removed rather
  than preserved: dual-read (already rejected), the per-tree audit, the `kLegacy`
  resolve status, per-profile ballast records, the migration runbook, and the
  disk-headroom constraint (§1.2, §9.2).
- Consequently the format can be **designed clean instead of inferred**: converters
  must emit unambiguous `dtype` spellings (rules D1–D5, §1.3), and the reader is
  allowed to *reject* ambiguity rather than resolve it.

What remains, restated precisely:

> **I3 (cold start).** `used_codecs(artifact) ⊆ SupportedCodecs(active_target)` for
> every artifact the converters produce from now on. An artifact whose codecs are
> not all natively executable on the active target is refused at load with a
> message naming the missing codecs, the target, and the profiles that would work.
> Verification of the refusal is table-driven and needs no GPU; verification of the
> accepted path is the per-model suite, which requires the bootstrap conversion
> (§6) before it has anything to run.

Why the original brief's hardest requirement disappears rather than gets solved: it
was "100 % backward compatibility with 45 already-converted trees, without
reconversion". Starting from scratch removes the premise — there are no trees to be
compatible with — and with it a whole class of dual-path code that would have
outlived its usefulness.

### 0.3 Goals

- A build matrix that produces native SASS for `sm_120/121` **and**
  `sm_86/sm_87` from one tree, under CUDA 13.5 (3090) and CUDA 12.6 (Orin).
- A codec contract that is *checkable without hardware*: pure functions, table
  tests, no CUDA in the header.
- A storage layer that makes an artifact self-describing and relocatable.
- A codec-profile mechanism that produces a **BF16** profile (Branch 2) and an
  **INT8 W8A16** profile (Branch 3) for Ampere.

### 0.4 Non-goals

- No new model families. No changes to any family's math.
- No CUDA graph / TMA / tcgen05 work for Ampere. Ampere has no TMA.
- No AMD backend: `amd:gfx*` is a *table row and an enum bit*, and
  `SupportedCodecs` returns 0 for it, which the gate reports as
  "target not implemented". No kernels.
- No `model.index.json` replacement. It keeps geometry and the VRAM profile, and
  loses its codec fields (§2.4.3 rule 5). See §2.4 for the boundary.
- No CPU *inference* implementation. `cpu:avx512*` gets a target identity, a
  capability probe and a mask; kernels are future work.
- No promotion of NVFP4/FP8 to Ampere by any means, including "just this once".

### 0.5 Anti-goals (would make this a failure even if it works)

- A gate that can be bypassed by a code path that does not go through
  `BeginLoad` (there are 40 family call sites; all 40 already route here).
- A second source of truth: a codec mask in `artifact.fb` **and** a re-derived
  one at runtime that can disagree silently.
- Per-family codec knowledge. That is the `validator.cpp` mistake
  (`docs/LSXFABRIC_AUDIT.md` F1/F7), which was fixed by deleting the file.
- A build that "supports" Ampere by compiling the Blackwell object libs for
  `sm_86` and letting them fail at launch.

---

## 1. Evidence base (what was measured, not assumed)

Recon on 2026-11-24, re-measured 2026-11-27, against
`/mnt/fast_storage/lsx_model` (39 checkpoints over 18 orgs) and `models/*.json`
(47 manifest rows).

### 1.1 Inventory (cold start, 2026-11-27)

**The operator deleted every converted tree.** The vault volume now holds sources
only; the artifact store starts empty.

| quantity | value | consequence |
|---|---|---|
| manifest rows (`models/*.json`) | **47** | the registry is the contract for what can be converted |
| source checkpoints on disk | **39** over 18 orgs, of which 38 are representable by the manifest | `ERNIE-5.5-Flash-Base-BF16` has **no manifest row**; the store is `<org>/<model>` |
| converted trees on disk | **0** | **the whole real-weight suite is dark until bootstrap** |
| `.arrow` files anywhere in the vault | **0** | nothing to read, nothing to compare against |
| free space on the vault volume | **2.6 TB of 15 TB (82 % used)** | the disk constraint of the previous revision is **gone** |
| surviving non-artifact evidence | `tests/fixtures/{fsi,code,det}/*.json`, `tests/perf/perf_baselines.json`, `models/*.json` | goldens and perf rows are independent of artifacts and **must not be lost** |

#### 1.1.1 The deletion is complete (verified 2026-11-27)

The last leftover — `Inkling-2-Mini/logestix/`, 18 GB — was the final tree; the
operator deleted it too. Measured now: **0 `logestix/` directories and 0 `.arrow`
files anywhere under the source store.** The store is sources only.

Three consequences that the rest of this document now assumes:

1. **There is no control tree, anywhere, for any model.** No converter can be
   cross-checked against a previously converted artifact, and no reader path can be
   tested against pre-contract bytes. The contract is proven by **fixtures we write
   ourselves** (Branch 1 Tasks 1–9) and by **the first real conversions** (Task 9a),
   not by parity with history.
2. **`Resolve` has exactly one happy path.** A source dir can never also be an
   artifact dir, so `kSourceOnly` and `kArtifact` cannot be confused by a stray
   directory — the ambiguity that motivated `kLegacy` no longer physically exists.
3. **Disk is not an issue** (2.6 TB free), so artifact writes are bounded by
   conversion time only.

### 1.2 Why this is a simplification, and what it costs

Removing every legacy tree removes, at once:

| dropped concern | previous revision |
|---|---|
| a bit-exact backward-compatibility obligation | 30 heterogeneous trees had to keep working on sm_120 |
| the legacy codec-inference audit | 30 per-tree cross-checks against the old format |
| dual-read (fb → JSON fallback) | already rejected on 2026-11-24; now moot |
| the "`used_codecs` for ballast" dilemma | GigaChat4's `mtp/` FP8 forced a stored-vs-executed decision |
| the multi-profile migration runbook | convert → verify → delete, per model |
| the disk-headroom hard constraint | 478 GB free against a 378 GiB artifact |
| the "keep one reference tree" question | the last leftover (18 GB) was itself deleted |

**What it costs — and this is the important part — is that nothing proves the
contract against real data any more.** The `dtype` inventory in §1.3, the six
subtree kinds in §1.4, and the "no BF16-only tree existed" finding are all
*measured from data that no longer exists* — and now not even one tree survives to
serve as a specimen. They remain valid as the **evidence that the contract was
designed against reality rather than invented**, and they are preserved here for
that reason. They are no longer a compatibility requirement.

One sharpening since the previous revision: there is no *specimen* either. The
`Inkling-2-Mini` tree was the last chance to run the emitter against real bytes and
is gone, so the first real conversion (Task 9a) is simultaneously the first test of
the emitter against anything larger than a fixture. Two mitigations, both already in
the plan: `EmitArtifact` re-reads and re-verifies its own output before returning
(Task 6), and `lsxforge` fails the conversion if the written mask disagrees with the
declared one (Task 8) — so a bad emitter produces a *failed conversion*, not a
plausible-looking artifact.

Two structural consequences:

1. **Invariant 3 loses its backward-compatibility clause.** It is now purely about
   *new* artifacts: `used_codecs(artifact) ⊆ SupportedCodecs(active_target)` for
   everything the converters produce from this point on. There is no legacy set to
   honour, and no reading of I3 should be stretched to cover one.
2. **Bootstrap becomes the first real deliverable.** With zero artifacts, 22 test
   files that resolve `LSX_TEST_*_ROOT` and 27 that touch a converted tree now
   SKIP — and per AGENTS.md *a gate that SKIPs is not a pass*. Until the emitter
   exists and two models are converted, the engine has **no real-weight
   verification at all**. Branch 1 therefore ends with conversions, not with a
   library (§6, Task 9a).

### 1.3 The `dtype` inventory that informed the contract (historical, frozen)

Every `dtype` value found in a non-experts plane across the 30 now-deleted trees.
Recorded because the contract was derived from it; **not** a runtime requirement.

| dtype | tensors | what it was | codec it meant |
|---|---|---|---|
| `BF16` | 21 594 | bf16 weights | `kBF16` |
| `F32` | 3 453 | fp32 norms / scales / biases | `kF32` |
| `U8` | **2 472** | NVFP4 nibbles (`*.packed`) + E4M3 scale bytes (`*.scales`) | `kNvFp4`, **as a pair** |
| `F8_E8M0` | 1 670 | MXFP8 block exponents | companion of `F8_E4M3` |
| `F8_E4M3` | 914 | fp8 weights | `kMxFp8` |
| `I8` | 768 | int8 weights | `kI8Sym` |
| `NVFP4` | 116 | NVFP4 named explicitly | `kNvFp4` |
| `I64` | 6 | index/metadata tensors | metadata, no codec |
| *(no `dtype` column)* | 514 | 9-column `ExpertSchema`, 6-column `DenseNvfp4Schema` | codec implied by schema shape |

**The lesson this table teaches, and the rule it becomes.** The old format had
**nine spellings for five codecs plus two kinds of non-codec**, and five of those
spellings were ambiguous without context:

- `U8` meant NVFP4 *only when* a `.scales` sibling existed;
- `F8_E8M0` and `I64` looked like planes but were companions and metadata;
- two whole files (514 planes) had no `dtype` column at all and were classified by
  their Arrow schema width.

Since **we now write the format from scratch** and nothing has to be read back from
the old one, the clean contract is not "map nine spellings correctly" but **"emit
unambiguous spellings, and reject ambiguity"**:

| rule | statement |
|---|---|
| **D1** | A quantized weight plane names its codec explicitly: `dtype = "NVFP4"`, `"F8_E4M3"`, `"I8"`, `"BF16"`, `"F32"`. Never `"U8"` for a weight. |
| **D2** | Companion tensors say so in their name — `.scale`, `.scales`, `.global_scale` — and carry the same `dtype` as the plane they accompany (`U8` for E4M3 scale bytes, `F32` for a global scale). |
| **D3** | Metadata tensors (index tables, bitmaps) are `dtype = "I64"`/`"U8"` **and** live under a declared `role = "meta"` plane, so no codec bit can be inferred from them. |
| **D4** | `VerifyArtifact` **fails** on an ambiguous weight plane — `U8` without a `.scales` sibling, an unknown `dtype`, or a plane whose `role` is `"meta"` yet whose bytes look like weights. Ambiguity is a conversion defect, not something the reader resolves. |
| **D5** | The mask is a set union, never a sum. |

D1–D5 replace the nine-row mapping table with five rules and make the
`U8`-pair heuristic **unnecessary in the reader and mandatory in the writer** — which
is strictly smaller surface in the runtime, and puts the failure where the defect is.

One consequence worth stating: the old `gemma` and `Muse` trees spelled NVFP4 as a
`U8` pair and the old `Nemotron-NVFP4` tree spelled it `NVFP4`. New conversions use
`NVFP4` for all of them. That is a converter change, in Branch 1, and it is the
kind of change "start from scratch" makes free.

Four facts about that old data remain worth keeping, because they explain design
choices rather than impose requirements:

1. **`U8` was not a codec** — it was how NVFP4 was *spelled* in three trees
   (Muse-Glimmer, gemma-4-26B, gemma-4-31B): `X.weight.packed` (nibbles) plus
   `X.weight.scales` (E4M3 codes). Rule D1 exists so this never recurs.
2. **Companions and metadata must not create codec bits** — `F8_E8M0`, `I64` and
   `BOOL` planes are real and legitimate; rules D2/D3 give them a home that cannot
   be mistaken for a weight.
3. **The mask is a set union, never a sum** — one codec spelled twice in one tree
   must be harmless (D5).
4. **`I8` storage already exists in code.** The generic passthrough at
   `converter.cpp:660` (`(dtype == "I8" || dtype == "U8") && !src_scale.empty()`)
   already carries int8 planes through the conversion, and `converter.cpp:309`
   already accepts `I8` as a source dtype. What does not exist is an int8
   **quantizer** and an int8 **inference kernel** — so Branch 3 is smaller than it
   first looked (§8.5), and this is a code fact, not a deleted-data fact.

### 1.4 Six subtree kinds, and the ballast rule that decides `used_codecs`

Converted trees were not two directories (`pinned/`, `experts/`) but **six**. This
is preserved because the new format must still have a place for each:

| subtree | trees that had it | contents | read at inference? |
|---|---|---|---|
| `pinned/` | 30 | dense weights, norms, tokenizer arrows | **yes** |
| `experts/` | 22 | routed experts (9-column `ExpertSchema`) | **yes** |
| `mtp/` | 9 | multi-token-prediction head | **no — ballast** |
| `router/` | 6 | router weights | **yes** |
| `shared/` | 6 | shared experts | **yes** |
| `ple/` | 1 | per-layer-embedding planes | **yes** |

Only `mtp/` is ballast: written by nine converters, declared in `model.index.json`
as `deferred: {"mtp": "stored", ...}` — "stored-not-applied" in the AGENTS.md sense,
never read at inference. In GigaChat4-37B and -708B it carried six `F8_E4M3`
tensors each, which is what made the meaning of `used_codecs` ambiguous.

**Cold-start decision (revises the 2026-11-27 ruling, and is simpler):**

> **v1 ships no ballast in any profile.** `mtp/` is not written. A plane that the
> load path cannot reach is not stored, so `used_codecs` has exactly one meaning —
> *the codecs of the planes in this artifact, all of which the loader can reach* —
> and there is no stored-vs-executed distinction anywhere in the gate.

Why this is now the right call and is not the earlier "drop it per profile" rule:

- The earlier rule existed only because *existing* nvfp4 trees already contained
  `mtp/` and had to keep working. Nothing does now.
- With one rule for all profiles, `dropped_planes[]` and `ArtifactSpec::drop_ballast`
  become unnecessary — two schema fields and a branch in the emitter, gone.
- Dropping is not data loss: the sources stay read-only and an MTP feature, if ever
  built, is a **new profile** (`nvfp4+mtp`) that declares the FP8 plane it needs —
  a gated, declared, testable event rather than an inherited silence.

The rejected alternative is unchanged: an `executed_codecs` field beside
`used_codecs` would be a second source of truth that can disagree (§0.5).

Test obligation that replaces the old per-profile ballast test: a conversion of a
family that *has* an MTP head must produce a tree **with no `mtp/` directory**, and
`VerifyArtifact` must not report a missing plane for it.

### 1.5 Tokenizer and chat template are **not** in any tree

Measured over all 30 deleted trees: **0 of 30** contained `chat_template.jinja`; all
30 contained exactly the two tokenizer arrows in `pinned/`. The template lived only
in the read-only source checkout. Invariant 5's "artifact is self-sufficient
including a copy of `chat_template.jinja`" was therefore **entirely unimplemented**
— and with the sources still present, every new artifact must copy it in (Task 6).
A model whose source has no `chat_template.jinja` records `chat_template = ""`
rather than inventing one.

### 1.6 How codec identity was previously expressed (historical)

Kept as the justification for `artifact.fb`, not as a requirement:

- 26 distinct `format` tags across 30 indexes, and 16 of the 30 trees declared **no**
  `format_tag` in their manifest row at all.
- `expert_nvfp4` appeared in the index of only 6 of 22 expert-bearing trees; the
  other 16 had no per-layer codec declaration.
- The dense plane codec was implied by the file suffix `_nvfp4` in some trees, by a
  `dtype = "NVFP4"` column in others, and by a `U8` `.packed`/`.scales` pair in
  three more. **Three spellings of one codec.**
- `arrow_tree_verify.cpp` classifies files by *schema shape* (9-column
  `ExpertSchema`, 6-column `DenseNvfp4Schema`, 4-column `PinnedSchema`, vocab,
  merges) — the system infers, it does not declare.

This is exactly the ambiguity the operator is deleting, and §1.3's table is the
proof: 9 `dtype` spellings for 5 real codecs plus metadata.

### 1.7 NVFP4 on Ampere is technically *possible* — hence the gate is policy

`lsx_inference_nvfp4.cu` has three execution paths for an NVFP4 weight:

1. TC `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale`
   (`lsx_tc_gemm.cu`, object lib built only for `120a;121a`);
2. seq==1 weight-streaming GEMV (scalar core);
3. bit-exact scalar tile kernel.

Paths 2 and 3 would *run* on sm_86 today. I1 forbids them for a model whose
declared profile is NVFP4 — the mask must exclude NVFP4 on Ampere **despite
the code being able to execute it**. The gate is therefore a policy gate with a
hardware table, and its tests are table tests, not GPU tests.

### 1.8 There is no F16/BF16 tensor-core GEMM in the tree

The only `mma` instruction in the repository is the NVFP4 `mxf4nvf4` one.
cuBLAS is **not linked** (two comments in `deepseek_moe_*_inference_attn.cu`
mention `cublasGemmStridedBatchedEx` as history; no header, no link). Every
linear layer goes through:

- `lsxcommon::gemv::RunLxGemmVec` / `RunLxGemmVecWT` — dtype-generic
  (`kF16`, `kBF16`, `kF32`), one warp per output row, fixed geometry,
  device-property-independent (built to satisfy `docs/numeric-contract.md`
  §1.1), and therefore **portable to sm_86/87 as-is**; or
- the NVFP4 layer above.

So on Ampere: **decode works immediately; prefill degrades to token-by-token
GEMV** (~10–20× slower than Blackwell prefill). Operator chose option **(c)**:
do not write a BF16 MMA GEMM now; write the GEMM once, in Branch 3, as the
INT8/W8A16 path (weights 1 byte/elt, dequant→f16 epilogue), which serves
decode and prefill.

### 1.9 Disk budget: a guard, no longer a constraint

The deletion freed the vault: **2.6 TB free of 15 TB (82 % used)**, of which 11 TB
is sources and ~1.6 TB the rest. The previous revision's hard constraint is gone —
the largest artifact anyone would write (GigaChat4-708B, 378 GiB) now fits several
times over.

Downgraded from a design constraint to a cheap guard:

- `lsxforge` still preflights free space against the estimated artifact size and
  refuses to start what cannot finish. Not because room is scarce, but because a
  half-written tree with no `artifact.fb` is refused by the gate for a reason that
  looks like a codec bug — the worst possible failure mode to debug.
- The 2× staging rule is dropped: there is nothing to stage against any more.
- Conversions can be batched. There is no interleaving requirement and no
  convert-verify-delete cycle; §6 is a cold-start cookbook, not a migration.

---

## 2. Architecture

### 2.1 Layering (new)

```
                      apps/ (lsxforge, lsxfabric, lsxbenchmark, lsxcli, taskmosaic)
                                        │
                              libs/lsxcommon  (families, ModelBase, kernels)
                                        │
                             libs/lsxstorage    (ns lsx::storage)
                                  │            │
                            FlatBuffers     spdlog
```

`libs/lsxstorage` is the **bottom** library. It deliberately does **not**
depend on `libs/lsxcommon` — otherwise the gate (in `ModelBase`, inside
`lsxcommon`) could not read `artifact.fb`. It depends only on FlatBuffers and
spdlog. Consequence: `path_registry` **moves** into `lsxstorage` and
`lsxcommon/path_registry.h` keeps its two existing free functions as thin
forwarders, so the 40+ call sites of `lsx_model_models_dir()` /
`lsx_cache_path()` are untouched.

**Where each piece lives:**

| concern | library | header reachable from `.cu`? |
|---|---|---|
| `HardwareTarget`, `PlaneCodec`, `SupportedCodecs`, `GateDecision` | `lsxcommon` (`include/lsxcommon/hardware_target.h`) | **yes** — so: no absl, no flatbuffers, no Arrow, no exceptions, C++20 POD only |
| artifact.fb read/write/verify, root resolution, profile select | `lsxstorage` | no — `.h` exposes plain `struct`s only; flatbuffers types stay in `.cpp` |
| the gate itself | `lsxcommon/src/model_base.cpp` (`BeginLoad`) | n/a |
| `--codec-profile` parsing + artifact emission | `apps/lsxforge` | n/a |
| autoselect for a launch | `apps/lsxfabric` | n/a |

The "no flatbuffers in headers" rule is not cosmetic: `libs/lsxcommon` headers
are pulled into ~200 `.cu` TUs. A flatbuffers include there would add seconds
per TU and a second NVCC exposure.

### 2.2 `PlaneCodec` — storage encoding, not execution path

`PlaneCodec` names **how a plane's bytes are encoded on disk**. It is
orthogonal to which kernel consumes it. A bitmask over it answers exactly one
question: *which of these encodings does the target execute natively?*

```cpp
// libs/lsxcommon/include/lsxcommon/hardware_target.h
namespace lsxcommon::hw {

enum class PlaneCodec : uint64_t {
    kNone   = 0,
    kF32    = 1ull << 0,   // IEEE binary32
    kBF16   = 1ull << 1,   // bfloat16
    kF16    = 1ull << 2,   // IEEE binary16
    kI8Sym  = 1ull << 3,   // signed int8, symmetric per-channel, f32 scale companion
    kMxFp8  = 1ull << 4,   // E4M3, block-32, E8M0 scale
    kNvFp4  = 1ull << 5,   // E2M1, block-16, UE4M3 scale (+ per-tensor f32 global)
    kMxFp4  = 1ull << 6,   // E2M1, block-32, E8M0 scale
    // 1<<7 .. 1<<31 reserved for future plane encodings
    // 1<<32 .. vendor extensions (never in SupportedCodecs of another vendor)
};
using CodecMask = uint64_t;

constexpr CodecMask Mask(PlaneCodec c) { return static_cast<CodecMask>(c); }
constexpr const char *PlaneCodecName(PlaneCodec c);
// Parses "nvfp4" / "mxfp8" / "bf16" / "i8" ...; returns false on unknown.
bool ParsePlaneCodec(std::string_view name, PlaneCodec *out);
} // namespace lsxcommon::hw
```

Deliberate omissions: no `kFp8E5M2` (nothing in the tree produces it), no
`kI4`/`kI2` (no converter), no `kF64` (nothing stores it). Adding a bit later is
a one-line change plus one table cell; speculative bits are YAGNI.

### 2.3 `HardwareTarget` and `SupportedCodecs`

```cpp
enum class Vendor : uint8_t { kNvidia, kCpu, kAmd /* reserved */, kUnknown };

struct HardwareTarget {
    Vendor vendor = Vendor::kUnknown;
    std::string arch;        // "sm_120", "sm_121", "sm_86", "sm_87",
                             // "avx512", "avx512+bf16+vnni", "neon", "gfx942"...
    // Normalized identity used by artifact.fb profile_id and by the table key:
    //   "nvidia:sm_120", "cpu:avx512+bf16+vnni", "amd:gfx942"
    std::string Id() const;
    bool operator==(const HardwareTarget &) const;
};

// The table. Pure, constexpr-ish, no CUDA, no globals, no env.
CodecMask SupportedCodecs(const HardwareTarget &t);

// I3 as a pure function: empty string = allowed.
struct GateDecision { bool allowed; CodecMask missing; std::string message; };
GateDecision CheckCodecs(CodecMask used, const HardwareTarget &t);

// The active target of this process. Resolved once: CUDA device 0 when a
// device is present, else CPUID-derived cpu:*.
const HardwareTarget &ActiveTarget();
// Tests only: overrides ActiveTarget for the rest of the process. No env var,
// no CLI flag — a test cannot accidentally gate a production run.
void SetActiveTargetForTesting(const HardwareTarget &t);
void ResetActiveTargetForTesting();
} // namespace lsxcommon::hw
```

**The table** (target → natively executed packings), with the hardware reason:

| target | mask | why |
|---|---|---|
| `nvidia:sm_120`, `nvidia:sm_121` | F32,BF16,F16,I8,MxFp8,NvFp4,MxFp4 | `mma ... kind::mxf4nvf4.block_scale` covers both block-scale FP4 flavours; `kind::f8f6f4` covers E4M3 |
| `nvidia:sm_89` | (not a row) | Ada has FP8 tensor cores. Excluded by **policy**, not capability — documented so the table is not mistaken for a capability list |
| `nvidia:sm_86`, `nvidia:sm_87` | F32,BF16,F16,**I8** | `mma.m16n8k16` f16/bf16 and `mma.m16n8k32.s8.s8.s32` (IMMA) are native on GA10x. **No FP8, no FP4**: no E4M3/E2M1 tensor-core instruction exists on GA10x |
| `cpu:avx512` (+feature suffix) | F32, and BF16 **iff** `avx512_bf16`, F16 **iff** `avx512_fp16`, I8 **iff** `avx512_vnni` | feature-derived, never assumed from the family name |
| `cpu:neon+dotprod` (Orin host side) | F32,BF16,F16,I8 | AArch64 NEON + dotprod; **no AVX512 exists on Jetson**, so `cpu:avx512` can never be the only CPU target |
| `amd:gfx*` | **0** | reserved; gate reports "target not implemented". Honest emptiness beats a fake row |

sm_87 does **not** execute sm_86 SASS, so an Ampere fatbin needs both
`code=sm_86` and `code=sm_87` (see §5).

### 2.4 `libs/lsxstorage` — specification

#### 2.4.1 Roots and resolution

```
LSX_SOURCE_ROOT    read-only HF checkpoints.        default: /mnt/fast_storage/lsx_model
LSX_ARTIFACT_ROOT  converted artifacts (writable).  default: UNSET
LSX_MODEL_ROOT     the pre-split spelling of the source root; still consulted
                   after LSX_SOURCE_ROOT so an older environment keeps working
LSX_CACHE_DIR      per-user host cache (unchanged)
```

The store is **org-nested**: `<SourceRoot>/<org>/<model>`, the layout Hugging
Face uses for a download. Measured on 2026-11-27: 39 checkpoints with weights
over 23 orgs (`Qwen/Qwen3.5-9B`, `microsoft/phi-4`, `thinkingmachines/Inkling`,
…), plus **9 empty placeholder directories** recording where an evicted
checkpoint belongs (`moonshotai/Kimi-K3`, `inclusionAI/Ling-3.1-flash`,
`RecursiveSI/flint`, `reflection/Beam-1.0-501B-A23B`, `MiniMaxAI/MiniMax-M3`,
`mistralai/Mistral-Large-4-1T-A52B`, `stepfun-ai/Step-5-Preview`,
`zai-org/GLM-5.2`, `XiaomiMiMo/MiMo-V2.6-Flash-RL`). Every one of the 47
manifest rows now has a directory: 39 with weights, 9 placeholders, 0 absent.

A placeholder is not a checkpoint: `ListSourceModels()` ignores it, so nothing
resolves to a directory with no weights, and `FindSourceModel()` reports the
placeholder by org instead of suggesting a typo. `ListPlaceholderDirs()` is the
accessor, so a tool can enumerate what is missing from the store without
scanning directories itself.

Resolution is a pure function with a documented precedence; it never guesses
codecs, only *location*:

```cpp
// libs/lsxstorage/include/lsxstorage/roots.h
namespace lsx::storage {

std::filesystem::path SourceRoot();      // LSX_SOURCE_ROOT   or kModelRootBase
std::optional<fs::path> ArtifactRoot();  // LSX_ARTIFACT_ROOT or nullopt
fs::path CacheRoot();

// Where artifacts of a model live, per profile:
//   <ArtifactRoot>/<org>/<model_dir>/<profile>/{artifact.fb,pinned,...}
//
// The path MIRRORS the source store, so moving between a checkpoint and its
// artifacts means changing only the root:
//   source     /mnt/fast_storage/lsx_model/microsoft/phi-4
//   artifact   /mnt/fast_storage/lsx_artifact/microsoft/phi-4/nvfp4
//
// Both segments are the STORE's spelling ("microsoft", "Qwen3.5-9B"), not the
// manifest's family/model_type: an operator navigates by what `ls` shows. The
// manifest identity is still recorded — inside artifact.fb as model_type/family
// — so the two spellings can never be confused at load time.
//
// No intermediate `logestix/` segment: the profile IS the artifact (§2.4.2).
fs::path ArtifactDir(const std::string &org, const std::string &model_dir,
                     const std::string &profile);

// Source checkpoints, one level down: <SourceRoot>/<org>/<model>.
struct SourceModelRef { std::string org, model; fs::path dir; };

// Every checkpoint under the source root, sorted by (org, model). A directory
// directly under the root is an ORG; only one that also holds a checkpoint does
// not qualify, so <root>/microsoft never appears as a model.
//
// "Is this a checkpoint?" is config.json OR params.json OR any *.safetensors
// file. config.json alone is NOT enough: Mistral's own download layout ships
// params.json + consolidated*.safetensors and no config.json (measured on
// mistralai/Mistral-Small-4-119B), so that predicate hides real models.
std::vector<SourceModelRef> ListSourceModels();

// Empty <org>/<model> directories: deliberate records of where an evicted
// checkpoint belongs. They are NOT checkpoints (nothing can load from them) and
// never enter ListSourceModels(); they exist so a diagnostic can distinguish
// "the directory is here, the weights are not" from "you mistyped the name".
std::vector<fs::path> ListPlaceholderDirs(const std::string &model);

// Exactly one checkpoint named `model`, which may be bare ("phi-4") or
// org-qualified ("microsoft/phi-4"). A qualified name matches exactly rather
// than falling back to a search. Returns nullopt when there is none OR when the
// bare name matches under more than one org; *why names the orgs. An ambiguous
// name is never resolved by picking one, because the choice would depend on
// directory iteration order.
std::optional<fs::path> FindSourceModel(const std::string &model,
                                        std::string *why);

// NO kLegacy status: with the artifact store starting empty there is nothing
// legacy to resolve to, and a status that could only ever fail is a trap for
// the next reader. A source dir resolves to kSourceOnly.
//
// There is also no LegacyDir(): its premise was a flat <Model> root, so with an
// org-nested store it could not even be computed, and nothing loadable ever
// lived there. FindSourceModel answers the question that mattered.
enum class ResolveStatus { kArtifact, kSourceOnly, kNotFound };
struct Resolved {
    ResolveStatus status;
    fs::path dir;                 // the directory handed to the loaders
    std::string profile;          // "" for kSourceOnly / kNotFound
    std::string detail;           // human-readable reason
};
Resolved Resolve(const std::string &model_or_path,
                 const lsxcommon::hw::HardwareTarget &t);
} // namespace lsx::storage
```

`Resolve` accepts: an artifact profile dir (already resolved → return it), a
source model dir (→ kSourceOnly, so the caller can say "convert it"), a bare
model name, an org-qualified name (`microsoft/phi-4`), or an org-nested path
(`<root>/Qwen/Qwen3.5-9B`).

#### 2.4.2 The `logestix/` leaf is gone — the path form is the profile dir

The first draft kept a `logestix/` segment inside every artifact directory
(`<profile>/logestix/pinned/…`) so that the 500-odd existing
`model_path / "logestix"` joins in the tree would keep working untouched. **The
operator rejected it, correctly, and this section records why the rejection is right
and what it costs.**

```
<LSX_ARTIFACT_ROOT>/<org>/<model_dir>/<profile>/
    artifact.fb
    model.index.json
    pinned/          experts/         router/      shared/      ple/
    chat_template.jinja              tokenizer.json            (provenance)
```

`cfg_.model_path` is the **profile dir**; every content subtree hangs directly off it.

**Why the leaf was wrong** (three reasons, in order of weight):

1. **The saving was imaginary.** Measured: **538** `"logestix"` path segments in
   **226** files — but they split as `libs/` 144 / 84, `apps/` 3 / 1, `tests/` **391 /
   142**. The tests build `<vault>/<Model>/logestix/pinned/…`, and the root moves from
   the vault to `LSX_ARTIFACT_ROOT` in Branch 1, so **all 391 test sites change
   regardless of whether the leaf survives.** The leaf would have protected ~147
   production sites while the test half was rewritten anyway — and left the two halves
   looking at different path forms.
2. **The leaf was a format marker, and a false one.** "Is this a converted tree?" is
   answered today by the presence of a directory *named* `logestix`
   (`arrow_tree_verify.cpp:506`, `deepseek_moe_repack.cpp:83-85`). The contract's
   marker is `artifact.fb`; leaving the leaf lets those sites keep checking a name
   instead of the contract.
3. **It duplicated the store's own name** — `<ROOT>/phi/phi-4/bf16/logestix/pinned` in
   a tree called logestix — and it carried four dead fallback branches inside
   `ResolvePinnedDir()` (`logestix-qwen35moe`, `logestix-glm`, `logestix-kimi`,
   `logestix-gemma`) for a layout that no longer exists.

**What the change costs, precisely:** `fs::path(X) / "logestix"` → `fs::path(X)`, a
one-form mechanical edit at 144 `libs/` + 3 `apps/` + 391 `tests/` sites; deletion of
the four `ResolvePinnedDir()` fallbacks; and the two detector sites rewritten to test
for `artifact.fb` instead of a directory name. Verified safe: **no production code
derives a model name from the basename of `cfg_.model_path`** (checked `filename()` /
`parent_path()` across `libs/`), and the converter's default model name comes from
`input_path`, which is a source directory and does not move.

`"logestix"` legitimately survives in exactly one place: `path_registry.cpp`'s
`~/.cache/logestix` / `%LOCALAPPDATA%\logestix\cache`, which is the per-user host
cache and has nothing to do with the artifact layout.

**Guard (new, mirrors `test_logging_guard`):** a `fast` test asserts that no source
file under `libs/`, `apps/` or `tests/` contains `"logestix"` as a path segment except
the cache-root allowlist — so the leaf cannot come back.

This is operator-decided (2026-11-27), not my call; the earlier "keeps Branch 1
bounded" argument is recorded here only as the reasoning that was withdrawn.

#### 2.4.3 `artifact.fb` — FlatBuffers schema (v1)

FlatBuffers is already a submodule built static
(`CMakeLists.txt:514-522`), pinned in `tests/test_core.cpp`, and unused inside
`libs/` — so this costs no new dependency. It gives zero-copy mmap reads, a
schema, a verifier, and forwards/backwards-compatible evolution.

```fbs
// libs/lsxstorage/schema/artifact.fbs
namespace lsx.storage.fb;

enum Vendor : byte { Nvidia, Cpu, Amd }
enum PlaneCodec : ulong (bit_flags) {
  F32 = 1, BF16 = 2, F16 = 4, I8Sym = 8, MxFp8 = 16, NvFp4 = 32, MxFp4 = 64
}

table Target { vendor:Vendor; arch:string; }        // arch "sm_86", "avx512+bf16"

table Plane {                                       // one storage plane
  codec:PlaneCodec;
  role:string;            // "dense.mlp", "expert.routed", "embed", "norm", ...
  path:string;            // relative to the artifact dir; "" = in model.index.json
  bytes:ulong;
  sha256:[ubyte];         // 32; empty = not digested (not allowed for I/O planes)
}

table Tokenizer {
  vocab:string = "pinned/tokenizer_vocab.arrow";
  merges:string = "pinned/tokenizer_merges.arrow";
  chat_template:string = "chat_template.jinja";     // "" when the model has none
  chat_template_sha256:[ubyte];
  source_tokenizer_sha256:[ubyte];
}

table Source {                                      // provenance, not identity
  model_name:string;        // "Phi-4"
  source_dir:string;        // "phi-4" relative to SourceRoot
  config_sha256:[ubyte];
  safetensors_index_sha256:[ubyte];
}

table Artifact {
  format_version:uint = 1;
  artifact_id:string;       // 32 hex: sha256(family|model|profile|mask|build)
  model_type:string;        // factory registry id, e.g. "phi-4"
  family:string;            // "phi"
  profile:string;           // "bf16", "nvfp4", "int8-w8a16"
  target:Target;            // the target this artifact is INTENDED for
  used_codecs:ulong;        // I2/I3 mask — the single source of truth
  planes:[Plane];
  // Per-layer codec mask, index = physical layer. Replaces the codec fields that
  // used to live in model.index.json (mlp_nvfp4 / attn_nvfp4 / head_nvfp4 /
  // expert_nvfp4[] / expert_fp8[]), which six load and inference files read to
  // pick a kernel per layer. Those fields are REMOVED from model.index.json
  // (spec §2.4.3 rule 5), so this array is the only per-layer codec record.
  layer_codec:[ulong];
  // Store coordinates, so the artifact records where it belongs and
  // ListProfiles() can enumerate siblings from artifact.fb alone:
  //   <root>/<org>/<model_dir>/<profile>
  org:string;               // "microsoft"
  model_dir:string;         // "phi-4" (the store's own spelling)
  tokenizer:Tokenizer;
  source:Source;
  layer_count:int;
  pinned_tensors:int;
  expert_files:int;
  converter_version:string;
  converter_git_rev:string;
  codec_profile_arg:string; // verbatim "--codec-profile" value
  created_utc:string;       // ISO-8601
  enterprise_reserved:ulong;// 0 in open edition
  root_type Artifact;
}
```

Rules that make it a *contract* rather than metadata:

- `used_codecs` **must** equal the OR of every `Plane.codec`. `lsxforge --verify`
  recomputes it from the `planes[]` list and fails on mismatch, so a mask cannot
  drift from the content it describes.
- `planes[]` **must** cover every I/O plane of the tree, and **every plane in the
  tree must be reachable by the load path** (§1.4: no ballast in v1). `--verify`
  enforces both directions: an unlisted plane and an unshipped listed plane are
  each a failure. The existing `Classify()`/schema logic in
  `arrow_tree_verify.cpp` is reused for *verification only*, never for runtime
  inference.
- **Plane identity follows rules D1–D5 (§1.3):** a quantized weight names its codec
  explicitly (`NVFP4`, `F8_E4M3`, `I8`, `BF16`, `F32` — never `U8`); companions are
  named `*.scale`/`*.scales`/`*.global_scale` and carry the same dtype as their
  plane; metadata is `I64`/`U8` under `role = "meta"`. An ambiguous weight plane is
  a conversion defect that `--verify` **fails**, never something the reader guesses.
- `target` is advisory provenance. **The gate uses `used_codecs`, not `target`** —
  otherwise copying a sm_120 artifact to a Jetson and editing one string would
  bypass I1.
- **Rule 5 — `model.index.json` owns geometry and the VRAM profile; it owns nothing
  about codecs.** Fields that identify a codec (`mlp_nvfp4`, `attn_nvfp4`,
  `head_nvfp4`, `expert_nvfp4[]`, `expert_fp8[]`, `fp8_kv_scales_dropped`) are
  **removed** from it, and their per-layer role moves to `Artifact::layer_codec[]`.
  Six files read those fields today to pick a kernel per layer
  (`inkling_moe_load.cpp`, `inkling_moe_loop_load.cpp`,
  `inkling_moe_inference_pipeline.cu`, `inkling_moe_loop_inference_pipeline.cu`,
  `gemma_moe_load.cpp`, `gemma_moe_inference.cu`); they read the artifact instead
  (§2.4.6). `--verify` **fails** an artifact whose `model.index.json` still carries
  one of those keys, so the second source of truth cannot come back unnoticed.
- **Why one file per artifact, not one per model.** `model.index.json` is written
  per artifact because its `vram` block is genuinely profile-dependent:
  `EstimateVram` charges `numel/2 + numel/16` only when `is_nvfp4(name)` and
  `raw_bytes` otherwise (`vram_estimate.cpp:52`), so a `bf16` profile's
  `min/optimal/max` differ from the `nvfp4` one by roughly 4×. A shared file would
  report the wrong VRAM for one of the two.
- **What must agree across profiles.** Every codec-independent field of
  `model.index.json` (`num_layers`, `hidden_size`, `vocab_size`, `eos_id`, …) must be
  byte-identical across all profiles of one model; only the `vram` block may differ.
  A mismatch is a converter defect, and `--explain-artifact` reports it (§2.7).
- `model.index.json` **stays** and remains the owner of model *geometry*
  (`num_layers`, `hidden_size`, `eos_id`, `vram`, …). `artifact.fb` owns
  *content identity*: codecs, planes, per-layer codec map, tokenizer, provenance.
  Neither duplicates the other. The eventual merge of geometry into `artifact.fb` is
  explicitly deferred (YAGNI, and it would touch 40 loaders).

#### 2.4.4 Read path

```cpp
// libs/lsxstorage/include/lsxstorage/artifact.h   (no flatbuffers types here)
namespace lsx::storage {

struct PlaneInfo {
    std::string role;
    uint64_t    codec_bit;      // one PlaneCodec bit
    std::string path;           // relative
    uint64_t    bytes;
    std::array<uint8_t,32> sha256; bool has_digest;
};
struct ArtifactInfo {
    uint32_t format_version = 0;
    std::string artifact_id, model_type, family, profile;
    std::string target_id;      // "nvidia:sm_86"
    uint64_t used_codecs = 0;
    std::vector<PlaneInfo> planes;
    std::vector<uint64_t> layer_codec;   // index = layer; per-layer codec mask
    std::string tokenizer_vocab, tokenizer_merges, chat_template;
    std::string model_name, source_dir;
    int layer_count = 0, pinned_tensors = 0, expert_files = 0;
};

enum class ReadStatus { kOk, kMissing, kUnreadable, kBadVersion, kMalformed };
struct ReadResult { ReadStatus status; ArtifactInfo info; std::string detail; };

// mmap + flatbuffers::Verifier. Never throws. kMissing is a distinct status:
// its caller (BeginLoad) turns it into the "not a LogestiX artifact" message.
ReadResult ReadArtifact(const std::filesystem::path &artifact_dir);
ReadResult ReadArtifactFile(const std::filesystem::path &artifact_fb);

// Recomputes masks/digests from disk and compares with what the file claims.
struct VerifyReport { bool ok; std::vector<std::string> problems;
                      uint64_t declared_mask, actual_mask; };
VerifyReport VerifyArtifact(const std::filesystem::path &artifact_dir,
                            bool deep /* digest every plane */);
} // namespace lsx::storage
```

AGENTS.md requires Arrow reads to be mmap-only (`test_arrow_usage_guard`), so
`ReadArtifact` mmaps the `.fb` (flatbuffers verifier over the mapped range) and
does not `ifstream` it into a string.

#### 2.4.6 Per-layer codec access for the loaders

Six files pick a kernel per layer and must stop reading `model.index.json` for it
(rule 5, §2.4.3). The artifact reaches them without a second parse:

```cpp
// libs/lsxcommon/include/lsxcommon/model_base.h  (protected, in lsxcommon)
// Populated by BeginLoad stage 2. Non-null iff the gate passed.
const lsx::storage::ArtifactInfo *artifact() const;

// The one helper the six call sites use. `layer` out of range, or a layer with no
// declared codec, returns false — callers keep their existing hard failure.
bool LayerUsesCodec(int layer, lsxcommon::hw::PlaneCodec codec) const;
```

So `expert_nvfp4[l] ? "nvfp4" : "mxfp8"` becomes
`LayerUsesCodec(l, PlaneCodec::kNvFp4) ? "nvfp4" : "mxfp8"`, with the same shape for
gemma's `mlp_nvfp4`/`attn_nvfp4`/`head_nvfp4`. Re-reading `artifact.fb` per layer is
forbidden: the artifact is mmapped once at load and its lifetime is the model's.

Cost, stated honestly: this is **6 files in 2 families** (inkling ×2 variants, gemma
×2 variants), not the 215-file blast radius that §2.4.2 avoids. It is the price of
one source of truth, and it is bounded because only these two families have a
per-layer codec mix today — a family with one codec for all layers never needs the
helper.

#### 2.4.5 Write path (converter side)

```cpp
// libs/lsxstorage/include/lsxstorage/artifact_writer.h
struct PlaneDecl { std::string role; uint64_t codec_bit;
                   std::string path; uint64_t bytes; };
struct ArtifactSpec {  // everything the emitter needs; no IO in the struct
    std::string model_type, family, profile, model_name, source_dir;
    std::string target_id;
    std::vector<PlaneDecl> planes;
    std::vector<uint64_t> layer_codec;  // per-layer codec mask (rule 5)
    std::string chat_template_src;   // absolute source path, "" if none
    std::string source_config;       // absolute; digested
    std::string source_index_json;   // absolute; digested
    int layer_count = 0, pinned_tensors = 0, expert_files = 0;
};
// Writes <artifact_dir>/artifact.fb atomically (tmp + rename),
// digests every plane, copies chat_template.jinja + tokenizer.json, then
// re-verifies what it wrote and fails if the re-verify disagrees.
uint64_t EmitArtifact(const ArtifactSpec &spec,
                      const std::filesystem::path &artifact_dir,
                      std::string *error);
} // namespace lsx::storage
```

`EmitArtifact` returns the mask it wrote; `lsxforge` compares it with the mask
the profile *promised* and fails the conversion on mismatch. That is the
"контрольная сверка конвертера" — and with the old trees gone it is now the *only*
check between a converter bug and a bad artifact, which is why it re-reads and
re-verifies its own output instead of trusting the write.

### 2.5 The gate — one choke point, three stages

`ModelBase::BeginLoad` currently does one thing (load the tokenizer DB). New
sequence, all inside it:

```cpp
bool ModelBase::BeginLoad(const std::string &model_path)
{
    if (loaded_) return true;
    // (1) RESOLVE  — source dir / bare name / profile dir -> profile dir
    auto r = lsx::storage::Resolve(model_path, hw::ActiveTarget());
    if (r.status == lsx::storage::ResolveStatus::kNotFound) { log; return false; }

    // (2) DECLARE  — artifact.fb is mandatory; no inference, no fallback masks
    auto a = lsx::storage::ReadArtifact(r.dir);
    if (a.status != lsx::storage::ReadStatus::kOk) {
        log::error(absl::StrFormat(
            "[load] %s is not a LogestiX artifact: %s. Every artifact must "
            "carry a verified artifact.fb. Convert (or reconvert) it with "
            "`lsxforge --<family> --input %s --codec-profile "
            "<bf16|nvfp4|int8-w8a16>`.", …));
        return false;
    }

    // (3) GATE  — I3, pure function, no hardware call
    const auto d = hw::CheckCodecs(a.info.used_codecs, hw::ActiveTarget());
    if (!d.allowed) {
        log::error("[load] artifact %s uses %s, which %s does not execute "
                   "natively. Available profiles for this model: %s",
                   a.info.artifact_id, hw::FormatMask(d.missing),
                   hw::ActiveTarget().Id(), ListCompatibleProfiles(…));
        return false;
    }

    cfg_.model_path = r.dir.string();   // rebind: every later join now resolves
    return LoadTokenizerDb(cfg_.model_path);
}
```

Ordering notes, each load-bearing:

- Resolution **before** the tokenizer load, because the tokenizer lives inside
  the resolved artifact.
- The rebind of `cfg_.model_path` is what makes the 40 family `Load()`
  implementations, and their content-subtree joins (`cfg_.model_path / "pinned"`,
  `/ "experts"`, …), correct without touching their bodies. It happens only after
  the gate passes, so a refused model never has a mutated config. The joins
  themselves lose their `"logestix"` segment once, in Task 4a (§2.4.2).
- No env bypass. Tests inject a target with `SetActiveTargetForTesting`, which
  cannot be reached from a production binary.
- `tokenizer_optional_` (synthetic fixtures) still applies, but **after** the
  gate: a synthetic test model has no artifact and must declare
  `cfg_.synthetic = true` to skip stages 1–3. This is an explicit config field,
  not a silent fallback — otherwise the gate would be bypassable by every
  fixture in the tree.

### 2.6 Codec profiles — the converter contract

`lsxforge` gains one string flag and one derived contract:

```cpp
ABSL_FLAG(std::string, codec_profile, "",
          "native codec profile to produce: bf16 | nvfp4 | mxfp8 | "
          "int8-w8a16. Empty = the family's historical default (nvfp4).");
```

| profile | planes produced | targets it satisfies | branch |
|---|---|---|---|
| `nvfp4` | NvFp4 weights + BF16/F32 rest; **no `mtp/`** | sm_120/121 | 1 |
| `bf16` | BF16 weights, F32 norms/scales; **no `mtp/`** | **every** target incl. sm_86/87, cpu:* | 2 |
| `int8-w8a16` | I8Sym weights + F32 per-channel scales, BF16/F32 acts & rest; **no `mtp/`** | sm_86/87, sm_120 | 3 |
| `mxfp8` | MxFp8 where the family supports it | sm_120/121 | not scheduled |

No profile ships ballast (§1.4). `nvfp4` is no longer described as "today's
layout": there is no today's layout to preserve — the suffix `_nvfp4`, the `U8`
pair spelling and the `mtp/` subtree all retire with the deleted trees.

The mask is **declared by the pipeline, verified by the emitter**: each
pipeline knows which planes it writes (it wrote them), so `main.cpp` builds the
`ArtifactSpec` from a small per-pipeline table; `EmitArtifact` recomputes the
mask from the files it finds and `--verify` fails on disagreement. Branch 1
therefore does **not** rewrite 40 converters — it adds one emitter call in
`apps/lsxforge/src/main.cpp` after each pipeline returns OK, plus one
per-pipeline plane table.

Rejected alternative: deriving the mask inside the runtime from the tree
(what the old code did). That is precisely what the operator deleted, and it
cannot distinguish "produced by the nvfp4 profile" from "produced by the bf16
profile then partially overwritten".

### 2.7 Autoselect (`lsxfabric`, invariant 4)

Selection is deterministic and total:

1. Enumerate profiles of `<org>/<model_dir>/` under the artifact root.
2. Keep those with `used_codecs ⊆ SupportedCodecs(active_target)`.
3. Rank by the profile rank table (highest first):
   `int8-w8a16` (if the target's INT8 kernel exists) ≥ `nvfp4` >
   `mxfp8` > `bf16`. Rationale: smallest bytes-per-element that the *target can
   execute* wins, which is also the throughput order; a profile whose kernels
   are not implemented on that target is excluded by step 2, not by rank.
4. Tie-break on `profile` string ascending (total order, reproducible), and log
   the full candidate list at `info` with the reason each was rejected.

`lsxfabric` prints the chosen artifact id and mask at startup; `--explain-artifact`
(the only new diagnostic flag) prints step 2/3 for every candidate and exits.

### 2.8 Jetson / unified memory (invariant 5 tail)

`HardwareProfile` (`resource_governor.h`) gains one field and one enum:

```cpp
enum class MemoryModel { kDiscrete, kUnified };
struct HardwareProfile {
    …
    MemoryModel memory_model = MemoryModel::kDiscrete;
    std::uint64_t unified_shared_bytes = 0;   // kUnified: RAM the GPU shares
};
```

Probe: `prop.integrated` (or `cudaDevAttrIntegrated`) → `kUnified`; on unified,
`vram_total_bytes` from `cudaMemGetInfo` is **not** free system RAM (the
carve-out is fixed at boot), so the governor must budget against
`unified_shared_bytes`, not the carve-out.

Two knobs stop being constants on unified memory:

- `lifecycle::kExpertCacheVramReserveBytes` = 8 GiB, calibrated on a 5090
  (`resource_governor.h`). On a 16 GB Orin NX that reserve is half the machine.
  Change: `reserve = min(8 GiB, 25 % of the budget)`, expressed as a function
  `ExpertCacheReserveBytes(budget)` next to the existing `ExpertCacheSlots`
  so both call sites share one rule.
- `ResourceGovernor` must count a permanent-weight lease on unified memory
  against **one** budget shared with host RAM, or a 16 GB Orin will accept a
  15 GB model and then OOM the OS.

No new GPU semantics are introduced for discrete cards: `kDiscrete` reproduces
today's arithmetic exactly, which is what keeps the 5090 baselines valid.

---

## 3. Registry of hard dependencies on sm_120 / NVFP4 / CUDA 13.5

Every item below was read in this tree on 2026-11-24. "Fix" names the branch.

### 3.1 Build: architecture selection

| # | location | what it hard-codes | impact | fix |
|---|---|---|---|---|
| B1 | `CMakeLists.txt:35` | `set(LSX_CUDA_ARCH "120;121" CACHE …)` default | configuring for `86;87` requires an explicit override that then trips B2 | 2 |
| B2 | `CMakeLists.txt:38-46` | validator `if(NOT arch MATCHES "^12[01][af]?$") → FATAL_ERROR "not a Blackwell consumer target… pre-Blackwell arches are deliberately unsupported"` | **blocks the build outright** for `86`/`87` | 2 |
| B3 | `CMakeLists.txt:47-54` | `lsx_cuda_gencode_flags()` emits `arch=compute_X,code=[compute_X,sm_X]` — emits PTX for every arch | for `86;87` this embeds PTX; fine, but must include **both** 86 and 87 SASS (sm_87 does not execute sm_86 SASS) | 2 |
| B4 | `libs/lsxcommon/CMakeLists.txt:640-660` | `lsxcommon_tc_cu` with literal `--generate-code=arch=compute_120a,code=sm_120a; …121a` | `kind::mxf4nvf4` PTX does not assemble for `sm_86/87`; the object lib must be **omitted** (not just empty) when the arch list has no 12x | 2 |
| B5 | `tests/CMakeLists.txt:1377-1392` | the same literal `120a;121a` pair for a TC test object lib | same as B4 | 2 |
| B6 | `tests/CMakeLists.txt:1433-1455` | `test_cuda_toolkit_guard` registration; compiles `.cu` with `LSX_CUDA_GENCODE_FLAGS` and `cuda_fp4.h` | on CUDA 12.6 `cuda_fp4.h` does not exist (it arrived in 12.8) → **compile error**, must be `#if __has_include(<cuda_fp4.h>)`-guarded | 2 |
| B7 | `CMakeLists.txt:76-100` | toolkit-vs-nvcc same-minor FATAL_ERROR | correct on both toolchains; no change (verified) | — |
| B8 | `CMakeLists.txt:104-160` | `lsx_stamp_cuda_toolkit` header/nvcc agreement + stamp file | correct on both; no change (verified) | — |
| B9 | `CMakeLists.txt:226` | Windows `… /arch:AVX512 …` | irrelevant to Jetson (aarch64) but relevant to the `cpu:avx512` story: x86-only | 2 (note only) |
| B10 | `CMakeLists.txt:249-260` | `LSX_ARCH_ZNVER4` → `-march=znver4` | illegal on aarch64; the Jetson preset must not inherit it | 2 |
| B11 | `CMakePresets.json` | 6 presets, all x86-64 (win/linux, debug/release/znver4/aocc) | no aarch64/CUDA-12.6 preset exists | 2 |
| B12 | `CMakeLists.txt:514-522` | FlatBuffers static submodule | already available for `artifact.fb`; **no new dependency** | 1 |

### 3.2 Runtime: device-property assumptions

| # | location | code | behaviour on sm_86/87 | fix |
|---|---|---|---|---|
| R1 | `lsx_tc_gemm.cu:909`, `:1149`, `:1317` | `if (prop.major != 12) return cudaErrorNotSupported;` | correct as a *capability* test, but callers must degrade, not fail | 2 (audit callers) |
| R2 | `lsx_tc_gemm.cu:1087` | `return prop.major == 12;` (TC availability probe) | fine | — |
| R3 | `lsx_inference_nvfp4.cu:957-975`, `:1179` | dispatch: TC → seq1 GEMV → scalar tile, tolerating `cudaErrorNotSupported`/`InvalidValue` | would silently fall back to scalar NVFP4 on Ampere — **I1 violation if reachable** | 1 (gate makes it unreachable) |
| R4 | `gpu_perf.cpp:116` | `k.tc_fast_math_default = (g.cc_major == 12);` | correct default (off on Ampere) | — |
| R5 | `gpu_perf.h:35-42` + `.cpp:50` | profile carries `cc_major/cc_minor` only | enough for the table; `hw::ActiveTarget` derives `sm_86` from it | 1 |
| R6 | `qwen_moe_exp_inference.cu:919` | comment: "smem on sm_120 is 99KB/block" | the smem budget differs: sm_86 99 KB, **sm_87 163 KB** — a kernel sized for 99 KB runs on both (verified from `docs/NVIDIA_DOCS_AMPERE.md` A1) | 2 (audit `__launch_bounds__`/dynamic smem requests) |
| R7 | `resource_governor.cpp` + `resource_governor.h` | discrete-VRAM budgeting, 8 GiB expert reserve | wrong on Jetson unified memory | 1 (§2.8) |
| R8 | `vram_estimate.cpp` | conversion-time `vram` block, discrete assumption | artifact metadata; advisory only on unified memory | 1 (record `memory_model` in the profile, do not gate on it) |

### 3.3 Tests and docs that assert Blackwell-only

| # | location | assertion | fix |
|---|---|---|---|
| T1 | `tests/test_cuda_toolkit_guard.cu:14` | `kMinCudaRuntime = 13020` | must become `>= 12060` with a per-branch floor, and the "13.x only" clause removed |
| T2 | `:50-51` | `REQUIRE(CUDART_VERSION >= kMinCudaRuntime); REQUIRE(CUDART_VERSION/1000 == 13);` | the second line fails on JetPack 12.6 | 2 |
| T3 | `:59-85` | three cases exercising `cuda_fp4.h` (`__nv_cvt_float_to_fp4`) | header absent < 12.8 → guard with `__has_include` and SKIP, never silently pass | 2 |
| T4 | `:88-96` | `REQUIRE(prop.major == 12)` for every present device | must accept membership in a supported set `{12.0,12.1,8.6,8.7}` and reject anything else | 2 |
| T5 | `:118-133` | stamp must equal this TU's toolkit | keep — it is the guard that caught the 13.4/13.5 skew; it is toolchain-agnostic | — |
| T6 | `:145-170` | driver-compatibility rule (major match only) | correct on JetPack (driver 12.6 = runtime 12.6) | — |
| T7 | `docs/NVIDIA_DOCS_AMPERE.md:22-27` | "Ampere (sm_80) is a FORBIDDEN build target … Never add `80-real`/`86-real`/`87-real` targets to CMake" | directly contradicts this work; the doc must be amended to "supported for sm_86/sm_87 only, sm_80 still forbidden" | 2 |
| T8 | `AGENTS.md` "Forbidden architectures: Hopper (90), Ada (89), Ampere (80)" | policy | requires an explicit operator edit (I do not edit AGENTS.md unilaterally); proposed wording: "Forbidden: Hopper (90), Ada (89), Ampere sm_80. Supported: Blackwell 12.0/12.1, Ampere 8.6/8.7." | 2 |
| T9 | `tests/test_ctest_labels_guard.cpp`, `test_infrastructure_guard.cpp` | new targets need one label, `RESOURCE_LOCK "gpu"` for device work, deps only via `lsx_dep_includes` | every new test below must comply | 1/2/3 |

### 3.4 Verified *absences* (good news, load-bearing)

- **No CUDA 13.5-only driver API is called** anywhere: `cuTensorMapSetAllocationClass`,
  `cuGraphAddMemcpyBatchNode`, `cudaMemsetBatchAsync`, `cuFabricCpyBatchAsync`
  appear only in `AGENTS.md` prose. So the 13.5→12.6 runtime skew is a *header
  and flag* problem, not an API problem.
- **cuBLAS is not linked** (comments only) — one fewer library whose heuristic
  could differ per device.
- **NCCL is not linked** — no collective to port.
- **No `__CUDA_ARCH__` branches** except one guard in `lsx_reduce.cuh:17`, so
  there is no hidden per-arch numerics switch to audit.
- **No 13.x-only language features** in device code: `--extended-lambda`,
  `-fmad=false`, `--res-usage` all exist in 12.6.

---

## 4. Numeric contract amendment (required, blocking)

`docs/numeric-contract.md` §1 states R3 is bit-identical **"across the
supported GPU-model set"** and §1.1.4 forbids "device-conditional numeric
paths". Splitting codecs by hardware breaks both *literally*: an Ampere build
cannot produce the same bits as a Blackwell build, because the executed
arithmetic differs (f16/BF16 MMA vs mxf4nvf4).

Proposed amendment (three sentences, one table):

> **R3 is scoped to a hardware target.** Two devices are in one R3 set only if
> they execute the same codec profile natively. `{sm_120, sm_121}` is one set;
> `{sm_86, sm_87}` is another; `{sm_86, sm_120}` is not an R3 relationship
> unless both run the `bf16` profile, in which case they are (both execute
> `mma.m16n8k16` f16/bf16 with f32 accumulate; reduction geometry is fixed by
> problem shape). §1.1.4 is refined to: "a *target capability* may select a
> kernel *family*; within a family, no device property may select numerics."
>
> A profile's goldens are captured **on that profile**. FSI/CODE/DET goldens are
> therefore per `(model_type, profile)`, stored as
> `<LSX_*_GOLDEN_DIR>/<model_type>@<profile>.json`, with the un-suffixed name
> reserving the historical `nvfp4` profile so no existing golden file moves.

Consequence for the gates: on Ampere, SelfCheck ("Paris", ≥3 bit-exact cycles),
FSI, CODE and DET must all be captured **for the `bf16` and `int8-w8a16`
profiles** before that profile is called released. Branch 2 captures `bf16`;
Branch 3 captures `int8-w8a16`.

---

## 5. CUDA 12.6 / JetPack 6.2.1 specifics

| concern | decision |
|---|---|
| compiler | `/usr/local/cuda-12.6/bin/nvcc`, host `g++-11` (JetPack default). Separate build tree: JetPack's aarch64 CUDA cannot share the x86 build dir. |
| arch flags | `-DLSX_CUDA_ARCH="86;87"` on Jetson; `-DLSX_CUDA_ARCH="86"` on the 3090 host. Both produce real SASS; PTX is emitted as a by-product and is harmless. |
| arch-`a` objects | `lsxcommon_tc_cu` is emitted **only if** `LSX_CUDA_ARCH` contains a `12x` entry. Otherwise CMake skips the object lib and the link still succeeds (its symbols are only referenced behind the `prop.major == 12` probes, which the `kAuto`/`kOff` scalar paths never reach on Ampere). |
| minimal CUDA floor | 12.8 for the `cuda_fp4.h` test cases; 12.6 for the library. Enforced as: library floor `12060`, `cuda_fp4.h` cases `>= 12080` via `__has_include`. |
| `cuda_fp4.h` on the 3090 host (13.5) | present, cases run. On Orin (12.6) → SKIP with an explicit reason, never a silent pass. |
| LTO / zen4 | the Jetson preset sets `LSX_LTO=OFF`, `LSX_ARCH_ZNVER4=OFF`, no `-march`. |
| AOCC preset | x86-only; `CMAKE_INSTALL_RPATH` entries in that preset are x86 paths — do not inherit. |
| **NEW preset** `jetson-orin-release` | `CMAKE_BUILD_TYPE=Release`, `BUILD_TESTING=ON`, `LSX_CUDA_ARCH=86;87`, `LSX_LTO=OFF`, toolchain file pinning `cuda-12.6`. |
| **NEW preset** `linux-release-sm86` | x86 host + RTX 3090: `LSX_CUDA_ARCH=86`, CUDA 13.5, `LSX_ARCH_ZNVER4` allowed. |
| `test_cuda_toolkit_guard` on 12.6 | T1–T4 fixes above; the stamp case (T5) stays and now proves the JetPack tree is internally consistent too. |

---

## 6. Cold-start runbook (post-Branch-1, operator-executed)

There is nothing to migrate. The store is empty; this is the order in which it
fills, and why that order matters.

**Why order matters:** with zero artifacts, 22 test files that resolve
`LSX_TEST_*_ROOT` and 27 that touch a converted tree currently SKIP, and a gate
that SKIPs is not a pass (AGENTS.md). The engine has **no real-weight verification
until two models exist**. So the first conversion is not a convenience — it is what
turns the suite back on.

1. **Bootstrap two models, `bf16` first.** `bf16` runs on every target, so it gives
   the 5090 suite something to load *and* immediately unblocks Ampere acceptance:

   ```
   LSX_ARTIFACT_ROOT=<store> lsxforge --phi  --input <SRC>/phi-4      --codec-profile bf16
   LSX_ARTIFACT_ROOT=<store> lsxforge --qwen --input <SRC>/Qwen3.5-9B --codec-profile bf16
   lsxforge --verify --input <store>/phi/phi-4/bf16
   lsxforge --verify --input <store>/qwen/qwen3.5-9b/bf16
   ```

   ≈ 47 GB written. Expected mask `bf16,f32`; **no `mtp/`**; `artifact.fb`,
   `chat_template.jinja` and `tokenizer.json` at the profile dir root (§2.4.2 — no
   intermediate `logestix/` segment); path segment `<model_type>`, so `Qwen3.5-9B`
   becomes `qwen3.5-9b`.

2. **Turn the suite back on and record the baseline** on the 5090: SelfCheck ×3,
   FSI, CODE, DET, perf row — for these two models only. Every other model stays
   SKIP, and the reason is visible: no artifact yet.

3. **Add `nvfp4` for the same two models.** This is what proves invariant 4 on a
   real machine: one model, two profiles, the fabric picking by target.

   ```
   LSX_ARTIFACT_ROOT=<store> lsxforge --phi --input <SRC>/phi-4 --codec-profile nvfp4
   lsxfabric --localmodel phi-4 --explain-artifact      # shows both candidates
   ```

4. **Grow the set as needed, `bf16` before `nvfp4` for any model that has to run on
   Ampere.** Sources are read-only and permanent; artifacts are cheap to rebuild, so
   there is no ordering hazard any more and no staging requirement (§1.9).

5. **Record** the new artifact set in `docs/OPEN_TASKS.md` and regenerate
   `docs/MODELS.md` via `tools/gen_model_manifest.py` so the catalogue points at the
   artifact root rather than the vault.

Rollback: delete the artifact directory and reconvert. Sources are untouched by
construction, so the worst case is conversion time — which is exactly why the
artifact root must never be inside the source vault (§2.4.1).

---

## 7. Risks

| risk | likelihood | mitigation |
|---|---|---|
| **Nothing proves the contract against real data** — the evidence base (§1.2–§1.4) was deleted, so a `dtype`/plane rule that was wrong about reality can now only be caught by fixtures written from the same understanding | **high** | the nine-row inventory is frozen in §1.3 as the provenance record; `--verify` enforces D1–D5 on every conversion; the first real conversion (runbook step 1) is treated as a contract test and any surprise is written back into §1.3 |
| **Bootstrap debt**: 22 vault-dependent test files SKIP until artifacts exist, and a SKIP is not a pass | **high** | runbook step 1 is a Branch-1 exit criterion, not a follow-up; the branch is not "done" until two models convert, verify and pass SelfCheck on the 5090 |
| Branch 1's 500-site path assumption breaks a family that joins paths differently | medium | `Resolve` accepts an already-resolved profile dir, so passing `cfg_.model_path` directly still works; the bootstrap conversions exercise two families early, and each later conversion exercises another |
| Mask declaration drifts from what a pipeline actually writes | **high** (40 pipelines) | `EmitArtifact` recomputes and fails on mismatch; `--verify` re-checks at any time; one emitter call site in `main.cpp` keeps the declared side auditable in one file |
| A converter still writes ballast (`mtp/`) because nine of them have the code | medium | §1.4 makes "no ballast in v1" a contract rule; `--verify` fails an artifact containing a plane the load path cannot reach, so a forgotten `mtp/` write is caught at conversion time, not at load time |
| A converter still spells NVFP4 as a `U8` pair (gemma/Muse/kimi/mimo/inkling all do today) | **high** | rule D1 plus `--verify` failing an ambiguous weight plane; the affected converters are edited in the same task that wires the emitter, and a fixture asserts `dtype == "NVFP4"` |
| A tree set that changes under the work invalidates counts in tests | medium | no literal counts anywhere: iterate `FindConvertedTrees()`, report what was found, assert only relations |
| Ampere prefill is unusably slow before Branch 3 | high | accepted by operator (option c); Branch 2 acceptance is "loads, gates, SelfCheck correct", not throughput; Branch 3 delivers the GEMM |
| INT8 W8A16 numerics fail FSI/CODE on Ampere | medium | W8A16 is a *storage* win (weights s8, dequant→f16, f32 accumulate): no activation quantization, so the error is bounded by per-channel weight rounding. Goldens are captured per profile (numeric-contract amendment), and a failure is a finding to report, not a blocker to hide |
| `sm_87` 163 KB smem lets a kernel compile that would OOM on `sm_86` 99 KB | low | one `__launch_bounds__`/smem audit in Branch 2; both arches are in every fatbin so a mismatch fails at the first Orin/3090 cross-run |
| Jetson unified memory accounting double-counts | medium | `MemoryModel` is a separate axis; `kDiscrete` arithmetic is unchanged (baselines stay valid), `kUnified` is a new, separately-tested path |
| AGENTS.md policy edit is missed, so a future session re-forbids Ampere | medium | Branch 2's definition of done includes the AGENTS.md + `NVIDIA_DOCS_AMPERE.md` amendments; a guard test asserts `SupportedCodecs(nvidia:sm_86)` is non-empty so a policy revert fails CI |

---

## 8. Open questions (do not block Branches 1–2)

1. **`model.index.json` → `artifact.fb` geometry merge** (§2.4.3): deferred, and now
   with a settled boundary — rule 5 moved the codec fields out, so what remains in
   the index is geometry + `vram`, which no loader reads differently per profile.
   Revisit when a loader rewrite is already happening for another reason.
2. **Profile rank between `nvfp4` and `int8-w8a16` on sm_120**: on Blackwell,
   NVFP4 has hardware support and INT8 is also native; the rank table currently
   prefers `int8-w8a16`. It should be measured, not assumed — Branch 3 records a
   perf row on both targets and the table is corrected from evidence if needed.
3. **`demoji`/`cpu:*` inference kernels**: no date. The target identity exists so
   that the *gate* is already correct for a CPU-only box.
4. **Goldens naming for profiles** (§4): `@profile` suffix chosen; whether the
   `nvfp4` profile keeps the bare name forever or is migrated to `@nvfp4` later
   is a hygiene decision, not a functional one.
5. **Branch 3 is smaller than first scoped, and the plan must say so.** `kI8Sym`
   storage already exists in code — the generic passthrough at
   `converter.cpp:660`, plus the `I8`/`U8`+scale rule at `converter.cpp:309`. What
   is missing is a per-channel **quantizer** (BF16 → I8) and an **inference
   kernel** (W8A16 GEMV/GEMM). So Branch 3 = quantizer + kernels + profile wiring,
   not a new storage format.
6. ~~**The leftover `Inkling-2-Mini/logestix/` — keep or delete?**~~ **Closed
   2026-11-27: deleted.** The cold start is absolute and there is no specimen tree
   anywhere (§1.1.1). The consequence — the emitter meets its first real bytes at
   Task 9a — is accepted, with `EmitArtifact`'s self-re-verify (Task 6) and the
   mask cross-check (Task 8) as the guards.

   Task-level estimate after this revision: Branch 1 ≈ 13 tasks (9 + Task 4a +
   Task 9a + Task 9b + the governor task), Branch 2 ≈ 6, Branch 3 ≈ 6.

---

## 9. Builder check (self-review, per skill: no subagents in this repo)

I read this document as an implementer who has not seen the tree, and fixed:

- **Ambiguity: "which target does the gate use?"** Was implied twice
  (`artifact.target` vs device). Fixed: §2.4.3 states the gate uses
  `used_codecs` and the *device*, and that `target` is advisory — otherwise one
  edited string bypasses I1.
- **Ambiguity: "does Branch 1 rewrite converters?"** Fixed: §2.6 — one emitter
  call in `apps/lsxforge/src/main.cpp` + a per-pipeline plane table; explicitly
  not 40 converter changes.
- **Gap: synthetic fixture models** would be blocked by the gate. Fixed: §2.5,
  explicit `cfg_.synthetic`, not a silent fallback.
- **Gap: how a refused model leaves `cfg_`** untouched. Fixed: rebind after the
  gate only.
- **Gap: golden files for new profiles.** Fixed: §4 `@profile` suffix, bare name
  reserved for `nvfp4` so no file moves.
- **Unstated dependency:** `lsxstorage` at the *bottom* of the library graph, and
  `path_registry` moving there. Stated in §2.1 with the cycle argument.
- **Known unknown left open:** profile rank on sm_120 (§8.2) is deferred to a
  measurement rather than asserted.

### 9.1 What the 2026-11-27 re-measurement caught that the first draft missed

*(Superseded in part by §9.2 — items 1 and 5 describe rulings that the cold start
then simplified away. Kept because they are how the design found its shape.)*

Five defects, all found by reading the data rather than the code:

1. **`used_codecs` had no defined meaning for ballast.** The first draft assumed
   mask = executed codecs. GigaChat4's `mtp/` carries `F8_E4M3` that the load path
   never touches, so that reading would have blocked a model for a codec it never
   executes — and the opposite reading opens the hole I1 exists to close. The
   first fix recorded a per-profile `dropped_planes[]`; §9.2 removed the need for
   it entirely by dropping ballast from every profile.
2. **`U8` would have been mis-mapped as a codec.** 2 472 tensors spell NVFP4 as a
   `U8` `.packed`/`.scales` pair. A naive `U8 → ?` mapping either invents a codec
   or fails three trees. §1.3 rule 1 fixes it.
3. **`I64`/`F8_E8M0` would have failed verification** on legitimate metadata and
   companion tensors (Qwen3.8-Flash-Next `ple/`, MXFP8 scales). §1.3 rule 2.
4. **The "31 trees" figure was already stale.** The count moved to 30 within three
   days, and three families lost their control tree. §1.1 and the risk table now
   forbid literal counts anywhere.
5. **The runbook assumed disk it does not have.** At the time, 478 GB free against
   a 378 GiB artifact made "convert all → verify → delete" impossible, so §1.9 and
   §6 became per-model and interleaved. The cold start freed the disk and reversed
   that (§9.2), which is why §6 is now a cookbook rather than a schedule.

Also corrected: `chat_template.jinja` is in **0 of 30** trees (the first draft's
claim that invariant 5 was "partly satisfied" was wrong — a probe that counted the
source-level file, not the artifact-level one).

### 9.2 What the cold start (2026-11-27, later the same day) removed

The operator deleted every converted tree. That is not an inventory change; it
deletes whole sections of the design, and the residue had to be pruned rather than
left to rot:

| removed | was | why it goes |
|---|---|---|
| the backward-compatibility clause of I3 | 30 trees had to keep working on sm_120 | nothing legacy exists; I3 is about new artifacts only |
| the per-tree `--emit-artifact` audit | runbook step 1 | no trees to audit; replaced by treating the first real conversion as the contract test |
| `ResolveStatus::kLegacy` and the fallback path | a compatibility route into `<LSX_MODEL_ROOT>/<Model>/logestix/` | a status that can only ever fail is a trap; `kSourceOnly` says the same thing honestly |
| `dropped_planes[]`, `ArtifactSpec::drop_ballast`, the per-profile ballast table | three schema/driver surfaces | v1 ships no ballast at all, so there is nothing to record |
| the disk-headroom constraint, the 2× staging rule, the interleaved migrate/verify/delete cycle | §1.9, §6 | 2.6 TB free; conversions can be batched |
| the nine-row `dtype` mapping table | §1.3 | replaced by five writer-side rules (D1–D5) that make ambiguity impossible instead of resolving it |

And one thing the cold start **added**, which is the largest schedule risk in the
document: **a bootstrap dependency.** With no artifacts, the real-weight suite is
dark, and a SKIP is not a pass. Two conversions are now a Branch-1 exit criterion,
not a follow-up. The nine-row inventory is retained in §1.3 precisely because it is
now the *only* surviving record of what the contract was designed against.

### 9.3 What the "per model or per artifact?" question exposed (2026-11-27d)

The operator asked whether `model.index.json` and `artifact.fb` are one per model.
Answering it precisely found a **defect in this document**, not a wording problem:

1. **Both files are per *artifact* (per model × profile), not per model** — and for
   `model.index.json` that is correct rather than wasteful, because its `vram` block
   is genuinely profile-dependent (`EstimateVram` charges NVFP4 bytes only when
   `is_nvfp4(name)`, `vram_estimate.cpp:52`, so `bf16` reads ~4× larger). §2.4.3 now
   says this, and adds the invariant it implies: **codec-independent fields must be
   byte-identical across profiles; only `vram` may differ.**
2. **`model.index.json` still carried codec declarations** — `mlp_nvfp4`,
   `attn_nvfp4`, `head_nvfp4`, `expert_nvfp4[]`, `expert_fp8[]` — and **six files
   read them to pick a kernel per layer** (inkling ×2, gemma ×2, each in a loader and
   an inference pipeline). The first draft said "`artifact.fb` owns codecs" and never
   said what happens to those fields, so the design shipped with a second source of
   truth — the exact thing §0.5 anti-goals forbid, and a live I1 hole: an artifact
   declaring `{nvfp4}` with a layer that `model.index.json` calls `mxfp8` would pass
   the model-level gate and then execute FP8 on an Ampere card.

   Fixed by **rule 5** (§2.4.3): codecs leave `model.index.json` entirely; the
   per-layer map becomes `Artifact::layer_codec[]`; the six readers go through
   `ModelBase::LayerUsesCodec()` (§2.4.6); `--verify` fails an artifact whose index
   still carries one of the five keys. Cost, stated: **6 files in 2 families**, not
   the 215-file blast radius §2.4.2 avoids.

The lesson recorded for the next reviewer: "which file owns X" is a question that
must be answered **per field**, not per file. The first draft answered it per file
("`artifact.fb` owns codecs") and was wrong by five fields.

### 9.4 The `logestix/` leaf was withdrawn (2026-11-27e)

The operator asked why an artifact directory needs a `logestix/` segment inside a
store that is itself logestix. Re-measuring to answer it withdrew my earlier ruling:

- The leaf protected **~147 production sites** (144 `libs/` + 3 `apps/`) while the
  **391 test sites** were rewritten anyway, because Branch 1 moves the root from the
  vault to `LSX_ARTIFACT_ROOT`. The saving was half imaginary.
- The leaf was also the de-facto "is this a converted tree?" marker
  (`arrow_tree_verify.cpp:506`, `deepseek_moe_repack.cpp:83-85`), so keeping it meant
  three detection sites kept checking a directory **name** instead of `artifact.fb`.
- It carried four dead `ResolvePinnedDir()` fallbacks for a layout that no longer
  exists.

Replaced by §2.4.2: no leaf, artifact = profile dir, plus a `fast` guard test that no
source file may spell `"logestix"` as a path segment (cache root allowlisted). Cost is
one-form mechanical editing, verified safe because nothing derives a model name from
the basename of `cfg_.model_path`.

Recorded pattern across §9.3 and §9.4: **both defects came from optimising the diff
instead of the contract.** The first draft asked "what change is cheapest to type",
not "what does the contract require" — and the answers were a duplicated source of
truth and a false format marker.
