# NVFP4 full-E4M3 block-scale codec — design

Date: 2026-10-06
Status: design, awaiting operator review
Scope item: `docs/OPEN_TASKS.md` section **L**
Branch: `feat/model-Nemotron3.5-nvfp4-ingest` (this node of the work)

## 1. Goal

Replace the repo's **power-of-two-only** E4M3 NVFP4 block-scale codec with the
**hardware-native unsigned E4M3 (UE4M3)** codec, on both the encode and decode
side. Today the encoder throws away the 3 mantissa bits and the decoder ignores
them; the Blackwell `mma ... kind::mxf4nvf4 ... .ue4m3` operand the repo already
targets consumes the **full** 3-bit mantissa. Closing that gap:

- lets the converter reproduce the source scale exactly instead of paying the
  measured ~11 % requantization loss on pre-quantized checkpoints (section L);
- gives BF16→NVFP4 conversions finer block scales than the 1.33x-median inflation
  forcing a power-of-two costs today.

The change is **decode-backward-compatible**: a power-of-two byte has
`mantissa == 0`, so a full-E4M3 decode returns the same value for every existing
`logestix/` tree. Only the **encoder** changes, which is why the operator deletes
and regenerates the converted variants to actually get the benefit.

Deliverables in this work item:

1. the codec + call-site change, TDD, `fast` label green;
2. per-model **model-lifecycle stage report** (convert → load → model →
   selfcheck → fsi/code/det → perf) for the vault trees, after re-conversion;
3. updated **`nvfp4-blackwell-conventions` skill**; the codec semantics are
   pinned in-code + by the new regression test (the NVIDIA doc mirror lives
   outside the repo at `/home/alexmv2025/projects/cuda/docs`, so it is updated
   there, not committed here).

## 2. Current state (the deviation being removed)

`libs/lsxcommon/src/nvfp4_scale.h`:

- `FloatToE4m3P2(s)`: `e4 = ceil(log2(s)) + 7`, clamped `[1,14]`,
  `byte = e4 << 3` (mantissa always 0, rounded **up**).
- `E4m3P2ToFloat(b)`: `((b>>3)&0xF)==0 -> 0`, else `2^(e4-7)`.

`libs/lsxcommon/src/nvfp4_codec.h::E4M3ScaleToFloat` is the device duplicate of
the decode. The deviation is documented in the external NVIDIA mirror
(`/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md` §1).

The hardware fact: PTX tcgen05/`mma.sync` NVFP4 uses a **UE4M3** scale factor at
`.block16` / `.scale_vec::4X` — unsigned, 4 exponent + 3 mantissa bits, MSB
zero-padded, NaN only at `0x7F`. The mma in `lsx_tc_gemm.cu:353` passes the raw
byte as `.ue4m3`, so a full positive E4M3 byte is already hardware-native.

## 3. Canonical semantics (pinned)

Unsigned E4M3 (`ue4m3`), valid bytes `0x00..0x7F`:

```
value(b) = 2^(e-7) * (1 + m/8)     e = (b>>3)&0xF, m = b&0x7
```

| condition | decode |
|---|---|
| `e == 0` | `m * 2^-9` — **subnormal** (`0x00 -> 0`) — see the 2026-10-07 addendum |
| `1 <= e <= 15`, `m <= 6`, and not `(e==15 && m==7)` | `2^(e-7)*(1+m/8)` |
| `e == 15 && m == 7` (`0x7F`, reserved NaN) | NaN — **never emitted** |

Range: min subnormal `2^-9` (`0x01`), min normal `2^-6` (`0x08`), max finite
`448` (`0x7E`).

Encode `FloatToE4m3(s)` (RNE, ties-to-even):

1. `s == 0` (or negative/NaN) → `0x00`.
2. below `2^-6`: round RNE on the `2^-9` subnormal grid, `m==8` carrying up to
   the min normal `0x08`; below half the min subnormal (`2^-10`) → `0x00`;
   above `448` → `0x7E`.
3. take the binade exponent `E` of the value and the E4M3 mantissa
   `me = round_nearest_even(mantissa_fraction * 8)`; a carry `me==8` becomes
   `me=0, E++`.
4. `E==15 && me==7` (would be the NaN slot) → `0x7E`.
5. `byte = (E<<3) | me`.

The implementation is **pure integer bit manipulation** (no `frexpf`/`rintf`) so
host and device are bit-identical by construction. Exact powers of two encode to
the **same byte** as today (`2^k -> e<<3` with `m=0`).

### 3.1 Backward-compatibility proof

Old decode: `e4==0 -> 0`, else `(e4+120)<<23` = `2^(e4-7)`.
New decode with `m==0`: `(e+120)<<23 | 0<<20` = `2^(e-7)`.
For every pow2 byte the two expressions are the **same 32-bit float**. Existing
`logestix/` trees therefore decode bit-identically; SelfCheck / FSI / CODE / DET
and every logits hash stay put until a tree is re-converted.

## 4. Blast radius

### 4.1 Codec + shared helpers

- `nvfp4_scale.h` — add canonical `FloatToE4m3` / `E4m3ToFloat` (host/device,
  integer only). Retire the `FloatToE4m3P2` / `E4m3P2ToFloat` names (no caller
  wants a pow2-only flavor). `E8m0ToE4m3Pow2Pair` and `ChooseE8m0ScaleShift`
  stay: E8M0 is an exact power of two, so mantissa-0 is already exact there.
- `nvfp4_codec.h::E4M3ScaleToFloat` — delegate to `nvfp4::E4m3ToFloat` (one
  textual source).
- `schema.cpp` / `schema.h` — rename `float_to_e4m3_p2` → `float_to_e4m3`,
  `e4m3_p2_to_float` → `e4m3_to_float`; update callers
  (`safetensors_gpu.cu` offline quantizer, `lsx_tc_gemm.cu`).
- `lsx_tc_gemm.cu::ActQuantKernel` — runtime act-quant uses the full encode and
  full decode for `1/scale`. **This is the one change that moves TC numerics**
  (`kAuto` fast-math path) without any re-conversion.
- `lsx_kernels_quant.cuh::Nvfp4DecodeScale`, `shared_experts.cu`,
  `lsx_inference_nvfp4.cu` decode sites — inherit the full decode via
  `E4M3ScaleToFloat`.
- `repack_e8m0_ue4m3_kernel` — unchanged (`e4 = e8-120`, exact; `e4=15` already
  emitted for `e8>135`, valid `0x78=256`).

### 4.2 What does NOT change

- E2M1 nibble codec, packing order, global-scale handling.
- The zero test (`(b>>3)&0xF == 0`), and therefore `0x80..0x87` still decode 0.
- E8M0 → E4M3 repack paths (exact).
- Decode of every existing pow2 tree (bit-identical).
- No new kernels, no new streams: the same functions change implementation.

## 5. Testing (TDD, red first)

New host test `tests/test_nvfp4_scale.cpp` (label `fast`):

1. **exhaustive decode**: all 128 positive bytes against a hard-coded table;
2. **round-trip**: `E4m3ToFloat(FloatToE4m3(v)) ≈ v` within half a ULP for a
   sweep, and exact when `v` is representable;
3. **RNE points**: `1.0->0x38`, `3.0->0x44` (was `0x48` under pow2),
   `0.1`, `448->0x7E`, `449->0x7E`, `2^-6->0x08`, subnormal `->0x08`, `0->0x00`;
4. **pow2 identity**: for every `k`, `FloatToE4m3(2^k) == old pow2 byte`;
5. **NaN slot**: never emitted; `0x7F` decodes to NaN;
6. **field-zero contract**: `0x00..0x07 -> 0`, `0x80..0x87 -> 0`.

Device parity (`.cu`, label `fast`): kernel decodes all 128 bytes, matches the
host table bit-for-bit; kernel encodes the RNE points and matches
`float_to_e4m3`.

Update the tests that pin pow2 **encoder** output for non-pow2 inputs
(`test_lsx_kernels.cu` encoder case, `test_blackwell_tc_mma.cu` actquant parity,
`test_deepseek_moe_loader.cpp` subnormal/magnitude cases,
`test_deepseek_moe_inference_kernels.cu` magnitude-preservation). Tests that
hand-build pow2 **input** scales and only decode stay green unchanged — this is
the backward-compat regression signal.

Gate: full `fast` label green, then the affected `inference` selfcheck targets
on existing pow2 trees must be **bit-identical** (same `logits_hash`).

## 6. Rollout and model-lifecycle verification

After the `fast` label is green:

1. operator deletes the stale `logestix/` converted subdirs (the encoder output
   changed);
2. re-convert each family with its `lsxforge` flag;
3. re-run the full lifecycle per model — convert → loader → model → selfcheck →
   FSI/CODE/DET → perf — and write the per-model stage report the operator asked
   for (which target, pass/fail/skip, `logits_hash`, t/s);
4. re-record `tests/perf/perf_baselines.json` hashes for the models whose
   `kAuto` TC act-quant numerics move (all TC-path families);
5. FSI/CODE/DET goldens are captured from source inputs, so they should be
   unchanged or improved; any divergence is investigated, not re-captured away.

## 7. Skill and documentation updates

- `.opencode/skills/nvfp4-blackwell-conventions/SKILL.md` — NVFP4 section: the
  block scale is full unsigned E4M3 (mantissa-aware), the codecs are
  `FloatToE4m3`/`E4m3ToFloat`, RNE encode; drop the "pow2-only" statements.
- external mirror `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md`
  §1 — replace the "logestix deviation (deliberate)" paragraph with the now-native
  UE4M3 contract; keep the zero-field regression note (still the contract). Not
  committed to this repo.
- `AGENTS.md` — only if it references the codec (it does not today).
- `docs/OPEN_TASKS.md` section **L** — move to `docs/open-tasks/` when resolved.

## 8. Non-goals

- No change to the E2M1 encode tie flavors or the packing order.
- No new pow2-only encoder flavor (no caller needs one).
- No change to `E8m0ToE4m3Pow2Pair` / DeepSeek repack.
- No device-conditional numeric path (the codec is device-independent).

## 9. Risks

- **TC numerics move** on the `kAuto` path: hashes and perf baselines must be
  re-recorded; the golden *tokens* are gated by SelfCheck/FSI/CODE/DET.
- **Encoder boundary**: the `0x7F` NaN slot must be unreachable; covered by the
  exhaustive encode test.
- **Subnormals**: the original design mapped `e==0` to 0; superseded by the
  2026-10-07 addendum below (E4M3FN subnormals), covered by the exhaustive
  `0x00..0x07` decode + RNE encode + bit-7-mask tests.

## 10. Addendum 2026-10-07 — E4M3FN subnormals (supersedes the field-zero contract)

A scan of the real `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` checkpoint
(5 935 `F8_E4M3` `*.weight_scale` tensors, 1 886 631 936 scale bytes) found
**2 995 063 positive subnormal bytes** (`0x01..0x07`) and zero sign-bit / zero
`0x7F` bytes. The ModelOpt source therefore uses the E4M3FN subnormal range, and
the §3 “`e==0 -> 0`” contract silently zeroed those blocks. The canonical codec
now decodes the **complete E4M3FN value table** over the valid domain
`0x00..0x7F` (`e==0 -> m*2^-9`), encodes RNE on the subnormal grid, and masks
bit 7 (the reserved sign bit of the unsigned `float_ue4m3_t` hardware scale) —
bit-identical to `cuda_fp8` / `torch.float8_e4m3fn`. Backward compatible: no
existing byte changes value except `e==0, m!=0`, which no logestix-converted tree
emits (`FloatToE4m3` never produces it). The Nemotron dequant kernel was
simplified to call this canonical codec (its private `E4m3Full` / `E2m1` copies
are gone).
