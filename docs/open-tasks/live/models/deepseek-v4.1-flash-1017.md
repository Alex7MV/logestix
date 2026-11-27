# DeepSeek-V4.1-Flash-1017 — open items

Sibling of [`deepseek-v4.1-flash`](deepseek-v4.1-flash.md) (stage-0 triage
2026-11-22: byte-identical `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `inference/config.json` and
`model.safetensors.index.json`; re-trained payload bytes). One converter, one
loader, one kernel set and one generation body serve both; this row owns a
re-declared passport (`lsxcommon::deepseek_v41_flash1017`), its own registry id
(`deepseek-v4.1-flash-1017`) and its own gates.

Converted 2026-11-22 into `/mnt/fast_storage/lsx_model/deepseek-ai/DeepSeek-V4.1-Flash-1017/logestix`
(294 GB, `logestix-v41-nvfp4`, 1246 pinned tensors, 15360 experts, deferred
`engram=12 mtp=2401 vision=266` — the parent's counts exactly).

### S1. CODE 2/3: CODE-03 is a false red from a literal anchor (instruments `I1`)

- **Status: open, not a model defect.**
- Measured 2026-11-22 on `linux-release-znver4-aocc`, one GPU, serial:
  `CODE-TEST-01` pass (`{"root":{"nodes":{"leaf":true}}}`), `CODE-TEST-02` pass
  (`m=l+((h-l)/2)`), `CODE-TEST-03` **fail** — `expected='\b[0-9A-F]{4}\b'`,
  `extracted=''`, while the raw final answer is `\b[A-F0-9]{4}\b`.
- Two semantically identical PCRE expressions, different member order.
  `ExtractRegex` finds its answer by the literal anchor `[0-9A-F]`, so it returns
  an empty string and the gate reports a form failure.
- Not fixed here on purpose: the anchor is shared by all 43 rows' CODE gates, so
  canonicalizing it is an instruments change with its own corpus-wide
  re-measurement. Filed as `I1` in
  [`instruments-and-guards.md`](../instruments-and-guards.md).
- The parent fails CODE-03 too, for an unrelated reason (a stray `?` appended to
  a correctly ordered class), so the two rows' CODE verdicts must not be read as
  the same defect.

### S2. Goldens are template-derived, so they pin the converted path only

- **Status: open, no local fix.**
- The 552B FP8/FP4 tilelang reference (`inference/generate.py`, 2×80 GB target)
  cannot execute on one 32 GB card, so `tests/fixtures/{fsi,code,det}/deepseek-v4.1-flash-1017.json`
  carry the prompt set's Expected Template as `expected` (captured with
  `tools/{fsi,code,det}_capture.py`, framed through the release's own
  `encoding/encoding.py` in thinking mode).
- Provenance is machine-checked rather than asserted: re-running the capture for
  the PARENT reproduces its committed goldens byte for byte — `input_ids` and
  `expected`, all three kinds — which is exactly the procedure the sibling's
  goldens were produced with.
- A source-faithful golden needs the vendor stack on 2×80 GB.

### S3. Throughput sits at the parent's shape

- **Status: measured, no action pending.** 16.27 t/s (`perf` and `r3`), TTFT
  488 ms, 231 W average at the canonical 2048-token protocol. The parent's `perf`
  row is 16.43 — identical geometry, re-trained weights, same expert-store
  residency path (`954 slots × 19.9 MiB`, hit rate 0.35 on the Paris probe).
  The parent's `r3` row (7.7, 2026-09-23) is stale and is **not** copied here;
  this row seeds r3 at the measured value.
