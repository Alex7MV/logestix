# mimo weight upload: the host-side BF16→FP16 pass (E3, closed 2026-10-17)

Item **§E3** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), the last purely technical
item in the deferred-decision cluster, closed on branch
`audit/lifecycle-abort-contract`. Kept verbatim with the resolution appended.

The three deferred §E items that needed a product or numerics decision — §E1
(concurrent multi-model dispatch), §E2 (which logits row a family publishes) and
§E4 (temperature/top_p in the OpenAI bridge) — are **still open and deliberately
so**; they are not defects and not this snapshot's business. E3 was different: it
was real work sitting on a known slower path, with an existing neutral kernel to
route it through.

---

### E3. `mimo` loads its weights through a host-side BF16→FP16 conversion
- `UploadHalf` (`mimo_moe_model.cpp:47`) decodes every BF16 tensor into a
  temporary host `std::vector`, then issues a synchronous pageable
  `cudaMemcpy` on the default stream — `(void)st;` ignores the model's stream.
  For ~11 GiB of weights this is slow and bypasses the persistent context.
- Correctness is fine and MiMo's perf gate is withdrawn (vendor checkpoint
  defect, A2), so this is a load-time cost only. Fixing it properly means a
  device-side BF16→FP16 kernel + stream-ordered upload, i.e. the same fast-load
  path GLM uses (`glm_moe_load.cu::GlmBf16ToHalfInPlace`).


---

## Resolution (2026-10-17)

`UploadHalf` decoded every BF16 element on the host into a temporary
`std::vector<__half>` and then issued a synchronous pageable `cudaMemcpy` on the
**default** stream — `(void)st;` ignored the model's stream entirely. Two costs
for ~11 GiB of weights: a serial host pass per tensor, and a full pipeline flush
per copy, both bypassing the persistent context.

Now the raw BF16 bits are copied verbatim (stream-ordered) and converted in place
on the device by the model-neutral kernel that GLM and the DeepSeek fast loader
already use, `fastload::Bf16ToHalfInPlace` (`bf16_convert.cu`). No host staging
buffer, no default-stream copy.

**No value moves.** The kernel is a bit-exact mirror of the host loop the code
used to run — bf16 bits `<<16` reinterpreted as FP32, rounded to nearest-even
FP16 — and that equivalence is not assumed here, it is already pinned:
`tests/test_deepseek_moe_load_fast.cu` carries "fastload Bf16ToHalfInPlace
bit-exact vs host reference" and a ragged-tail case. Those are the same assertions
DeepSeek relies on; MiMo now inherits them.

`UploadBf16` and `UploadF32` had the same `(void)st;` defect for the same reason
(pageable default-stream synchronous copies). Their bytes are already final, so
they only needed to become stream-ordered (`cudaMemcpyAsync` on `st`). All three
`(void)st;` casts are gone.

**Verified 2026-10-17** (RTX 5090/Linux, `linux-debug`): the five MiMo model-stage
targets green (`model`, `convert`, `loader`, `kernels`, `selfcheck` — the last
SKIPs, no converted checkpoint here), all four `*_load_fast` targets green, fast
suite 94/94.
