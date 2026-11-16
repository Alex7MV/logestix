# Arrow read-path matrix — model → code → read path

Machine-checked extract of the 2026-11-01 Apache Arrow usage audit. The
full audit draft (internal, Russian-language) was removed before
open-sourcing; this matrix is the part `tests/test_arrow_usage_guard.cpp`
reads (rule R-A8): every adapted model must have exactly one row, the
`verdict` must be a known read-path class, and every converter/loader TU a
row names must exist in the source tree. Adding a model without a row here
is a test failure, not a silent hole. The audit's findings and their
resolutions are recorded in English in
`docs/open-tasks/2026-11-01-arrow-audit-and-arena-decision.md`.

`verdict` is the host-storage class of the routed experts (dense models are
marked `views+dense`). Pinned tensors in every family are read through the
zero-copy view helpers; the former gemma exception (audit finding A-07) was
fixed by moving gemma to views.

The `Kimi-K3.1-Pro` row was added after the audit itself (2026-11-05, with
the variant): it reuses the `kimi_moe_k3_load.cpp` loader and adds only an
FP16 indexer plane to it, so no new read path appears. The other rows are
the 2026-11-01 snapshot.

<!-- arrow-matrix:start -->
| model | converter | loader | read path | write path | verdict |
|---|---|---|---|---|---|
| AliceAI-Foundation-80B-A3B-Base | alice_moe_convert.cpp | alice_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Beam-1.0-501B-A23B | beam_moe_convert.cpp | beam_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| DeepSeek-V4-Flash | deepseek_moe_repack.cpp | deepseek_moe_load.cu | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| DeepSeek-V4-Pro-0813 | deepseek_moe_pro_convert.cpp | deepseek_moe_pro_load.cu | views + neutral ExpertStore (mmap-stream) | schema.cpp | views+ExpertStore |
| DeepSeek-V4.1-Flash | deepseek_moe_v41_convert.cpp | deepseek_moe_v41_load.cu | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| GLM-5.3-Flash | glm_moe_flash_convert.cpp | glm_moe_flash_load.cpp | unpack + pinned arena | schema.cpp | unpack+pinned-arena |
| GigaChat3.5-432B-A28B-Reasoning | gigachat_moe_convert.cpp | gigachat_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Hy-Memory-30B-A3B | hy_moe_convert.cpp | hy_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| Hy-SimulMT-30B-A3B | hy_moe_simulmt_convert.cpp | hy_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| Kimi-K2.7-Code | kimi_moe_convert.cpp | kimi_moe_load.cpp | views + PinChunked | schema.cpp | views+PinChunked |
| Kimi-K3 | kimi_moe_k3_convert.cpp | kimi_moe_k3_load.cpp | views+ExpertStore (+ transient pinned unpack) | schema.cpp | views+dense |
| Kimi-K3.1-Pro | kimi_moe_pro_convert.cpp | kimi_moe_k3_load.cpp | views+ExpertStore (+ transient pinned unpack, + FP16 indexer plane) | schema.cpp | views+ExpertStore |
| Ling-3.1-flash | ling_moe_convert.cpp | ling_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| MiniMax-M3 | minimax_moe_convert.cpp | minimax_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| MiniMax-M3.1-Flash | minimax_moe_flash_convert.cpp | minimax_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Mistral-Small-4-119B-2603-NVFP4 | mistral_moe_convert.cpp | mistral_moe_load.cpp | views + neutral ExpertStore (shared: unpack) | schema.cpp | views+ExpertStore |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | nemotron_moe_convert.cpp, nemotron_moe_nvfp4_convert.cpp | nemotron_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Ornith-1.5-35B-A3B | ornith_moe_convert.cpp | ornith_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Qwen3.5-35B-A3B | qwen_moe_convert.cpp | qwen_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Qwen3.5-9B | qwen_dense_convert.cpp | qwen_inference_load.cpp | views + dense | schema.cpp | views+dense |
| Qwen3.6-35B-A3B | qwen_moe_36_convert.cpp | qwen_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Qwen3.8-27B | qwen_dense_convert.cpp | qwen_inference_load.cpp | views + dense | schema.cpp | views+dense |
| Qwen3.8-Flash-Next | qwen_moe_exp_convert.cpp | qwen_moe_exp_load.cpp | unpack + pinned arena | schema.cpp | unpack+pinned-arena |
| Step-5-Preview | step_moe_convert.cpp | step_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Xing4.0-29B-A4B | xing_moe_convert.cpp | xing_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| flint | flint_moe_convert.cpp | flint_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| gemma-4-26B-A4B-it | gemma_moe_convert.cpp | gemma_moe_load.cpp | views + PinChunked (dense triple: views) | schema.cpp | views+PinChunked |
| gemma-4-31B-it | gemma_dense_convert.cpp | gemma_dense_load.cpp | views + dense (triple: views) | schema.cpp | views+dense |
| granite-4.2-30b | granite_dense_convert.cpp | granite_dense_load.cpp | views + dense | schema.cpp | views+dense |
| granite-switch-4.1-30b | granite_dense_switch_convert.cpp | granite_dense_switch_load.cpp | views + dense (+LoRA planes) | schema.cpp | views+dense |
| phi-4 | phi_dense_convert.cpp | phi_dense_load.cpp | views + dense | schema.cpp | views+dense |
<!-- arrow-matrix:end -->
