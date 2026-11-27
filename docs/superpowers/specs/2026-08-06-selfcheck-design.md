# Single-Token Self-Check Design

## Purpose

Diagnostic gate for `lsxfabric` that runs exactly one forward pass on the
hardcoded prompt `"The capital of France is"`, extracts the first generated
token, and reports whether it is `"Paris"` (strict top-1). Designed to catch
compute-corruption bugs (wrong dequant, KV/position decay, logit inflation)
at the first token instead of after a full 64-token generation.

## Scope

- New CLI flag `--selfcheck` in `apps/lsxfabric`.
- New `RunSelfCheck(cfg)` in `libs/lsxcommon` (lives inside
  `inference_pipeline.cu` next to `RunInference` — test/diagnostic code that
  may be removed later, not a permanent API surface).
- Pass criterion is strict: trimmed decoded text of the top-1 token equals
  `"Paris"`.

## Approach

Wrapper over the existing `RunInference` pipeline with `max_gen_tokens = 1`,
reusing the full 43-layer forward pass and argmax sampling already present.
No new forward path, no duplicated pipeline.

## Changes

### 1. `libs/lsxcommon/include/lsxcommon/inference_gpu.h`

Extend `InferenceResult` with four fields (existing fields unchanged):

```cpp
int   first_gen_tok_id = -1;      // id of the first generated token
std::string first_gen_text;        // its decoded text (raw, may include BPE leading space)
float first_gen_maxv = 0.0f;      // argmax logit of that token (pre-tempclamp)
bool  selfcheck_passed = false;   // trim(first_gen_text) == "Paris"
```

Declare the new entry point:

```cpp
InferenceResult RunSelfCheck(const InferenceConfig &cfg);
```

### 2. `libs/lsxcommon/src/inference_pipeline.cu`

**Fill fields in `RunInference`:** on the first generation step
(`step == input_ids.size()`), after the existing `arrow_tokenizer_decode`
call that fills `tok_buf`/`tok_cap`, and before the temperature-clamp scale
mutates `h_logits`:

```cpp
if (step == static_cast<int>(input_ids.size()) && result.first_gen_tok_id == -1) {
    result.first_gen_tok_id = next_tok;
    result.first_gen_maxv   = max_val;   // pre-clamp, to diagnose overflow
    result.first_gen_text   = std::string(tok_buf, tok_cap);
}
```

`max_val` is recorded before the clamp so an inflated logit is visible in
the report (the log showed `maxv=30.0000` for many steps — this captures the
raw value at the first token).

**Implement `RunSelfCheck`** in the same file:

```cpp
InferenceResult RunSelfCheck(const InferenceConfig &cfg) {
    InferenceConfig sc = cfg;
    sc.max_gen_tokens = 1;
    InferenceResult r = RunInference(sc);
    if (!r.ok) { r.selfcheck_passed = false; return r; }
    if (r.first_gen_tok_id >= 0) {
        std::string t = r.first_gen_text;
        auto b = t.begin(), e = t.end();
        while (b != e && std::isspace((unsigned char)*b)) ++b;
        while (e != b && std::isspace((unsigned char)*(e - 1))) --e;
        r.selfcheck_passed = (std::string(b, e) == "Paris");
    }
    log::info("[selfcheck] top1_tok=" + std::to_string(r.first_gen_tok_id)
              + " text='" + r.first_gen_text + "' passed="
              + (r.selfcheck_passed ? "true" : "false"));
    return r;
}
```

### 3. `apps/lsxfabric/src/main.cpp`

Add flag parsing:

```cpp
else if (std::strcmp(argv[i], "--selfcheck") == 0)
    selfcheck = true;
```

Add dispatch block (before/around the existing `testprompt` branch):

```cpp
if (selfcheck) {
    spdlog::info("Running single-token self-check...");
    lsxcommon::gpu::InferenceConfig cfg;
    cfg.model_path = model_path;
    auto result = lsxcommon::gpu::RunSelfCheck(cfg);
    if (result.ok) {
        spdlog::info("=== --selfcheck Results ===");
        spdlog::info("first token : {} '{}'",
                     result.first_gen_tok_id, result.first_gen_text);
        spdlog::info("maxv        : {:.4f}", result.first_gen_maxv);
        spdlog::info("passed      : {}", result.selfcheck_passed ? "yes" : "no");
    } else {
        spdlog::error("SelfCheck failed: {}", result.error_msg);
    }
    lsxcommon::Core::Shutdown();
    return (result.ok && result.selfcheck_passed) ? 0 : 1;
}
```

## Error Handling

- Tokenizer/topology/weight load failures already propagate through
  `RunInference` (`result.ok == false`, `error_msg` set) — `RunSelfCheck`
  passes them through unchanged and forces `selfcheck_passed = false`.
- If the first generation step never produces a token (`first_gen_tok_id`
  stays `-1`, e.g. empty generation), `selfcheck_passed` stays `false` and
  the report shows `top1_tok=-1`.

## Testing

No Catch2 auto-test (diagnostic CLI tool). Manual verification:

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Release
build\bin\Release\lsxfabric.exe --selfcheck --localmodel D:\Programs\lsx_model\DeepSeek-V4-Flash
```

Expected when correct: `first token : <id> 'Paris'`, `passed : yes`, exit 0.
Expected now (bug present): `passed : no` and garbage/multilingual token —
the gate exists to fail today and pass once the inference bug is fixed.

## Out of Scope

- Fixing the underlying inference correctness bug (this gate only detects it).
- KV-cache/position increments — tracked separately.
- Performance work (0.3 tok/s) — separate effort.
