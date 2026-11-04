# Qwen family framing: the two "3.8 deltas" were family-wide (2026-11-04)

Branch `fix/qwen-template-fidelity`, commit `b3eaa066`. Closes the second §R16
item (`Open (framing, latent for the whole qwen family)`) that the
[Qwen3.8-Flash-Next lifecycle session](2026-11-04-qwen38-flash-next-lifecycle.md)
left behind. The first §R16 item — the serialized 5.58 ms/token expert H2D —
stays open in `docs/OPEN_TASKS.md` §R16; it is a kernel-ordering change, not
framing.

## What the previous session got wrong

It recorded three byte-level 3.8 template deltas and carried all three behind a
new `QwenRenderPolicy`, with the older members left on their old wire:

1. the closed think block on every assistant history turn,
2. jinja `tojson` tool serialisation,
3. `render_content(...)|trim`.

Deltas (2) and (3) are **not** 3.8-only. `diff Qwen3.6-35B-A3B/chat_template.jinja
Qwen3.8-Flash-Next/chat_template.jinja` shows both lines identical; so does a
`grep -c 'render_content.*|trim'` / `grep -c 'tool | tojson'` sweep over all six
checkpoints in the vault (`Qwen3.5-9B`, `Qwen3.5-35B-A3B`, `Qwen3.6-35B-A3B`,
`Qwen3.8-27B`, `Ornith-1.5-35B-A3B`, `Qwen3.8-Flash-Next`): 4 trim sites and 1
tojson site each. What the 3.8 template genuinely adds is `reasoning_effort`, a
`preserve_thinking` default of true, and an empty-system-turn guard
(`{%- if content %}` around the tool-less system turn).

So the shared codec had been emitting the compact tool JSON and the untrimmed
content for **every** qwen member, and the "keep the older members byte-unchanged"
plan would have frozen that deviation in place. Worse, the other direction: the
3.8 member applied `trim_content` at every `esc()` call site, including tool
names and descriptions, which the template renders through `tool | tojson` —
verbatim.

## Oracle (external, not our renderer)

`transformers 5.15.1`, `apply_chat_template(..., tokenize=False,
add_generation_prompt=True, enable_thinking=False)` on the checkpoints' own
`chat_template.jinja`, with a deliberately padded conversation:

```
system "  You are a helper.  \n"  ->  "<|im_start|>system\nYou are a helper.<|im_end|>\n"
user   "\n  Weather in Paris?\n " ->  "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
tool description "  Get the weather  \n" ->  "description": "  Get the weather  \n"
```

Both the 3.6 and the 3.8 tokenizer agree: content is trimmed, tool definitions
are not. Empty system turn, same oracle:

| system content | 3.8-flash-next | 3.6-35B-A3B |
|---|---|---|
| `""` | `<|im_start|>user\nhi…` (no system turn) | `<|im_start|>system\n<|im_end|>\n<|im_start|>user…` |
| `"   "` | same as above | same as above |

## Change

* **`trim_content` and `jinja_tool_json` are gone** as policy fields. Trimming and
  jinja-tojson are unconditional in `qwen_tool_format.cpp`, because no qwen
  template makes them configurable — a flag implied a choice that does not exist,
  and the previous session's flag then shipped switched off for nine members.
* Trim is scoped to message content (`esc_content`). Tool definitions, parameter
  names and argument values go through `esc_verbatim` (control-marker
  neutralisation only), which is what `tool | tojson` does.
* `content|trim` also decides the separator before the first `<tool_call>`: the
  template tests `content|trim`, so whitespace-only assistant content renders as
  empty here too.
* `QwenRenderPolicy` keeps exactly the three genuinely per-family fields:
  `enable_thinking`, `history_think_block` (3.8 `preserve_thinking` default) and
  the new `suppress_empty_system` (the 3.8 `{%- if content %}` guard).

## Evidence

| instrument | result |
|---|---|
| `tests/test_qwen_tool_framing.cpp` | **102 assertions / 15 cases** (was 86/11); four new cases — trimmed content, tojson tool wire, untrimmed tool description, template-dependent empty system turn — each written red first against the oracle text above. One legacy assertion that had pinned the compact wire (`"name":"get_weather"` → `"name": "get_weather"`) was updated |
| `tests/test_qwen_moe_exp_model.cpp` | 9 assertions / 3 cases; pins the policy the model renders with (`history_think_block`, `suppress_empty_system`, `enable_thinking == false`) |
| real-weight tool round-trip, six members | all `get_weather {"city":"Paris"}` on the new wire: qwen3.6 (`42/3`), qwen3.8-27b (`30/2`), ornith (`58/3`), qwen3.8-flash-next (`44/2`), **plus two that had none** — qwen3.5-9b (`30/2`, `a2af669548e59c73`) and qwen3.5-35b-a3b (`42/3`, `f4c91be5f2206d56`) |
| SelfCheck `logits_hash` after the change | unchanged: `2e3bcd090441ad65` (3.8-flash-next), `8bf2a2bf937eca91` (3.8-27b), `c3fe24e9dbc55b1e` (ornith), `a2af669548e59c73` (3.5-9b), `f4c91be5f2206d56` (3.5-35b-a3b) |
| `ctest --preset linux-debug -L fast` | **134/134** |

The parity gates could not have caught either deviation: FSI/CODE/DET consume
pre-tokenised `input_ids` from the goldens (`req.input_ids = c.input_ids` in
`fsi_selfcheck.cpp:493`, `code_selfcheck.cpp:369`, `det_selfcheck.cpp:258`), so
the renderer is not in their path at all, and none of the nine capture prompts
(`tools/{fsi,code,det}_prompts.json`) carries surrounding whitespace or tools.

## The instrument gap this closes

qwen3.5-9b and qwen3.5-35b-a3b had **no real-weight tool round-trip** at all,
while their framing was being changed. Both now have one. That is the missing
instrument, not a missing assertion: a codec unit test cannot show that a
particular model's `BuildConversationInputIds` wires the codec correctly, and the
goldens never touch it.
