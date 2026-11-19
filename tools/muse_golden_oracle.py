#!/usr/bin/env python3
"""Muse-Glimmer-30B text-tower oracle for golden capture.

The released checkpoint is a multimodal wrapper
(MuseGlimmerForConditionalGeneration) whose 59.5 GB BF16 tree does not fit the
32 GB card. The text tower alone (muse_glimmer_text, ~55 GB BF16) runs on the
750 GB host CPU. This oracle loads ONLY the language-model + lm_head weights
into a text-config wrapper, skipping the vision tower / adapter / projection
(the engine strips the same 809 tensors as ballast). Greedy generation through
the standard GenerationMixin, framed by the checkpoint's chat_template.jinja.

Used by tools/{fsi,code,det}_capture.py via OVERRIDES loader="muse_text".

This module is also the single place that knows the checkpoint's ATEM framing
facts a capture needs:

* `TEMPLATE_KWARGS` — the chat template's `reasoning_strength` switch. The
  template renders `Reasoning strength: <value>.` into the system block and
  defaults to `high`; at `high` the model spends the whole token budget inside
  `assistant to=self` reasoning and re-emits its answer in a loop, which is why
  the first FSI capture never produced a clean answer line (docs/OPEN_TASKS.md
  R21). `low` is the documented off switch.
* `ATEM_ANSWER_CHANNEL` / `ATEM_REASONING_CHANNEL` — the decoded channel
  headers. The `<|start|>`/`<|message|>`/`<|eom|>` markers are special tokens
  (decode drops them), so channel headers survive as literal text.
* `select_answer_channel()` — the capture-side twin of the C++ harness rule
  (`fsi::SelectAtemAnswerChannel`): the FIRST COMPLETE answer channel, never
  the reasoning tail and never the last (budget-truncated) answer.

The torch/transformers imports are lazy so this module can be imported by
`tools/fsi_capture.py --selftest` (and by the pure-string tests) on a host
without the checkpoint stack.
"""
import json
import os

# ATEM channel headers as they appear in DECODED text (markers are special
# tokens, so they are dropped by skip_special_tokens=True).
ATEM_ANSWER_CHANNEL = "assistant to=user"
ATEM_REASONING_CHANNEL = "assistant to=self"

# Chat-template variables the capture must pass. `reasoning_strength: low` is
# rendered as `Reasoning strength: low.` in the system block; the template's
# default is `high`.
TEMPLATE_KWARGS = {"reasoning_strength": "low"}


def capture_opts():
    """Capture overrides contributed by this loader (merged by the capture
    tools after the loader module is imported)."""
    return {
        "template_kwargs": dict(TEMPLATE_KWARGS),
        "answer_channel": ATEM_ANSWER_CHANNEL,
        "reasoning_channel": ATEM_REASONING_CHANNEL,
        # The reasoning model can keep talking past its answer, but the answer
        # channel is complete long before the budget ends; the full budget is
        # what the verifier replays, so pin it.
        "ignore_eos": True,
    }


def select_answer_channel(text, answer_channel=ATEM_ANSWER_CHANNEL,
                          reasoning_channel=ATEM_REASONING_CHANNEL):
    """Return the FIRST COMPLETE answer channel of an ATEM stream.

    Must stay behaviourally identical to
    `lsxcommon::atem::SelectAtemAnswerChannel` (libs/lsxcommon/src/atem_channel.cpp)
    so the golden's `expected` is the answer the C++ harness would extract from
    the same raw text. A channel is COMPLETE when another channel header
    follows it; the last channel in the stream can be cut by the token budget.
    Falls back to the last (possibly truncated) answer channel, then to the
    input unchanged.
    """
    if not answer_channel:
        return text
    answer = ""
    pos = 0
    while pos < len(text):
        a = text.find(answer_channel, pos)
        if a == -1:
            break
        body = a + len(answer_channel)
        end = len(text)
        nxt = text.find(answer_channel, body)
        if nxt != -1:
            end = min(end, nxt)
        if reasoning_channel:
            nxt = text.find(reasoning_channel, body)
            if nxt != -1:
                end = min(end, nxt)
        if end < len(text) and not answer:
            answer = text[body:end]
        pos = end
    if answer:
        return answer
    last = text.rfind(answer_channel)
    if last != -1:
        return text[last + len(answer_channel):]
    return text


def _text_causal_lm_class():
    """Build the text-only causal LM class (imports the checkpoint stack)."""
    import torch
    from torch import nn
    from transformers import GenerationMixin
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerPreTrainedModel,
        MuseGlimmerTextModel,
    )

    class _MuseGlimmerTextForCausalLM(MuseGlimmerPreTrainedModel,
                                      GenerationMixin):
        """Text-only causal LM: language_model + lm_head, no vision tower."""

        _tied_weights_keys = {}
        accepts_loss_kwargs = False

        def __init__(self, config):
            # config is the FULL MuseGlimmerConfig; the text tower reads
            # text_config.
            super().__init__(config)
            self.model = MuseGlimmerTextModel(config.text_config)
            self.lm_head = nn.Linear(config.text_config.hidden_size,
                                     config.text_config.vocab_size,
                                     bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        def forward(self, input_ids=None, attention_mask=None,
                    position_ids=None, past_key_values=None, inputs_embeds=None,
                    use_cache=None, logits_to_keep=0, **kwargs):
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
            hidden = outputs.last_hidden_state
            slice_indices = (slice(-logits_to_keep, None)
                             if isinstance(logits_to_keep, int)
                             else logits_to_keep)
            logits = self.lm_head(hidden[:, slice_indices, :])
            # The wrapper's softcap (output_multiplier then tanh) lives here too.
            tc = self.config.text_config
            mult = getattr(tc, "output_multiplier", None)
            cap = getattr(tc, "final_logit_softcapping", None)
            if mult is not None:
                logits = logits * mult
            if cap is not None:
                logits = cap * torch.tanh(logits / cap)
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=logits, past_key_values=outputs.past_key_values)

    return _MuseGlimmerTextForCausalLM


def _load_text_weights(model, path):
    """Read only model.language_model.* + lm_head.* from the safetensors shards."""
    from safetensors import safe_open
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = {}
    state = model.state_dict()
    loaded = 0
    # Map checkpoint names -> wrapper names: the wrapper's text tower is
    # self.model (MuseGlimmerTextModel), so checkpoint
    # "model.language_model.X" -> "model.X"; "lm_head.weight" stays.
    for ck_name, shard in weight_map.items():
        if ck_name.startswith("model.language_model."):
            local = "model." + ck_name[len("model.language_model."):]
        elif ck_name == "lm_head.weight":
            local = ck_name
        else:
            continue  # vision ballast
        if local not in state:
            continue
        if shard not in shards:
            shards[shard] = safe_open(os.path.join(path, shard), framework="pt")
        tensor = shards[shard].get_tensor(ck_name)
        state[local].copy_(tensor.to(state[local].dtype))
        loaded += 1
    print(f"muse_text oracle: loaded {loaded} text tensors "
          f"(skipped {len(weight_map) - loaded} vision/other)")
    return model


def load(model_path, device_map=None):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    # eager attention: the CPU reference, no flash/sdpa kernel variance.
    cfg._attn_implementation = "eager"
    if getattr(cfg, "text_config", None) is not None:
        cfg.text_config._attn_implementation = "eager"
    model = _text_causal_lm_class()(cfg)
    model = _load_text_weights(model, model_path)
    model.eval()
    model = model.to("cpu")
    return model
