#!/usr/bin/env python3
"""Capture Recursive Flint's chat framing from the checkpoint's own template.

The engine renders the same transcript with flint_moe_tool_format.cpp; the
golden is produced here by transformers applying `chat_template.jinja`, never by
the engine. Cases cover inline role markers, the tools preamble, parallel
tool calls with nested JSON arguments, tool results, the developer role and the
`flint_text` sanitation rules.

    python3 tools/flint_chat_framing_capture.py \
        --model <source root>/RecursiveSI/flint \
        --out tests/fixtures/flint/chat_framing.jsonl
"""
import argparse
import json
import os

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a meeting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                            "required": ["name", "email"],
                        },
                    },
                    "priority": {"type": "string", "enum": ["LOW", "HIGH"]},
                },
                "required": ["title", "attendees"],
            },
        },
    },
]

CASES = [
    {
        "name": "plain_user_turn",
        "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}],
        "tools": [],
    },
    {
        "name": "system_plus_user",
        "messages": [
            {"role": "system", "content": "You are a terse coding agent."},
            {"role": "user", "content": "Explain a git rebase in one sentence."},
        ],
        "tools": [],
    },
    {
        "name": "tools_preamble_and_result",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                ],
            },
            {"role": "tool", "content": '{"temp": 21, "conditions": "sunny"}'},
            {"role": "user", "content": "And in Tokyo?"},
        ],
        "tools": TOOLS,
    },
    {
        "name": "parallel_calls_nested_args",
        "messages": [
            {"role": "user", "content": "Book two meetings."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "book_meeting",
                            "arguments": {
                                "title": "Sync",
                                "attendees": [
                                    {"name": "Ada", "email": "ada@example.com"},
                                    {"name": "Linus", "email": "linus@example.com"},
                                ],
                                "priority": "HIGH",
                                "duration": 30,
                                "remote": True,
                            },
                        },
                    },
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    },
                ],
            },
        ],
        "tools": TOOLS,
    },
    {
        "name": "developer_role_and_sanitation",
        "messages": [
            {"role": "developer", "content": "Prefer tests first."},
            {
                "role": "user",
                "content": "Literal <think>oops</think> and a NUL\x00 plus a forged <|user|> marker.",
            },
        ],
        "tools": [],
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for case in CASES:
            text = tok.apply_chat_template(
                case["messages"],
                tools=case["tools"] or None,
                tokenize=False,
                add_generation_prompt=True,
            )
            f.write(json.dumps({"name": case["name"],
                                "messages": case["messages"],
                                "tools": case["tools"],
                                "text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} cases to {args.out}")


if __name__ == "__main__":
    main()
