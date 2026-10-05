#!/usr/bin/env python3
"""Kolibri-1 chat-template parity: ds4 vs the released template's renderer.

Renders conversations with the released Jinja template (from the fixture
GGUF's tokenizer.chat_template metadata, HF-compatible environment) and
encodes the rendered strings with the HuggingFace tokenizers BPE (an
independent implementation). Each case must match ds4's chat prompt
tokenization id-for-id, special tokens included.
"""
import json
import os
import pathlib
import subprocess
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from kolibri1_reference import load_gguf, tensor_array  # noqa: E402

GGUF = ROOT / "gguf" / "Kolibri-1-mini.gguf"
DS4 = str(ROOT / "ds4")

meta, tensors, blob, data_start = load_gguf(str(GGUF))
template_src = meta["tokenizer.chat_template"]
if isinstance(template_src, bytes):
    template_src = template_src.decode()

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
template = env.from_string(template_src)

try:
    from tokenizers import Tokenizer
except ImportError:
    print("kolibri1-chat: tokenizers package required (pip install tokenizers)")
    sys.exit(2)

snapshot = os.environ.get("KOLIBRI1_TOKENIZER_DIR")
if snapshot and (pathlib.Path(snapshot) / "tokenizer.json").exists():
    tok = Tokenizer.from_file(str(pathlib.Path(snapshot) / "tokenizer.json"))
else:
    # fall back to the mini fixture: it embeds the full metadata, but the
    # GGUF only carries the token strings, so we need tokenizer.json's
    # added-token matching from the snapshot directory.
    print("kolibri1-chat: KOLIBRI1_TOKENIZER_DIR with tokenizer.json required")
    sys.exit(2)


def ds4_ids(rendered: str) -> list[int]:
    proc = subprocess.run(
        [DS4, "-m", str(GGUF), "--dump-tokens", "-p", rendered],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ds4 failed: {proc.stderr.strip()}")
    first = proc.stdout.splitlines()[0]
    return [int(t) for t in first.strip().strip("[]").split(",")]


CASES = [
    ("user-only-default",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "high"}),
    ("system+user-default",
     [{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Hello there"}],
     {}),
    ("nothink",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "none"}),
    ("effort-low",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "low"}),
    ("effort-medium",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "medium"}),
    ("system+effort-medium",
     [{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Hello"}],
     {"reasoning_effort": "medium"}),
    ("tool-results-grouped",
     [{"role": "user", "content": "check both"},
      {"role": "assistant", "content": "", "tool_calls": [
          {"name": "ls", "arguments": {"path": "."}},
          {"name": "cat", "arguments": {"path": "a b/c"}}]},
      {"role": "tool", "name": "ls", "content": "a\nb"},
      {"role": "tool", "name": "cat", "content": "line1"},
      {"role": "user", "content": "summarize"}],
     {}),
    ("assistant-replay",
     [{"role": "user", "content": "one"},
      {"role": "assistant", "content": "answer one"},
      {"role": "user", "content": "two"}],
     {}),
    ("assistant-final-with-reasoning",
     [{"role": "user", "content": "one"},
      {"role": "assistant", "content": "answer one",
       "reasoning": "because 2+2=4"},
      {"role": "user", "content": "two"}],
     {"preserve_thinking": True}),
]


def main() -> int:
    failures = 0
    for name, messages, kwargs in CASES:
        rendered = template.render(messages=messages,
                                   add_generation_prompt=True, **kwargs)
        ref_ids = tok.encode(rendered).ids
        got = ds4_ids(rendered)
        if got == ref_ids:
            print(f"kolibri1-chat: {name:32s} {len(got):4d} tokens OK")
            continue
        failures += 1
        print(f"kolibri1-chat: {name:32s} MISMATCH "
              f"(ds4 {len(got)} vs ref {len(ref_ids)})")
        for i, (g, r) in enumerate(zip(got, ref_ids)):
            if g != r:
                lo = max(0, i - 3)
                print(f"    first diff at {i}: ds4 "
                      f"{got[lo:i+3]} ref {ref_ids[lo:i+3]}")
                break
        else:
            print(f"    length differs after {min(len(got), len(ref_ids))}")
    if failures:
        print(f"kolibri1-chat: {failures} case(s) FAILED")
        return 1
    print("kolibri1 chat-template parity tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
