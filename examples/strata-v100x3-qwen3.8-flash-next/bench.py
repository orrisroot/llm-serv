#!/usr/bin/env python3
"""Minimal OpenAI-compatible benchmark for an llm-serv inference server.

Measures, per one request:
  - prompt tokens/s  = prompt_tokens / TTFT        (prefill speed)
  - decode tokens/s  = completion_tokens / (total - TTFT)
Streaming is used so TTFT can be measured. Stdlib only.

Usage:
  python3 bench.py --url http://127.0.0.1:8000/v1 --key SECRET \
                   --prompt-tokens 4096 --output-tokens 256 --rounds 3 \
                   --effort none
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("The quick brown fox jumps over the lazy dog near the river bank "
         "while a curious cat watches from a sunny windowsill").split()


def make_prompt(n_tokens: int) -> str:
    # ~0.75 tokens per word for English; pad generously.
    words = []
    rng = random.Random(42)
    while len(words) < int(n_tokens * 0.9):
        words.append(rng.choice(WORDS))
    return " ".join(words)


def stream_chat(url: str, api_key: str, model: str, prompt: str, max_tokens: int,
                effort: str, temperature: float = 0.0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
    }
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        start = time.monotonic()
        ttft = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            now = time.monotonic()
            if ttft is None:
                ttft = now - start
            choices = chunk.get("choices") or []
            for ch in choices:
                delta = ch.get("delta") or {}
                txt = delta.get("content") or delta.get("reasoning_content") or ""
                if txt:
                    del txt  # characters are not tokens; usage comes later
        total = time.monotonic() - start
    return ttft, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="strata",
                    help="model id the server accepts (any non-empty id works)")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="none",
                    help="reasoning_effort: none|low|medium|high")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    prompt = make_prompt(args.prompt_tokens)
    # warmup (also loads the model / builds caches)
    for _ in range(args.warmup):
        stream_chat(args.url, args.key, args.model, "hi", 8, args.effort)

    results = []
    for i in range(args.rounds):
        ttft, total = stream_chat(args.url, args.key, args.model, prompt,
                                  args.output_tokens, args.effort)
        # token counts come from a second, non-streaming call to keep it cheap:
        results.append((ttft, total))

    # Non-streaming call for token counts (usage)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.output_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    if args.effort:
        body["reasoning_effort"] = args.effort
    req = urllib.request.Request(
        args.url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {args.key}"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        usage = json.load(resp)["usage"]
    ptok = usage["prompt_tokens"]
    ctok = usage["completion_tokens"]

    print(f"prompt_tokens={ptok} completion_tokens={ctok}")
    for i, (ttft, total) in enumerate(results):
        pps = ptok / ttft if ttft else float("nan")
        dts = ctok / max(total - ttft, 1e-6)
        print(f"round {i+1}: TTFT={ttft:.2f}s total={total:.2f}s "
              f"prompt={pps:.0f} tok/s decode={dts:.1f} tok/s")
    good = [(r[0], r[1]) for r in results]
    if good:
        ttft = min(t for t, _ in good)
        total = sum(t for _, t in good) / len(good)
        # report the best-prompt and mean-decode figures
        pps = ptok / ttft
        dts = ctok / (total - (sum(t for t, _ in good) / len(good)))
        print(f"summary: prompt(best)={pps:.0f} tok/s "
              f"decode(mean)={dts:.1f} tok/s")


if __name__ == "__main__":
    main()
