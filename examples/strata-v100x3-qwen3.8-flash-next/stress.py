#!/usr/bin/env python3
"""Concurrent OpenAI-compatible throughput benchmark (stdlib only).

Usage:
  python3 stress.py --url http://127.0.0.1:8000/v1 --key SECRET \
                    --workers 6 --requests 12 --prompt-tokens 1024 --output-tokens 128
"""
import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request

WORDS = ("The lazy fox sleeps beneath a quiet maple tree while distant hills "
         "catch the evening light and birds drift home across the fields").split()


def make_prompt(n_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words = []
    while len(words) < int(n_tokens * 0.9):
        words.append(rng.choice(WORDS))
    return " ".join(words)


def call(url, key, model, prompt, max_tokens, temperature=0.0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": temperature,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.load(resp)
    dt = time.monotonic() - t0
    usage = data.get("usage") or {}
    return dt, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="strata",
                    help="model id the server accepts (any non-empty id works)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=128)
    args = ap.parse_args()

    prompts = [make_prompt(args.prompt_tokens, 1000 + i) for i in range(args.requests)]
    # warmup/load
    call(args.url, args.key, args.model, prompts[0], 4)
    call(args.url, args.key, args.model, prompts[0], 4)

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(call, args.url, args.key, args.model,
                          prompts[i % len(prompts)], args.output_tokens)
                for i in range(args.requests)]
        for f in concurrent.futures.as_completed(futs):
            jobs.append(f.result())

    lats = [dt for dt, _, _ in jobs]
    ptok = sum(p for _, p, _ in jobs)
    ctok = sum(c for _, _, c in jobs)
    span = max(lats)
    lats.sort()
    p50 = statistics.median(lats)
    p95 = lats[int(len(lats) * 0.95) - 1] if lats else 0
    print(f"requests={len(jobs)} ok  workers={args.workers}")
    print(f"prompt_tokens={ptok}  completion_tokens={ctok}")
    print(f"aggregate throughput: {ctok / span:.1f} tok/s (output over wall span {span:.2f}s)")
    print(f"latency p50={p50:.2f}s p95={p95:.2f}s max={max(lats):.2f}s  avg={statistics.mean(lats):.2f}s")


if __name__ == "__main__":
    main()
