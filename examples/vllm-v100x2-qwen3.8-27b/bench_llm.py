#!/usr/bin/env python3
"""Minimal stdlib-only LLM endpoint benchmark for A/B comparisons.

Usage: python3 bench_llm.py single|parallelN|prefill9k|prefill9k-warm|ptokens

Reads VLLM_API_KEY from $KEY_FILE (default: $HOME/.llm-serv-vllm1cat.env on the host)

Prefix-cache hygiene: the serving fork enables prefix caching by default for
this hybrid model, so identical prompts are served from cache. Unless
NONCE=<value> is set, every invocation appends a unique marker so prompt
prefills are cold; pass the same NONCE to re-measure a warm (cache-hit) run.
Metrics: e2e aggregate, decode-only aggregate (from earliest first token),
per-request TTFT / decode tok/s.
"""
import json
import os
import sys
import threading
import time
import urllib.request
import uuid

BASE = os.environ.get("LLM_BASE", "http://localhost:8000/v1")
KEY_FILE = os.environ.get("KEY_FILE", os.path.expanduser("~/.llm-serv-vllm1cat.env"))
KEY = None
for line in open(KEY_FILE):
    line = line.strip()
    if line.startswith("VLLM_API_KEY="):
        KEY = line.split("=", 1)[1].strip().strip("\"'")
assert KEY, "no key"

NONCE = os.environ.get("NONCE", uuid.uuid4().hex[:8])

PARA = (
    "The rapid growth of open source machine learning has produced a rich ecosystem "
    "of serving stacks, each optimized for specific hardware families. Inference engines "
    "must balance memory bandwidth, tensor core utilization, and scheduling latency to "
    "deliver acceptable throughput on modest GPU clusters. Recent work on low precision "
    "quantization reduces weight memory, letting older architectures serve modern language "
    "models despite their limited peak compute. "
)


def prompt(mult=16, label=""):
    # Marker goes at the START so the prefix cache cannot satisfy the common
    # suffix across runs (a suffix-only marker still shares the cached prefix).
    marker = "[%s-%s] " % (NONCE, label)
    return marker + PARA * mult


def call(p, max_tokens=192, timeout=1800):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,  # force exact length; EOS ignored
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    first = None
    last = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or []
            if choices and choices[0].get("delta"):
                now = time.time()
                if first is None:
                    first = now
                last = now
    return {"t0": t0, "first": first, "last": last or time.time(),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


def run(mode):
    if mode == "single":
        res = call(prompt(16, "s"))
        ct = res["completion_tokens"] or 0
        span = (res["last"] - res["first"]) if res["first"] else None
        print(json.dumps({"mode": mode, "nonce": NONCE,
                          "prompt_tokens": res["prompt_tokens"],
                          "completion_tokens": ct,
                          "ttft_s": round(res["first"] - res["t0"], 3) if res["first"] else None,
                          "decode_s": round(span, 3) if span else None,
                          "tg_tok_s": round(ct / span, 2) if span else None}))
    elif mode.startswith("parallel"):
        n = int(mode.replace("parallel", "")) or 1
        results = []

        def worker(i):
            results.append(call(prompt(16, "w%d" % i)))

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.time() - t0
        total = sum(r["completion_tokens"] or 0 for r in results)
        firsts = [r["first"] for r in results if r["first"]]
        lasts = [r["last"] for r in results]
        dec_span = (max(lasts) - min(firsts)) if firsts else None
        ttft = [(r["first"] - r["t0"]) for r in results if r["first"]]
        print(json.dumps({"mode": mode, "n": n, "nonce": NONCE,
                          "wall_s": round(wall, 3),
                          "e2e_agg_tok_s": round(total / wall, 2),
                          "decode_agg_tok_s": round(total / dec_span, 2) if dec_span else None,
                          "total_tokens": total,
                          "ttft_mean_s": round(sum(ttft) / len(ttft), 3) if ttft else None,
                          "ttft_max_s": round(max(ttft), 3) if ttft else None}))
    elif mode.startswith("prefill9k"):
        # Cold when NONCE is fresh; run again with the same NONCE for a warm
        # (prefix-cache-hit) measurement.
        p = prompt(125, "p")
        res = call(p, max_tokens=16)
        pt = res["prompt_tokens"]
        ttft = (res["first"] - res["t0"]) if res["first"] else None
        print(json.dumps({"mode": mode, "nonce": NONCE, "prompt_tokens": pt,
                          "ttft_s": round(ttft, 3) if ttft else None,
                          "prefill_tok_s": round(pt / ttft, 1) if ttft else None}))
    elif mode == "ptokens":
        print(json.dumps({"mode": mode, "prompt_tokens": 0}))
    else:
        sys.exit("unknown mode")


if __name__ == "__main__":
    run(sys.argv[1])
