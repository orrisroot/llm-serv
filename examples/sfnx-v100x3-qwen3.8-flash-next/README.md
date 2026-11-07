# sFNX (V100 ×3) — Qwen3.8-Flash-Next UD-Q4_K_XL

[sFNX](https://github.com/johnfink8/sfnx) is a CUDA inference engine that splits **every layer** of
Qwen3.8-Flash-Next across three PCIe V100 cards (no NVLink needed): experts (512 a layer) split by
FF width, recurrent Gated DeltaNet by heads, attention K/V by position, output head by vocabulary.
Its sums between cards run inside the window's CUDA graph around a ring, which is what makes the split
pay off without NVLink. The serving front end is a fork of Strata's `serve` (web UI, OpenAI/Anthropic/
Responses APIs, tools, pictures).

Validated on a 3× V100-32GB host (PCIe, no NVLink).

## Assumptions

| Item | Value |
| --- | --- |
| GPU | 3x V100-32GB, **every pair a CUDA peer** (`nvidia-smi topo -p2p r` → OK), not in a translated IOMMU domain |
| CUDA | Toolkit 12.9, paired with g++ 14 |
| Compiler | g++ 14 (install it on distributions that do not ship it, or add `g++-14`/`gcc-14` symlinks), `ninja-build` |
| Model | Unsloth **UD-Q4_K_XL** (4 GGUF shards, 111 GB) + `mmproj-F16.gguf`, under `models/` |
| Draft layer | `data/mtp-rt/` (see below) + `data/tokenizer`, `data/draft_vocab_en.bin`, `data/expert-routes.bin` (in repo) |
| Runtime | `make` → `bin/sfnx`, `bin/sfnx-vision`; front venv (Python ≥3.14); llama.cpp @ `a37cc46` |

## Building

```sh
git clone https://github.com/johnfink8/sfnx.git && cd sfnx
# llama.cpp at the pinned commit, as ./llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp   # then fetch+checkout a37cc46
ln -s /opt/llm-serv/models/sfnx models   # shared store (or /path/to/models)
curl -LsSf https://astral.sh/uv/install.sh | sh        # for the front venv and hf downloads
uv python install 3.14
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python .
make -j$(nproc)                          # bin/sfnx and bin/sfnx-vision
```

The draft layer's runtime files (`data/mtp-rt`, not in git) come from Strata's tools plus sfnx's
`tools/mtp_experts.py`:

```sh
# 1. fetch the 31 mtp.* tensors from the BF16 checkpoint (range requests, ~5 GB; pinned revision)
python <strata checkout>/tools/mtp_fetch.py inventory --out <dir>   # then: fetch, verify
# 2. pack (mtp_pack.py), then extract the *dense* runtime files with mtp_rt.py —
#    skip its expert block (the routed experts come from experts-q4_0.bin, step 3).
# 3. routed experts:
python tools/mtp_experts.py q4_0 <mtp-fetch>/tensors data/mtp-rt/experts-q4_0.bin
```

## Running inside llm-serv

This directory is the `llm-serv` instance template (`run`, `env.example`, plus `bench.py` and
`stress.py` for measurements):

- `/opt/llm-serv/sfnx/run` ← this `run`
- `/etc/llm-serv/sfnx.env` ← `env.example` (`LLM_SERV_HOST`, `LLM_SERV_PORT`, `STRATA_API_KEY`,
  `MODEL_NAME`)
- `systemctl enable --now llm-serv@sfnx` — the checkout lives at `${SFNX_HOME}` (default
  `/opt/llm-serv/sfnx/engine`), must be owned by the service account, and its `models/` link points
  into the shared `/opt/llm-serv/models/` store; the config's paths are relative to it.
  Keep that store on the fastest disk available (the engine reads weights from it while decoding).

`run` starts the front server in the foreground (so `llm-serv@` manages it), reading the API key from
the env file. The API-exposed model name comes from `MODEL_NAME` in the env file (default
`qwen3.8-flash-next`; `run` merges it into a per-instance copy of the engine config at start). The
unit template already raises `LimitMEMLOCK=infinity`, which sFNX needs for its host-side tables.

Or use the project's own unit: `make install` (a port-8080 unit) or `serve/run.sh serve/sfnx.json`
(the project's transient test unit, port 8095) — one at a time, since the engines share the GPUs. This
example uses the direct `front.server` invocation in `run` so the llm-serv template unit manages it.

## Limits / operational notes

- One engine at a time: VRAM is full at idle; nothing else may open a CUDA context while it serves.
- `STRATA_ENGINE_READY_S=0` makes the front wait for the engine's READY line as long as it takes
  (default 900 s; the template unit has no start timeout).
- Prompts are read one at a time; other lanes keep decoding between a read's chunks. Six lanes.
- Size `--conversation-cache-mib` to the host's RAM — the parked-conversation cache shares host
  memory (8 GiB is a good fit for a 128 GB-class host).

## Benchmarks (`bench.py`/`stress.py`, greedy, reasoning off)

| Workload | sFNX (UD-Q4_K_XL) |
| --- | --- |
| Prompt read 3.9k | ~2,800 tok/s (TTFT ≈1.4 s) |
| Prompt read 30.9k | ~3,840 tok/s (TTFT ≈8.1 s) |
| Prompt read ~262k (max) | ~2,600 tok/s (TTFT ≈100 s) |
| Decode 4k ctx (single stream) | ~157 tok/s |
| Decode 32k ctx | ~142 tok/s |
| Decode 262k ctx (max) | ~95 tok/s |
| 6 concurrent (1k prompt, 128 out) | ~153 tok/s aggregate, p50 7.7 s |
