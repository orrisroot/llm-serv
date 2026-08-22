# 1Cat-vLLM / V100 32GB ×2 / Qwen3.8 27B FP8

Serves Qwen3.8 27B through [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a vLLM fork focused on SM70 / Tesla V100, using its `FLASH_ATTN_V100` attention backend and 2-way tensor parallelism. Validated on this exact host.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA Tesla V100-PCIE-32GB ×2 |
| CUDA | 12.8 (`/usr/local/cuda-12.8`) — the last toolkit supporting Volta |
| Engine | 1Cat-vLLM 1.3.0 (prebuilt wheel), PyTorch 2.10.0+cu128 |
| Python | 3.12 (standalone build with dev headers — see below) |
| Model | `Qwen/Qwen3.8-27B-FP8` (28.8 GiB, includes MTP head) |
| Listen address | `0.0.0.0:8000` (API key required) |

### Why FP8, not FP16

The FP16 checkpoint (51.8 GiB) does not fit alongside a KV cache on two V100s: at `gpu-memory-utilization 0.95` the engine reports *"1.15 GiB KV cache is needed, which is larger than the available KV cache memory (0.12 GiB)"* even at a 32k context. The FP8 checkpoint halves the weights (28.8 GiB), leaving roughly 30 GiB for KV cache, and is the only viable path for this model on the host. Measured single-stream speed is on par with the FP16 path and with llama.cpp serving the Q8 GGUF (37.8 vs 38.7 tok/s).

## Measured performance

Native OpenAI endpoint, temperature 0, 256-token outputs, 8 concurrent × 192-token generations. First request is warmup (slow on V100, excluded).

| Profile | Single TG | Parallel aggregate |
| --- | --- | --- |
| 4 slots × 256k context | 37.8 tok/s | 128.6 tok/s (4-way) |
| **8 slots × 128k context (this example)** | 37.8 tok/s | **242.3 tok/s (8-way)** |

Prompt processing is ~1300 tok/s for a 2.7k-token prompt. Raising concurrency from 4 to 8 slots costs no single-stream speed; the memory freed by FP8 is what makes it possible.

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `vllm1cat` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/vllm1cat
```

`run` derives the venv and env file paths from its own install location, so any instance name works without editing the script.

## Runtime virtualenv

The virtualenv is created at its final path — a venv bakes absolute paths into its scripts, so it cannot be built elsewhere and moved. Its interpreter has to sit somewhere `llm-serv` can reach, so install that under `/opt/llm-serv` as well. Use the standalone Python build: the system interpreter on this host lacks `Python.h`, and Triton compiles a small launcher at runtime, so a header-less interpreter fails during model inspection.

```sh
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv python install 3.12
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/vllm1cat/.venv
```

Install the release wheel. Download it from the [latest release](https://github.com/1CatAI/1Cat-vLLM/releases/latest):

```sh
curl -sL -o /tmp/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl \
  https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.3.0/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl

sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm1cat/.venv uv pip install \
  --no-cache-dir --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  /tmp/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl

rm -f /tmp/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl
```

`--index-strategy unsafe-best-match` is required. By default uv takes a package from the first index that carries it at all; the CUDA index carries `flashinfer-python`, but not the `0.6.11.post2` this wheel pins, so the resolve fails outright without ever consulting PyPI. The flag makes uv consider every index, which is what `--extra-index-url` already means to pip.

The wheel bundles `flash_attn_v100` and the SM70 CUDA extensions; no source build or lmdeploy tree is required.

Verify the environment before starting:

```sh
cd /tmp   # outside any 1Cat-vLLM source checkout
sudo -u llm-serv /opt/llm-serv/vllm1cat/.venv/bin/python - <<'PY'
import torch, vllm, flash_attn_v100
print("vllm", vllm.__version__, "| torch", torch.__version__, "| gpus", torch.cuda.device_count())
PY
```

## Installation

### Scripts

Install this directory's contents into place:

```sh
sudo install -o llm-serv -g llm-serv -m 0750 run         /opt/llm-serv/vllm1cat/run
sudo install -o llm-serv -g llm-serv -m 0640 env.example /etc/llm-serv/vllm1cat.env
sudoedit /etc/llm-serv/vllm1cat.env   # replace the placeholder with the real key
```

### Model files

Fetch the weights into the shared model store:

```sh
sudo env "PATH=$PATH" uvx hf download Qwen/Qwen3.8-27B-FP8 \
  --local-dir /opt/llm-serv/models/Qwen/Qwen3.8-27B-FP8/
sudo chown -R llm-serv:llm-serv /opt/llm-serv/models/Qwen
```

### Start

```sh
sudo systemctl enable --now llm-serv@vllm1cat
```

Model loading takes a few minutes; follow it with `sudo tail -f /var/log/llm-serv/vllm1cat-stderr.log`.

## Configuration rationale

| Flag | Value | Rationale |
| --- | --- | --- |
| `--tensor-parallel-size 2` | 2 | One rank per GPU across two cards. TP4 is the fork's public reference on 4-GPU hosts — set `CUDA_VISIBLE_DEVICES` there |
| `--gpu-memory-utilization 0.95` | 0.95 | ~30.2 GiB used per GPU after load; the FP8 weights leave just enough KV headroom |
| `--max-model-len 131072` / `--max-num-seqs 8` | — | 8 slots × 128k. The parallelism profile: 242 tok/s aggregate at zero single-stream cost. The 4-slot × 256k profile (`LLM_MAX_MODEL_LEN=262144 LLM_MAX_NUM_SEQS=4`) serves long context at 128.6 tok/s |
| `--max-num-batched-tokens 8192` | 8192 | Prefill batch budget from the fork's public profiles |
| `--kv-cache-dtype fp8_e5m2` | fp8_e5m2 | Halves KV memory (830k-token cache vs 429k at FP16); the FP8 V100 KV path this fork ships |
| `--attention-backend FLASH_ATTN_V100` | — | 1Cat-vLLM's SM70 attention path (decode + prefill) — the reason this fork exists |
| `--tool-call-parser qwen3_coder` / `--enable-auto-tool-choice` | — | OpenAI-compatible tool calling, validated with this model family |
| `--reasoning-parser qwen3` | — | Keeps the reasoning (thinking) content in responses (in the `reasoning` field), like llama.cpp's `--reasoning-preserve` |
| `LLM_REASONING_EFFORT` (env) | unset | Default thinking effort applied to requests that do not set one. Valid: `low`, `medium`, `xhigh` (the Qwen3.8 template's default is `xhigh`); wired through `--default-chat-template-kwargs`. A per-request `reasoning_effort` still overrides it |
| `--served-model-name qwen3.8-27b` | — | Model name exposed by the API, decoupling clients from the on-disk layout |

MTP speculative decoding is an opt-in in this example, matching the fork's V100 public profile: set `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1` (validated on this host) or pass an explicit `--speculative-config`. Measured trade-off on this host (fp8_e5m2 KV, TP2):

| Workload | MTP off | MTP4 (opt-in) |
| --- | --- | --- |
| Single-stream, short context | 37.8 tok/s | 48.4 tok/s (+30%) |
| 8 concurrent × 192 tokens | 242 tok/s | 159 tok/s (-35%) |
| 64k-context decode | 1.9 tok/s | 1.6 tok/s |

Enable MTP for low-concurrency, latency-sensitive serving; keep it off for high-concurrency or long-context workloads.

Image inputs are enabled by default on the `FLASH_ATTN_V100` path (one image per prompt); pass `--limit-mm-per-prompt '{"image":0,"video":0}'` for text-only serving.

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0:8000` by default. Both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance.