# vLLM / H100 ×1 / Qwen3.8 27B FP8

Serves **`Qwen/Qwen3.8-27B-FP8`** on a single NVIDIA H100 PCIe 80 GB using **vLLM from PyPI**. This is the recommended model for this host: the FP8 checkpoint is 28.8 GiB, so a single H100 hosts the weights, a large FP8 KV pool, and comfortable headroom.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 PCIe 80 GB ×1 (Hopper / sm_90) |
| CUDA | 13.4 (`/usr/local/cuda-13.4`; driver 615.71) |
| Python | 3.12 (uv-managed Venv) |
| Engine | stock `vllm` from PyPI — latest release (`vllm==0.32.0`, 2026-10) |
| Model | `Qwen/Qwen3.8-27B-FP8` (28.8 GiB on disk, FP8 block-scaled, includes MTP draft head) |
| Listen address | `0.0.0.0:8000` (API key required) |

## Prerequisites

The host must already be provisioned — service account, shared directories, and template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md).

Pick an instance name — `vllm` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o root -g root /opt/llm-serv/vllm
```

## Install the engine (stock wheel)

`uv` comes from [Provisioning the host](../../README.md#tooling). Build the virtualenv at its final path (a venv bakes absolute paths into its scripts, so it cannot be moved):

```sh
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv python install 3.12
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/vllm/.venv

# Make the venv writable by you for the install, then drop it again if desired
sudo chown -R "$USER" /opt/llm-serv/vllm/.venv

VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install "vllm==0.32.0"

sudo chown -R root:llm-serv /opt/llm-serv/vllm/.venv
```

`vllm==0.32.0` is the current PyPI release and the version this example was verified against. The exact pin keeps the artifact reproducible when a newer release ships.

Two notes on the installed environment:

- The wheel bundles `flashinfer-python`, which vLLM uses for the attention path on H100. It is a binary dependency that ships with the wheel, so it needs no local build.
- vLLM's JIT tooling needs a CUDA toolkit and `ninja` on `PATH` (the `run` script adds the toolkit's `bin/` and the venv's `bin/`). This is for compiling small per-host kernels at startup, not for building vLLM itself.

Validate the environment (run as the service account, from a directory outside any vLLM source checkout):

```sh
sudo -u llm-serv /opt/llm-serv/vllm/.venv/bin/python - <<'PY'
import torch, vllm, flashinfer
print("vllm", vllm.__version__, "| torch", torch.__version__,
      "| gpus", torch.cuda.device_count(), "| cap", torch.cuda.get_device_capability(0))
PY
```

Expected: one GPU, capability `(9, 0)`.

## Installation

### Scripts

```sh
sudo install -o root -g llm-serv -m 0750 run         /opt/llm-serv/vllm/run
sudo install -o root -g llm-serv -m 0640 env.example /etc/llm-serv/vllm.env
sudoedit /etc/llm-serv/vllm.env   # replace the placeholder with the real key
```

### Model files

```sh
sudo env "PATH=$PATH" uvx hf download Qwen/Qwen3.8-27B-FP8 \
  --local-dir /opt/llm-serv/models/Qwen/Qwen3.8-27B-FP8/
```

### Start

```sh
sudo systemctl enable --now llm-serv@vllm
```

Model loading takes a couple of minutes. Watch it:

```sh
sudo tail -f /var/log/llm-serv/vllm-stderr.log
```

## Configuration rationale

| Flag | Value | Rationale |
| --- | --- | --- |
| `--tensor-parallel-size <n>` | 1 | One H100; TP would only add NCCL overhead on a single rank. `n` defaults to the number of visible devices |
| `--gpu-memory-utilization <f>` | 0.95 | Weights 28.8 GiB, so 0.95 leaves roughly 47 GiB of FP8 KV pool (≈1.2M tokens at 262k). `LLM_GPU_MEMORY_UTILIZATION` overrides |
| `--max-model-len <n>` | 262144 | The model's native context. Lower it to trade KV memory for nothing (there is plenty). 1M context is possible via YaRN `--hf-overrides`, off by default |
| `--max-num-seqs <n>` | 64 | Concurrency ceiling; cheap at this KV budget. `LLM_MAX_NUM_SEQS` overrides |
| `--max-num-batched-tokens <n>` | 8192 | Per-step scheduling budget |
| `--kv-cache-dtype fp8` | fp8 | Halves KV memory vs bf16; the checkpoint is FP8 anyway |
| `--reasoning-parser qwen3` | — | The chat template opens every assistant turn with ` thing`, so without this the whole reasoning block lands in `message.content`. Not optional in practice |
| `--tool-call-parser qwen3_coder` / `--enable-auto-tool-choice` | — | OpenAI-compatible tool calling |
| `--trust-remote-code` | — | The checkpoint ships a custom `Qwen3_5ForConditionalGeneration` model that vLLM imports |

### Environment

| Variable | Default | Rationale |
| --- | --- | --- |
| `VLLM_API_KEY` | — | API key; `run` exits if unset |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU exposed; TP degree derives from the device count unless `LLM_TENSOR_PARALLEL_SIZE` is set |
| `LLM_TENSOR_PARALLEL_SIZE` | (device count) | Forces the tensor-parallel degree |
| `LLM_MAX_MODEL_LEN` / `LLM_MAX_NUM_SEQS` | `262144` / `64` | Context per request and concurrency ceiling |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory ceiling |
| `LLM_REASONING_EFFORT` / `LLM_ENABLE_THINKING` | unset | Default chat-template kwargs (`low`/`medium`/`xhigh`, or `false`) |
| `LLM_ENABLE_MM` | unset | `true` adds `--mm-encoder-tp-mode data` for multimodal prompts — the vision tower is a single small ViT, so TP on it costs more than it saves |
| `LLM_MTP_TOKENS` | `4` (recommended) | Enables the in-checkpoint MTP draft head (`qwen3_5_mtp`) at the given depth. Measured fastest at 4 on the reference host — see [Measured on the target host](#measured-on-the-target-host). Verify acceptance (`vllm:spec_decode_num_{accepted,draft}_tokens_total`); a drafter that is ignored loses throughput 
| `LLM_SERV_HOST` / `LLM_SERV_PORT` | `0.0.0.0` / `8000` | Listen address |

## Verification

```sh
source <(sudo cat /etc/llm-serv/vllm.env)

curl -sf http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${VLLM_API_KEY}"

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Give me three primes above 100."}],"max_tokens":2048}'
```

## Measured on the target host

Measured 2026-10-27 on this exact host (single H100 PCIe 80 GB, FP8 TP1, vLLM 0.32.0). `bench_llm.py` in this directory, 1173-token prompts, 192-token completions, thinking off. KV pool at 262k context: 1,126,934 tokens.

### MTP depth sweep (max-num-seqs 64, gpu-memory-utilization 0.95)

| MTP | acceptance | single (tok/s) | 8-way e2e (tok/s) | 8-way decode (tok/s) | TTFT (s) |
| --- | --- | --- | --- | --- | --- |
| off | — | 55.9 | 332 | 364 | 0.388 |
| 3 | 71.5% | 128.9 | 555 | 757 | 0.184 |
| **4** | **62.1%** | **164.4** | 556 | 759 | **0.183** |
| 5 | 60.3% | 149.2 | 570 | 784 | 0.143 |
| 7 | 47.2% | 118.2 | 473 | 699 | 0.182 |

**MTP=4 wins**: single-stream is fastest (164 tok/s, 2.94× vs off) and TTFT stays lowest. Deeper drafts (5) draft a bit more accurately under parallel load but pay back less per draft token; 7 is net negative.

### Parallel scaling with MTP=4

| concurrency | e2e (tok/s) | decode (tok/s) | TTFT mean (s) |
| --- | --- | --- | --- |
| 1 | — | 164.4 | 0.183 |
| 8 | 556 | 759 | 0.84 |
| 16 | 762 | 867 | 1.45 |
| 32 | 930 | 1,016 | 2.46 |
| 64 (seqs=128) | 965 | 1,027 | 4.89 |

Aggregate decode saturates around **1,000 tok/s** on this card. Raising `LLM_MAX_NUM_SEQS` past 64 buys another ~35 tok/s at 64-way but costs single-stream (125 vs 164 tok/s), so the default 64 is kept.

### Prefill

9k-token cold prefill: **10,394 tok/s** (TTFT 0.87 s).

### gpu-memory-utilization 0.95 vs 0.97

0.97 grows the KV pool 6% (1,193k tokens) but CUDA graph memory profiling eats the graph reserve, and single-stream drops 21% (129 vs 164 tok/s). 0.95 is the better operating point on this card; parallel throughput is unchanged.

### Thinking on

Single-stream with thinking on (default): **107 tok/s** (MTP=4). Lower than the thinking-off figure because the thinking block is the draft target, still ~2× the MTP-off baseline.

## Sources

- Model: <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>
- vLLM recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-27B>
- Engine: <https://pypi.org/project/vllm/>

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated. It binds to `0.0.0.0:8000` by default; both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance.
