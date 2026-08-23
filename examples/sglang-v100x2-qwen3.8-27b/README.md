# sglang-V100 / V100 32GB ×2 / Qwen3.8 27B FP8

Serves Qwen3.8 27B through [sglang-V100](https://github.com/haohervchb/sglang-V100), a fork of SGLang focused on SM70 / Tesla V100, using its TileLang FlashAttention backend and native SM70 kernels, with 2-way tensor parallelism (one rank per V100).

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA Tesla V100-PCIE-32GB ×2 (Volta / sm_70) |
| CUDA | 12.8 (`/usr/local/cuda-12.8`) — **the last toolkit supporting Volta** |
| Engine | sglang-V100 (SGLang fork), built from source into a virtualenv |
| Runtime | venv at `<instance>/.venv` — Python 3.12, torch 2.9.1+cu128, tilelang 0.1.8 |
| Model | `Qwen/Qwen3.8-27B-FP8` (28.8 GiB, includes MTP head) |
| Listen address | `0.0.0.0:8000` (API key required) |

### Why this fork

Volta cannot run upstream SGLang's attention kernels, and the hybrid Qwen3.8 27B (attention + GatedDeltaRule SSM layers) needs SM70 paths for both. The fork ships `tilelang_fa_v100` (attention prefill + decode on TileLang), a TurboMind-based FP8 weight layout, a native E5M2 KV-cache path, and DSpark / DFlash2 speculative decoding — the same family of SM70 work as the 1Cat-vLLM example in this repository, on the SGLang serving stack.

### Measured performance (caveat emptor)

The fork's published numbers were measured on **4× V100-SXM2-32GB with NVLink**, not on this host's 2× V100-PCIE pair. The reference table (256 greedy output tokens, one cold-cache request):

| Profile | 1K prefill | 1K decode | 25K prefill | 25K decode |
| --- | ---: | ---: | ---: | ---: |
| Target only, E5M2 KV (TP4) | 2,992 tok/s | 58.2 tok/s | 3,714 tok/s | 50.8 tok/s |
| DSpark-7, FP16 KV (TP4) | 2,749 tok/s | 107.8 tok/s | 3,140 tok/s | 78.3 tok/s |

TP2 was benchmarked by the fork only for the DSpark-7 profile (13-point sweep, FP16 KV, 32k context): **1K prefill 1,761 tok/s / decode 76.5 tok/s**, 25K 1,686 / 51.9. By geometric mean TP2 delivers about 55% of TP4 prefill and 69% of TP4 decode — the SXM2 cards and NVLink contribute the rest. The DFLASH2 profile this example ships was measured on this host (next section).

### Measured on this host

All measurements on this exact host (V100-PCIE 32GB ×2, TP2, greedy, 256-token outputs; aggregate = 8 concurrent × 192-token generations, wall-clock end to end; the first request after each server boot is excluded). The fork's reference profile is single-request; the concurrency ceiling of this fork on 2×32GB is **3 slots at 128k** — 4 slots OOMs at pool allocation.

| Configuration | Single | 8-concurrent aggregate | Prefill (1.6k-token prompt) |
| --- | ---: | ---: | ---: |
| Target only, 128k, 1 slot | 33.7 tok/s | 32.6 tok/s | ~500 tok/s |
| Target only, 128k, 4 slots | 33.7 tok/s | 18.6 tok/s | ~500 tok/s |
| Target only, 256k, 2 slots | 33.7 tok/s | 62.3 tok/s | 514 tok/s |
| DFLASH2, 128k, 1 slot | 61.9 tok/s | 73.5 tok/s | 519 tok/s |
| DFLASH2, 128k, 2 slots | 59.2 tok/s | 88.5 tok/s | — |
| **DFLASH2, 128k, 3 slots (this example)** | **61.2 tok/s** | **112.7 tok/s** | ~500 tok/s |
| DFLASH2, 256k, 1 slot | 60.1 tok/s | 71.0 tok/s | 503 tok/s |

Target-only decode does not batch well on this fork (the Triton GDN decode at batch >1 is slower than separate streams), so raising slots from 1 to 4 *halves* the aggregate throughput. DFLASH2 removes that penalty — the draft proposals keep the batch busy — and 3 slots is the memory ceiling: the KV pool is capped at ~780k tokens by the 32GB cards, so 4 slots × 128k does not fit (OOM at pool allocation), and at 256k context only 1 slot fits.

### KV cache quantization (fp8_e5m2)

The fork's 8-bit compact-KV path, verified on this host (all figures DFLASH2, 128k, 1 slot, 131072-token pool):

| KV dtype | KV memory (K+V at 131k tokens) | Single | 8-concurrent | Outputs vs FP16 |
| --- | ---: | ---: | ---: | --- |
| `fp8_e5m2` (default) | 2.0 GiB/GPU (1.0 + 1.0) | 61.9 tok/s | 73.5 tok/s | 4/5 prompts byte-identical, 1 same answer |
| `auto` (FP16) | 4.0 GiB/GPU (2.0 + 2.0) | 61.8 tok/s | 71.0 tok/s | reference |
| `fp8_e4m3` | 2.0 GiB/GPU | 61.8 tok/s | 75.2 tok/s | n/a |

The 8-bit cache halves KV memory with **no measured speed difference** — the decode bottleneck is the Triton GDN path, not KV reads — and that capacity headroom is exactly what enables the 3-slot profile. Greedy outputs are effectively identical to FP16: four of five test prompts byte-identical, the fifth same answer with different wording, which is the expected E5M2 numerical drift.

Expect a slow first request after boot: Triton/TileLang compile their kernels into the instance `$HOME` on first use.

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `sglang` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/sglang
```

`run` derives the venv and env file paths from its own install location, so any instance name works without editing the script.

## Building the engine

The engine is compiled from the fork's source into a plain virtualenv (see `build` in this directory), following the same shape as the other examples in this repository.

### Prerequisites

Beyond the provisioned host, the build needs these tools (all checked by `build`):

```sh
# Rust, for the sglang-grpc extension (maturin/pyo3)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

# protoc, for the grpc build script (or download a binary and set PROTOC=)
sudo dnf install -y protobuf-compiler      # RHEL-family; apt: protobuf-compiler
```

CUDA 12.8 (`/usr/local/cuda-12.8`) and cmake must be present; ninja comes from pip. On this host nvcc 12.8 compiles fine with the default gcc 14; on hosts with an older gcc the build picks `g++-13/12/11` first.

### Build

```sh
# 1. Fork checkout, readable by the service account
sudo install -d -o "$USER" -g "$(id -gn)" /opt/llm-serv/sglang-v100
git clone https://github.com/haohervchb/sglang-V100.git \
  /opt/llm-serv/sglang-v100/sglang-V100

# 2. Virtualenv at its final path, like the other examples in this repo
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/sglang/.venv
sudo chown -R "$USER" /opt/llm-serv/sglang/.venv

# 3. Build (torch cu128, patched FlashInfer, TurboMind SM70, sglang-kernel,
#    Marlin; runs the fork's smoke test at the end)
SGLANG_SRC_DIR=/opt/llm-serv/sglang-v100/sglang-V100 \
SGLANG_VENV=/opt/llm-serv/sglang/.venv \
  bash examples/sglang-v100x2-qwen3.8-27b/build
```

The build takes a couple of hours on a first run (nvcc compiles several extensions); `build` caps parallelism by available RAM. It is incremental: re-running it reuses the checkouts and only rebuilds what changed. The venv's Python is the shared uv-managed CPython under `/opt/llm-serv/python` (used by the vLLM examples too) — the system Python on this host lacks `_xxsubinterpreters`, which tilelang requires.

Validate:

```sh
/opt/llm-serv/sglang/.venv/bin/python -c \
  "import sglang, torch, tilelang; print('sglang ok | torch', torch.__version__)"
```

## Installation

### Scripts

Install this directory's contents into place:

```sh
sudo install -o llm-serv -g llm-serv -m 0750 run         /opt/llm-serv/sglang/run
sudo install -o llm-serv -g llm-serv -m 0640 env.example /etc/llm-serv/sglang.env
sudoedit /etc/llm-serv/sglang.env   # replace the placeholder with the real key
```

### Model files

Fetch the weights and the DFLASH2 draft checkpoint (used by the default speculative profile) into the shared model store:

```sh
sudo env "PATH=$PATH" uvx hf download Qwen/Qwen3.8-27B-FP8 \
  --local-dir /opt/llm-serv/models/Qwen/Qwen3.8-27B-FP8/
sudo env "PATH=$PATH" uvx hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir /opt/llm-serv/models/z-lab/Qwen3.8-27B-DFlash2/
sudo chown -R llm-serv:llm-serv /opt/llm-serv/models/Qwen /opt/llm-serv/models/z-lab
```

For the alternative DSpark profile, fetch `RadixArk/Qwen3.8-27B-DSpark` the same way.

### Start

```sh
sudo systemctl enable --now llm-serv@sglang
```

Model loading takes a few minutes, then the first request triggers a one-time kernel compile; follow it with `sudo tail -f /var/log/llm-serv/sglang-stderr.log`.

## Configuration rationale

The defaults are the fastest measured profile on this host: DFLASH2 speculative decoding, 128k context, 3 concurrent slots, fp8_e5m2 KV. Tuning values come from the env file; `run` shows the exact command line.

### Attention and model path

| Flag | Rationale |
| --- | --- |
| `--attention-backend tilelang_fa_v100` | The fork's SM70 attention path (prefill + decode) — the reason this fork exists |
| `--linear-attn-prefill-backend tilelang` / `--linear-attn-decode-backend triton` | The model's SSM (GatedDeltaRule) layers: chunked GDN prefill runs on TileLang, decode stays on Triton (the fork's native recurrent decoder is correct but slower at one-token shapes) |
| `--dtype float16` | Volta has no bf16; the FP8 weights run under FP16 activations |
| `--kv-cache-dtype fp8_e5m2` | The fork's 8-bit compact-KV route on V100 (native E5M2 cache writes). Measured: halves KV memory versus FP16 with no speed difference — see the quantization section. `auto` selects FP16 KV |
| `--trust-remote-code` | The model ships custom modelling code that sglang must import |

### Parallelism and memory

| Flag | Value | Rationale |
| --- | --- | --- |
| `--tensor-parallel-size <n>` | 2 | One rank per GPU, derived from the number of visible devices. The fork validates TP4 on SXM2/NVLink; TP2 is the 2-card profile |
| `--mem-fraction-static <f>` | 0.95 | Measured ceiling for the 3-slot profile. Each TP2 rank holds 14.8 GiB of weights; the fork reserves 4.5 GiB of JIT headroom, leaving ~11 GiB for the KV pool |
| `--context-length` / `--max-total-tokens` | 131072 / 393216 | 3 slots × 128k. The KV pool is the shared resource: the fork caps it at ~780k tokens (the 32GB cards hold no more), so 4 slots × 128k or 2 slots × 256k does not fit — both OOM |
| `--max-running-requests` | 3 | The measured concurrency sweet spot: 112.7 tok/s aggregate at 8 users. 1 slot: 73.5; 2 slots: 88.5; 4 slots: OOM at pool allocation |
| `--cuda-graph-max-bs` / `--cuda-graph-bs` | 3 / `1 2 3` | CUDA graphs captured for batch 1..3, matching `--max-running-requests` |
| `--chunked-prefill-size` | 8192 | Prefill chunk budget from the reference command |
| `--mamba-full-memory-ratio` / `--mamba-scheduler-strategy` | 0.1 / `extra_buffer` | SSM state budget and scheduler strategy from the reference profile |
| `--enable-nccl-nvls` | — | NCCL NVLS hooks for prefill-heavy requests; no NVLS transport is created without NVLink, so it is harmless on this host |

### API surface

| Flag | Rationale |
| --- | --- |
| `--reasoning-parser qwen3` | Keeps the reasoning (thinking) content in responses (in the `reasoning` field), like llama.cpp's `--reasoning-preserve` |
| `--tool-call-parser qwen3_coder` | OpenAI-compatible tool calling, validated with this model family |
| `--served-model-name qwen3.8-27b` | Model name exposed by the API, decoupling clients from the on-disk layout |

### Speculative decoding

DFLASH2 is the default (`LLM_SPEC_ALGORITHM=DFLASH`): the `z-lab/Qwen3.8-27B-DFlash2` draft (block 8, window 2048) roughly doubles single-stream speed over target-only decode — 61.2 vs 33.7 tok/s — and, crucially, makes decode batch-friendly: target-only aggregate *drops* when slots rise (triton decode), while DFLASH2 keeps climbing to 112.7 tok/s at 3 slots. The target KV stays in E5M2; the small 5-layer draft cache runs FP16.

Set `LLM_SPEC_ALGORITHM=none` for plain target-only decode (half the speed, no draft model needed), or `DSPARK` for the `RadixArk/Qwen3.8-27B-DSpark` draft (the fork's TP2 sweep measured 76.5 tok/s single-stream).

## Environment variables

| Variable | Default | Rationale |
| --- | --- | --- |
| `SGLANG_API_KEY` | — | API key; `run` exits if unset |
| `SGLANG_VENV` | `$INSTANCE_DIR/.venv` | Runtime virtualenv built from source; override if the venv lives elsewhere |
| `LLM_SPEC_ALGORITHM` | `DFLASH` | Speculative algorithm: `DFLASH` (fastest), `DSPARK`, or `none` for target-only |
| `LLM_SPEC_DRAFT_MODEL` | `z-lab/Qwen3.8-27B-DFlash2` | Draft checkpoint under the model store; for DSPARK set `RadixArk/Qwen3.8-27B-DSpark` |
| `LLM_SPEC_DFLASH_BLOCK_SIZE` / `LLM_SPEC_DRAFT_WINDOW_SIZE` | `8` / `2048` | DFLASH2 proposal block size and draft KV window |
| `LLM_SPEC_DSPARK_BLOCK_SIZE` | `7` | DSpark proposal block size |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs exposed to sglang; TP degree derives from the count unless `LLM_TENSOR_PARALLEL_SIZE` is set |
| `LLM_TENSOR_PARALLEL_SIZE` | (device count) | Forces the tensor-parallel degree instead of deriving it |
| `LLM_MEM_FRACTION_STATIC` | `0.95` | GPU memory ceiling |
| `LLM_CONTEXT_LENGTH` / `LLM_MAX_TOTAL_TOKENS` | `131072` / `393216` | Context per request and shared KV pool (3 × 128k) |
| `LLM_MAX_RUNNING_REQUESTS` | `3` | Concurrent slots; raise with `LLM_CUDA_GRAPH_MAX_BS` up to the memory ceiling |
| `LLM_CUDA_GRAPH_MAX_BS` | `3` | CUDA-graph batch ceiling; `run` passes the list `1..n` |
| `LLM_CHUNKED_PREFILL_SIZE` | `8192` | Prefill chunk budget |
| `LLM_KV_CACHE_DTYPE` | `fp8_e5m2` | 8-bit KV cache; `auto` for FP16 KV (half the pool, same speed) |
| `FLASHINFER_DISABLE_VERSION_CHECK` / `NCCL_P2P_LEVEL` / `NCCL_SHM_DISABLE` / `SGLANG_CUSTOM_ALLREDUCE_ALGO` / `SGLANG_MAMBA_CONV_DTYPE` / `SGLANG_MAMBA_SSM_DTYPE` / `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | fork-validated values | Runtime environment from the fork's serve commands; all overridable from the env file |

## Hardware notes

- **No NVLink on this host.** The upstream commands set `NCCL_P2P_LEVEL=NVL` (validated on SXM2 cards with NVLink). This host's V100-PCIE pair has none, so `run` leaves it unset — NCCL then uses P2P over PCIe — and sets `NCCL_SHM_DISABLE=1`, matching what the 1Cat-vLLM example on this host found necessary. On an NVLink host, set `NCCL_P2P_LEVEL=NVL` in the env file.
- **One-time JIT warmup.** Triton and TileLang compile kernels into `$HOME` (`/var/lib/llm-serv/sglang/`, created by systemd) on first use; the first request after boot is slow.
- **No start timeout and no memory cap** apply, as everywhere in this framework — a slow model load simply takes as long as it takes.

## Authentication and networking

`run` validates `SGLANG_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

Unlike the other examples, the key cannot ride the environment into the server: this fork's sglang reads no `*_API_KEY` environment variable (verified in `server_args.py` — the key is argparse-only). `run` therefore passes it as `--api-key`, which makes it visible in `ps` output and in the stderr log's launch echo. The env file itself stays `0640 root:llm-serv`, and the key is never tracked in this repository. If the visibility bothers you, front the instance with a reverse proxy that maps per-user keys onto this single backend key, as the L40S example does.

The server binds to `0.0.0.0:8000` by default. Both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance.