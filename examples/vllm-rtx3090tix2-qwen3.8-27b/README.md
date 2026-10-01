# vllm-backport / RTX 3090 Ti ×2 / Qwen3.8 27B AWQ (nicosuter)

Serves **`nicosuter/Qwen3.8-27B-AWQ`** through [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) with 2-way tensor parallelism (one rank per RTX 3090 Ti). This is the recommended model for this hardware: it is an AWQ **W4A16** quant with the GatedDeltaNet input projections kept at **INT8**, so it preserves near-FP8 accuracy (see [Accuracy notes](#accuracy-notes)) while running Ampere's fast Marlin W4A16 kernel — the best measured balance of speed, concurrency, and quality on this host.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 Ti ×2 (Ampere / sm_86, 24 GiB each) |
| CUDA | 13.x (`/usr/local/cuda` toolkit symlink; Ampere is fully supported) |
| Engine | **vllm-backport** (vLLM fork focused on A6000 / 3090 / A100), pinned revision |
| Runtime | venv at `<instance>/.venv`, built from source with `uv` |
| Model | `nicosuter/Qwen3.8-27B-AWQ` (~15 GiB on disk; ~12 GiB/GPU at TP2) |
| Listen address | `0.0.0.0:8000` (API key required) |

## Why this model (and why vllm-backport)

### nicosuter AWQ — best accuracy/speed balance on Ampere

The full comparison against other Qwen3.8-27B quants is in the [appendix measurements](#measured-on-the-target-host-2x-rtx-3090-tp2). In short, measured on this exact hardware class (2× RTX 3090, TP2, 256K context):

- Single-stream decode: ~53 tok/s (vs 49 NVFP4, 40 FP8, 56 RedHat INT4)
- 8-way aggregate: ~248 tok/s
- KV pool: ~599k tokens → ~2 concurrent 256K requests (vs 1 for FP8)
- **Accuracy: within ~0.2 pt of FP8 in the author's published evals** — this is the key reason to prefer it over a plain 4-bit quant

### vllm-backport

[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) is a vLLM engineering fork whose stated focus is **running frontier models on older cards like A6000, 3090 and A100** — exactly this hardware class. It publishes per-architecture images (`latest-sm86` for Ampere) and tracks upstream closely. Its fork-specific gains that matter here:

- **`FULL_AND_PIECEWISE` CUDA-graph mode** — captures the whole decode step, including the NCCL all-reduce, into one graph. Documentation notes it improves decode on Ampere, provided NCCL is pinned (`NCCL_ALGO=Ring NCCL_PROTO=Simple` + `--disable-custom-all-reduce`) so graph replay re-issues the exact captured collective.
- Qwen3.8 (Qwen3.5 hybrid) architecture support: `qwen3_5.py` / `qwen3_5_mtp.py` ship in-tree.
- Marlin W4A16 / FP8 kernels tuned for Ampere.

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `vllmawq` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o root -g root /opt/llm-serv/vllmawq
```

`run` derives the venv and env file paths from its own install location, so any instance name works without editing the script.

## Building the runtime

The engine is compiled from the vllm-backport source at a **pinned revision** (so the artifact is reproducible), into the runtime venv at its final path.

### Build tooling

`uv` comes from [Provisioning the host](../../README.md#tooling). The source checkout lives under `/opt/llm-serv` (readable by the service account) and is kept out of this repository:

```sh
sudo install -d -o "$USER" -g "$(id -gn)" /opt/llm-serv/vllm-backport
git clone https://github.com/wtdcode/vllm-backport.git \
  /opt/llm-serv/vllm-backport/vllm-backport
```

### Build

```sh
cd /opt/llm-serv/vllm-backport/vllm-backport
git checkout <PINNED_SHA>            # see note below: pin the revision you verified against

sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv python install 3.12
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/vllmawq/.venv

# The venv is created at its final path; make it writable by you for the build.
sudo chown -R "$USER" /opt/llm-serv/vllmawq/.venv

export CUDA_HOME=/usr/local/cuda
export VIRTUAL_ENV=/opt/llm-serv/vllmawq/.venv
export PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6        # Ampere sm86 only, for a lean local build
export MAX_JOBS=16 NVCC_THREADS=2

# Build a wheel, then install it -- like the other fork examples in this
# repository. Editable installs would let Python import the source tree,
# which `run` must avoid (see the launch comment).
uv pip install --no-build-isolation build
python -m build --wheel --no-isolation
uv pip install --no-deps dist/vllm_backport-*.whl
```

> **Pinning note:** vllm-backport tracks upstream fast and publishes versioned
> Docker tags (`:v0.13.0-sm86`) but no GitHub release tags, so `git tag`
> will be empty. Pin by commit instead: `git log --oneline -1` after you have
> a revision you verified, then record that SHA here and in any deploy notes.

| Variable | Rationale |
| --- | --- |
| `git checkout <sha>` | Pins the exact revision this setup was verified against; the fork's `master` moves fast |
| `TORCH_CUDA_ARCH_LIST=8.6` | Builds just the sm86 kernels this host needs — matches vllm-backport's `-sm86` images; on other Ampere cards use 8.0 |
| `MAX_JOBS` / `NVCC_THREADS` | Caps build parallelism so the compile does not exhaust host memory |

Validate (run as the service account, outside the source tree):

```sh
sudo -u llm-serv /opt/llm-serv/vllmawq/.venv/bin/python - <<'PY'
import torch, vllm
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
print("vllm-backport", vllm.__version__, "| torch", torch.__version__, "| gpus", torch.cuda.device_count())
PY
```

The server should log `Using MarlinLinearKernel for CompressedTensorsWNA16` at startup (this is the W4A16 path this model uses on Ampere).

## Installation

### Scripts

Install this example's files into place:

```sh
cd examples/vllm-rtx3090tix2-qwen3.8-27b

sudo install -o root -g llm-serv -m 0750 run         /opt/llm-serv/vllmawq/run
sudo install -o root -g llm-serv -m 0640 env.example /etc/llm-serv/vllmawq.env
sudoedit /etc/llm-serv/vllmawq.env   # replace the placeholder with the real key
```

### Model files

Fetch the weights into the shared model store:

```sh
sudo env "PATH=$PATH" uvx hf download nicosuter/Qwen3.8-27B-AWQ \
  --local-dir /opt/llm-serv/models/nicosuter/Qwen3.8-27B-AWQ/
```

### Start

```sh
sudo systemctl enable --now llm-serv@vllmawq
```

Model loading takes a few minutes. Watch the first startup for the expected log lines:

```sh
sudo tail -f /var/log/llm-serv/vllmawq-stderr.log
# expect: Using MarlinLinearKernel for CompressedTensorsWNA16
```

## Configuration rationale

| Flag | Value | Rationale |
| --- | --- | --- |
| `--tensor-parallel-size <n>` | 2 | One rank per GPU, derived from the number of visible devices |
| `--linear-backend marlin` | (set on Ampere) | The 4-bit layers run as W4A16 GEMMs on Ampere; Marlin is the fast dequant path. `run` gates on capability < 10; auto is left to Blackwell |
| `--kv-cache-dtype fp8` | fp8 | Halves KV memory vs bf16; the model ships FP8 KV calibration. This is what makes ~2 concurrent 256K requests possible |
| `--compilation-config` | `FULL_AND_PIECEWISE` | The fork's whole-decode CUDA graph, the main Ampere win over stock. Requires the NCCL pin and `--disable-custom-all-reduce`; capture sizes derived from `--max-num-seqs` |
| `NCCL_ALGO=Ring` / `NCCL_PROTO=Simple` | env | Pin the collective so FULL-graph replay is exact — the fork's documented fix for "crash on Ampere" graph capture |
| `--disable-custom-all-reduce` | — | Pairs with the NCCL pin for graph-capturable collectives |
| `--max-model-len 262144` / `--max-num-seqs 32` | 256k / 32 | Maximum context; the 32-slot ceiling is cheap but real concurrency is KV-bound (~2 for full-context requests) |
| `--gpu-memory-utilization 0.95` | 0.95 | ~22.4 GiB/GPU addressable; ~12 GiB weights, ~10 GiB left for FP8 KV + activations. Up to 0.98 starts on the target host (0.99 fails at startup); 0.90 is the conservative fallback if graph warmup OOMs |
| `--tool-call-parser qwen3_coder` / `--enable-auto-tool-choice` | — | OpenAI-compatible tool calling |
| `--reasoning-parser qwen3` | — | Keeps thinking content in the `reasoning` field |
| `LLM_REASONING_EFFORT` / `LLM_ENABLE_THINKING` (env) | unset | Default template kwargs for requests that do not set their own |

### MTP / speculative decoding — off by default

Measured on this hardware class: MTP is positive on NVFP4 (W4A4) targets but **negative on W4A16 quant** such as this AWQ — the draft verification cost outweighs its acceptance against an already-fast Marlin decode. **Benchmark before enabling**, and read acceptance from `vllm:spec_decode_num_{accepted,draft}_tokens_total` rather than token/s.

### LMCache (optional)

vllm-backport bundles a paired [LMCache fork](https://github.com/wtdcode/LMCache) for production KV-cache serving. It is a separate process and is not wired into this `run`; enable it only if you need long-lived cross-restart KV reuse for a chat workload.

## Environment variables

| Variable | Default | Rationale |
| --- | --- | --- |
| `VLLM_API_KEY` | — | API key; `run` exits if unset |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs exposed; TP degree derives from the count unless `LLM_TENSOR_PARALLEL_SIZE` is set |
| `LLM_TENSOR_PARALLEL_SIZE` | (device count) | Forces the tensor-parallel degree |
| `LLM_LINEAR_BACKEND` | `marlin` (Ampere) / `auto` (Blackwell) | W4A16 GEMM backend; both the env override and the auto capability gate are supported |
| `LLM_CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | CUDA-graph mode; `FULL_DECODE_ONLY` is the fallback if capture OOMs |
| `LLM_CUDAGRAPH_CAPTURE_SIZES` | `[1..max-num-seqs]` | JSON list of batch sizes to capture; FULL graphs hold private pools, so cap it |
| `NCCL_ALGO` / `NCCL_PROTO` | `Ring` / `Simple` | Pinned collective for graph replay |
| `LLM_MAX_MODEL_LEN` / `LLM_MAX_NUM_SEQS` | `262144` / `32` | Context per request and concurrency ceiling |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory ceiling; verified to start up to 0.98 on the target host, 0.99 fails; drop to 0.90 if graph warmup OOMs |
| `LLM_KV_CACHE_DTYPE` | `fp8` | KV cache precision; `bf16` doubles KV memory |
| `LLM_MTP_TOKENS` | unset | Enables `--speculative-config {"method":"mtp",...}`; negative for this model — keep unset |
| `LLM_REASONING_EFFORT` / `LLM_ENABLE_THINKING` | unset | Default chat-template kwargs |
| `LLM_SERV_HOST` / `LLM_SERV_PORT` | `0.0.0.0` / `8000` | Listen address |

## Sources

- Engine: <https://github.com/wtdcode/vllm-backport> (Ampere/A6000/3090/A100 focus; `FULL_AND_PIECEWISE`, NCCL pin, per-arch `-sm86` images)
- Model: <https://huggingface.co/nicosuter/Qwen3.8-27B-AWQ> (W4A16+INT8 mixed, FP8-level evals)
- Quantization accuracy research (FP8 lossless, W4A16 1-3%): Kurtić et al., "Give Me BF16 or Give Me Death?", <https://www.alphaxiv.org/abs/2411.02355>
- RedHat INT4 official evals: <https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4>
- vLLM Qwen3.8 recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-27B>
- Ampere NVFP4 measurements + MTP economics: club-3090 [discussion #662](https://github.com/noonghunna/club-3090/discussions/662)

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated. It binds to `0.0.0.0:8000` by default; both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance.

## Appendix

### Measured on the target host (2× RTX 3090, TP2)

All figures measured on the target host at `gpu-memory-utilization 0.90`, with this exact configuration (256K context, fp8 KV, Marlin, `FULL_AND_PIECEWISE`). Single-stream = 300-token completion at temperature 0, thinking off/on; 8-way = eight concurrent 300-token completions, wall-clock aggregate. "Recovery" = published quantized score ÷ reference (BF16 or FP8) score.

| Model | Single (off) | 8-way | KV pool | 256K conc. | Recovery (published) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **nicosuter AWQ (default)** | 52.9 | 248 | 599k | 2.29x | ~99.9% vs FP8 | near-FP8 accuracy; GDN layers stay INT8 |
| nicosuter AWQ + MTP3 | 37.9 | 198.6 | 505k | 1.93x | — | MTP hurts W4A16 targets, keep off |
| cyankiwi AWQ-INT4 | 53.6 | **264** | 618k | 2.36x | n/a | fastest plain 4-bit |
| RedHatAI INT4 | **56.3** | 220 | **663k** | **2.53x** | ~99.4% vs BF16 | fastest single-stream, official evals |
| Pilcothink AutoRound | 53.8 | **311** | 612k | 2.33x | ~99%+ (mixed) | highest 8-way throughput |
| Qwen FP8 (official) | 40.2 | 191 | 304k | 1.16x | ~100% (lossless) | accuracy floor, slowest |
| unsloth NVFP4 | 49.2 | 181 | 541k | 2.06x | var. (agentic -10pt) | Blackwell-native, weak on Ampere |

The default is nicosuter AWQ: near-FP8 accuracy (see [Accuracy notes](#accuracy-notes)) at ~53 tok/s single and ~248 tok/s at 8-way, with ~2 concurrent 256K requests — the best combined value where accuracy matters. The alternatives are trade-offs on one axis: `Pilcothink/MixedInt4-AutoRound` (or `cyankiwi`) if raw aggregate throughput is the priority, `RedHatAI INT4` for single-stream latency, and FP8 if accuracy alone decides. Swap `MODEL_PATH` in `run` to switch.

MTP does not help this model: it is positive on NVFP4 (thinking-on single stream +20-31%) but negative on W4A16 quants such as this AWQ (single −25-30%, 8-way −15-20%); keep `LLM_MTP_TOKENS` unset.

### Accuracy notes

**FP8 is effectively lossless** — that characterization is correct. A 500k-evaluation study across the Llama-3.1 family (Kurtić et al.) confirmed FP8 (W8A8) is effectively lossless at every model scale, and Qwen's official FP8 builds assume the same. If you want to maximize accuracy alone, FP8 is the answer. But as measured above, on 2× 24 GB it is the slowest profile (40 tok/s, 1 concurrent 256K request) — for many workloads that is more accuracy than is worth the throughput it costs.

**4-bit is not "badly broken", it is "1-3% degradation plus task-dependent skew":**
- The same study found a well-calibrated INT4 weight-only (W4A16) model degrades 1-3%, on par with INT8.
- RedHatAI INT4's official evals (vs BF16): IFEval 99.67%, MMLU-Pro 98.81%, GSM8K 101.1%, MATH-500 99.52%, GPQA 98.49%, AIME 98.69% — an average of ~99.4% recovery, i.e. roughly 0.5-1.5% degradation. Reasoning tasks lose 1-1.5%, while instruction following and GSM8K sit near 100% (quantization noise occasionally even helps).
- Degradation tends to surface in **edge cases** — agentic tool loops, long context, hard reasoning — rather than in benchmark averages. club-3090 reported an NVFP4 (W4A4) agentic 8-pack ~10 points below FP8. **W4A4 is particularly weak for agentic workloads; W4A16 is notably more robust.**

**Why nicosuter AWQ is preferred on accuracy grounds**: it keeps the GatedDeltaNet input projections (`in_proj_qkv`, `in_proj_z` — the heart of the hybrid-SSM layers) at **INT8** in an otherwise-W4A16 quant. The author's published evals put it within **0.2 points of FP8** (BFCL 100.07%, GPQA 98.60%, LiveCodeBench 101.63%). That means "near-FP8 accuracy" at 1.3× the FP8 throughput and ~2× the 256K concurrency. It is safer than a plain all-4-bit W4A16 quant (the GDN layers stay 8-bit) and practically on par with FP8 — the best accuracy/speed/concurrency position measured here.

**Open questions:**
- All of these evals are English-centric. Rankings can shift on Japanese (e.g. JAS) or domain-specific benchmarks — measure your own workload before committing.
- `Pilcothink/MixedInt4-AutoRound` keeps important layers at BF16/FP8 inside a mixed 4-bit quant; it has the fastest 8-way throughput (311 tok/s) and likely good accuracy too — worth checking if throughput is the priority.
- Where accuracy is a hard requirement, decide whether FP8's single-concurrent-request-at-256K constraint is acceptable.
