# 1Cat-vLLM / V100 32GB ×2 / Qwen3.8 27B FP8

Serves Qwen3.8 27B through [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a vLLM fork focused on SM70 / Tesla V100, using its `FLASH_ATTN_V100` attention backend (tensor parallelism defaults to 2, one rank per V100). The engine is compiled from the fork's source at the revision pinned below, so the procedure yields the same artifact whenever it is run.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA Tesla V100-PCIE-32GB ×2 |
| CUDA | 12.8 (`/usr/local/cuda-12.8`) — the last toolkit supporting Volta |
| Engine | 1Cat-vLLM `main` @ `d30469863287471a7082842500ae73299a697e0d` (`1.5.1.dev925+gd30469863.cu128`), built from source, PyTorch 2.10.0+cu128 |
| Python | 3.12.15 (uv-managed standalone build with dev headers — see below) |
| Model | `Qwen/Qwen3.8-27B-FP8` (28.8 GiB, includes MTP head) |
| Listen address | `0.0.0.0:8000` (API key required) |

### Why FP8, not FP16

The FP16 checkpoint (51.8 GiB) fills the two 32 GB cards by itself, leaving no room for a usable KV cache. The FP8 checkpoint halves the weights (28.8 GiB), leaving roughly 30 GiB for KV cache; it is the only viable way to serve this model on two V100-32GB.

## Measured performance

Native OpenAI endpoint, measured with the pinned build on the target hardware using `bench_llm.py` in this directory. Default profile: 256k context (the model's maximum), up to 16 concurrent sequences, MTP off, temperature 0, 192-token outputs (`min_tokens`, first request after boot excluded). A per-run nonce at the start of the prompt defeats the fork's default prefix cache, so "cold" runs prefill from scratch; reusing a nonce measures the prefix-cache hit path.

| Profile | Throughput |
| --- | --- |
| Prefill, cold 9k-token prompt | ~1,440 tok/s (TTFT ≈6.3 s) |
| Prefill, prefix-cache hit | ~9,250 tok/s (TTFT ≈1.0 s) |
| Single-stream decode | 39.0 tok/s |
| Decode aggregate, 4-way | 99.7 tok/s |
| Decode aggregate, 8-way | 143.2 tok/s |
| Decode aggregate, 16-way | 182 tok/s |

Decode aggregate excludes prefill (from the earliest first token). Prompt processing is compute-bound on both GPUs; at `gpu-memory-utilization 0.95` the server uses ≈29.3 GiB per GPU after load.

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `vllm1cat` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o root -g root /opt/llm-serv/vllm1cat
```

`run` derives the venv and env file paths from its own install location, so any instance name works without editing the script.

## Building the engine

The wheel is compiled from the fork's source at a pinned revision as your normal user. Run the build from the root of the llm-serv checkout: the source lands in `1Cat-vLLM/`, which `.gitignore` keeps out of the repository, and the later steps return to the checkout root. The build bundles the SM70 Flash-V100 kernels and the fork's other CUDA work into the wheel.

### Build tooling

`uv` comes from [Provisioning the host](../../README.md#tooling). The rest is specific to this build, installed once per host:

```sh
# rust, for the optional vllm-rs frontend (rust-toolchain.toml pins the
# 1.95 channel, which rustup installs on demand)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# perl core modules used by the vendored OpenSSL build inside vllm-rs
# (RHEL-family perl is minimal — Debian/Ubuntu's perl is already complete)
sudo dnf install -y perl-FindBin perl-IPC-Cmd perl-Time-Piece

# protoc and its well-known type includes, for vllm_grpc.proto
sudo dnf install -y protobuf-compiler protobuf-devel
```

### Build

```sh
git clone https://github.com/1CatAI/1Cat-vLLM.git
cd 1Cat-vLLM
git checkout d30469863287471a7082842500ae73299a697e0d   # tip of main, 2026-09-29

uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=cu128 -r requirements/build/cuda.txt

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_CUDA_ARCHITECTURES=70
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=16 NVCC_THREADS=2

python -m build --wheel --no-isolation
ls dist/
deactivate
```

| Variable | Rationale |
| --- | --- |
| `git checkout <sha>` | Pins the exact revision this setup was verified against, so the build is reproducible no matter when it runs; the tip of `main` moves |
| `--torch-backend=cu128` | Resolves the `torch==2.10.0` pin to the CUDA 12.8 build (`2.10.0+cu128`) from the PyTorch index |
| `TORCH_CUDA_ARCH_LIST="7.0"` / `CMAKE_CUDA_ARCHITECTURES=70` | SM70 only — the fork's recommended setting for V100 |
| `CUDA_HOME` / `PATH` | Points the build at the CUDA 12.8 toolkit, matching the `cu128` torch ABI |
| `VLLM_TARGET_DEVICE=cuda` | Explicit CUDA device target; auto-detected from torch otherwise |
| `MAX_JOBS=16` / `NVCC_THREADS=2` | Caps build parallelism so the compile does not exhaust host memory |

The build compiles the full vLLM CUDA sources plus the fork's `flash_attn_v100` and SM70 TurboMind kernels, and takes roughly 90 minutes to a couple of hours on a 16-core host. The wheel lands as `dist/1cat_vllm-<version>-cp312-cp312-linux_x86_64.whl`; the version derives from the fork's git history (tag plus commit distance, `v1.5.0-925-gd30469863` yields `1.5.1.dev925+gd30469863.cu128`).

The `vllm-rs` Rust frontend is built and bundled alongside the Python package — rustup pulls the pinned 1.95 channel, the perl modules satisfy the vendored OpenSSL build, and protoc compiles `vllm_grpc.proto`. It is optional either way: the Python frontend is the runtime default (`VLLM_USE_RUST_FRONTEND=0`), and if any of those tools is missing the build tolerates the rust failure and the wheel still works without `vllm-rs`.

## Runtime virtualenv

The virtualenv is created at its final path — a venv bakes absolute paths into its scripts, so it cannot be built elsewhere and moved. Its interpreter has to sit somewhere `llm-serv` can reach, so install that under `/opt/llm-serv` as well. Use the standalone Python build: distro interpreters commonly lack dev headers (`Python.h`), and Triton compiles a small launcher at runtime, so a header-less interpreter fails during model inspection.

```sh
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv python install 3.12
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/vllm1cat/.venv
```

Install the runtime dependencies, then the wheel just built. The fork's `requirements/cuda.txt` pins `nvidia-cutlass-dsl[cu13]`, whose `cu13` extra does not exist for the CUDA 12.8 build — drop the extra the same way the fork's own CI does:

```sh
mkdir -p /tmp/1cat-reqs
cp requirements/common.txt requirements/cuda.txt /tmp/1cat-reqs/
sed -i 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' /tmp/1cat-reqs/cuda.txt

sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm1cat/.venv uv pip install \
  --torch-backend=cu128 --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r /tmp/1cat-reqs/common.txt -r /tmp/1cat-reqs/cuda.txt

sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm1cat/.venv uv pip install \
  --no-deps dist/1cat_vllm-*.whl

rm -rf /tmp/1cat-reqs
cd ..   # back to the llm-serv checkout root
```

`--index-strategy unsafe-best-match` is required. By default uv takes a package from the first index that carries it at all; the CUDA index carries `flashinfer-python`, but not the `0.6.11.post2` this build pins, so the resolve fails outright without ever consulting PyPI. The flag makes uv consider every index, which is what `--extra-index-url` already means to pip.

Verify the environment before starting. The shell must be outside the `1Cat-vLLM/` source tree — Python would otherwise import the source instead of the wheel's CUDA extensions:

```sh
sudo -u llm-serv /opt/llm-serv/vllm1cat/.venv/bin/python - <<'PY'
import torch, vllm, flash_attn_v100
print("vllm", vllm.__version__, "| torch", torch.__version__, "| gpus", torch.cuda.device_count())
PY
```

## Installation

### Scripts

Install this example's files into place:

```sh
cd examples/vllm-v100x2-qwen3.8-27b

sudo install -o root -g llm-serv -m 0750 run         /opt/llm-serv/vllm1cat/run
sudo install -o root -g llm-serv -m 0640 env.example /etc/llm-serv/vllm1cat.env
sudoedit /etc/llm-serv/vllm1cat.env   # replace the placeholder with the real key
```

### Model files

Fetch the weights into the shared model store:

```sh
sudo env "PATH=$PATH" uvx hf download Qwen/Qwen3.8-27B-FP8 \
  --local-dir /opt/llm-serv/models/Qwen/Qwen3.8-27B-FP8/
```

### Start

```sh
sudo systemctl enable --now llm-serv@vllm1cat
```

Model loading takes a few minutes; follow it with `sudo tail -f /var/log/llm-serv/vllm1cat-stderr.log`.

## Configuration rationale

| Flag | Value | Rationale |
| --- | --- | --- |
| `--tensor-parallel-size <n>` | 2 | One rank per GPU, derived from the number of visible devices. On a 4-GPU host, set `CUDA_VISIBLE_DEVICES` to four devices (or set `LLM_TENSOR_PARALLEL_SIZE`) |
| `--gpu-memory-utilization <f>` | 0.95 | ≈29.3 GiB used per GPU after load; the FP8 weights leave just enough KV headroom. Set `LLM_GPU_MEMORY_UTILIZATION` to override |
| `--max-model-len 262144` / `--max-num-seqs 16` | — | The model's maximum context (262,144, per `config.json`), up to 16 concurrent sequences. The engine auto-sizes the attention block to align with the mamba page size (1568 tokens on this build). `LLM_MAX_MODEL_LEN` / `LLM_MAX_NUM_SEQS` override either; lowering the context is the cheapest way to free KV memory |
| `--max-num-batched-tokens 8192` | 8192 | Prefill batch budget from the fork's public profiles |
| `--kv-cache-dtype fp8_e5m2` | fp8_e5m2 | Halves KV memory vs FP16; the fork's E5M2 V100 KV path also enables the bounded short-context decode CUDA graph (8192 tokens) |
| `--attention-backend FLASH_ATTN_V100` | — | 1Cat-vLLM's SM70 attention path (decode + prefill) — the reason this fork exists |
| `--tool-call-parser qwen3_coder` / `--enable-auto-tool-choice` | — | OpenAI-compatible tool calling |
| `--reasoning-parser qwen3` | — | Keeps the reasoning (thinking) content in responses (in the `reasoning` field), like llama.cpp's `--reasoning-preserve` |
| `LLM_REASONING_EFFORT` (env) | unset | Default thinking effort applied to requests that do not set one. Valid: `low`, `medium`, `xhigh` (the Qwen3.8 template's default is `xhigh`); wired through `--default-chat-template-kwargs`. A per-request `reasoning_effort` still overrides it |
| `LLM_ENABLE_THINKING` (env) | unset | Turns thinking off entirely when set to `false` (`enable_thinking=false`), skipping the reasoning path. The model default is on |
| `--served-model-name qwen3.8-27b` | — | Model name exposed by the API, decoupling clients from the on-disk layout |

Some defaults are applied automatically by the fork at startup and need no flags:

| Auto setting | Value | Rationale |
| --- | --- | --- |
| `enable_prefix_caching` / `mamba_cache_mode=align` | on / align | SM70 serving default for hybrid (linear-attention) models; prefix cache reuse requires the mamba cache to align with the attention pages |
| `VLLM_SM70_RMSNORM_GATED_EXACT` | 1 | Fork default for the no-MTP profile (`config/vllm.py`): native FP32 gated-RMSNorm arithmetic. Set `VLLM_SM70_RMSNORM_GATED_EXACT=0` in the env file for the fused FP16 path |
| `VLLM_SM70_FP8_KV_DECODE_CONTEXT_BUCKETS` | 8192 | Bounds the FP8-KV short-context decode graph so the short D256 GQA graph stays scalar-only |
| compile-graph policy | `VLLM_COMPILE`, cudagraph `FULL_AND_PIECEWISE`, capture `(1,2,4,8,16)` | The fork's SM70 compile-graph policy for this profile |

MTP speculative decoding is opt-in and not part of this profile: set `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1` or pass an explicit `--speculative-config`.

Image inputs are enabled by default on the `FLASH_ATTN_V100` path (one image per prompt); pass `--limit-mm-per-prompt '{"image":0,"video":0}'` for text-only serving.

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0:8000` by default. Both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance.
