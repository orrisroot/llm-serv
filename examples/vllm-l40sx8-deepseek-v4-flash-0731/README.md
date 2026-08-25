# vLLM / L40S ×8 / DeepSeek V4 Flash 0731

Serves DeepSeek V4 Flash 0731 across eight L40S GPUs with vLLM, using 8-way tensor parallelism, DSA sparse attention and dspark speculative decoding.

vLLM is not installed from PyPI here. Ada (sm_89) support for this model comes from the [`yhfgyyf/vllm-deepseek-v4-sm89`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) fork, and the wheel is compiled locally.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA L40S ×8 (Ada / sm_89) |
| CUDA | 13.2 (`/usr/local/cuda-13.2`) |
| Python | 3.12 |
| Engine | `yhfgyyf/vllm-deepseek-v4-sm89`, built from source |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Listen address | `0.0.0.0:8000` (API key required) |

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `vllm` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o root -g root /opt/llm-serv/vllm
```

`run` derives the virtualenv and env file paths from its own install location, so any instance name works without editing the script.

### Build tooling

`uv` comes from [Provisioning the host](../../README.md#tooling). The rest is specific to this build, installed once per host as your normal user:

```sh
# the interpreter this build uses; the runtime venv gets its own copy later
uv python install 3.12

# rust, needed by the vLLM build
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# gh, used to pull the fork's prebuilt wheels
curl -OL https://github.com/cli/cli/releases/download/v2.97.0/gh_2.97.0_linux_amd64.tar.gz
tar zxvf gh_2.97.0_linux_amd64.tar.gz
mv gh_2.97.0_linux_amd64/bin/gh ~/.local/bin/
rm -rf gh_2.97.0_linux_amd64 gh_2.97.0_linux_amd64.tar.gz
```

## Building the engine

This example compiles the engine wheel from the fork's HEAD in a scratch directory as your normal user:

```sh
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89

uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.13.0 --torch-backend=cu130 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130

export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.2

# Version stamp, built from the fork's release tag plus the commits since it
tag=$(git describe --tags --abbrev=0)     # v0.23.1rc1.dev904-g998fd644b-cu132-sm89
distance=$(git rev-list --count "${tag}..HEAD")
base=${tag#v}; base=${base%%-*}           # 0.23.1rc1.dev904
export VLLM_VERSION_OVERRIDE="${base%.dev*}.dev$(( ${base#*.dev} + distance ))+g$(git rev-parse --short=9 HEAD).cu132"

export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=16 NVCC_THREADS=2

.venv/bin/python -m build --wheel --no-isolation
ls dist/
deactivate
cd ..
```

The fork tags its releases as `v<version>-g<hash>-cu132-sm89`, which setuptools-scm cannot parse — hence the override, rather than letting the build compute its own version. The tag must be present, so run `git fetch --tags` if the clone does not have it.

| Variable | Rationale |
| --- | --- |
| `TORCH_CUDA_ARCH_LIST="8.9+PTX"` | Ada only, plus PTX so the kernels still load on newer cards |
| `VLLM_MAIN_CUDA_VERSION=13.2` | Matches the CUDA 13.2 toolkit and the `cu130` torch build |
| `VLLM_VERSION_OVERRIDE` | Stamps the wheel. Checked out at the tag above it yields `0.23.1rc1.dev904+g998fd644b.cu132`, and the dev counter keeps climbing with each commit past the tag. The `.cu132` local segment is what the install glob below matches |
| `MAX_JOBS=16` / `NVCC_THREADS=2` | Caps build parallelism so the compile does not exhaust host memory |

## Installation

### Runtime virtualenv

The virtualenv is created at its final path — a venv bakes absolute paths into its scripts, so it cannot be built elsewhere and moved. Its interpreter has to sit somewhere `llm-serv` can reach, so install that under `/opt/llm-serv` as well:

```sh
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv python install 3.12
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/vllm/.venv
```

Install the base environment from the fork's release wheels, then overwrite vLLM itself with the wheel just compiled. The release wheel has to go in first: it is what pulls vLLM's dependency tree, which the final `--no-deps` install deliberately leaves alone so the pinned versions survive. The vLLM, FlashInfer-python and FlashInfer-cubin wheels must come from the same release, so all three are pulled by `gh release download`:

```sh
gh release download --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_cubin-0.6.17-*.whl' \
  --pattern 'flashinfer_python-0.6.17*sm89*.whl' \
  --pattern 'vllm-*.cu132-cp312-cp312-linux_x86_64.whl' \
  --dir /tmp/vllm-sm89-release

sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  torch==2.13.0 --torch-backend=cu130
sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  /tmp/vllm-sm89-release/flashinfer_cubin-0.6.17-*.whl
sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  /tmp/vllm-sm89-release/flashinfer_python-0.6.17*sm89*.whl
sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  /tmp/vllm-sm89-release/vllm-*.cu132-cp312-cp312-linux_x86_64.whl --torch-backend=cu130
sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  --force-reinstall --no-deps ./vllm-deepseek-v4-sm89/dist/vllm-*.cu132-*.whl

rm -rf /tmp/vllm-sm89-release
```

### FlashInfer TP8 patch

The fork bundles `flashinfer 0.6.17+sm89.1`, whose sparse-MLA prefill kernel only handles `num_heads ≥ 16`. At TP=8, the 64 attention heads divide into 8 per rank, and the kernel rejects the configuration. Backport the prefill kernel fix from [upstream flashinfer#4380](https://github.com/flashinfer-ai/flashinfer/pull/4380) (commit `24d7dfb2`) to enable `num_heads=8`.

The four files below are **adapted** from that commit, not verbatim upstream copies: they extend dispatch to `page_block_size ∈ {64, 256}` and add SM89 (`SPARSE_MLA_USE_SM89_PRIMS`) conditionals that this fork needs. Installing the verbatim upstream files instead would silently drop `page_block_size=256` support. Always use these patched copies, and only update them by re-adapting the newer upstream sources.

The four patched source files live alongside this README:

```sh
FLASHINFER_DATA=/opt/llm-serv/vllm/.venv/lib/python3.12/site-packages/flashinfer/data

sudo cp patches/flashinfer/csrc/sparse_mla_sm120_prefill.cu    "$FLASHINFER_DATA/csrc/"
sudo cp patches/flashinfer/csrc/sparse_mla_sm120_decode_dsv4.cu "$FLASHINFER_DATA/csrc/"
sudo cp patches/flashinfer/csrc/sparse_mla_sm120_jit_binding.cu "$FLASHINFER_DATA/csrc/"
sudo cp patches/flashinfer/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh \
            "$FLASHINFER_DATA/include/flashinfer/attention/sparse_mla_sm120/"
```

Fresh installs need no further steps — the JIT cache is empty and the patched sources are compiled on first start. When updating an existing installation, clear the cache so the patched sources are recompiled:

```sh
sudo rm -rf /var/lib/llm-serv/vllm/.cache/flashinfer
```

The patch can be dropped once the fork updates its bundled flashinfer to a release that includes the upstream fix.

### Scripts

```sh
sudo install -o root -g llm-serv -m 0750 run         /opt/llm-serv/vllm/run
sudo install -o root -g llm-serv -m 0640 env.example /etc/llm-serv/vllm.env
sudoedit /etc/llm-serv/vllm.env   # replace the placeholder with the real key
```

### Model files

```sh
sudo env "PATH=$PATH" uvx hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /opt/llm-serv/models/deepseek-ai/DeepSeek-V4-Flash-0731/
```

### Start

```sh
sudo systemctl enable --now llm-serv@vllm
```

Loading a model of this size across eight GPUs takes several minutes; follow it with `sudo tail -f /var/log/llm-serv/vllm-stderr.log`.

## Configuration rationale

### Parallelism and memory

| Flag | Value | Rationale |
| --- | --- | --- |
| `--tensor-parallel-size <n>` | 8 | One rank per L40S; the model does not fit on fewer. `n` defaults to the number of visible devices; set `LLM_TENSOR_PARALLEL_SIZE` to force it |
| `--gpu-memory-utilization <f>` | 0.88 | Measured ceiling on this host — 0.90 fails, 0.88 holds. The remainder absorbs activation spikes and the NCCL buffers. Set `LLM_GPU_MEMORY_UTILIZATION` to override |
| `--max-model-len <n>` | 512k | Context length per sequence. Set `LLM_MAX_MODEL_LEN` to override |
| `--max-num-seqs <n>` | 16 | Concurrent sequences, bounded by KV cache at this context length. Set `LLM_MAX_NUM_SEQS` to override |
| `--max-num-batched-tokens 4096` | 4096 | Per-step scheduling budget; high enough to absorb the spec-decode slot overhead (see the env table) without letting prefill chunks stretch decode latency under long-prompt load |
| `--kv-cache-dtype fp8_ds_mla` | FP8 | DeepSeek MLA-specific FP8 KV cache; what makes 512k context affordable |
| `--block-size 256` | 256 | Large paged-attention blocks, matched to the MLA sparse kernel |

### DeepSeek V4 specifics

| Flag | Rationale |
| --- | --- |
| `--trust-remote-code` | The model ships custom modelling code that vLLM must import |
| `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` | DSA sparse attention path from the FlashInfer build in this fork |
| `--speculative-config '{"method":"dspark",...}'` | dspark speculative decoding, 4 draft tokens, probabilistic draft sampling |
| `--reasoning-parser deepseek_v4` | Splits reasoning content out of the response |
| `LLM_REASONING_EFFORT` (env) | Default thinking effort for requests that do not set one. The DeepSeek V4 template treats `none` as off and `max`/`xhigh` as maximum; any other value falls back to the model default. Wired through `--default-chat-template-kwargs`; a per-request `reasoning_effort` still overrides it |
| `LLM_ENABLE_THINKING` (env) | Turns thinking off entirely when set to `false` (`enable_thinking=false`), skipping the reasoning path. The model default is on |
| `--tool-call-parser deepseek_v4` / `--enable-auto-tool-choice` | Parses the model's tool-call format and lets it pick tools itself |
| `--served-model-name deepseek-v4-flash-0731` | Model name exposed by the API, decoupling clients from the checkout path |

### Environment

| Variable | Rationale |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Tensor-parallel GPUs. Defaults to all eight; the degree is derived from the count unless `LLM_TENSOR_PARALLEL_SIZE` is set |
| `LLM_TENSOR_PARALLEL_SIZE` | Forces the tensor-parallel degree instead of deriving it from the device count |
| `LLM_MAX_MODEL_LEN` / `LLM_MAX_NUM_SEQS` | Context length and concurrent sequences, defaulting to 512k / 16 |
| `LLM_MAX_NUM_BATCHED_TOKENS` | Per-step scheduling budget, defaulting to 4096. Needs to fit the spec-decode overhead — roughly `max_num_seqs × (1 + num_speculative_tokens)` — or vLLM caps `max_num_scheduled_tokens` below the budget and batch throughput suffers |
| `LLM_NUM_SPECULATIVE_TOKENS` | DSpark draft depth, defaulting to 4. Acceptance falls off sharply past ~4-5 positions for this draft model, so the old 7 mostly wasted draft compute |
| `LLM_GPU_MEMORY_UTILIZATION` | GPU memory ceiling, defaulting to 0.88 |
| `TILELANG_CACHE_DIR` / `TRITON_CACHE_DIR` / `FLASHINFER_WORKSPACE_BASE` | JIT kernel caches, pinned under the per-instance `$HOME` (`/var/lib/llm-serv/<instance>/.cache/`) so Triton, TileLang and FlashInfer kernels compiled on one start are reused on the next instead of recompiled |
| `NCCL_SHM_DISABLE=1` | Shared-memory transport is unusable between the ranks here; NCCL falls back to peer-to-peer |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | The FlashInfer wheel is pinned to this fork and fails the stock version check |
| `LD_LIBRARY_PATH` | The venv's bundled shared objects (`PyNvVideoCodec`, `lib/`) followed by the CUDA 13.2 runtime |

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0:8000` by default. Both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance. The reference deployment keeps the engine on a private network and fronts it with an httpd reverse proxy that maps per-user API keys onto this single backend key.

Note that `--trust-remote-code` executes Python shipped in the model repository. Pin the model revision if the checkout is refreshed from upstream.
