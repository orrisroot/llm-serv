# vLLM / L40S ×8 / DeepSeek V4 Flash 0731

Serves DeepSeek V4 Flash 0731 across eight L40S GPUs with vLLM, using 8-way tensor parallelism, DSA sparse attention and dspark speculative decoding.

vLLM is not installed from PyPI here. Ada (sm_89) support for this model comes from the [`yhfgyyf/vllm-deepseek-v4-sm89`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) fork, and the wheel is compiled locally.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA L40S ×8 (Ada / sm_89) |
| CUDA | 13.0 (`/usr/local/cuda-13.0`) |
| Python | 3.12 |
| Engine | `yhfgyyf/vllm-deepseek-v4-sm89`, built from source |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Listen address | `0.0.0.0:8000` (API key required) |

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `vllm` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/vllm
```

`run` derives the virtualenv and env file paths from its own install location, so any instance name works without editing the script.

### Build tooling

`uv` comes from [Provisioning the host](../../README.md#tooling). The rest is specific to this build, installed once per host as your normal user:

```sh
# the interpreter this build and the runtime venv use
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

The published release wheel is missing [commit `26d02fd`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/commit/26d02fdbbd3cbaf07e8bb94998925c78b5dbe215), without which the server does not run. The engine therefore has to be compiled from the fork's HEAD, in a scratch directory as your normal user:

```sh
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89

uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.11.0 --torch-backend=cu130 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements/build/cuda.txt

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.0

# Version stamp, built from the fork's release tag plus the commits since it
tag=$(git describe --tags --abbrev=0)     # v0.23.1rc1.dev904-g8e321cc4f-cu130-sm89
distance=$(git rev-list --count "${tag}..HEAD")
base=${tag#v}; base=${base%%-*}           # 0.23.1rc1.dev904
export VLLM_VERSION_OVERRIDE="${base%.dev*}.dev$(( ${base#*.dev} + distance ))+g$(git rev-parse --short=9 HEAD).cu130"

export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=16 NVCC_THREADS=2

.venv/bin/python -m build --wheel --no-isolation
ls dist/
deactivate
cd ..
```

The fork tags its releases as `v<version>-g<hash>-cu130-sm89`, which setuptools-scm cannot parse — hence the override, rather than letting the build compute its own version. The tag must be present, so run `git fetch --tags` if the clone does not have it.

| Variable | Rationale |
| --- | --- |
| `TORCH_CUDA_ARCH_LIST="8.9+PTX"` | Ada only, plus PTX so the kernels still load on newer cards |
| `VLLM_MAIN_CUDA_VERSION=13.0` | Matches the CUDA 13.0 toolkit and the `cu130` torch build |
| `VLLM_VERSION_OVERRIDE` | Stamps the wheel. Checked out at the tag above it yields `0.23.1rc1.dev904+g8e321cc4f.cu130`, and the dev counter keeps climbing with each commit past the tag. The `.cu130` local segment is what the install glob below matches |
| `MAX_JOBS=16` / `NVCC_THREADS=2` | Caps build parallelism so the compile does not exhaust host memory |

## Installation

### Runtime virtualenv

The virtualenv is created at its final path — a venv bakes absolute paths into its scripts, so it cannot be built elsewhere and moved:

```sh
sudo env "PATH=$PATH" uv venv --python 3.12 --seed /opt/llm-serv/vllm/.venv
```

Install the base environment from the fork's release wheels, then overwrite vLLM itself with the wheel just compiled:

```sh
gh release download --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_python-0.6.14*sm89*.whl' \
  --pattern 'vllm-*.cu130-cp312-cp312-linux_x86_64.whl' \
  --dir /tmp/vllm-sm89-release

sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  torch==2.11.0 flashinfer-cubin==0.6.13 --torch-backend=cu130
sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  /tmp/vllm-sm89-release/flashinfer_python-0.6.14*sm89*.whl
sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv uv pip install \
  --force-reinstall --no-deps ./vllm-deepseek-v4-sm89/dist/vllm-*.cu130-*.whl

rm -rf /tmp/vllm-sm89-release
```

### Scripts

```sh
sudo install -o llm-serv -g llm-serv -m 0750 run         /opt/llm-serv/vllm/run
sudo install -o llm-serv -g llm-serv -m 0640 env.example /etc/llm-serv/vllm.env
sudoedit /etc/llm-serv/vllm.env   # replace the placeholder with the real key
```

### Model files

```sh
sudo env "PATH=$PATH" uvx hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /opt/llm-serv/models/deepseek-ai/DeepSeek-V4-Flash-0731/

sudo chown -R llm-serv:llm-serv /opt/llm-serv/models/deepseek-ai
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
| `--tensor-parallel-size 8` | 8 | One rank per L40S; the model does not fit on fewer |
| `--gpu-memory-utilization 0.85` | 0.85 | Leaves headroom on each card for activation spikes and the NCCL buffers |
| `--max-model-len 524288` | 512k | Context length per sequence |
| `--max-num-seqs 16` | 16 | Concurrent sequences, bounded by KV cache at this context length |
| `--max-num-batched-tokens 2048` | 2048 | Caps the prefill chunk, keeping decode latency stable under long-prompt load |
| `--kv-cache-dtype fp8_ds_mla` | FP8 | DeepSeek MLA-specific FP8 KV cache; what makes 512k context affordable |
| `--block-size 256` | 256 | Large paged-attention blocks, matched to the MLA sparse kernel |

### DeepSeek V4 specifics

| Flag | Rationale |
| --- | --- |
| `--trust-remote-code` | The model ships custom modelling code that vLLM must import |
| `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` | DSA sparse attention path from the FlashInfer build in this fork |
| `--speculative-config '{"method":"dspark",...}'` | dspark speculative decoding, 6 draft tokens, greedy draft sampling |
| `--reasoning-parser deepseek_v4` | Splits reasoning content out of the response |
| `--tool-call-parser deepseek_v4` / `--enable-auto-tool-choice` | Parses the model's tool-call format and lets it pick tools itself |
| `--served-model-name deepseek-v4-flash-0731` | Model name exposed by the API, decoupling clients from the checkout path |

### Environment

| Variable | Rationale |
| --- | --- |
| `NCCL_SHM_DISABLE=1` | Shared-memory transport is unusable between the eight ranks here; NCCL falls back to peer-to-peer |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | The FlashInfer wheel is pinned to this fork and fails the stock version check |
| `LD_LIBRARY_PATH` | The venv's bundled shared objects (`PyNvVideoCodec`, `lib/`) followed by the CUDA 13.0 runtime |

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0`. If the server should not be reachable from outside the host, restrict port 8000 at the firewall or change `HOST` in `run` to `127.0.0.1`. The reference deployment keeps the engine on a private network and fronts it with an httpd reverse proxy that maps per-user API keys onto this single backend key.

Note that `--trust-remote-code` executes Python shipped in the model repository. Pin the model revision if the checkout is refreshed from upstream.

Port 8000 is also used by the llama.cpp example; give one of them a different `PORT` if both run on the same host.
