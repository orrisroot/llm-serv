# llama.cpp / V100 32GB ×2 / Qwen3.8 27B

Serves the multimodal Qwen3.8 27B (GGUF) through llama.cpp's OpenAI-compatible server, split across two V100 GPUs.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA Tesla V100 32GB ×2 (64GB total, Volta / sm_70) |
| CUDA | 12.8 (`/usr/local/cuda-12.8`) |
| Engine | llama.cpp, built from source for sm_70 |
| Model | `unsloth/Qwen3.8-27B-GGUF` — `Qwen3.8-27B-UD-Q8_K_XL.gguf` |
| Vision projector | `mmproj-F16.gguf` from the same repository |
| Listen address | `0.0.0.0:8000` (API key required) |

Volta does not support bf16 or FP8, which is why both the weights and the KV cache use Q8-family quantization.

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `llama` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/llama /opt/llm-serv/llama/bin
```

`run` derives the engine binary and env file paths from its own install location, so any instance name works without editing the script.

## Building the engine

llama.cpp is compiled locally for Volta and installed under `/opt/llm-serv/llama/bin/`.

```sh
git clone https://github.com/ggml-org/llama.cpp

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

cmake llama.cpp -B llama.cpp/build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70

cmake --build llama.cpp/build --config Release -j$(nproc)
```

| Option | Rationale |
| --- | --- |
| `-DGGML_CUDA=ON` | Enable the CUDA backend |
| `-DCMAKE_CUDA_ARCHITECTURES=70` | Volta (sm_70) only. Targeting a single architecture keeps build time and binary size down; the result will not run on other GPU generations |
| `-DBUILD_SHARED_LIBS=OFF` | Link ggml/llama statically, so deployment is a single binary with no accompanying `.so` files to install |

Install the result:

```sh
sudo install -m 0755 llama.cpp/build/bin/llama /opt/llm-serv/llama/bin/llama
```

The CUDA runtime libraries are still linked dynamically, which is why `run` re-exports the same `CUDA_HOME` and `LD_LIBRARY_PATH` at startup. Build and runtime must point at the same CUDA installation.

## Installation

### Scripts

Install this directory's contents into place:

```sh
sudo install -o llm-serv -g llm-serv -m 0750 run         /opt/llm-serv/llama/run
sudo install -o llm-serv -g llm-serv -m 0640 env.example /etc/llm-serv/llama.env
sudoedit /etc/llm-serv/llama.env   # replace the placeholder with the real key
```

### Model files

Fetch the weights and the vision projector into the shared model store:

```sh
sudo env "PATH=$PATH" uvx hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  mmproj-F16.gguf \
  --local-dir /opt/llm-serv/models/unsloth/Qwen3.8-27B-GGUF/
```

`env "PATH=$PATH"` keeps `uvx` reachable, since `sudo` drops `~/.local/bin` from the path.

The resulting layout, which `run` refers to by these exact paths:

```
/opt/llm-serv/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf
/opt/llm-serv/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf
```

Hand the files to the service account:

```sh
sudo chown -R llm-serv:llm-serv /opt/llm-serv/models/unsloth
```

### Start

```sh
sudo systemctl enable --now llm-serv@llama
```

Model loading takes a few minutes; follow it with `sudo tail -f /var/log/llm-serv/llama-stderr.log`.

## Configuration rationale

### GPU allocation

| Flag | Value | Rationale |
| --- | --- | --- |
| `-ngl 99` | all layers | Offload every layer to GPU (a value above the real layer count is the idiomatic way to say "all") |
| `-ts 34,30` | 34:30 | Layer split across the two cards, weighted to account for the other buffers GPU0 carries |

### Context and slots

| Flag | Value | Rationale |
| --- | --- | --- |
| `-np 2` | 2 | Parallel slots — two concurrent requests |
| `-c 524288` | 512k | **Total** context across all slots, i.e. 256k tokens per slot |
| `-ctk q8_0` / `-ctv q8_0` | Q8_0 | Quantized KV cache. Roughly halves KV memory versus F16, which is what makes this context length fit in 64GB |
| `--no-kv-unified` | — | Keeps each slot's KV cache independent, so a long session cannot crowd out the other slot |

### Model and generation

| Flag | Rationale |
| --- | --- |
| `--mmproj mmproj-F16.gguf` | Vision projector; enables image input |
| `--image-min-tokens 1024` | Minimum token budget per image, so small images are not downscaled too aggressively |
| `--spec-type draft-mtp` / `--spec-draft-n-max 3` | Speculative decoding via the model's built-in MTP head — no separate draft model needed |
| `--reasoning-preserve` | Keeps the reasoning (thinking) content in the response |
| `--jinja` | Uses the chat template embedded in the GGUF |
| `--alias qwen3.8-27b` | Model name exposed by the API, decoupling clients from the on-disk filename |

## Authentication and networking

`run` validates `LLAMA_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0`. If the server should not be reachable from outside the host, restrict port 8000 at the firewall or change `HOST` in `run` to `127.0.0.1`.
