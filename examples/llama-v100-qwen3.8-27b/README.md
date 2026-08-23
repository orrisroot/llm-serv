# llama.cpp / V100 ×2 or ×3 / Qwen3.8 27B

Serves the multimodal Qwen3.8 27B (GGUF) through llama.cpp's OpenAI-compatible server, split across two or three V100 GPUs. The 3-GPU profile is the configuration validated and benchmarked on this exact host.

## Target environment

| Item | Value |
| --- | --- |
| GPU | Tesla V100-PCIE-32GB ×2 or ×3 (Volta / sm_70) |
| CUDA | 12.8 (`/usr/local/cuda-12.8`) — **the last toolkit supporting Volta** |
| Driver | 580.x (final series for V100) |
| Engine | llama.cpp, built from source for sm_70 with `GGML_CUDA_FORCE_MMQ=ON` |
| Model | `unsloth/Qwen3.8-27B-GGUF` — `Qwen3.8-27B-UD-Q8_K_XL.gguf` |
| Vision projector | `mmproj-F16.gguf` from the same repository |
| Listen address | `0.0.0.0:8000` (API key required) |

Volta does not support bf16 or FP8, which is why both the weights and the KV cache use Q8-family quantization. Flash attention is active: quantized V cache (`-ctv q8_0`) forces `-fa auto` to resolve to enabled, and FA runs on Volta through the tile kernels.

## Measured performance (3-GPU host)

Native `/completion` endpoint, temperature 0, short-prompt decode (`n_predict 256`), ~2.6k-token prompt processing, and 8 concurrent × 192-token generations:

| Configuration | Single TG | Single PP | 8-parallel aggregate |
| --- | --- | --- | --- |
| Default build (no MMQ) | 38.7 tok/s | 725 tok/s | ~62.5 tok/s |
| **FORCE_MMQ build (this example)** | 38.7 tok/s | 616 tok/s | **~66 tok/s** |

The two builds trade prompt processing against concurrent generation throughput. This example ships the MMQ build (concurrent-throughput priority, +6%); to favour prompt-heavy workloads (RAG, image-heavy input), build without `-DGGML_CUDA_FORCE_MMQ` and keep everything else identical.

## Prerequisites

The host must already be provisioned — service account, shared directories, and the template unit. See [Provisioning the host](../../README.md#provisioning-the-host), or follow [`docs/SETUP.md`](../../docs/SETUP.md) for the whole procedure in order.

Pick an instance name — `llama` throughout this document — and create its directory:

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/llama /opt/llm-serv/llama/bin
```

`run` derives the engine binary and env file paths from its own install location, so any instance name works without editing the script.

## Building the engine

llama.cpp is compiled locally for Volta with the MMQ (quantized matmul) kernels forced on, and installed under `/opt/llm-serv/llama/bin/`.

```sh
git clone https://github.com/ggml-org/llama.cpp

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

cmake llama.cpp -B llama.cpp/build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70

cmake --build llama.cpp/build --config Release -j$(nproc)
```

| Option | Rationale |
| --- | --- |
| `-DGGML_CUDA=ON` | Enable the CUDA backend |
| `-DGGML_CUDA_FORCE_MMQ=ON` | Force quantized MMQ kernels. Measured +6% concurrent throughput at the cost of -15% prompt processing; omit for the prompt-processing-priority build |
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

The same `run` serves either GPU count; only the three values below change. The defaults are the 2-GPU profile; the 3-GPU profile is set via the env file (see [Environment variables](#environment-variables)).

### GPU allocation

| Flag | Value | Rationale |
| --- | --- | --- |
| `-ngl 99` | all layers | Offload every layer to GPU (a value above the real layer count is the idiomatic way to say "all") |
| `-ts 34,30` | 34:30 | 2-GPU split across the two cards, weighted to account for the other buffers GPU0 carries |
| `-ts 34,35,27` | 34:35:27 | 3-GPU split. Uneven on purpose. All cards are identical 32GB, but GPU2 carries fixed per-device allocations that equal splits cannot accommodate: `34,33,33` fails to start (OOM at MTP context creation) and `32,34,34` loads but crashes with a CUDA OOM during prompt processing |

### Context and slots

| Flag | Value | Rationale |
| --- | --- | --- |
| `-np 2` / `-np 8` | 2 / 8 | Parallel slots. 2 (default) or 8 (3-GPU profile) concurrent requests |
| `-c 524288` / `-c 1048576` | 512k / 1M | **Total** context across all slots — 256k (×2) or 128k (×3) per slot |
| `-ctk q8_0` / `-ctv q8_0` | Q8_0 | Quantized KV cache. Roughly halves KV memory versus F16, which is what makes the context fit. On the 3-GPU profile the model is hybrid (full attention every 4th layer, SSM elsewhere — 17 attention layers of 65), so the full 1M-token KV cache costs only ~37 GiB |
| `--no-kv-unified` | — | Keeps each slot's KV cache independent, so a long session cannot crowd out the other slot |

### Batch sizes

| Flag | Value | Rationale |
| --- | --- | --- |
| `-ub` (default) | 512 | The physical batch is left at the default. `-ub 1024` does not fit (OOM), and `-ub 768` fits but measured **-19% prompt processing** — larger microbatches hurt the PCIe-pipelined split on this card |
| `-b` (default) | 2048 | Logical batch; no measured effect |

### Latency and monitoring

| Flag | Rationale |
| --- | --- |
| `--cache-reuse 256` | Reuses displaced KV chunks for diverging prompts; zero measured cost, large TTFT win for multi-turn chat and shared system prompts |
| `--metrics` | Prometheus endpoint, for validating these numbers in production |
| `--mmproj mmproj-F16.gguf` | Vision projector; enables image input |
| `--image-min-tokens 1024` | Minimum token budget per image, so small images are not downscaled too aggressively |

### Speculative decoding

| Flag | Value | Rationale |
| --- | --- | --- |
| `--spec-type draft-mtp` | — | Uses the model's built-in MTP head (`nextn_predict_layers=1`) — no separate draft model needed |
| `--spec-draft-n-max 3` | 3 | Measured draft acceptance 0.5–0.67 at 3, which is exactly break-even against plain decode. `--spec-draft-n-max 6` is strictly worse (-17% single, -20% parallel) and is not used |

### Model and generation

| Flag | Rationale |
| --- | --- |
| `--reasoning-preserve` | Keeps the reasoning (thinking) content in the response |
| `--jinja` | Uses the chat template embedded in the GGUF |
| `--alias qwen3.8-27b` | Model name exposed by the API, decoupling clients from the on-disk filename |

## Environment variables

| Variable | Default | Rationale |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs exposed to llama-server. Set `0,1,2` for the 3-GPU host (with `LLM_TENSOR_SPLIT=34,35,27`) |
| `LLM_TENSOR_SPLIT` | `34,30` | `-ts` layer split. Set `34,35,27` for the 3-GPU host |
| `LLM_N_PARALLEL` | `2` | `-np` parallel slots. Set `8` for the 3-GPU host |
| `LLM_TOTAL_CTX` | `524288` | `-c` total context across all slots. Set `1048576` for the 3-GPU host |

## Authentication and networking

`run` validates `LLAMA_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0:8000` by default. Both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance.
