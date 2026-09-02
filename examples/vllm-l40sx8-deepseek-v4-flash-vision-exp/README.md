# vLLM / L40S ×8 / DeepSeek V4 Flash Vision Exp

Serves DeepSeek V4 Flash Vision Exp across eight L40S GPUs with vLLM, using 8-way tensor parallelism with expert parallelism over the MoE layers, DSA sparse attention and dspark speculative decoding.

vLLM is not installed from PyPI here. Ada (sm_89) support for this model comes from the [`yhfgyyf/vllm-deepseek-v4-sm89`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) fork, which publishes prebuilt CUDA-13.2 wheels on its release page.

## Target environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA L40S ×8 (Ada / sm_89) |
| CUDA | 13.2 (`/usr/local/cuda-13.2`) |
| Python | 3.12 |
| Engine | `yhfgyyf/vllm-deepseek-v4-sm89` (prebuilt `vision7` wheel) |
| FlashInfer | `0.6.18+glm53.dsv4.vision1` (bundled, TP8-capable) |
| Model | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
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

# gh, used to pull the fork's prebuilt wheels
curl -OL https://github.com/cli/cli/releases/download/v2.97.0/gh_2.97.0_linux_amd64.tar.gz
tar zxvf gh_2.97.0_linux_amd64.tar.gz
mv gh_2.97.0_linux_amd64/bin/gh ~/.local/bin/
rm -rf gh_2.97.0_linux_amd64 gh_2.97.0_linux_amd64.tar.gz
```

### Runtime virtualenv

The virtualenv is created at its final path — a venv bakes absolute paths into its scripts, so it cannot be built elsewhere and moved. Its interpreter has to sit somewhere `llm-serv` can reach, so install that under `/opt/llm-serv` as well:

```sh
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv python install 3.12
sudo env "PATH=$PATH" UV_PYTHON_INSTALL_DIR=/opt/llm-serv/python \
  uv venv --python 3.12 --seed /opt/llm-serv/vllm/.venv

gh release download v0.28.1rc0-vision-sm89-sm120-cu130 \
  --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_python-0.6.18+glm53.dsv4.vision1.sm89sm120.cu130.pt213-*.whl' \
  --pattern 'vllm-*glm53.dsv4.vision7.sm89sm120.cu130-*.whl' \
  --pattern SHA256SUMS \
  --dir /tmp/vllm-sm89-sm120-vision-release

(cd /tmp/vllm-sm89-sm120-vision-release && sha256sum -c SHA256SUMS)

sudo env "PATH=$PATH" VIRTUAL_ENV=/opt/llm-serv/vllm/.venv UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  uv pip install /tmp/vllm-sm89-sm120-vision-release/vllm-*glm53.dsv4.vision7.sm89sm120.cu130-*.whl \
  --torch-backend=cu130

rm -rf /tmp/vllm-sm89-sm120-vision-release
```

Use the `vision7` wheel (`vllm-0.28.1rc0.dev293+gcb7a435391.glm53.dsv4.vision7.sm89sm120.cu130-*.whl`) and the bundled FlashInfer `0.6.18+glm53.dsv4.vision1` from this release. The FlashInfer wheel is TP=8 capable out of the box — the SparseMLA kernels handle the 8-head-per-rank split natively and need no patching.

The `vision7` vLLM wheel itself needs **one** post-install patch before it will serve at `max-num-seqs` above ~4. This adds `.contiguous()` to the C128A sparse-attention decode index tensor before it is handed to FlashInfer; it edits the wheel in place, so a reinstall or venv recreation must re-apply it. Apply it straight from this example's `patches/` directory at venv build time (install `patch` first if not already present):

```sh
sudo patch -p1 -d /opt/llm-serv/vllm/.venv/lib/python3.12/site-packages \
  < patches/vllm-c128a-eidx-contiguous.patch
```

### Scripts

```sh
sudo install -o root -g llm-serv -m 0750 run         /opt/llm-serv/vllm/run
sudo install -o root -g llm-serv -m 0640 env.example /etc/llm-serv/vllm.env
sudoedit /etc/llm-serv/vllm.env   # replace the placeholder with the real key
```

### Model files

```sh
sudo env "PATH=$PATH" uvx hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir /opt/llm-serv/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/
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
| `--enable-expert-parallel` | on | Shards the MoE experts across all eight ranks, so each GPU holds far fewer expert weights. The leftover router/shared expert layers stay tensor-parallel. This is the mechanism that makes the model + 512k KV cache + DSpark draft model fit at all |
| `--gpu-memory-utilization <f>` | 0.960 | CUDA graph memory profiling is on by default, which makes 0.90 behave like ~0.82 and leaves only ~12.7 GiB (2.5x at 512k) for KV. 0.960 restores ~15.4 GiB KV (≈3.05x at 512k) while keeping the concurrency above 3x. Do not push toward ~0.978: that leaves only ~180 MiB free, and a vision/multimodal request that JIT-compiles kernels at inference (Triton `fp8_mqa_logits_triton`, TileLang fused vision/norm recompile) overflows the last ~190 MiB and crashes all ranks with CUDA OOM. The lower value keeps a few hundred MiB of headroom to absorb that transient workspace spike. Set `LLM_GPU_MEMORY_UTILIZATION` to override |

| `--max-model-len <n>` | 512k | Context length per sequence. Set `LLM_MAX_MODEL_LEN` to override |
| `--max-num-seqs <n>` | 32 | Concurrent sequences. At the full 512k context the KV cache only allows ~3.2 concurrent, but for the shorter real-world contexts this model is served with (a few k to tens of k tokens per request) the ceiling is far higher, so 32 lets one batch carry far more sequences — measured +30% output token throughput and ~1.3x lower time-to-complete versus 16. Values above ~4 require the C128A contiguity patch described in the build section. Set `LLM_MAX_NUM_SEQS` to override |
| `--max-num-batched-tokens 4096` | 4096 | Per-step scheduling budget; high enough to absorb the spec-decode slot overhead (see the env table) without letting prefill chunks stretch decode latency under long-prompt load |
| `--kv-cache-dtype fp8_ds_mla` | FP8 | DeepSeek MLA-specific FP8 KV cache; what makes 512k context affordable |
| `--block-size 256` | 256 | Large paged-attention blocks, matched to the MLA sparse kernel |

### DeepSeek V4 specifics

| Flag | Rationale |
| --- | --- |
| `--trust-remote-code` | The model ships custom modelling code that vLLM must import |
| `--tokenizer-mode deepseek_v4` | Uses the DeepSeek V4 tokenizer implementation (chat-template aware tokenization) instead of the stock one; matches how the fork's sample config runs this model |
| `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` | DSA sparse attention path from the FlashInfer build in this fork |
| `--moe-backend auto` | Picks the MoE kernel backend to match the loaded weights (needed with expert parallelism) |
| `--interleave-mm-strings` | Fits multimodal inputs into the prompt context window so vision and text interleave without exhausting it |
| `--enable-prefix-caching` | Reuses KV across chat turns that share a prefix (prompt caching), which helps most with the multimodal prefill |
| `--speculative-config '{"method":"dspark",...}'` | dspark speculative decoding, 3 draft tokens, probabilistic draft sampling |
| `--reasoning-parser deepseek_v4` | Splits reasoning content out of the response |
| `LLM_REASONING_EFFORT` (env) | Default thinking effort for requests that do not set one. The DeepSeek V4 template treats `none` as off and `max`/`xhigh` as maximum; any other value falls back to the model default. Wired through `--default-chat-template-kwargs`; a per-request `reasoning_effort` still overrides it |
| `LLM_ENABLE_THINKING` (env) | Turns thinking off entirely when set to `false` (`enable_thinking=false`), skipping the reasoning path. The model default is on |
| `--tool-call-parser deepseek_v4` / `--enable-auto-tool-choice` | Parses the model's tool-call format and lets it pick tools itself |
| `--served-model-name deepseek-v4-flash-vision-exp` | Model name exposed by the API, decoupling clients from the checkout path |

### Environment

| Variable | Rationale |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Tensor-parallel GPUs. Defaults to all eight; the degree is derived from the count unless `LLM_TENSOR_PARALLEL_SIZE` is set |
| `LLM_TENSOR_PARALLEL_SIZE` | Forces the tensor-parallel degree instead of deriving it from the device count |
| `LLM_MAX_MODEL_LEN` / `LLM_MAX_NUM_SEQS` | Context length and concurrent sequences, defaulting to 512k / 32. At 512k the KV cache only sustains ~3.2 concurrent; lower context lengths scale the ceiling roughly as `num_gpu_blocks × block_size ÷ context`, which is why 32 concurrent pays off at the short contexts this model is usually served at |
| `LLM_MAX_NUM_BATCHED_TOKENS` | Per-step scheduling budget, defaulting to 4096. Needs to fit the spec-decode overhead — roughly `max_num_seqs × (1 + num_speculative_tokens)` — or vLLM caps `max_num_scheduled_tokens` below the budget and batch throughput suffers |
| `LLM_NUM_SPECULATIVE_TOKENS` | DSpark draft depth, defaulting to 3 (the vision profile validated on this host). Acceptance falls off sharply past ~4-5 positions |
| `LLM_GPU_MEMORY_UTILIZATION` | GPU memory ceiling, defaulting to 0.960 to compensate for the CUDA graph memory profiling reservation while leaving headroom for the vision path's JIT-at-inference spike; see the `--gpu-memory-utilization` row |
| `TILELANG_CACHE_DIR` / `TRITON_CACHE_DIR` / `FLASHINFER_WORKSPACE_BASE` | JIT kernel caches, pinned under the per-instance `$HOME` (`/var/lib/llm-serv/<instance>/.cache/`) so Triton, TileLang and FlashInfer kernels compiled on one start are reused on the next instead of recompiled |
| `NCCL_SHM_DISABLE=1` | Shared-memory transport is unusable between the ranks here; NCCL falls back to peer-to-peer |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | The FlashInfer wheel is pinned to this fork and fails the stock version check |
| `LD_LIBRARY_PATH` | The venv's bundled shared objects (`PyNvVideoCodec`, `lib/`) followed by the CUDA 13.2 runtime |

## Authentication and networking

`run` validates `VLLM_API_KEY` at startup and exits if it is unset, so the server is never exposed unauthenticated.

It binds to `0.0.0.0:8000` by default. Both are overridable from `/etc/llm-serv/<instance>.env` without touching `run` — set `LLM_SERV_HOST=127.0.0.1` to keep it on the loopback, or `LLM_SERV_PORT=` to move it out of the way of another instance. The reference deployment keeps the engine on a private network and fronts it with an httpd reverse proxy that maps per-user API keys onto this single backend key.

Note that `--trust-remote-code` executes Python shipped in the model repository. Pin the model revision if the checkout is refreshed from upstream.
