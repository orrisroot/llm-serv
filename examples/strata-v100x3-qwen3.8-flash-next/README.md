# Strata (V100 ×3) — Qwen3.8-Flash-Next IQ3_S

[Strata](https://github.com/Niko1221/Strata) is a Mixture-of-Experts inference engine that splits one
checkpoint across GPU VRAM, host RAM and NVMe and keeps the experts that are actually used in VRAM by
adapting to the conversation. It is the practical way to run Qwen3.8-Flash-Next (a 125B-total / 6B-active
MoE with a 28.8 GB n-gram lookup table and an MTP/DFlash2 drafter) on a machine whose GPUs cannot hold it.

## Assumptions

| Item | Value |
| --- | --- |
| Host | Linux x86-64 multi-GPU host, PCIe, no NVLink |
| GPU | 3x Tesla V100-PCIE-32GB (compute capability 7.0, Volta), driver 580.x |
| CPU | Slow modest-core CPU (see the tuning notes) |
| RAM / disk | Expert arena in host memory; model store in the shared `/opt/llm-serv/models/` tree |
| Engine | Strata Terminal (release), **CUDA 12 engine package** — CUDA 13 dropped Volta, so the project ships a separate `engine-cuda12` package (`-DSTRATA_EXPERIMENTAL_SM60=ON`). It is selected automatically on a Volta-only machine, and the startup log says `CUDA 12: ...`. |
| Model | Qwen3.8-Flash-Next, ISTA-DASLab **IQ3_S** GSQ-RCO pack (about 84 GB download; ~50 GiB of experts + dense weights; native 262,144-token context) |
| Layout | 3-GPU layer split (`--gpu 0,1,2`): experts cached in VRAM per card, expert arena in RAM, n-gram lookup table read from the SSD |

Verified per the project's older-GPU matrix and V100 page
([OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/main/docs/OLDER_GPUS.md),
[NVIDIA_V100.md](https://github.com/Niko1221/Strata/blob/main/docs/NVIDIA_V100.md)): Volta runs on the
experimental CUDA 12 engine with dedicated kernels (BF16 projections on FP16 tensor cores, Volta prompt
attention, expert mode 8). The kernels are contributed and measured by the community; treat speed numbers
as machine-specific.

## One-time provisioning

Run the other examples' provisioning steps (service account, base files, log dir) first — see the repo
README. Then:

1. **Instance files.** Install `run` and the env file; put the API key (from the operator's secret store,
   see `env.example` — the value is never committed) into the env file:

   ```sh
   sudo install -d -m 0755 -o root -g root /opt/llm-serv/strata
   sudo install -m 0750 -o root -g llm-serv run /opt/llm-serv/strata/run
   sudo install -m 0640 -o root -g llm-serv env.example /etc/llm-serv/strata.env
   # edit /etc/llm-serv/strata.env and set STRATA_API_KEY=...
   ```

2. **Strata Terminal for the service account.** The service user has no home directory, so the installer
   runs with `HOME` pointed at the directory the unit will later create for the instance:

   ```sh
   sudo install -d -o llm-serv -g llm-serv /var/lib/llm-serv/strata
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o /tmp/strata-install.sh
   sudo -u llm-serv env HOME=/var/lib/llm-serv/strata bash /tmp/strata-install.sh
   ```

3. **Model store.** Make the model folder readable (and writable for pack preparation) by the service
   user. The download and preparation are also done as that user below, so own it outright:

   ```sh
   sudo install -d -o llm-serv -g llm-serv /opt/llm-serv/models/strata
   ```

4. **One-time setup+download as `llm-serv`** (84 GB + engine package; resumes if interrupted):
   this downloads the model files, prepares the pack, fetches the CUDA 12 engine and *draft layer* for
   this account, writes the run config, and leaves the prepared state for `strata serve`:

   ```sh
   sudo -u llm-serv bash -c 'cd /var/lib/llm-serv/strata && \
     env HOME=/var/lib/llm-serv/strata \
     /var/lib/llm-serv/strata/.local/bin/strata launch \
     --model iq3_s --data-dir /opt/llm-serv/models/strata --context 256k --vision gpu \
     --gpu 0,1,2 --access lan --no-api-key --port 8000 --yes --no-tui'
   sudo -u llm-serv bash -c 'cd /var/lib/llm-serv/strata && \
     env HOME=/var/lib/llm-serv/strata \
     /var/lib/llm-serv/strata/.local/bin/strata launch --stop'
   ```

   (`cd` into the service account's home as shown: Strata resolves state relative to the working
   directory.)

   `--no-api-key` is required: the key comes from `STRATA_API_KEY` in the env
   file (exported by `run` and read by `strata serve`), and a launch-generated
   key would shadow it. `strata launch --no-tui` starts a daemonized server —
   fine for the setup run; the instance itself runs the foreground
   `strata serve`, which `run` invokes. The generated run config lands in
   `$HOME/.local/share/strata/install/strata-*.json`; the `run` script finds it
   automatically (or set `STRATA_PROFILE` in the env file).

5. **Start the instance.**

   ```sh
   sudo systemctl enable --now llm-serv@strata
   systemctl status llm-serv@strata
   curl http://127.0.0.1:8000/health
   ```

## Operational notes

- **API key.** `run` requires `STRATA_API_KEY` from `/etc/llm-serv/strata.env` (never committed to this
  repository). The setup launch runs with `--no-api-key`, so the secret never enters the run config.
- **Update.** `sudo -u llm-serv env HOME=... strata update`; then restart the instance. The engine is
  fetched from the signed release feed, so no CUDA toolkit is needed on the host.
- **First start.** Pack preparation plus loading ~30 GiB of pinned arena takes a few minutes.

## Tuning

The launch flow supports `--tune quick` (minutes) / `--tune thorough` and, per the Strata V100 page,
these opt-in switches measured on Volta (verified on this 3-GPU split; see "Tuned settings" below):

| Switch | Effect (contributed measurements) |
| --- | --- |
| `STRATA_SM70_TABLE=1` | three Volta decode kernels together (bitwise the defaults) |
| `STRATA_PF_WMMA=1` | FP16 tensor-core prompt experts (~+10% prefill) |
| `STRATA_PF_WMMA2=1` | second prompt kernel (+2.5-4.3%) |
| `STRATA_Q8_GATHER_WIDE=0` | int8 KV gather (~+11% on that kernel) |
| `STRATA_GDN_REC_WARP=2` | DeltaNet recurrence (+6-9%) |
| `STRATA_ARGMAX_ROWS_SM70=1`, `STRATA_GR_NORM_FAST=1` | small decode gains |

They are measured on a single V100; verify each on this 3-GPU split before keeping it. Reset with
`strata launch --set KEY=off` or by removing the key from the run config and restarting.

## Benchmarks

Measured with this example's settings (IQ3_S, 256K context, 3-GPU layer split,
`"parallel": 6`; `bench.py`/`stress.py`, reasoning off, greedy, 256 output tokens; prompt reads are
fresh — no prefix cache).

| Workload | Result |
| --- | --- |
| Prompt read, 3.9k tokens | ~2,130 tok/s (TTFT ≈1.9 s) |
| Prompt read, 30.9k tokens | ~3,535 tok/s (TTFT ≈8.7 s) |
| Prompt read, ~262k tokens (max context) | ~3,257 tok/s (TTFT ≈80 s) |
| Decode, 4k context (single stream) | 80.8 tok/s |
| Decode, 4k context (6 parallel) | 60.3 tok/s |
| Decode, 32k context | 84.4 tok/s |
| Decode, 262k context (max) | 49.2 tok/s |
| Decode range across content | 40–157 tok/s (DFlash2 acceptance is content-dependent; code/continuations draft best) |
| 6 concurrent (1k prompt, 128 out) | ~127 tok/s aggregate; p50 10.2 s, p95 11.1 s |

Against the same host's `vllm-v100x2-qwen3.8-27b` example (Qwen3.8 27B FP8, vLLM, 2×V100):
single-stream decode 38.8 tok/s → **about 2.2× here**, prefill 1,770 tok/s at a cold 9k prompt →
2,130 (3.9k) to 3,535 (30.9k) tok/s here.

The [sfnx example](../sfnx-v100x3-qwen3.8-flash-next/) (UD-Q4_K_XL) on the same host measured under
the same harness: decode 4k **156.9** tok/s (single stream), decode 262k **96.5** tok/s, 262k prefill
1,907 tok/s, 6-way aggregate **138** tok/s (p50 6.6 s). Strata wins long-prompt prefill (3,257 vs
1,907 tok/s); sFNX wins decode and concurrency.

## Tuned settings

The run config (`$HOME/.local/share/strata/install/strata-*.json`) carries these; on another
V100 box add them to the setup launch with `--set KEY=VALUE`:

- `"parallel": 6` — six decode slots, matching the sfnx example's six lanes for the comparison rows
  above. The default is 1 (single-stream decode 80.8 tok/s at 4k).
- `--prefill-pipe 1024` — prompt chunk pipelining on the layer split: 4k prefill +41% over default.
- `env`: `STRATA_SM70_TABLE=1`, `STRATA_PF_WMMA=1`, `STRATA_PF_WMMA2=1`,
  `STRATA_Q8_GATHER_WIDE=0`, `STRATA_GDN_REC_WARP=2`, `STRATA_ARGMAX_ROWS_SM70=1`,
  `STRATA_GR_NORM_FAST=1`, `STRATA_FUSE_DECODE=1` — the V100 (sm_70) kernels from
  [Strata's NVIDIA_V100.md](https://github.com/Niko1221/Strata/blob/main/docs/NVIDIA_V100.md):
  prefill +14–15%, decode within noise on this 3-GPU split.
  `STRATA_PF_WMMA=1` and `STRATA_GDN_REC_WARP=2` change FP rounding (answers can differ in the
  last bits); the rest are bitwise the defaults.

All experts are resident in VRAM (24,576 of 24,576 profiled pairs, 8,192 slots per
card), so the CPU pool is idle during decode and decode speed tracks the GPU kernels and draft
acceptance.
