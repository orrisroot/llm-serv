# llm-serv

A minimal framework for running LLM inference servers as **systemd services** on a local GPU machine.

Everything that depends on the engine (llama.cpp, vLLM, ...) or the model is confined to a single launch script named `run`. The systemd unit itself is an engine-agnostic template: adding a new engine or model never requires touching the unit file.

## Design principles

1. **One template unit covers every configuration.** Instantiating `llm-serv@.service` is all it takes to bring up a new engine or model.
2. **Launch flags are hardcoded in `run`.** Configuration is not scattered across `Environment=` directives or ambient variables. Reading `run` tells you exactly how that instance starts.
3. **Only secrets live in `EnvironmentFile`.** API keys go to `/etc/llm-serv/<instance>.env` and are never tracked in this repository.
4. **Logs stay out of the journal.** Inference servers are chatty, so output goes to dedicated files managed by logrotate.

## Repository layout

```
README.md
docs/SETUP.md                           End-to-end setup walkthrough
etc/                                    ← Installed to the host as-is
  systemd/system/llm-serv@.service        Template unit
  logrotate.d/llm-serv                    Log rotation policy
  tmpfiles.d/llm-serv.conf                Creates /var/log/llm-serv at boot
examples/                               ← Reference configurations, meant to be copied
  llama-v100x2-qwen3.8-27b/
    README.md                             Assumptions and rationale for this setup
    run                                   Launch script
    env.example                           Environment file template
```

Every file under `etc/` is installed to the identical path on the host, and holds the engine-independent base. Engine-specific pieces live under `examples/` and are copied into place at deploy time.

Directories that only need to exist — `/opt/llm-serv`, `/etc/llm-serv`, `/var/log/llm-serv` — are created with the right ownership in [Provisioning the host](#provisioning-the-host).

### Deployment mapping

| Repository path | Host path | Ownership / mode |
| --- | --- | --- |
| `etc/systemd/system/llm-serv@.service` | same | `0644 root:root` |
| `etc/logrotate.d/llm-serv` | same | `0644 root:root` |
| `etc/tmpfiles.d/llm-serv.conf` | same | `0644 root:root` |
| `examples/<name>/run` | `/opt/llm-serv/<instance>/run` | `0750 llm-serv:llm-serv` |
| `examples/<name>/env.example` | `/etc/llm-serv/<instance>.env` | `0640 llm-serv:llm-serv` |
| (untracked) | `/opt/llm-serv/models/` | shared model store |
| (created by tmpfiles.d) | `/var/log/llm-serv/` | `0750 root:root` |

## Provisioning the host

These steps are shared by every instance and only need to be done once. They are written for a normal account with `sudo`.

### Service user

All engines run as a dedicated unprivileged system account with no home directory and no login shell:

```sh
sudo useradd --system --no-create-home --shell /usr/sbin/nologin llm-serv
```

If the NVIDIA device nodes on the host are not world-accessible, add the account to the group that owns `/dev/nvidia*` as well.

### Directories

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv /opt/llm-serv/models
sudo install -d -m 0750 -o root     -g llm-serv /etc/llm-serv
```

The log directory is created by `tmpfiles.d` in the next step.

### Base files

```sh
sudo install -m 0644 etc/systemd/system/llm-serv@.service /etc/systemd/system/
sudo install -m 0644 etc/logrotate.d/llm-serv             /etc/logrotate.d/
sudo install -m 0644 etc/tmpfiles.d/llm-serv.conf         /etc/tmpfiles.d/

sudo systemd-tmpfiles --create /etc/tmpfiles.d/llm-serv.conf   # creates the log dir now
sudo systemctl daemon-reload
```

`systemd-tmpfiles --create` creates the log directory now; later boots handle it automatically. `daemon-reload` is only needed here, when the template unit itself is installed or changed.

## Instance conventions

Running `systemctl start llm-serv@<instance>` expands `%i` in the template unit to `<instance>`, which resolves to:

| Item | Path |
| --- | --- |
| Working directory | `/opt/llm-serv/<instance>/` |
| Entry point | `/opt/llm-serv/<instance>/run` |
| Environment file | `/etc/llm-serv/<instance>.env` (prefixed with `-`, so it is optional) |
| stdout / stderr | `/var/log/llm-serv/<instance>-stdout.log` / `-stderr.log` |
| Engine binary | `/opt/llm-serv/<instance>/bin/` (convention only; the path is referenced from `run`) |

`<instance>` is an arbitrary identifier — an engine name (`llama`) or an engine-plus-model name (`llama-qwen3`) both work. To run multiple configurations of the same engine side by side, give them distinct instance names and assign separate ports.

Services run as the dedicated `llm-serv` user, so model files and engine binaries must be readable by that user.

## Adding a new configuration

1. Copy the closest match from `examples/` and adjust `run` — model paths, GPU split, port, and so on.
2. Create the instance directory: `sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/<instance>`.
3. Install `run` to `/opt/llm-serv/<instance>/run` (`0750 llm-serv:llm-serv`).
4. If the engine needs an API key or similar, create `/etc/llm-serv/<instance>.env` (`0640 llm-serv:llm-serv`).
5. Start it with `sudo systemctl enable --now llm-serv@<instance>`. No `daemon-reload` is needed, since the unit itself is unchanged.

To keep the configuration in this repository as well, add it under `examples/<engine>-<hardware>-<model>/` and document its assumptions — GPU layout, CUDA version, model, and how the engine binary was built — in a `README.md` alongside it.

## Shared resource limits and restart policy

These are applied uniformly to every instance by the template unit.

| Setting | Value | Rationale |
| --- | --- | --- |
| `TasksMax` | infinity | Lifts the thread-count limit for inference engines |
| `LimitNOFILE` | 65536:524288 | Raises the soft limit for a keep-alive HTTP server, keeping the default hard limit |
| `LimitMEMLOCK` | infinity | Page-locked memory for `--mlock` and RDMA/GPUDirect paths |
| `Restart` / `RestartSec` | on-failure / 10s | Restart only on abnormal exit |
| `StartLimitBurst` / `IntervalSec` | 3 / 300s | Give up after 3 failures within 300 seconds |
| `TimeoutStopSec` | 120s | SIGKILL 120 seconds after SIGTERM |
| `KillMode` | mixed | SIGTERM reaches the engine only, so multi-process engines stop their own workers |
| `UMask` | 0027 | Log files created `0640` rather than `0644` |
| `After=` | `network-online.target`, `nvidia-persistenced.service` | Start after the GPU driver is initialized |

There is no start timeout: a slow model load simply takes as long as it takes.

The template sets no memory cap. With the model resident in VRAM, the only host memory of any size is the memory-mapped weight file, and those pages are reclaimable — a `MemoryMax=` would mostly add a way for the cgroup OOM killer to kill a healthy server. On a shared host where the engine must be boxed in, add one per instance.

### Per-instance overrides

Any unit setting can be changed for a single instance without touching the template:

```sh
sudo systemctl edit llm-serv@<instance>
```

```ini
[Service]
MemoryMax=96G
```

This writes `/etc/systemd/system/llm-serv@<instance>.service.d/override.conf`, which survives updates to the template unit.

## Logging

- Destination: `/var/log/llm-serv/<instance>-stdout.log` and `<instance>-stderr.log`
- Rotation: monthly, 120 generations (~10 years), gzip compressed
- `copytruncate` is used: systemd's `append:` holds the file open, so the log must be truncated in place rather than renamed.

The directory is created by `etc/tmpfiles.d/llm-serv.conf`, which runs early in boot. `LogsDirectory=` in the unit would be too late, since systemd opens the log files before a unit's exec directories exist.

### Ownership and permissions

Logs are root-owned end to end and are read with `sudo`: the directory is `0750 root:root`, the files inside `0640 root:root`. systemd opens them while still privileged and hands the descriptors to the service, so the `llm-serv` account needs no access here.

`UMask=0027` in the unit sets the file mode, and the rotated `.gz` copies inherit it. The logrotate stanza has no `create` directive because `copytruncate` never recreates the live file, and no `su` directive because rotation must run as root to truncate root-owned files.

Checking on a running instance:

```sh
systemctl status llm-serv@<instance>
sudo tail -f /var/log/llm-serv/<instance>-stderr.log
```

The journal only records service start/stop events; inference server output is not duplicated there.

## Setup

[`docs/SETUP.md`](docs/SETUP.md) walks through a bare host to a verified endpoint in order, including engine build, model download, verification, and troubleshooting. The sections above are the reference material behind it; each example's `README.md` covers the parts specific to that engine and hardware.

## Operational notes

- Real API keys are never tracked here. `examples/` only ships templates (`env.example`).
- Model weights and engine binaries are untracked as well. Models belong under `/opt/llm-serv/models/` and are shared across instances; engine binaries are per-instance, and the build procedure is documented in each example.

## Available examples

| Directory | Engine | Hardware | Model |
| --- | --- | --- | --- |
| [`examples/llama-v100x2-qwen3.8-27b`](examples/llama-v100x2-qwen3.8-27b) | llama.cpp | Tesla V100 32GB ×2 / CUDA 12.8 | Qwen3.8 27B (GGUF, UD-Q8_K_XL, multimodal) |
