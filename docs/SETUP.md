# Setup guide

End-to-end walkthrough for taking a bare GPU host to a running, verified inference endpoint.

The steps up to and including [Base files](#3-base-files) are shared by every engine and model. Everything after that is per-instance; this guide follows [`examples/llama-v100x2-qwen3.8-27b`](../examples/llama-v100x2-qwen3.8-27b) as the concrete case, using the instance name `llama`. For the reasoning behind each piece of configuration, see the [top-level README](../README.md) and the example's own README.

All commands are run as root unless stated otherwise.

## 0. Prerequisites

| Requirement | Check |
| --- | --- |
| systemd | `systemctl --version` |
| NVIDIA driver, GPUs visible | `nvidia-smi` |
| CUDA toolkit matching the example (12.8) | `ls /usr/local/cuda-12.8` |
| Build toolchain (`git`, `cmake`, `gcc`) | `cmake --version` |
| `uv` (for `uvx hf download`) | `uvx --version` |

The engine is built from source, so the toolchain is needed on the host itself — or on a machine with the same CUDA version, from which only the resulting binary is copied over.

## 1. Service account

```sh
useradd --system --no-create-home --shell /usr/sbin/nologin llm-serv
```

A system account with no home directory and no login shell. Every instance runs as this user, so model files and the engine binary must be readable by it.

If `/dev/nvidia*` is not world-accessible on this host, add the account to the owning group:

```sh
ls -l /dev/nvidia*                      # check the group
usermod -aG <group> llm-serv            # only if needed
```

## 2. Directories

```sh
install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv /opt/llm-serv/models
install -d -m 0750 -o root     -g llm-serv /etc/llm-serv
```

`/var/log/llm-serv` is created by `tmpfiles.d` in the next step.

## 3. Base files

From a checkout of this repository:

```sh
install -m 0644 etc/systemd/system/llm-serv@.service /etc/systemd/system/
install -m 0644 etc/logrotate.d/llm-serv             /etc/logrotate.d/
install -m 0644 etc/tmpfiles.d/llm-serv.conf         /etc/tmpfiles.d/

systemd-tmpfiles --create /etc/tmpfiles.d/llm-serv.conf
systemctl daemon-reload
```

Verify:

```sh
ls -ld /var/log/llm-serv                    # drwxr-x--- root root
logrotate --debug /etc/logrotate.d/llm-serv
```

Everything above is engine-independent; adding a second engine later reuses all of it.

## 4. Instance directory

Pick the instance name — it becomes the `%i` in `llm-serv@<instance>` and determines every per-instance path. This guide uses `llama`.

```sh
install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/llama /opt/llm-serv/llama/bin
```

## 5. Build the engine

For the V100 example, llama.cpp is compiled for sm_70. Run this as an unprivileged user; only the install step needs root.

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

```sh
install -m 0755 llama.cpp/build/bin/llama /opt/llm-serv/llama/bin/llama
```

Confirm the binary runs as the service account. The CUDA runtime is linked dynamically, so the library path has to be given here the same way `run` sets it:

```sh
sudo -u llm-serv env LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
  /opt/llm-serv/llama/bin/llama --version
```

See the [example README](../examples/llama-v100x2-qwen3.8-27b/README.md#building-the-engine) for what each cmake option buys.

## 6. Models

```sh
uvx hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  mmproj-F16.gguf \
  --local-dir /opt/llm-serv/models/unsloth/Qwen3.8-27B-GGUF/

chown -R llm-serv:llm-serv /opt/llm-serv/models/unsloth
```

The weights run to tens of GB; make sure `/opt` has room. The paths must match what `run` refers to:

```sh
sudo -u llm-serv ls -l /opt/llm-serv/models/unsloth/Qwen3.8-27B-GGUF/
```

## 7. Environment file

```sh
install -o llm-serv -g llm-serv -m 0640 \
  examples/llama-v100x2-qwen3.8-27b/env.example /etc/llm-serv/llama.env
```

Then edit `/etc/llm-serv/llama.env` and replace the placeholder with the real key:

```sh
LLAMA_API_KEY="sk-..."
```

`run` refuses to start when this is unset, so the server is never exposed unauthenticated by accident.

## 8. Launch script

```sh
install -o llm-serv -g llm-serv -m 0750 \
  examples/llama-v100x2-qwen3.8-27b/run /opt/llm-serv/llama/run
```

Review the flags before starting — GPU split, context length, and port are all hardcoded here by design. Paths are not: the script derives the engine binary and env file locations from where it is installed, so it works under any instance name.

## 9. Start and verify

```sh
systemctl enable --now llm-serv@llama
systemctl status llm-serv@llama
```

Model loading takes a while; watch it happen. The logs are root-owned, so this needs root or `sudo`:

```sh
tail -f /var/log/llm-serv/llama-stderr.log
```

Confirm both GPUs are populated:

```sh
nvidia-smi
```

Then exercise the API:

```sh
source /etc/llm-serv/llama.env

curl -sf http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${LLAMA_API_KEY}"

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello."}]}'
```

The model should report itself under the alias `qwen3.8-27b` rather than the GGUF filename.

Finally, confirm it will come back on its own after a reboot:

```sh
systemctl is-enabled llm-serv@llama
```

## Operations

**Restart after editing `run` or the env file.** Neither is read by systemd, so no `daemon-reload` is involved:

```sh
systemctl restart llm-serv@llama
```

**Change a unit setting for one instance only.** Use a drop-in instead of editing the template:

```sh
systemctl edit llm-serv@llama       # writes .../llm-serv@llama.service.d/override.conf
systemctl restart llm-serv@llama
```

**Update the engine.** Rebuild, reinstall the binary, restart. Keeping the previous binary as `llama.prev` makes rollback a copy and a restart.

**Update the model.** Download alongside the existing files, point `run` at the new filename, restart. The old weights can be removed once the new ones are confirmed working.

**Run a second configuration.** Repeat from step 4 with a different instance name and a different `PORT` in its `run`. The base install from steps 1–3 is shared.

**Remove an instance.**

```sh
systemctl disable --now llm-serv@llama
rm -rf /opt/llm-serv/llama /etc/llm-serv/llama.env
rm -f  /var/log/llm-serv/llama-*
```

## Troubleshooting

Read the exit status from `systemctl status llm-serv@<instance>`. systemd's own failures appear as `<code>/<NAME>` before the engine runs.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `209/STDOUT` | `/var/log/llm-serv` missing | `systemd-tmpfiles --create /etc/tmpfiles.d/llm-serv.conf` (step 3) |
| `200/CHDIR` | `/opt/llm-serv/<instance>/` missing | Step 4 |
| `203/EXEC` | `run` not executable, or wrong shebang | `chmod 0750 /opt/llm-serv/<instance>/run` |
| `Error: LLAMA_API_KEY is not set.` in stderr | env file missing, or unreadable by `llm-serv` | Step 7; check ownership and `0640` |
| Permission denied on a model or the binary | Files owned by root without group read | `chown -R llm-serv:llm-serv` the offending path |
| `start request repeated too quickly` | 3 failures within 300s tripped the rate limit | Fix the cause, then `systemctl reset-failed llm-serv@<instance>` |
| CUDA out of memory during load | Context or split too large for 64 GB | Lower `-c`, or retune `-ts` in `run` |
| Startup appears to hang | Large models take minutes to load, and there is no start timeout | Watch `<instance>-stderr.log` until the listen line appears |
| `Address already in use` | Another instance holds the port | Change `PORT` in `run` |

Logs live in `/var/log/llm-serv/<instance>-{stdout,stderr}.log`, not in the journal. `journalctl -u llm-serv@<instance>` only shows start/stop events.
