# Setup guide

End-to-end walkthrough for taking a bare GPU host to a running, verified inference endpoint.

Steps 0 to 3 are the same for every engine and model. From [step 4](#4-pick-an-example) on, the work depends on which example you are deploying: this guide gives the shape of each step and the parts that are common, and points at the example for the commands that differ. For the reasoning behind each piece of configuration, see the [top-level README](../README.md).

Commands that change the system are shown with `sudo`; everything else runs as your normal user. Running `sudo -v` first caches the credential so the longer steps do not stop to re-prompt.

## 0. Prerequisites

| Requirement | Check |
| --- | --- |
| systemd | `systemctl --version` |
| NVIDIA driver, GPUs visible | `nvidia-smi` |
| git | `git --version` |

Both examples build their engine from source, so the CUDA toolkit and compilers have to be on the host itself — or on a machine with the same CUDA version, from which only the build output is copied over. Which version, and which extra tools, depends on the example; see the table in [step 4](#4-pick-an-example).

`uv` is not assumed to be present. Install it for your own account — every example uses it to fetch models, and some to build the engine:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Because it lives in `~/.local/bin`, which `sudo` drops from the path, the steps that write into `/opt` invoke it as `sudo env "PATH=$PATH" uv ...`.

## 1. Service account

```sh
sudo useradd --system --no-create-home --shell /usr/sbin/nologin llm-serv
```

A system account with no home directory and no login shell. Every instance runs as this user, so model files and the engine binary must be readable by it.

If `/dev/nvidia*` is not world-accessible on this host, add the account to the owning group:

```sh
ls -l /dev/nvidia*                           # check the group
sudo usermod -aG <group> llm-serv            # only if needed
```

## 2. Directories

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv /opt/llm-serv/models
sudo install -d -m 0750 -o root     -g llm-serv /etc/llm-serv
```

`/var/log/llm-serv` is created by `tmpfiles.d` in the next step.

## 3. Base files

From a checkout of this repository:

```sh
sudo install -m 0644 etc/systemd/system/llm-serv@.service /etc/systemd/system/
sudo install -m 0644 etc/logrotate.d/llm-serv             /etc/logrotate.d/
sudo install -m 0644 etc/tmpfiles.d/llm-serv.conf         /etc/tmpfiles.d/

sudo systemd-tmpfiles --create /etc/tmpfiles.d/llm-serv.conf
sudo systemctl daemon-reload
```

Verify:

```sh
ls -ld /var/log/llm-serv                         # drwxr-x--- root root
sudo logrotate --debug /etc/logrotate.d/llm-serv
```

Everything above is engine-independent; adding a second engine later reuses all of it.

## 4. Pick an example

Everything from here depends on which example you deploy.

| | [llama.cpp](../examples/llama-v100x2-qwen3.8-27b) | [llama.cpp ×3](../examples/llama-v100x3-qwen3.8-27b) | [vLLM](../examples/vllm-l40sx8-deepseek-v4-flash-0731) |
| --- | --- | --- | --- |
| Hardware | V100 32GB ×2 (sm_70) | V100 32GB ×3 (sm_70) | L40S ×8 (sm_89) |
| CUDA | 12.8 | 12.8 | 13.0 |
| Extra build tools | cmake, gcc | cmake, gcc | rust, gh, Python 3.12 |
| Engine artifact | one static binary in `bin/` | one static binary in `bin/` | a virtualenv in `.venv/` |
| Model | `unsloth/Qwen3.8-27B-GGUF` | `unsloth/Qwen3.8-27B-GGUF` | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| API key variable | `LLAMA_API_KEY` | `LLAMA_API_KEY` | `VLLM_API_KEY` |
| Served model name | `qwen3.8-27b` | `qwen3.8-27b` | `deepseek-v4-flash-0731` |
| Instance name below | `llama` | `llama` | `vllm` |

The remaining steps write `<instance>` and `<example>` where the values from that table go. Create the instance directory:

```sh
sudo install -d -m 0755 -o llm-serv -g llm-serv /opt/llm-serv/<instance>
```

An example may call for a subdirectory as well — `bin/` for llama.cpp — which its README covers.

## 5. Install the engine

Neither example uses a prebuilt engine, and the two procedures have almost nothing in common, so each lives with its example:

- [llama.cpp ×2 — Building the engine](../examples/llama-v100x2-qwen3.8-27b/README.md#building-the-engine)
- [llama.cpp ×3 — Building the engine](../examples/llama-v100x3-qwen3.8-27b/README.md#building-the-engine)
- [vLLM — Building the engine](../examples/vllm-l40sx8-deepseek-v4-flash-0731/README.md#building-the-engine) and [Runtime virtualenv](../examples/vllm-l40sx8-deepseek-v4-flash-0731/README.md#runtime-virtualenv)

Two rules apply to both, and account for most of the failures at this step:

- Build as your normal user. Only the install into `/opt` needs `sudo`.
- Whatever the engine turns out to be, `llm-serv` must be able to read and execute it **and** everything it resolves at runtime — shared libraries, and for vLLM the Python interpreter the virtualenv links to. Anything left under `/root` is unreachable.

## 6. Models

The same shape for both: fetch into the shared store under the upstream org and repository name, then hand the files to the service account. The example README gives the exact repository and, for llama.cpp, which files to pick out of it.

```sh
sudo env "PATH=$PATH" uvx hf download <org>/<repo> \
  --local-dir /opt/llm-serv/models/<org>/<repo>/

sudo chown -R llm-serv:llm-serv /opt/llm-serv/models/<org>
```

The store is not writable by your account, hence `sudo`; `env "PATH=$PATH"` keeps `uvx` reachable, since `sudo` drops `~/.local/bin` from the path. Weights run to tens of GB, so check `/opt` has room. Confirm the service account can read what landed:

```sh
sudo -u llm-serv ls -l /opt/llm-serv/models/<org>/<repo>/
```

## 7. Environment file

```sh
sudo install -o llm-serv -g llm-serv -m 0640 \
  examples/<example>/env.example /etc/llm-serv/<instance>.env

sudoedit /etc/llm-serv/<instance>.env
```

Replace the placeholder with the real key. The variable is named for the engine — `LLAMA_API_KEY` or `VLLM_API_KEY` — because the engine itself reads it from the environment. `run` refuses to start when it is unset, so the server is never exposed unauthenticated by accident.

The same file accepts `LLM_SERV_HOST` and `LLM_SERV_PORT`. Both examples default to `0.0.0.0:8000`, so running them on one host means moving one of them.

## 8. Launch script

```sh
sudo install -o llm-serv -g llm-serv -m 0750 \
  examples/<example>/run /opt/llm-serv/<instance>/run
```

Review the flags before starting — GPU split, context length and batching are hardcoded here by design. Paths are not: the script derives the engine and env file locations from where it is installed, so it works under any instance name.

## 9. Start and verify

```sh
sudo systemctl enable --now llm-serv@<instance>
systemctl status llm-serv@<instance>
```

Loading a model of this size takes minutes; watch it happen. The logs are root-owned:

```sh
sudo tail -f /var/log/llm-serv/<instance>-stderr.log
```

Confirm every GPU is populated:

```sh
nvidia-smi
```

Then exercise the API, substituting the key variable and served model name from step 4:

```sh
source <(sudo cat /etc/llm-serv/<instance>.env)

curl -sf http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${LLAMA_API_KEY}"

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello."}]}'
```

The model should report itself under its served name rather than its on-disk path.

Finally, confirm it will come back on its own after a reboot:

```sh
systemctl is-enabled llm-serv@<instance>
```

## Operations

**Restart after editing `run` or the env file.** Neither is read by systemd, so no `daemon-reload` is involved:

```sh
sudo systemctl restart llm-serv@<instance>
```

**Change a unit setting for one instance only.** Use a drop-in instead of editing the template:

```sh
sudo systemctl edit llm-serv@<instance>   # writes .../override.conf
sudo systemctl restart llm-serv@<instance>
```

**Update the engine.** Rebuild and reinstall over the old artifact, then restart. Keep the previous one aside — a renamed binary, or a copy of the virtualenv — so rollback is a move and a restart.

**Update the model.** Download alongside the existing files, point `run` at the new path, restart. The old weights can be removed once the new ones are confirmed working.

**Run a second configuration.** Repeat from step 4 with a different instance name, setting `LLM_SERV_PORT=` in its env file. The base install from steps 1–3 is shared.

**Remove an instance.**

```sh
sudo systemctl disable --now llm-serv@<instance>
sudo systemctl clean --what=state llm-serv@<instance>
sudo rm -rf /opt/llm-serv/<instance> /etc/llm-serv/<instance>.env
sudo rm -f  /var/log/llm-serv/<instance>-*
```

## Troubleshooting

Read the exit status from `systemctl status llm-serv@<instance>`. systemd's own failures appear as `<code>/<NAME>` before the engine runs.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `209/STDOUT` | `/var/log/llm-serv` missing | `sudo systemd-tmpfiles --create /etc/tmpfiles.d/llm-serv.conf` (step 3) |
| `200/CHDIR` | `/opt/llm-serv/<instance>/` missing | Step 4 |
| `203/EXEC` | `run` not executable, or wrong shebang | `sudo chmod 0750 /opt/llm-serv/<instance>/run` |
| `Error: <ENGINE>_API_KEY is not set.` in stderr | env file missing, or unreadable by `llm-serv` | Step 7; check ownership and `0640` |
| Permission denied on a model or the engine | Files owned by root without group read | `sudo chown -R llm-serv:llm-serv` the offending path |
| `bad interpreter: Permission denied` | The virtualenv links to an interpreter `llm-serv` cannot reach | `readlink -f <venv>/bin/python3`; reinstall it somewhere readable (step 5) |
| `start request repeated too quickly` | 3 failures within 300s tripped the rate limit | Fix the cause, then `sudo systemctl reset-failed llm-serv@<instance>` |
| CUDA out of memory during load | Context or memory share too large for the cards | llama.cpp: lower `-c` or retune `-ts`. vLLM: lower `--gpu-memory-utilization` or `--max-model-len` |
| Startup appears to hang | Large models take minutes to load, and there is no start timeout | Watch `<instance>-stderr.log` until the listen line appears |
| `Address already in use` | Another instance holds the port | Set `LLM_SERV_PORT=` in `/etc/llm-serv/<instance>.env` |

Logs live in `/var/log/llm-serv/<instance>-{stdout,stderr}.log`, not in the journal. `journalctl -u llm-serv@<instance>` only shows start/stop events.
