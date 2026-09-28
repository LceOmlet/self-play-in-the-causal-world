# Isolated NumPy/SciPy tool mode

The model sees exactly `act(command: object)` and `numerics(code: string)`.
`act` delegates to the original CPT environment without changing permissions,
observation charges, terminal rewards or task generation. `numerics` executes
ordinary Python with complete NumPy/SciPy, preserving variables within one
answer and destroying state between answers. Only model-visible prompt and
observations enter the numerical session. Hidden worlds, graders, model weights
and host credentials are not mounted or passed to it.

TRL's native `environment_factory` discovers and dispatches tools. The project
does not replace its parser, Qwen chat template or tool loop. The task prompt
documents the native Qwen3.5 function/parameter protocol; `act.command` has an
object schema. The upstream environment API is still experimental.

## Isolation boundary

Both the official Jupyter client/parser and ipykernel run inside the same
answer-specific NsJail. The host uses bounded Python-standard-library byte
frames (`Connection.send_bytes` / `recv_bytes(maxlength)`), never pickle or host
evaluation of model text. NsJail and Linux provide the isolation boundary:

- One CPU, 1 GiB RAM including the parser, zero swap, 64 tasks per answer.
- Host-enforced 30-second deadline; at most 8192 UTF-8 output bytes.
- Read-only numerical rootfs, private bounded tmpfs, no external network or
  host project/socket mounts; namespaces, seccomp, no capabilities and
  `no_new_privs`.
- The entire answer cgroup freezes between calls. Timeout, cancellation,
  terminal answers, reward collection and teardown kill all descendants.
- Tool/role delimiters and terminal control characters in numerical output
  are escaped before returning text to the model.

The tested stack was NsJail 3.6 commit
`f78475530b46d0186111a9096b30725f816b55fe`, Kafel
`76d0f41bf3eb5c4008713d64b9767b461a9129a3`, Python 3.12.3,
NumPy 2.3.5, SciPy 1.18.1, ipykernel 7.3.0, jupyter_client 8.10.0 and
pyzmq 27.2.0. Local checks covered 15 adversarial groups, five original-owner
families, idle/cancellation lifecycle and 31 Qwen protocol checks. These are
bounded test results, not a proof against every possible escape.

The four deployed core files (`tool_mode.py`, `secure_backend.py`,
`sandbox_bridge.py`, `nsjail-secure.cfg.in`) are published unchanged. Build and
model paths in deployment helpers are parameters; private evidence is excluded.

## Provisioning and checks

This implementation requires Linux with cgroup v2 delegation, working user
namespaces and the fixed root-owned `/opt/cpt-numerics` installation. It fails
closed without those prerequisites. There is no unsandboxed fallback.

Use a trusted build directory containing `nsjail-source/nsjail` (built from the
fixed upstream source) and a Python 3.12 virtualenv in `runtime/` with the pinned
numerical/Jupyter packages. `build-rootfs.py --base /absolute/build` creates a
new `rootfs-staging/` and refuses to overwrite an existing tree. Review that
tree, then install it as `/opt/cpt-numerics/rootfs`, owned by root with no
group/world writable files. Do not replace a rootfs used by a running training
job. `sudo bash install-host.sh /absolute/build` installs the trusted NsJail
binary and its dedicated AppArmor user-namespace profile; the global userns
restriction is unchanged. `sudo bash install-bridge.sh /absolute/source`
installs the trusted bridge and fixed Jupyter kernelspec in that rootfs.

Run boundary-only tests without a sandbox:

```bash
cd research/numeric_tool_mode_20260928
python -m unittest -v test_tool_mode.py
```

For actual adversarial checks, copy the scripts into a disposable trusted
`/absolute/check/source` directory. Use a dedicated **user** service whose name
starts `cpt-numerics-` and ends `.service`, with delegation:

```bash
systemd-run --user --wait --pipe --collect --unit=cpt-numerics-security-check \
  -p Delegate=yes -p MemoryMax=5G -p TasksMax=384 \
  /absolute/check/runtime/bin/python /absolute/check/source/security_tests.py
```

Run `check_idle_and_cancellation.py` the same way. The owner and Qwen checks
need the original task source, the matching training runtime and the local
Qwen tokenizer. Set `CPT_MODEL_DIR=/absolute/model`; `check_qwen_protocol.py`
also accepts `--model` and requires a new `--output` directory. None of these
checks loads Qwen model weights. Their generated reports/trajectories stay
private. The full production Trainer fixture and launch entry are in
`../numeric_training_from_zero_20260928/`.
