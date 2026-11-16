# Portable private single-host deployment

`service.sh` is the primary, portable way to run the private diagnostic API and the
single-GPU worker: `deploy/service/service.sh {start|stop|status|verify}`. It hardcodes no
path, account, or port -- everything site-specific comes from the environment. Copy
`service.env.example`, fill in every `REPLACE_WITH_*` value, and either `source` your copy
before calling `service.sh` or point `UNFLATTEN_SERVICE_ENV` at it. `service.sh` creates
`$UNFLATTEN_DATA_DIR`'s subdirectories (`artifacts/`, `work/`, `run/`, `logs/`) at `0700` on
`start`. See [application operations](../../docs/RUNBOOK.md) for the current lifecycle guide and
[the main README](../../README.md) for the supported Docker launcher. For this native
alternative, provision the model bundle and sandbox described below before starting.
If encrypted storage is required for your deployment, provision it at the host level;
the script does not verify disk encryption.

- `start` runs `api_cli` and `worker_cli` each under `setsid` inside a restart loop (5s
  between restarts); pid files and logs live under `$UNFLATTEN_DATA_DIR/run` and
  `$UNFLATTEN_DATA_DIR/logs`.
- `stop` sends `SIGTERM` to each process group and waits on `kill -0` (never `pgrep -f` --
  memory `pgrep-self-match`, which self-matches and never returns) up to
  `UNFLATTEN_STOP_TIMEOUT_SECONDS` (default 150s), then `SIGKILL`.
- `status` reports pid liveness for both processes, plus a `curl /health/ready` check when
  `curl` is on `PATH`.
- `verify` runs `scripts/build_model_bundle.py verify`, checks the sandbox image is present
  under the configured runtime, and checks environment completeness -- it never starts a
  process.

`service.sh` runs no restart supervisor of its own beyond the loop above: whoever starts it
decides whether to also wrap it in systemd.

The deployment emits Prediction IR and produces Word output (`diagnostic_docx` in
the stable API) when the configured model and document assembly succeed. Check the
job's `docx_produced` result before downloading Word output.

## Optional systemd templates

`unflatten-diagnostic-{api,worker}.service` and `unflatten-diagnostic.target` are an
**optional** template for operators who want systemd supervision instead of (or on top of)
`service.sh`'s own restart loop; nothing requires them. Their `ExecStart` lines call the same
`api_cli`/`worker_cli` entry points with `--model-bundle` (D7); install them in
`/etc/systemd/system/`. Because systemd's `EnvironmentFile=` directive itself needs a literal
path, these templates keep example paths (`/etc/unflatten/service.env`, `/srv/unflatten`,
`/var/lib/unflatten-diagnostic`) that an operator using this path customizes to match their
own `service.env`; `service.sh` itself never hardcodes any of them.

The API is hard-bound to `127.0.0.1`; do not change that binding without the required code,
security, and governance reviews.

The worker requires rootless Docker or Podman access and access to the single configured
NVIDIA GPU. Its unit deliberately uses `PrivateDevices=no`: a private `/dev` would hide CUDA.
Instead, the closed systemd device policy permits only `/dev/nvidia0`, `/dev/nvidiactl`,
`/dev/nvidia-uvm`, and `/dev/nvidia-uvm-tools` in addition to systemd's standard pseudo
devices. Provision those nodes before starting the worker; changing the configured GPU
requires a matching `DeviceAllow` update. All mutable state is restricted to
`/var/lib/unflatten-diagnostic` in the template; place that directory (or your own
`UNFLATTEN_DATA_DIR`) on an encrypted host volume.

The API principal file has a stricter runtime contract: install it as
`/etc/unflatten/principals.json`, owned by `unflatten:unflatten`, with mode `0400`. The API
validates both the effective service-user ownership and exact mode before reading it, and
the API unit mounts that path read-only. Replace it atomically as root when rotating tokens;
never make it group- or other-readable. The worker does not receive this credential file.
The separate `/etc/unflatten/service.env` remains root-owned mode `0600` because systemd,
not the unprivileged process, reads it.

Provision the two files without opening either secret to a group or other users:

```bash
sudo install -d -o root -g root -m 0755 /etc/unflatten
sudo install -o unflatten -g unflatten -m 0400 principals.json /etc/unflatten/principals.json
sudo install -o root -g root -m 0600 service.env /etc/unflatten/service.env
```

After principal rotation, restart only `unflatten-diagnostic-api.service`; the API loads the
closed principal set at startup. A root-owned `0600` principal file is intentionally rejected
because the `unflatten` API process cannot read it, as are owner-writable and group/other modes.

The OCI service test is deliberately external-image scaffolding. Set
`UNFLATTEN_SERVICE_OCI_IMAGE` to a reviewed digest-pinned image to run it. This repository
does not declare an application image release complete merely because the sandbox image can
be built.
