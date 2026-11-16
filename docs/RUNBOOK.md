# Unflatten W 1.0 operations

For the complete English installation guide see [README](../README.md); the
Slovak guide is [README.sk.md](../README.sk.md). Start the supported local Docker
application with `./Start.sh` (or `./Spustit.sh`). Both names use the same data.

## Normal operation

- `./Start.sh status` shows the application services; `./Start.sh logs` follows
  their logs. The browser address is `http://127.0.0.1:8421/` by default.
- Opening the launcher again preserves a running conversion. `./Start.sh stop`
  stops the services and preserves accounts, jobs, results and the model cache.
- After an update, use `./Start.sh restart` when no conversion is running. The
  verified model cache is reused. The model must be loaded on a CUDA GPU before
  conversions can start.
- Use `./Start.sh reset-password` locally to set a new password and revoke old
  web sessions. Pass the same `UNFLATTEN_APP_DATA_DIR` if using a custom folder.
- Keep the adjacent `.launcher/` credential directory private. Never copy tokens,
  databases, models or user documents into source control or release archives.

## Alternative native service deployment

The units under `deploy/service/` are an advanced alternative to the main Docker
launcher. Follow that directory's README before enabling them. The API principal
file is owned by `unflatten:unflatten`, mode `0400`, and mounted read-only into the
API service. Its directory must also prevent other users from reading it. The
worker does not receive the API principal credentials.

The API's private-device restriction and the worker's explicit NVIDIA device
allowlist serve different purposes. Do not hide the worker's CUDA devices or
broaden its allowlist to unrelated devices. The PDF sandbox's Docker socket
access remains an explicit local-machine opt-in.
