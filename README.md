<div align="center">
  <img src="src/unflatten/hf_export/unflatten-w-mark.png" width="128" alt="Unflatten W">
  <h1>Unflatten W 1.0</h1>
  <p>Documents to editable Word<br>by <a href="https://www.evoliq.cz/sk">EVOLIQ s.r.o.</a></p>
  <a href="https://www.evoliq.cz/en">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/brand/evoliq-logo-white.png">
      <img src="docs/brand/evoliq-logo-dark.png" width="180" alt="EVOLIQ s.r.o.">
    </picture>
  </a>
</div>

**English** · [Slovenčina](README.sk.md)

The complete local **Unflatten W** application: convert PDFs and images, manage
jobs, download Word documents, compare previews, manage models and sign in with
your own password. Starting the app automatically downloads, verifies and loads
**Unflatten W Dense 1.0** onto your NVIDIA GPU.

[Tool source](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-) ·
[Model on Hugging Face](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0) ·
[EVOLIQ s.r.o.](https://www.evoliq.cz/sk)

## Requirements

- Linux, Docker Engine with the Compose plugin, NVIDIA Container Toolkit and a
  compatible NVIDIA driver. The locked runtime uses PyTorch with CUDA 13; check
  your driver's compatibility before installing.
- An NVIDIA CUDA GPU. At least **16 GB of GPU memory is recommended for the
  distributed BF16 model**. The app checks available memory and model suitability
  before loading. Other GPU workloads can reduce the available capacity.
- About 11 GB for downloaded model files, plus space for container images and
  your documents. The first build and download require internet access.
- The [model repository](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0)
  is public and can be downloaded without a Hugging Face account or token.
  Public availability does not change the personal-use and commercial licence terms.

## Start

Clone or extract this repository, then run from its directory:

```bash
./Start.sh
```

The first start explains the local Docker socket access used by the isolated PDF
sandbox. After you accept, the public model downloads automatically without a
Hugging Face login. An existing local login or an optional `HF_TOKEN` can also be
used; credentials are saved privately outside the source directory.

The launcher builds the services, opens **http://127.0.0.1:8421/** and prepares the
model on the GPU automatically. On your first visit, create a password and confirm
it. **Registration has no activation token or access-token field.** Model access
credentials are handled by the launcher, independently of the app password.

Follow model preparation in the setup wizard or **Models**. When ready, upload a
PDF or image and start a conversion. Open the job to follow per-page progress,
download the resulting Word document and compare available previews.

For a clickable shortcut in your applications menu, run once:

```bash
./Install-launcher.sh
```

Open **Unflatten W 1.0** in the applications menu afterwards. `Start.sh` and
`Install-launcher.sh` are aliases of `Spustit.sh` and `Instalovat-spustac.sh`; both
names use the same application, settings and data.

## API and headless conversion

The same service accepts requests without a browser login or application UI.
Open [Swagger UI](http://127.0.0.1:8421/docs) to try uploads, job status and Word
downloads, or retrieve [OpenAPI JSON](http://127.0.0.1:8421/openapi.json) for an
integration. API clients use a Bearer token, separately from the web password and
Hugging Face download token.

The [API guide](docs/API.md) covers authentication, supported request settings,
curl, Swagger and running the server with `--web-ui none`. The configurable
[Python example](examples/headless_client.py) needs only Python's standard library:

```bash
cp examples/headless-request.example.json request.json
# Edit request.json: local input/output paths, service URL and optional model/label.
# Set UNFLATTEN_API_TOKEN privately as described in docs/API.md.
python3 examples/headless_client.py request.json
```

It uploads the file, polls with a deadline and saves the resulting Word document.
The existing API output identifier for Word is `diagnostic_docx`.

## English and Slovak

Use **SK / EN** in the footer of any page, including login and first-run setup.
The choice applies immediately and is saved in your browser. A saved preference
wins over the browser language. Without one, Slovak browsers use Slovak and all
others use English. Changing the interface language does not translate documents.

Terminal messages follow the OS locale. To choose explicitly:

```bash
UNFLATTEN_APP_LANG=en ./Start.sh
UNFLATTEN_APP_LANG=sk ./Spustit.sh
```

## Data, repeated starts and updates

Accounts, the job database, results and model cache are stored in
`~/.local/share/unflatten-w-app-1.0/`, outside the source tree. An optional Hugging Face
download token is in the adjacent `~/.local/share/unflatten-w-app-1.0.launcher/`
directory with `0600` file permissions. It is mounted only into the worker,
outside the shared data mount.

Opening the launcher again opens the running app without interrupting conversions.
Missing or stopped services use the saved configuration without rebuilding.
Stopping preserves accounts, jobs, results and downloaded weights:

```bash
./Start.sh status
./Start.sh logs
./Start.sh stop
```

Choose a custom data folder and port on first start if needed:

```bash
UNFLATTEN_APP_DATA_DIR="/path/to/my data" UNFLATTEN_APP_PORT=8421 ./Start.sh
```

Later starts reuse the saved port. Paths may contain spaces. Backslashes, quotes,
`$`, `%`, `:`, `,` and control characters are unsupported and rejected before
writing configuration.

After changing credentials or updating the application, use an explicit restart
**when no conversion is active**:

```bash
./Start.sh restart
```

This rebuilds and recreates the services while preserving data and the verified
model cache. To reset a forgotten password locally:

```bash
./Start.sh reset-password
```

Enter the new password at the hidden prompt. Existing web sessions are revoked.
For a custom data folder, pass the same `UNFLATTEN_APP_DATA_DIR` when resetting.

For unattended installation, use `UNFLATTEN_APP_NONINTERACTIVE=1` and, after
reviewing Docker socket access, `UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1`. Public model
downloads need no credentials. Optional credentials can come from local Hugging Face
login or a locally supplied `HF_TOKEN`; never
commit tokens or place them in documentation. Standard `HTTP_PROXY`, `HTTPS_PROXY`
and `NO_PROXY` settings are forwarded for builds and downloads without being
written into the source.

## Release links

- [Unflatten W Dense 1.0 model and model card](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0)
- [Application releases and downloads](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-/releases)
- [Release facts, evaluation and limitations](docs/RELEASE.md)
- [API and headless integration](docs/API.md)

## Model and supported runtime

The distributed catalogue pins model revision
`9ebf7df1d6b01ae5d3a396f379fe454378a38ae7` and its manifest hash. It uses the native
CUDA engine, **BF16**, DeepStack, the bundled loop guard and `bf16_dynamic` KV cache.
It never silently changes model revision or falls back to CPU. Downloaded files
and a GPU-loaded model have separate states: **Load onto GPU** retries loading
verified cached weights without downloading them again. Cold starts check the
available memory even when a default model is already saved.

On integrated CUDA GPUs such as DGX Spark, reclaimable shared system memory is
considered using `MemAvailable`; swap does not count toward the estimate.

Unflatten W reconstructs editable Word documents from PDFs and images. Documents
are processed locally. PDFs are parsed in a separate restricted container.

The 12 GB RTX 4070 SUPER acceptance test uses a **separate FP8 test bundle**. This
repository does not distribute that bundle or silently select it for smaller
GPUs. The default remains the pinned BF16 release.

## Licence and credits

Original EVOLIQ contributions use the
[EVOLIQ Personal Use and Commercial License 1.0](LICENSE): free personal,
noncommercial use; commercial and business use require a paid agreement.
Contact [EVOLIQ s.r.o.](https://www.evoliq.cz/sk) for commercial use.

Third-party components keep their own licences. See [NOTICE](NOTICE),
[LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and notices under `vendor/`. Model weights
are downloaded separately and remain subject to their model-card licence and
upstream terms.

## Development and verification

Use the checked-in lockfiles rather than an unbounded dependency upgrade:

```bash
uv sync --group dev --group ml --group server
PYTHONPATH=src uv run pytest -m ''
cd web
npm ci
npm run verify
npx playwright test --config e2e/release-ui.config.ts
```

The frontend toolchain supports Node 22 (22.12 or later), Node 24 or Node 26 and later. `npm run verify`
checks generated API types, formatting, lint, types, coverage and the production
build. Default Python tests cover service, renderer and document AST contracts;
`-m ''` also includes their slow checks. Historical training and evaluation modules
are not supported application entry points.

See `distribution-verification.json` for the exact acceptance scope and limits.
The source package contains no model weights, credentials, customer documents or
live job data. `source-origin.json` records the source snapshot's provenance.
