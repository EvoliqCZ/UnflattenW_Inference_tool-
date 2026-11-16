# Two images from one build: `runtime-api` (small, no LibreOffice) and
# `runtime-worker` (adds LibreOffice + poppler + Carlito for the DOCX->PNG
# preview render path). compose.yaml builds each service at its own target.
#
# `torch` (uv.lock) pulls its own CUDA 13 runtime libraries as pip packages
# (`cuda-toolkit`, `nvidia-cudnn-cu13`, ...), so the base image only needs a
# glibc new enough for the manylinux_2_28 wheels; the NVIDIA driver itself is
# injected by the NVIDIA Container Toolkit at `docker run` time, not baked
# into the image.
#
# The hostile-PDF admission/raster step (untrusted input) still runs only in
# the separate, per-job `deploy/pdf-sandbox` container reached over the
# Docker socket -- LibreOffice here renders our OWN already-built DOCX for a
# side-by-side preview, not attacker-controlled PDF bytes.
ARG PYTHON_BASE=docker.io/library/python:3.12.11-slim-bookworm

# --- web: SPA build (Node 22 LTS, digest-pinned) ----------------------------
# `web/dist` is committed nowhere (web/.gitignore) -- every image build
# compiles the SPA from source. `npm run build` = generate (openapi-typescript)
# + tsc -b + vite build + postbuild check-dist.mjs (design §6.4 guard).
ARG NODE_BASE=docker.io/library/node:22-bookworm-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c

FROM ${NODE_BASE} AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM ${PYTHON_BASE} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG UV_VERSION=0.9.7
ARG CURL_VERSION=7.88.1-10+deb12u15
ARG CA_CERTIFICATES_VERSION=20230311+deb12u1
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl=${CURL_VERSION} \
        ca-certificates=${CA_CERTIFICATES_VERSION} \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Resolve and build the venv from the locked, pinned dependency set only --
# no dev tooling (ruff/pytest) in the runtime image. `--no-install-project`
# first so this layer caches across source-only changes; the project itself
# is installed in the next layer once src/ is present.
RUN uv sync --frozen --no-dev --group ml --group server --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY vendor ./vendor
RUN uv sync --frozen --no-dev --group ml --group server

FROM ${PYTHON_BASE} AS runtime-base
LABEL org.opencontainers.image.title="Unflatten diagnostic PDF->Word service"

RUN groupadd --gid 65532 unflatten \
    && useradd --uid 65532 --gid 65532 --create-home --shell /usr/sbin/nologin unflatten
# Docker seeds an empty NAMED volume's ownership/mode from whatever already
# exists at the mount point in the image, so the api/worker/init containers
# (all uid 65532, or root under compose.sandbox-socket.yaml for worker) can
# write into the `unflatten_data` volume from the start.
RUN mkdir -p /data && chown unflatten:unflatten /data

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/vendor ./vendor
COPY pyproject.toml uv.lock ./
COPY deploy/docker/entrypoint-api.sh deploy/docker/entrypoint-worker.sh deploy/docker/entrypoint-init.sh /usr/local/bin/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/unflatten

# --- api: no LibreOffice, no docker CLI, always non-root -------------------
FROM runtime-base AS runtime-api
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TINI_VERSION=0.19.0-1+b3
RUN apt-get update \
    && apt-get install --no-install-recommends -y tini=${TINI_VERSION} \
    && rm -rf /var/lib/apt/lists/*
# Built SPA lands exactly where api.py's `_WEB_DIST_DIR` looks for it
# (`resources.files("unflatten.service.ui") / "dist"`, design §6.1); the
# `legacy` UI mode (default) never reads this directory.
COPY --from=web-builder --chown=65532:65532 /web/dist /app/src/unflatten/service/ui/dist
ARG UNFLATTEN_GIT_SHA=unknown
ENV UNFLATTEN_GIT_SHA=${UNFLATTEN_GIT_SHA}
USER 65532:65532
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python3", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read(); sys.exit(0)"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint-api.sh"]

# --- worker: docker-cli (sandbox over the host socket) + LibreOffice
# writer + poppler-utils + pdftoppm's own dependency chain + Carlito
# (metric-compatible Calibri substitute, DEC-044 Dodatok C) for the
# DOCX -> PNG side-by-side preview render path. `HOME=/home/unflatten`
# (created above) is writable by the default non-root user, and by root
# under compose.sandbox-socket.yaml, for LibreOffice's per-process
# `-env:UserInstallation=` profile dir (src/unflatten/renderer/probe.py) and
# `/tmp`, which is world-writable (mode 1777) in this base image by default.
FROM runtime-base AS runtime-worker
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TINI_VERSION=0.19.0-1+b3
ARG DOCKER_CLI_VERSION=20.10.24+dfsg1-1+deb12u1+b6
ARG LIBREOFFICE_WRITER_VERSION=4:7.4.7-1+deb12u14
ARG POPPLER_UTILS_VERSION=22.12.0-2+deb12u3
ARG FONTS_CARLITO_VERSION=20220224-1
ARG FONTCONFIG_VERSION=2.14.1-4
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        tini=${TINI_VERSION} \
        docker.io=${DOCKER_CLI_VERSION} \
        libreoffice-writer=${LIBREOFFICE_WRITER_VERSION} \
        poppler-utils=${POPPLER_UTILS_VERSION} \
        fonts-crosextra-carlito=${FONTS_CARLITO_VERSION} \
        fontconfig=${FONTCONFIG_VERSION} \
    && rm -rf /var/lib/apt/lists/*
# Least privilege by default: the base compose file (no socket mounted)
# fails closed at PdfSandbox.require_ready() regardless of user anyway.
# compose.sandbox-socket.yaml's `user: "0:0"` override elevates only when
# that EXPLICIT opt-in file is used (root-owned docker.sock access).
USER 65532:65532
HEALTHCHECK NONE
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint-worker.sh"]
