# See: https://github.com/astral-sh/uv/blob/main/docs/guides/integration/docker.md

FROM ghcr.io/astral-sh/uv:0.10.11-python3.12-trixie-slim

ARG ZENOH_VERSION=1.8.0
ARG ZENOH_DOWNLOAD_URL="https://github.com/eclipse-zenoh/zenoh/releases/download/${ZENOH_VERSION}/zenoh-${ZENOH_VERSION}-aarch64-unknown-linux-gnu-debian.zip"

# System deps:
#   libportaudio2 -- sounddevice (acies-mic)
#   git           -- uv installs rawshake from GitHub
# Debug tools:
#   vim, less     -- editing and viewing files
#   just          -- running justfile recipes
#   procps        -- ps, kill
#   iproute2      -- ip, ss (network state)
#   iputils-ping  -- ping (connectivity check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    ca-certificates \
    curl \
    git \
    iputils-ping \
    iproute2 \
    just \
    less \
    libportaudio2 \
    procps \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install zenohd
RUN curl -fsSL "${ZENOH_DOWNLOAD_URL}" -o /tmp/zenoh-debian.zip \
    && unzip /tmp/zenoh-debian.zip -d /tmp/zenoh-debian \
    && dpkg -i /tmp/zenoh-debian/zenohd_${ZENOH_VERSION}_arm64.deb \
    && rm -rf /tmp/zenoh-debian.zip /tmp/zenoh-debian \
    | bash -s -- --to /usr/local/bin

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
# ENV UV_TOOL_BIN_DIR=/usr/local/bin

WORKDIR /acies-os

# -------------- deps layer: rebuild only when lock file changes --------------
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/sensors/pyproject.toml,target=packages/sensors/pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev

# ----------------- source layer: rebuilds when source changes -----------------
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml        packages/core/pyproject.toml
COPY packages/core/src                   packages/core/src
COPY packages/sensors/pyproject.toml     packages/sensors/pyproject.toml
COPY packages/sensors/src                packages/sensors/src
COPY packages/classifiers/pyproject.toml packages/classifiers/pyproject.toml
COPY packages/classifiers/src            packages/classifiers/src
COPY packages/buffers/pyproject.toml     packages/buffers/pyproject.toml
COPY packages/buffers/src                packages/buffers/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev \
        --package acies-sensors 

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
