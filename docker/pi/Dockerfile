FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG ZENOHD_VERSION=1.8.0

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
RUN curl -fsSL \
    "https://github.com/eclipse-zenoh/zenoh/releases/download/${ZENOHD_VERSION}/zenohd-${ZENOHD_VERSION}-aarch64-unknown-linux-gnu.zip" \
    -o /tmp/zenohd.zip \
    && unzip /tmp/zenohd.zip -d /tmp/zenohd \
    && mv /tmp/zenohd/zenohd /usr/local/bin/zenohd \
    && rm -rf /tmp/zenohd.zip /tmp/zenohd

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
# ENV UV_TOOL_BIN_DIR=/usr/local/bin

WORKDIR /acies-os

# --- deps layer: rebuilds only when lockfile changes ---
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml        packages/core/pyproject.toml
COPY packages/sensors/pyproject.toml     packages/sensors/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# --- source layer: rebuilds when source changes ---
COPY packages/core/src    packages/core/src
COPY packages/sensors/src packages/sensors/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
