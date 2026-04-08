# See: https://github.com/astral-sh/uv/blob/main/docs/guides/integration/docker.md

FROM ghcr.io/astral-sh/uv:0.10.11-python3.12-trixie-slim

ARG ZENOH_VERSION=1.8.0
ARG TARGETARCH

# System deps:
#   git      -- uv installs rawshake from GitHub
# Debug tools:
#   vim, less     -- editing and viewing files
#   just          -- running justfile recipes
#   procps        -- ps, kill
#   iproute2      -- ip, ss (network state)
#   iputils-ping  -- ping (connectivity check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    iputils-ping \
    iproute2 \
    just \
    less \
    openssh-client \
    procps \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install zenohd (multi-arch: amd64 or arm64)
RUN case "${TARGETARCH}" in \
        amd64) ZENOH_ARCH="x86_64-unknown-linux-gnu"  ; DEB_ARCH="amd64" ;; \
        arm64) ZENOH_ARCH="aarch64-unknown-linux-gnu"  ; DEB_ARCH="arm64" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" && exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/eclipse-zenoh/zenoh/releases/download/${ZENOH_VERSION}/zenoh-${ZENOH_VERSION}-${ZENOH_ARCH}-debian.zip" \
        -o /tmp/zenoh-debian.zip \
    && unzip /tmp/zenoh-debian.zip -d /tmp/zenoh-debian \
    && dpkg -i /tmp/zenoh-debian/zenohd_${ZENOH_VERSION}_${DEB_ARCH}.deb \
    && rm -rf /tmp/zenoh-debian.zip /tmp/zenoh-debian

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

WORKDIR /acies-os

# Set up SSH known hosts for git dependencies.
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts \
     && printf "Host github.com\n    User git\n  IdentityAgent SSH_AUTH_SOCK\n" > /root/.ssh/config \
     && chmod 600 /root/.ssh/config

# -------------- deps layer: rebuild only when lock file changes --------------
RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/buffers/pyproject.toml,target=packages/buffers/pyproject.toml \
    --mount=type=bind,source=packages/classifiers/pyproject.toml,target=packages/classifiers/pyproject.toml \
    --mount=type=bind,source=packages/controllers/pyproject.toml,target=packages/controllers/pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/sensors/pyproject.toml,target=packages/sensors/pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev

# ----------------- source layer: rebuilds when source changes -----------------
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml           packages/core/pyproject.toml
COPY packages/core/src                      packages/core/src
COPY packages/buffers/pyproject.toml        packages/buffers/pyproject.toml
COPY packages/buffers/src                   packages/buffers/src
COPY packages/classifiers/pyproject.toml    packages/classifiers/pyproject.toml
COPY packages/classifiers/src               packages/classifiers/src
COPY packages/controllers/pyproject.toml    packages/controllers/pyproject.toml
COPY packages/controllers/src               packages/controllers/src
COPY packages/sensors/pyproject.toml        packages/sensors/pyproject.toml
COPY packages/sensors/src                   packages/sensors/src
RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
        --package acies-core \
        --package acies-buffers \
        --package acies-classifiers \
        --package acies-controllers \
        --package acies-sensors

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
