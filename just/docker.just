# [private]
# default:
#     @just -f {{ justfile() }} --list
#
# Docker build, registry, and deployment recipes.
# Imported by justfile; can also be run directly:
#   just --justfile docker.justfile <recipe>
# Local registry address. Use laptop's LAN IP for Pi access, e.g. 192.168.1.10:5000.

registry_port := env("REGISTRY_PORT", "5100")
registry := env("REGISTRY", "localhost:" + registry_port)
registry_external := env("REGISTRY_EXTERNAL", "10.8.0.102:" + registry_port)
tag := env("TAG", "latest")

# Space-separated Pi SSH targets. Set in .env, e.g. PI_HOSTS="pi@192.168.1.20 pi@192.168.1.21"

pi_hosts := env("PI_HOSTS", "")
pi_app_dir := env("PI_APP_DIR", "/ws/acies/acies-os")

# ----------------------------------- build -----------------------------------

# One-time: install Docker Desktop and buildx plugin on Apple Silicon
setup-docker-apple-silicon:
    brew install docker docker-buildx colima
    mkdir -p ~/.docker/cli-plugins
    ln -sf $(brew --prefix)/opt/docker-buildx/bin/docker-buildx ~/.docker/cli-plugins/docker-buildx

# One-time: install buildx plugin (macOS) and create a builder that supports --platform.
builder-setup:
    colima start
    docker buildx rm acies-builder 2>/dev/null || true
    docker buildx create --name acies-builder --use --bootstrap --driver-opt network=host

# Cross-compile Pi image for linux/arm64 and push to registry.
# Native on Apple Silicon (no QEMU needed -- same arch, different OS).

# Run `just builder-setup` once before using this.
build-pi:
    docker buildx build \
        --platform linux/arm64 \
        -t {{ registry }}/acies-pi:{{ tag }} \
        --push \
        -f docker/pi/Dockerfile \
        .

# ------------------------------- local registry -------------------------------

# Start docker daemon with Colima if not already running. (macOS only)
docker-up:
    -colima stop --force
    colima start --runtime docker
    colima status

# Start a local Docker registry on port 5100. (on macOS 5000 is used by AirTunes)
registry-up:
    docker start acies-registry 2>/dev/null || docker run -d \
        --name acies-registry \
        -p {{ registry_port }}:5000 \
        --restart unless-stopped \
        registry:2

# Stop and remove the local registry.
registry-down:
    docker stop acies-registry
    docker rm acies-registry

# list images in the local registry.
registry-list:
    @echo "images in registry {{ registry }}:"
    @curl -s http://{{ registry }}/v2/_catalog | jq -r '.repositories[]'

# check if a port is in use (e.g. check registry port before starting registry-up).
check-port port=registry_port:
    @if lsof -nP -iTCP:{{ port }} -sTCP:LISTEN >/dev/null; then \
        echo "port {{ port }}: in use"; \
        lsof -nP -iTCP:{{ port }} -sTCP:LISTEN; \
    else \
        echo "port {{ port }}: free"; \
    fi

# One-time per Pi: allow Docker daemon to pull from the insecure local registry.

# Usage: just registry-trust pi@192.168.1.20
registry-trust pi:
    ssh {{ pi }} "echo '{\"insecure-registries\":[\"{{ registry_external }}\"]}' \
        | sudo tee /etc/docker/daemon.json \
        && sudo systemctl restart docker"

# --- deploy ---
# Pull updated image and restart services on all Pis.

# Requires PI_HOSTS set in .env or environment.
deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{ pi_hosts }}" ]; then
        echo "error: PI_HOSTS not set. Add PI_HOSTS='pi@192.168.1.20 pi@192.168.1.21' to .env"
        exit 1
    fi
    for pi in {{ pi_hosts }}; do
        echo "==> pushing latest images to $pi"
        ssh "$pi" "docker pull {{ registry_external }}/acies-pi:latest \
            && docker tag {{ registry_external }}/acies-pi:latest acies-pi:latest \
            && docker rmi {{ registry_external }}/acies-pi:latest"
    done
