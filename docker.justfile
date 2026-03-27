# Docker build, registry, and deployment recipes.
# Imported by justfile; can also be run directly:
#   just --justfile docker.justfile <recipe>
# Local registry address. Use laptop's LAN IP for Pi access, e.g. 192.168.1.10:5000.

registry := env("REGISTRY", "localhost:5000")
tag := env("TAG", "latest")

# Space-separated Pi SSH targets. Set in .env, e.g. PI_HOSTS="pi@192.168.1.20 pi@192.168.1.21"

pi_hosts := env("PI_HOSTS", "")

# --- build ---
# Cross-compile Pi image for linux/arm64 and push to registry.

# Native on Apple Silicon (no QEMU needed -- same arch, different OS).
build-pi:
    docker buildx build \
        --platform linux/arm64 \
        -t {{ registry }}/acies-pi:{{ tag }} \
        --push \
        -f docker/pi/Dockerfile \
        .

# --- local registry ---

# Start a local Docker registry on port 5000.
registry-up:
    docker run -d \
        --name acies-registry \
        -p 5000:5000 \
        --restart unless-stopped \
        registry:2

# Stop and remove the local registry.
registry-down:
    docker stop acies-registry
    docker rm acies-registry

# One-time per Pi: allow Docker daemon to pull from the insecure local registry.

# Usage: just registry-trust pi@192.168.1.20
registry-trust pi:
    ssh {{ pi }} "echo '{\"insecure-registries\":[\"{{ registry }}\"]}' \
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
        echo "==> deploying to $pi"
        ssh "$pi" "docker compose \
            -f ~/acies-os/docker/pi/docker-compose.yml \
            --project-directory ~/acies-os \
            up -d --pull always"
    done
