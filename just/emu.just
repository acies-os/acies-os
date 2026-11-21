# Network emulation with Mininet. See emulation/README.md.

emu_container := "acies-mn"
emu_data_dir := env("DATA_DIR", "/data")
emu_model_dir := env("MODEL_DIR", "/model")

[private]
default:
    @just -f {{ justfile() }} --list

# Build the Mininet image on top of acies-server (run `just build-server` first).
emu-build:
    docker build -t acies-mn:latest -f emulation/Dockerfile emulation

# Start emulated nodes and edge stack together; monitor at http://127.0.0.1:8780.
emu-up experiment="baseline":
    just emu-up-nodes {{ experiment }}
    just emu-up-infra

# Start the emulated nodes (emulation's edge-up-nodes; run first). Default: baseline, healthy links.
emu-up-nodes experiment="baseline":
    #!/usr/bin/env bash
    set -euo pipefail
    # Already running: reuse it (as `docker compose up` does for the edge stack).
    if [ "$(docker inspect -f '{{{{.State.Running}}' {{ emu_container }} 2>/dev/null)" = "true" ]; then
        running=$(docker inspect -f '{{{{join .Args " "}}' {{ emu_container }} | grep -o '[^/ ]*\.toml' | sed 's/\.toml$//')
        echo "emulated nodes already running (experiment: ${running}); reusing them"
        if [ "${running}" != "{{ experiment }}" ]; then
            echo "  to switch to '{{ experiment }}': just emu-down-all, then just emu-up {{ experiment }}"
        fi
        exit 0
    fi
    docker rm -f {{ emu_container }} >/dev/null 2>&1 || true  # a stopped leftover would block the name
    docker run -d --rm --name {{ emu_container }} \
        --privileged --init \
        --ulimit nofile=65536:65536 \
        -p 127.0.0.1:8765:8765 \
        -p 127.0.0.1:8780:8780 \
        -v {{ emu_data_dir }}:/data:ro \
        -v {{ emu_model_dir }}:/model:ro \
        -v ~/.acies:/root/.acies \
        -v {{ justfile_directory() }}/emulation:/emu:ro \
        -e SCENE -e RUN -e ACIES_LOG \
        -e VFM_ENCODER_WEIGHT -e VFM_VEHICLE_HEAD_WEIGHT -e VFM_LABELS -e VFM_VEHICLE_EMBED_DIM \
        acies-mn:latest python /emu/topo.py /emu/experiments/{{ experiment }}.toml
    echo "emulated nodes started; now: just emu-up-infra (logs: just emu-logs)"

# Start the edge stack on the emulated network (emulation's edge-up-infra), with recorder and monitor.
emu-up-infra *args:
    ACIES_NAMESPACE=$(hostname -s) docker compose -p edge-emu \
        --env-file .env \
        -f docker/edge.yml \
        -f docker/edge-classifier.yml \
        -f docker/emu-edge.yml \
        --profile replay --profile spar \
        {{ args }} \
        up

# Work on the web app with hot reload at http://127.0.0.1:5174 (needs Node 22; uses the running monitor).
emu-ui-dev:
    cd emulation/ui && npm install --no-audit --no-fund && npm run dev

# Build the web app on the host; the monitor serves it on the next page load (no image rebuild).
emu-ui-build:
    cd emulation/ui && npm ci --no-audit --no-fund && npm run build

# Follow the emulator log (topology, deploys, applied faults).
emu-logs:
    docker logs -f {{ emu_container }}

# Deploy a replay without the page (joint SPAR on all nodes); default target GLE350.
emu-deploy target="gle350" *args:
    docker exec {{ emu_container }} python /emu/deploy.py --target {{ target }} {{ args }}

# Change a link now, e.g. `just emu-fault rs3 down`, `just emu-fault all set --loss 20`.
emu-fault nodes action *args:
    docker exec {{ emu_container }} python /emu/fault.py {{ nodes }} {{ action }} {{ args }}

# Summarize the newest record (or pass a file under ~/.acies/emu/records).
emu-analyze *args:
    docker exec {{ emu_container }} python /emu/analyze.py {{ args }}

# Open a shell on an emulated node (namespace) or the container, e.g. `just emu-shell rs3`.
emu-shell node="":
    #!/usr/bin/env bash
    if [ -z "{{ node }}" ]; then
        exec docker exec -it {{ emu_container }} bash
    fi
    pid=$(docker exec {{ emu_container }} pgrep -f "^bash.*mininet:{{ node }}$" | head -1)
    [ -n "$pid" ] || { echo "no emulated node {{ node }}"; exit 1; }
    exec docker exec -it {{ emu_container }} mnexec -a "$pid" bash

# Stop the edge stack and the emulator.
emu-down-all:
    -docker compose -p edge-emu down
    -docker stop {{ emu_container }}
