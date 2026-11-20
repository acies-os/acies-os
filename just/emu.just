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

# Start the Mininet container with an experiment from emulation/experiments/ (detached).
emu-up experiment="baseline":
    docker run -d --rm --name {{ emu_container }} \
        --privileged --init \
        --ulimit nofile=65536:65536 \
        -p 127.0.0.1:8765:8765 \
        -v {{ emu_data_dir }}:/data:ro \
        -v {{ emu_model_dir }}:/model:ro \
        -v ~/.acies:/root/.acies \
        -v {{ justfile_directory() }}/emulation:/emu:ro \
        -e SCENE -e RUN -e ACIES_LOG \
        -e VFM_ENCODER_WEIGHT -e VFM_VEHICLE_HEAD_WEIGHT -e VFM_LABELS -e VFM_VEHICLE_EMBED_DIM \
        acies-mn:latest python /emu/topo.py /emu/experiments/{{ experiment }}.toml
    @echo "emulator started; follow it with: just emu-logs"

# Start the edge stack (zenohd, gateway, gps, spar, recorder, ui) inside the emulator's network.
emu-edge-up *args:
    ACIES_NAMESPACE=$(hostname -s) docker compose -p edge-emu \
        --env-file .env \
        -f docker/edge.yml \
        -f docker/edge-classifier.yml \
        -f docker/emu-edge.yml \
        --profile replay --profile spar \
        {{ args }} \
        up

# Follow the emulator log (topology, deploys, applied faults).
emu-logs:
    docker logs -f {{ emu_container }}

# Deploy a replay without the UI, e.g. `just emu-deploy miata` (joint SPAR on all nodes).
emu-deploy target *args:
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
emu-down:
    -docker compose -p edge-emu down
    -docker stop {{ emu_container }}
