acies_log := env("ACIES_LOG", "")
acies_namespace := env("ACIES_NAMESPACE", `hostname -s`)
edge_zenoh_connect := env("ZENOH_CONNECT", "tcp/127.0.0.1:7501")
edge_zenoh_bridge := env("ZENOH_BRIDGE", "")
pi_hosts := env("PI_HOSTS", "")
mqtt_broker := env("MQTT_BROKER")
mqtt_port := env("MQTT_PORT")
mqtt_topic := env("MQTT_TOPIC")

[private]
default:
    @just -f {{ justfile() }} --list

# Run zenohd in the server image.
edge-zenohd *args:
    @mkdir -p ~/.acies/logs
    echo {{ edge_zenoh_connect }}
    -@docker run --rm -it --init \
        --detach-keys="ctrl-z,z" \
        --network host \
        -v ~/.acies:/root/.acies \
        acies-server:latest zenohd --listen {{ edge_zenoh_connect }} {{ edge_zenoh_bridge }} {{ args }}

# Run acies-gateway in the server image.
edge-gateway *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -e ACIES_NAMESPACE={{ acies_namespace }}/infra \
        -e ACIES_NAME=gateway \
        --network host \
        -v ~/.acies:/root/.acies \
        acies-server:latest acies-gateway --acies-connect {{ edge_zenoh_connect }} {{ args }}

# Run acies-gateway directly (no Docker).
edge-gateway-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LOG={{ acies_log }} \
    ACIES_NAMESPACE={{ acies_namespace }}/infra \
    ACIES_NAME=gateway \
    uv run acies-gateway --acies-connect {{ edge_zenoh_connect }} {{ args }}

# run tracker locally
edge-tracker-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LOG={{ acies_log }} \
    ACIES_NAMESPACE={{ acies_namespace }}/infra \
    ACIES_NAME=tracker \
    env -u ACIES_LISTEN \
    uv run acies-tracker \
    --acies-connect {{ edge_zenoh_connect }} \
    --config packages/controllers/data/ict.toml
    {{ args }}

# Replay GPS data locally.
edge-replay-gps-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LOG={{ acies_log }} \
    uv run acies-gps --acies-namespace {{ acies_namespace }}/infra --acies-name gps \
        --acies-connect {{ edge_zenoh_connect }} {{ args }}

# MQTT
edge-mqtt-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LISTEN="" \
    ACIES_LOG={{ acies_log }} \
        uv run acies-mqtt \
        --acies-namespace {{ acies_namespace }}/infra \
        --acies-name gps \
        --acies-connect {{ edge_zenoh_connect }} \
        --broker {{ mqtt_broker }} \
        --port {{ mqtt_port }} \
        --mqtt-topic {{ mqtt_topic }}

# To run SPAR locally, you need to run`just edge-up-infra` (and typically `just edge-up-nodes`).
# Mirrors docker/edge-classifier.yml: fixed namespace `joint` so downstream
# subscribers find this spar's output, and uses SPAR_WEIGHT / SPAR_LABELS from .env.

# Run spar locally.
edge-spar-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LISTEN="" \
    ACIES_LOG={{ acies_log }} \
    uv run acies-spar \
        --acies-namespace joint --acies-name spar \
        --acies-connect {{ edge_zenoh_connect }} \
        --geo '**/geo' --mic '**/mic' \
        --weight "$MODEL_DIR/$SPAR_WEIGHT" \
        --labels "$SPAR_LABELS" \
        {{ args }}

# Start infra for an edge replay
edge-up-infra *args:
    ACIES_NAMESPACE=$(hostname -s) docker compose -p edge-replay \
        --env-file .env \
        -f docker/edge.yml \
        -f docker/edge-classifier.yml \
        {{ args }} \
        up

# Start nodes for an edge replay (replay only; no per-node VFM classifier)
edge-up-nodes *args:
    docker compose -p edge-replay --env-file .env \
        -f docker/replay.yml \
        -f docker/classifier.yml \
        {{ args }} \
        up

# Stop all containers for an edge replay
edge-down-all:
    docker compose -p edge-replay down

# Stop all containers for an edge replay and clear all images.
edge-clear:
    just edge-down-all
    docker image prune -f

# Pull the latest code and rebuild the server image.
edge-rebuild:
    -git pull
    just build-server
    just edge-clear

# Upload a file to /model/ on all PI_HOSTS.
[no-cd]
upload-model file:
    #!/usr/bin/env bash
    if [ -z "{{ pi_hosts }}" ]; then
        echo "error: PI_HOSTS is not set. Export PI_HOSTS='host1 host2 ...' first."
        exit 1
    fi
    for host in {{ pi_hosts }}; do
        echo "uploading {{ file }} -> ${host}:/model/"
        rsync -avz --progress "{{ file }}" "${host}:/model/"
    done
