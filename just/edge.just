acies_log := env("ACIES_LOG", "")
acies_namespace := env("ACIES_NAMESPACE", `hostname -s`)
edge_zenoh_connect := env("ZENOH_CONNECT", "tcp/127.0.0.1:7501")
edge_zenoh_bridge := env("ZENOH_BRIDGE", "")
pi_hosts := env("PI_HOSTS", "")

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
    @mkdir -p ~/.acies/logs ~/.acies/run
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
    @mkdir -p ~/.acies/logs ~/.acies/run
    ACIES_LOG={{ acies_log }} \
    ACIES_NAMESPACE={{ acies_namespace }}/infra \
    ACIES_NAME=gateway \
    uv run acies-gateway --acies-connect {{ edge_zenoh_connect }} {{ args }}

# Replay GPS data locally.
edge-replay-gps-local *args:
    @mkdir -p ~/.acies/logs ~/.acies/run
    ACIES_LOG={{ acies_log }} \
    uv run acies-gps --acies-namespace {{ acies_namespace }}/infra --acies-name gps \
        --acies-connect {{ edge_zenoh_connect }} {{ args }}

# Start infra for an edge replay
edge-up-infra *args:
    ACIES_NAMESPACE=$(hostname -s) docker compose -p edge-replay \
        --env-file .env \
        -f docker/edge.yml \
        {{ args }}
        up

# Start nodes for an edge replay
edge-up-nodes *args:
    docker compose -p edge-replay --env-file .env \
        -f docker/replay.yml \
        -f docker/classifier.yml \
        {{ args }} \
        up

# Stop all containers for an edge replay
edge-down-all:
    docker compose -p edge-replay down

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
