acies_log := env("ACIES_LOG", "")
acies_host := env("ACIES_HOST", `hostname -s`)
edge_zenoh_connect := env("ZENOH_CONNECT", "tcp/127.0.0.1:7501")
edge_zenoh_bridge := env("ZENOH_BRIDGE", "")

[private]
default:
    @just -f {{ justfile() }} --list

# Run zenohd in the server image. Listens on UDS for local apps; bridges to Pi devices via TCP.
edge-zenohd *args:
    @mkdir -p ~/.acies/logs
    echo {{ edge_zenoh_connect }}
    -@docker run --rm -it --init \
        --detach-keys="ctrl-z,z" \
        --network host \
        -v ~/.acies:/root/.acies \
        acies-server:latest zenohd --listen {{ edge_zenoh_connect }} {{ edge_zenoh_bridge }} {{ args }}

# Run acies-gateway in the server image. Arguments are forwarded to acies-gateway directly.
edge-gateway *args:
    @mkdir -p ~/.acies/logs ~/.acies/run
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -e ACIES_HOST={{ acies_host }} \
        -e ACIES_NAME=gateway \
        --network host \
        -v ~/.acies:/root/.acies \
        acies-server:latest acies-gateway --acies-connect {{ edge_zenoh_connect }} {{ args }}

# Run acies-gateway directly (no Docker). Arguments are forwarded to acies-gateway directly.
edge-gateway-local *args:
    @mkdir -p ~/.acies/logs ~/.acies/run
    ACIES_LOG={{ acies_log }} \
    ACIES_HOST={{ acies_host }} \
    ACIES_NAME=gateway \
    uv run acies-gateway --acies-connect {{ edge_zenoh_connect }} {{ args }}

# Replay GPS data locally. Pass --data-dir, --scene, --run, --label, etc.
edge-replay-gps-local *args:
    @mkdir -p ~/.acies/logs ~/.acies/run
    ACIES_LOG={{ acies_log }} \
    uv run acies-gps --acies-host {{ acies_host }} --acies-name gps \
        --acies-connect {{ edge_zenoh_connect }} {{ args }}
