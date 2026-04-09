pi_app_dir := env("PI_APP_DIR", "/ws/acies/acies-os")
acies_log := env("ACIES_LOG", "")
acies_namespace := env("ACIES_NAMESPACE", `hostname -s`)
pi_zenoh_connect := env("ZENOH_CONNECT", "tcp/127.0.0.1:7501")
pi_zenoh_bridge := env("ZENOH_BRIDGE", "")
rawshake_debug := env("RAWSHAKE_DEBUG", "0")

[private]
default:
    @just -f {{ justfile() }} --list

# Fix ownership of /data files to the current user
own-data:
    sudo chown -R $(id -u):$(id -g) /data

# Run an interactive shell in the Pi image.
pi-shell dev="false":
    @touch ~/.docker_bash_history
    @mkdir -p ~/.acies/logs
    -docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        -v $SSH_AUTH_SOCK:/ssh-agent \
        -e SSH_AUTH_SOCK=/ssh-agent \
        -e HISTFILE=/root/.bash_history \
        {{ if dev == "true" { "-v " + pi_app_dir + ":/acies-os" } else { "" } }} \
        -v /data:/data \
        -v ~/.docker_bash_history:/root/.bash_history \
        -v ~/.acies:/root/.acies \
        --group-add dialout \
        --group-add audio \
        --device /dev/serial0:/dev/serial0 \
        --device /dev/snd:/dev/snd \
        --device /dev/bus/usb:/dev/bus/usb \
        acies-pi:latest bash

# Run zenohd in the Pi image.
pi-zenohd *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it --init \
        --detach-keys="ctrl-z,z" \
        --network host \
        -v ~/.acies:/root/.acies \
        acies-pi:latest zenohd --listen {{ pi_zenoh_connect }} {{ pi_zenoh_bridge }} {{ args }}

# Run acies-mic in the Pi image.
pi-mic *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -e ACIES_NAMESPACE={{ acies_namespace }} \
        -e ACIES_NAME=mic \
        -e ACIES_CONNECT={{ pi_zenoh_connect }} \
        --network host \
        -v /data:/data \
        -v ~/.acies:/root/.acies \
        --group-add audio \
        --device /dev/snd:/dev/snd \
        acies-pi:latest acies-mic {{ args }}

# Run acies-geo in the Pi image.
pi-geo *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -e ACIES_NAMESPACE={{ acies_namespace }} \
        -e ACIES_NAME=geo \
        -e ACIES_CONNECT={{ pi_zenoh_connect }} \
        -e RAWSHAKE_DEBUG={{ rawshake_debug }} \
        --network host \
        -v /data:/data \
        -v ~/.acies:/root/.acies \
        --group-add dialout \
        --device /dev/serial0:/dev/serial0 \
        --device /dev/bus/usb:/dev/bus/usb \
        acies-pi:latest acies-geo {{ args }}

# Run acies-vfm in the Pi image.
pi-vfm *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -e ACIES_NAMESPACE={{ acies_namespace }} \
        -e ACIES_NAME=vfm \
        -e ACIES_CONNECT={{ pi_zenoh_connect }} \
        --network host \
        -v /data:/data \
        -v /model:/model \
        -v ~/.acies:/root/.acies \
        acies-pi:latest acies-vfm {{ args }}

# Run acies-vfm locally (no Docker).
pi-vfm-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LOG={{ acies_log }} \
    uv run acies-vfm --acies-namespace {{ acies_namespace }} --acies-name vfm \
        --acies-connect {{ pi_zenoh_connect }} \
        --weight /model/Parkland_TransformerV4_vehicle_classification_finetune_ict2024_1.0_multiclass_best.pt \
        --geo {{ acies_namespace }}/geo --mic {{ acies_namespace }}/mic \
        --labels "gle350,miata,cx30,mustang" \
        {{ args }}

# Replay sensor data locally.
pi-replay-local *args:
    @mkdir -p ~/.acies/logs
    ACIES_LOG={{ acies_log }} \
    uv run acies-replay --acies-namespace {{ acies_namespace }} \
        --acies-connect {{ pi_zenoh_connect }} {{ args }}

# Start Pi services (live sensors) via docker compose.
pi-up *args:
    ACIES_NAMESPACE={{ acies_namespace }} docker compose -p pi \
        --env-file .env \
        -f docker/pi.yml \
        {{ args }} \
        up

# Start Pi services (replay) via docker compose.
pi-up-replay *args:
    ACIES_NAMESPACE={{ acies_namespace }} docker compose -p pi \
        --env-file .env \
        -f docker/pi-replay.yml \
        {{ args }} \
        up

# Stop all Pi containers.
pi-down:
    docker compose -p pi down

# Record audio using arecord.
record output="/data/alsa-test.wav":
    arecord -D hw:ArrayUAC10,0 -f S16_LE -r 16000 -c 6 -d 40 {{ output }}
