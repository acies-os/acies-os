pi_app_dir := env("PI_APP_DIR", "/ws/acies/acies-os")
acies_log := env("ACIES_LOG", "")

# [private]
# default:
#     @just -f {{ justfile() }} --list

# Fix ownership of /data files to the current user
own-data:
    sudo chown -R $(id -u):$(id -g) /data

# Run an interactive shell in the Pi image. Pass dev=true to mount source code.
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

# Run acies-mic in the Pi image. Arguments are forwarded to acies-mic directly.
pi-mic *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -v /data:/data \
        -v ~/.acies:/root/.acies \
        --group-add audio \
        --device /dev/snd:/dev/snd \
        acies-pi:latest acies-mic {{ args }}

# Run acies-geo in the Pi image. Arguments are forwarded to acies-geo directly.
pi-geo *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -v /data:/data \
        -v ~/.acies:/root/.acies \
        --group-add dialout \
        --device /dev/serial0:/dev/serial0 \
        --device /dev/bus/usb:/dev/bus/usb \
        acies-pi:latest acies-geo {{ args }}

# Run acies-vfm in the Pi image. Arguments are forwarded to acies-vfm directly.
pi-vfm *args:
    @mkdir -p ~/.acies/logs
    -@docker run --rm -it \
        --detach-keys="ctrl-z,z" \
        -e PATH="/acies-os/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
        -e ACIES_LOG={{ acies_log }} \
        -v /data:/data \
        -v ~/.acies:/root/.acies \
        acies-pi:latest acies-vfm {{ args }}

# Record 40 seconds of audio from the USB microphone array and save it to /data/alsa-test.wav
record output="/data/alsa-test.wav":
    arecord -D hw:ArrayUAC10,0 -f S16_LE -r 16000 -c 6 -d 40 {{ output }}
