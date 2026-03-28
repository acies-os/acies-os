pi_app_dir := env("PI_APP_DIR", "/ws/acies/acies-os")

# [private]
# default:
#     @just -f {{ justfile() }} --list

# Fix ownership of /data files to the current user
fix-data-ownership:
    sudo chown -R $(id -u):$(id -g) /data

# Run an interactive shell in the Pi image with source code mounted
pi-shell:
    @touch ~/.docker_bash_history
    @mkdir -p ~/.acies/logs
    -docker run --rm -it \
        -e HISTFILE=/root/.bash_history \
        -e PROMPT_COMMAND='history -a' \
        -v {{ pi_app_dir }}:/acies-os \
        -v /data:/data \
        -v ~/.docker_bash_history:/root/.bash_history \
        -v ~/.acies:/root/.acies \
        --group-add dialout \
        --group-add audio \
        --device /dev/serial0:/dev/serial0 \
        --device /dev/snd:/dev/snd \
        --device /dev/bus/usb:/dev/bus/usb \
        acies-pi:latest bash
