pi_app_dir := env("PI_APP_DIR", "/ws/acies/acies-os")

# [private]
# default:
#     @just -f {{ justfile() }} --list

# Run an interactive shell in the Pi image with source code mounted
pi-shell:
    docker run --rm -it \
        -v {{ pi_app_dir }}:/acies-os \
        -v /data:/data \
        -v ~/.docker_bash_history:/root/.bash_history \
        --group-add dialout \
        --device /dev/serial0:/dev/serial0 \
        acies-pi:latest bash
