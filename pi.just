[private]
default:
    @just -f {{ justfile() }} --list

pi_app_dir := env("PI_APP_DIR", "/ws/acies/acies-os")

# Run an interactive shell in the Pi image with source code mounted
shell:
    docker run --rm -it \
        -v {{ pi_app_dir }}:/acies-os \
        -v /data:/data \
        acies-pi:latest bash
