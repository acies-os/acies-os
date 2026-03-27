# [private]
# default:
#     @just -f {{ justfile() }} --list

# Run an interactive shell in the Pi image with source code mounted
shell app_dir="/ws/acies/acies-os":
    docker run --rm -it \
        -v {{ app_dir }}:/acies-os \
        -v /data:/data \
        acies-pi:latest bash
