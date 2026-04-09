[private]
default:
    @just -f {{ justfile() }} --list

# Show all sensor logs for a replay.
replay:
    multitail $(find . \( -name geo.log -o -name mic.log -o -name gps.log \) | sort)

# Show mic and geo energy of the given node based on gateway log.
[script(bash)]
gateway-energy +node:
    set -euo pipefail

    base_cmd="find . -type f -name gateway.log -print0 | sort -z | xargs -0 -r tail -F"

    args=()
    for node in {{nodes}}; do
      args+=(
        -l "$base_cmd | grep --line-buffered -F 'energy from ${node}/geo'"
        -l "$base_cmd | grep --line-buffered -F 'energy from ${node}/mic'"
      )
    done

    multitail "${args[@]}"
