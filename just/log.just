[private]
default:
    @just -f {{ justfile() }} --list

# Show all sensor logs for a replay.
replay:
    find . \( -name geo.log -o -name mic.log -o -name gps.log \) -print0 | \
        sort -z | \
        xargs -0 -r multitail

# Show mic and geo energy of the given node based on gateway log.
gateway-energy node:
    multitail \
        -l "find . -type f -name gateway.log -print0 | xargs -0 -r tail -F | grep --line-buffered -F 'energy from {{ node }}/geo'" \
        -l "find . -type f -name gateway.log -print0 | xargs -0 -r tail -F | grep --line-buffered -F 'energy from {{ node }}/mic'"
