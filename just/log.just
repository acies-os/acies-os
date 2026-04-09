[private]
default:
    @just -f {{ justfile() }} --list

# Show all sensor logs for a replay.
replay:
    multitail **/geo.log **/mic.log **/gps.log

# Show mic and geo energy of the given node based on gateway log.
gateway-energy node:
    multitail \
        -l "tail -F ~/.acies/logs/**/gps.log | grep --line-buffered 'energy from {{ node }}/geo'" \
        -l "tail -F ~/.acies/logs/**/gps.log | grep --line-buffered 'energy from {{ node }}/mic'"
