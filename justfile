set dotenv-load := true
set allow-duplicate-variables := true
set allow-duplicate-recipes := true

import 'just/dev.just'
import 'just/docker.just'
import 'just/pi.just'
import 'just/edge.just'

[private]
default:
    @just -f {{ justfile() }} --list

# WP exp additioanl routes
add-routes:
    sudo route -n add -net 10.7.0.0/24 192.168.68.77
    sudo route -n add -net 10.7.1.0/24 192.168.68.78
    netstat -nr
