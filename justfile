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

# monitor routers
pings:
    @fping -l 192.168.68.77 192.168.68.78

# server the UI
serve-via-eugene:
    @ssh -N \
       -o ExitOnForwardFailure=yes \
       -R 127.0.0.1:15173:127.0.0.1:5173 \
       -R 127.0.0.1:18765:127.0.0.1:8765 \
       eugene

# receive the UI via eugene
receive-via-eugene:
    @ssh -N \
       -o ExitOnForwardFailure=yes \
       -L 5173:localhost:15173 \
       -L 8765:localhost:18765 \
       eugene
