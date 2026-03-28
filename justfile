set dotenv-load := true
set allow-duplicate-variables := true

import 'just/dev.just'
import 'just/docker.just'
import 'just/pi.just'

[private]
default:
    @just -f {{ justfile() }} --list
