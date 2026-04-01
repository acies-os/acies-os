set dotenv-load := true
set allow-duplicate-variables := true
set allow-duplicate-recipes := true

import 'just/dev.just'
import 'just/docker.just'
import 'just/pi.just'

[private]
default:
    @just -f {{ justfile() }} --list
