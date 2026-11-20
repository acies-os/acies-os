# Network emulation image: acies-server + Mininet.
#
# Mininet hosts are network namespaces inside this container. Each emulated
# node runs the same commands as a Pi (zenohd, replay, classifier) from the
# acies venv that comes with the base image.
#
# Build (after `just build-server`):
#   just emu-build

FROM acies-server:latest

# mininet       -- topology, mnexec, TCLink (tc/netem shaping)
# bridge-utils  -- brctl, used by Mininet's LinuxBridge switch (no OVS needed)
# net-tools     -- ifconfig, which Mininet uses to configure interfaces
RUN apt-get update && apt-get install -y --no-install-recommends \
    bridge-utils \
    iperf \
    mininet \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Mininet is installed for Debian's python3. Expose only the `mininet` package
# to the acies venv (python 3.12) so the runner can use mininet and zenoh in
# one process.
RUN mkdir -p /opt/mininet-py \
    && ln -s /usr/lib/python3/dist-packages/mininet /opt/mininet-py/mininet

ENV PATH=/acies-os/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/mininet-py

WORKDIR /emu
