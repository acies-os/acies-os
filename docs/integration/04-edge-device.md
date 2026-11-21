# 4. Edge Devices

Repeat these steps for every device. The examples use a Raspberry Pi named `rs1`.

## Operating system

Install a **64-bit** Linux and enable SSH. On a Raspberry Pi, use Raspberry Pi Imager with
*Raspberry Pi OS Lite (64-bit)*. In *Edit settings*, set the hostname (e.g. `rs1`), a user and
password, and the Wi-Fi network, then turn on SSH.

Check the architecture:

```bash
ssh rs1 uname -m                 # must print aarch64
```

## Install Docker and git

On the device:

```bash
sudo apt update && sudo apt install -y git just rsync
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"  # then log out and back in
docker run --rm hello-world
```

## Folders

```bash
sudo mkdir -p /data/replay /model /ws/acies
sudo chown -R "$USER":"$USER" /data /model /ws/acies
```

| Folder | Contents |
| --- | --- |
| `/model` | Model weight files |
| `/data/replay/<SCENE>/` | Recorded data this device replays |
| `/ws/acies/acies-os` | Copy of the repository (for the Docker compose file and `.env`) |

## Clone the repository

Log in with agent forwarding so the private clone uses your GitHub key:

```bash
ssh -A rs1
git clone git@github.com:acies-os/acies-os.git /ws/acies/acies-os
```

## Create `.env`

```bash
cd /ws/acies/acies-os
cp docker/.env.pi.example .env
```

Change these lines and uncomment the replay block. Keep every other line as it is in the
template.

```bash
ACIES_NAMESPACE=rs1                    # this device's name (unique per device)
ZENOH_BRIDGE=tcp/<EDGE_IP>:7500        # edge server address; port = edge ZENOH_LISTEN

SCENE=2024-03-29-ICT                   # same as the edge .env
RUN=2
NODE=rs1                               # recorded node to replay (usually = ACIES_NAMESPACE)
```

> [!IMPORTANT]
> Write `ZENOH_BRIDGE` as just `tcp/IP:PORT`. Do not add `--connect`.

## Allow the edge registry

Run this **on the edge server**, once per device:

```bash
just registry-trust rs1
```

It lets the device pull images from the edge server's registry. It overwrites
`/etc/docker/daemon.json` on the device.
