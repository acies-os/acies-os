# Acies Controller

## Setup

### Python Environment Management

This project uses [`uv`](https://docs.astral.sh/uv) (or its predecessor [`rye`](https://rye.astral.sh)) to manage the Python environment.

To check if `rye` is already installed, run:

```bash
which rye
```

- If the command prints a path, `rye` is available and you can skip this step.
- If not, we recommend installing `uv` for new setups. Follow [uv’s official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

For backward compatibility, you may also install [rye](https://rye.astral.sh/guide/installation/).

### Clone and install dependencies

Tarek's note: The instruction `git clone https://github.com/acies-os/controller.git` below did not work for me because github does not support https password authentication. Instead, one needs to use SSL, create a public key, tell GitHub which key it is, and then update a local .ssl file too to tell the local machine which identity to share with GitHub. If the steps below, did not work for you either, please follow the aforementioned process instead.

```bash
$ git clone https://github.com/acies-os/controller.git
$ cd controller
controller$ uv sync

# or, if using rye
controller$ rye sync
```

### Install `just`

Install `just` use [your package manager](https://just.systems/man/en/packages.html) or [pre-built binary](https://just.systems/man/en/pre-built-binaries.html).

### Install `zenohd`

For x86_64 linux:

```bash
wget https://github.com/eclipse-zenoh/zenoh/releases/download/0.11.0/zenoh-0.11.0-x86_64-unknown-linux-gnu-standalone.zip
```

For aarch64 linux:

```bash
wget https://github.com/eclipse-zenoh/zenoh/releases/download/0.11.0/zenoh-0.11.0-aarch64-unknown-linux-gnu-standalone.zip
```

For aarch64 macOS:

```bash
wget https://github.com/eclipse-zenoh/zenoh/releases/download/0.11.0/zenoh-0.11.0-aarch64-apple-darwin-standalone.zip
```

To unzip the downloaded file, use the following command:

```bash
unzip zenoh-0.11.0-<platform>-standalone.zip
```

Replace `<platform>` with the appropriate platform identifier (e.g., `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, or `aarch64-apple-darwin`).

To verify the installed version, run:

```bash
./zenohd --version
```

Ensure the output shows version `v0.11.0`.

## If using WSL as host:

### Get WSL IP Address

First of all, you need to locate the WSL's IP by running:

```bash
# activate WSL
windows$ wsl
# inside of WSL, run
wsl$ hostname -I
```

For example:

```bash
hermanwu713@MSI:~/acies-core$ hostname -I
172.17.26.25
```

Please note that your WSL IP **changes on every WSL restart**.

### Config Controller ZROUTER

In the root address of the controller repository, run:

```bash
$ cd controller
# Replace <WSL_IP> with the IP shown by `hostname -I` in WSL
controller$ just update-zrouter <WSL_IP>
```

Tarek's Note: If the above does not work, you can just manually change the environment variable called `ZROUTER` to `tcp/<WSL-Host-IP>:7447`. For example, if the WSL IP is: `172.17.26.25`, run: 

```bash
$ export ZROUTER="tcp/172.17.26.25:7447"
```

Check updated ZROUTER by running:

```bash
just echo-zrouter
```

Tarek's note: If you are doing this repeatedly, you may want to permanently set the environment variable, ZROUTER, so that you do not have to manually update it every time you open the WSL shell. To do so, you can open your .bashrc file (in the home directory), ~/.bashrc, for example using:

```bash
sudo vim ~/.bashrc
```

then add the following to the end:

```bash
CURRENT_IPS=$(hostname -I)
export ZROUTER="tcp/${CURRENT_IPS}:7447"
```
This will initialize ZROUTER to the VM's IP every time you open a WSL terminal.

### Configure Windows to Forward Port

Since devices on your LAN cannot directly reach WSL’s internal IP, use `netsh` portproxy to forward Windows ports to WSL.

In **Administrator Powershell**, run:

```powershell
# Reset old portproxy rules
netsh interface portproxy reset

Tarek's note: To run the admin powershell, search for PowerShell in the windows start menu and when it finds the icon, rightclick on it and choose "Run as Administrator".

# Replace <WSL_IP> with the IP shown by `hostname -I` in WSL
netsh interface portproxy add v4tov4 listenport=7447 listenaddress=0.0.0.0 connectport=7447 connectaddress=<WSL_IP>
```

To verify port forward rules:

```powershell
netsh interface portproxy show all
```

### Configure Windows Firewall

By default, Windows Firewall block inbound connection at port 7447. In **Administrator Powershell**, add rules to allow port 7447:

```powershell
New-NetFirewallRule -DisplayName "Zenoh TCP 7447" -Direction Inbound -Protocol TCP -LocalPort 7447 -Action Allow
```

To verify firewall rule:

```powershell
Get-NetFirewallRule | findstr 7447
```

### Connecting from Raspberry Pis

First, connect your devices to Windows' hotspot, you can verify your devices' IP addresses in `Windows Settings > Network & Internet > Mobile Hotspot`. 

SSH into your device:

```bash
ssh <Username>@<device_IP>
```

The Windows hotspot adapter usually have the IP Address `192.168.137.1`. You can verify this by running `ipconfig` in Powershell. Look for `Wireless LAN adapter Local Area Connection`.

Test if your device can reach Windows' IP.

```bash
# Run on the Raspberry pi, and you should see the following output
$ nc -vz 192.168.137.1 7447
Connection to 192.168.137.1 7447 port [tcp/*] succeeded!
```

Now, run this just command to set ZROUTER:

```bash
just -g set-zrouter tcp/192.168.137.1:7447
just -g reload-services
```

At this point, your device should connect to the controller through the Windows host, using the forwarded port. If you encounter connectivity issues, double-check the port forwarding and firewall rules on Windows, and ensure your devices are connected to the correct hotspot network.

## Usage

### Start `Zenohd`

```bash
$ cd controller
controller$ ./zenohd
```

In the output, check the line

```bash
2025-09-03T23:29:00.786023Z  INFO main ThreadId(01) zenoh::net::runtime::orchestrator: Zenoh can be reached at: tcp/172.17.26.25:7447
```

Ensure the `<protocal>/<ip_addr>:<port>` match the ZROUTER address you set.

### Start the controller

```bash
# Start the controller
# N = <number of nodes> * <number of services on each node>
# Usually there are 4 services on each node: geo, mic, aciesd, vfm
$ cd controller
controller$ just ctl <N>
```
