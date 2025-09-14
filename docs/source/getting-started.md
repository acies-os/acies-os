# Getting Started

## Installing Dependencies

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
