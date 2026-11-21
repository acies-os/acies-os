# Acies-OS

A content-centric platform for edge AI twinning and orchestration. Edge devices (e.g. Raspberry
Pis) stream sensor data and run classifiers; an edge server collects the results and shows them in
a web UI. Everything runs in Docker.

The latest release is `v0.3.0`.

## Documentation

**Docs website: <https://acies-os.github.io/acies-os/>**

The **[integration guide](docs/integration/index.md)** walks through running Acies-OS on your
own hardware, step by step:

1. [Overview](docs/integration/01-overview.md)
2. [Repository access and SSH](docs/integration/02-access.md)
3. [Edge server](docs/integration/03-server.md)
4. [Edge devices](docs/integration/04-edge-device.md)
5. [Network](docs/integration/05-network.md)
6. [Build and run](docs/integration/06-build-and-run.md)
7. [Data and model weights](docs/integration/07-data-and-weights.md)
8. [Network emulation](docs/integration/08-network-emulation.md) (optional)

To browse the docs as a website locally: `just live-doc`, then open <http://localhost:8000>.

## Network emulation

To test without edge devices, or to see what network failures do to localization, the
network emulator runs all eight devices on the edge server behind an emulated network
(Mininet). A web page controls the links (blackouts, loss, delay, bandwidth) and shows SPAR's
localization error as the replay plays:

```bash
just emu-build    # once
just emu-up       # then open http://127.0.0.1:8780
```

Setup: pages 2, 3, 7 and [8](docs/integration/08-network-emulation.md) of the integration
guide. Day-to-day use: [emulation/README.md](emulation/README.md).

## Citation

```bibtex
@inproceedings{li2024aciesos,
  title={Acies-OS: A Content-Centric Platform for Edge AI Twinning and Orchestration},
  author={Jinyang Li, Yizhuo Chen, Tomoyoshi Kimura, Tianshi Wang, Ruijie Wang, Denizhan Kara, Yigong Hu, Li Wu, Walid A. Hanafy, Abel Souza, Prashant Shenoy, Maggie Wigness, Joydeep Bhattacharyya, Jae Kim, Guijun Wang, Greg Kimberly, Josh Eckhardt, Denis Osipychev, Tarek Abdelzaher},
  booktitle={2024 33rd International Conference on Computer Communications and Networks (ICCCN)},
  pages={1--1},
  year={2024},
  organization={IEEE}
}
```
