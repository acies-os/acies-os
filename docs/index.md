# Acies-OS Documentation

Acies-OS is a content-centric platform for edge AI twinning and orchestration, built mostly around
**Docker**: every service on the edge server and the edge devices runs in a container, so setting it
up is mainly configuring and running Docker images. These pages show how to run it on your own
hardware: one **edge server** plus one or more **edge devices**.

## Integration guide

The [**integration guide**](integration/index.md) walks through the setup step by step:
edge server + edge devices, replayed sensor data with GPS, and vehicle classification. Its last
page, [**network emulation**](integration/08-network-emulation.md), runs all devices on the edge
server behind an emulated network for network-failure experiments.

```{toctree}
:hidden:
:caption: Integration guide

integration/index
```

## What you need

No code changes are needed. To run the system you:

1. get access to the private repositories and set up an SSH key for them,
2. install Docker and `just`,
3. download the replay data and model weights,
4. fill in two `.env` files (edge server and each device),
5. build the Docker images, and
6. connect the edge server and devices on one network and start everything.

The [overview](integration/01-overview.md) has the full checklist with links to each step.

## Citation

If you use Acies-OS in your research, please cite:

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
