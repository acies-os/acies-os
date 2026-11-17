# Acies-OS Integration Guide

This guide shows how to run Acies-OS on your own hardware: **one edge server** (a laptop or
workstation) and **one or more edge devices** (we use Raspberry Pis as the example). The edge
devices replay recorded sensor data and run vehicle classification. The edge server collects the
results and shows them in a web UI.

Read the pages in order:

1. [Overview](01-overview.md): what runs where
2. [Repository access and SSH](02-access.md): GitHub permissions and SSH keys
3. [Edge server](03-server.md): set up the edge server and build the images
4. [Edge devices](04-edge-device.md): prepare each device
5. [Network](05-network.md): IP addresses and ports, and how to check them
6. [Build and run](06-build-and-run.md): deploy and start everything
7. [Data and model weights](07-data-and-weights.md): what to download and where it goes

## Files you will edit

You do not need to read or change any code or Docker files. Everything is configured through these
three files:

| File | Where | Purpose |
| --- | --- | --- |
| `.env` (copied from `docker/.env.edge.example`) | edge server, repo root | Edge settings: device list, ports, replay scene, data and model folders |
| `.env` (copied from `docker/.env.pi.example`) | each edge device, repo root | Device name, edge server address, model weight filenames |
| `~/.ssh/config` | edge server | SSH shortcuts for GitHub and the devices |

The services themselves are defined in `docker/edge.yml` (edge server) and `docker/pi-replay.yml`
(edge devices). The `just` commands in this guide use them; you don't need to edit them.
