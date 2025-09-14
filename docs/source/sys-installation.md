# System Installation Guide

(sec-join-github-org)=

## Joining the Acies-OS Organization on GitHub

To use and/or contribute to this testbed, please make sure to:

1. **Join us on GitHub**
   Join the **acies-os** organization on GitHub (you will need to send your GitHub ID to the administrator).
   Org profile: [https://github.com/acies-os](https://github.com/acies-os)

2. **Authenticate your machine**
   If you're going to use your machine as a controller for the testbed (recommended if you're running experiments in the middle of nowhere with no Internet access), you will need to clone some of the GitHub repositories on your machine so you can compile and install controller software ahead of time.
   Create an **SSH key** on your machine and use it to authenticate to the **acies-os** organization; otherwise cloning/syncing will fail.

---

(sec-raspberry-pi-setup)=

## Installing Raspberry Pi Software

If you are using a new Raspberry Pi, follow the instruction here to set up your board:
[https://github.com/acies-os/fleet-command/blob/main/readme.md](https://github.com/acies-os/fleet-command/blob/main/readme.md)

---

(sec-install-on-edge-server)=

## Installing Edge Server Software

If you are using your own machine to run the **controller** and **GUI** during field experiments, you will need to install both from their repositories.
You must first join the acies-os organization and authenticate your machine--see [this section](#sec-join-github-org).

* **Install the controller**:
  [https://github.com/acies-os/controller/blob/main/README.md](https://github.com/acies-os/controller/blob/main/README.md)

* **Install the GUI**:
  [https://github.com/acies-os/ui/blob/main/readme.md](https://github.com/acies-os/ui/blob/main/readme.md)

---

(sec-field-network-setup)=

## Network Set-up for Field Data Collection

1. **Know your passwords**
   You will need three passwords to work on this testbed. They change over time, so they're not listed verbatim here. We refer to them as:

   * **Wi-Fi password**: When you power on the Raspberry Shakes, they are hard-coded to connect to a wireless network with SSID `iobt-2.4` and a specific network password. In the rest of this document, we refer to that as **the Wi-Fi password**.
   * **Shakes password**: To log in to the Shakes after they connect to the above network, you will use account name `myshake` and its password. We refer to this as **the Shakes password**.
   * **Server password**: If you are running experiments that involve a server (other than your own machine), you will need an account on that server for centralized testbed components. Usually the edge server is your own machine, but if you are using an existing server (e.g., Eugene), you will need credentials to the used account on that server.

2. **Connect peripherals**
   Connect a matching antenna (labeled with the same number as the Shake), a GPS, and a microphone to each Raspberry Pi via the USB ports.

3. **Power on the Raspberry Shakes**
   Red and blue (board power) LEDs should come on, plus a yellow one (indicating SD read/write activity) that may blink irregularly.

4. **Set up a Wi-Fi hotspot**
   Create a hotspot with SSID `iobt-2.4` and the corresponding **Wi-Fi password**. The Shakes are pre-configured to connect only to that SSID with that specific password.
   *(OS-specific steps omitted here; on Windows you can use **Settings -> Network & Internet -> Mobile hotspot**. Use the same SSID/password as above.)*

> For post-experiment data copying instructions, see the issue thread: **acies-os/postexp#2**.

---

(sec-field-analytics-deployment)=

## Field Analytics Deployment

If you also need to run analytics in the field:

1. **Copy model weights**

    Please refer to this link on how to download & copy model weights:

    [https://github.com/acies-os/vehicle-classifier?tab=readme-ov-file#download-model-weights](https://github.com/acies-os/vehicle-classifier?tab=readme-ov-file#download-model-weights)

2. **Run the classifier**
   Change to the classifier repo on the device and run the desired classifier:

   ```bash
   cd /ws/acies/classifier/vehicle-classifier
   just <classifier-name>
   # e.g.,
   # just vfm
   ```
