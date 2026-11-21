# Network emulator web app

The page served by the monitor (`emulation/monitor.py`) at http://127.0.0.1:8780. Vue 3 +
Vite + Leaflet, the same stack and look as the main UI (`ui/app`).

```bash
npm install
npm run dev      # http://127.0.0.1:5174, hot reload; /ws goes to the monitor on :8780
npm run build    # dist/, which the monitor serves when it is newer than the image's copy
```

From the repo root, `just emu-ui-dev` and `just emu-ui-build` do the same, and
`just emu-build` builds it into the emulator image (no Node needed on the host).
Set `MONITOR_URL` to point the dev server at a monitor elsewhere.

## Structure

```text
src/
├── main.js                  mounts App, loads the global styles
├── App.vue                  page layout only
├── config.js                UI choices: vehicle colours, targets per site, default target, tiles
├── composables/             state, shared by components (one store each)
│   ├── useMonitor.js        WebSocket to the monitor; snapshot, sites, toast; send(), fault()
│   ├── useExperiment.js     site, target, fault plan; deploy(), applyPlan()
│   └── useDeployOverlay.js  the configuring window's countdown to start_at
├── lib/                     plain functions, no Vue
│   ├── plan.js              plan rows <-> emulator schedule events, presets
│   ├── links.js             link state (Up / Impaired / Down / Unplugged), delay text
│   ├── chart.js             canvas helpers for the time charts
│   └── storage.js           guarded localStorage (last plan, last target)
├── components/
│   ├── AppHeader.vue        logo, title, link to the main UI
│   ├── ExperimentPanel.vue  site, target, status, Deploy; contains PlanTable
│   ├── PlanTable.vue        planned changes, presets, "apply to running replay"
│   ├── NodeChips.vue        link picker used by PlanTable
│   ├── NetworkMap.vue       Leaflet map: node states, truth, SPAR predictions, legend
│   ├── LinkPanel.vue        StatStrip + LinkTable + LinkForm
│   ├── LinkTable.vue        per-link state, settings, measurements, Down/Unplug buttons
│   ├── LinkForm.vue         set bw / delay / jitter / loss now
│   ├── SparPanel.vue        SPAR stats and two TimeCharts
│   ├── TimeChart.vue        a value per SPAR window over replay time (line or step)
│   ├── StatStrip.vue        row of labelled values
│   ├── ChangeLog.vue        applied link changes
│   ├── DeployOverlay.vue    the configuring window
│   └── ToastMessage.vue     replies to commands
└── styles/
    ├── tokens.css           colours and fonts (as the main UI)
    └── base.css             shared classes: .panel, .pill, .btn, .field, tables
```

Data flow: the monitor sends `sites` once and a `snapshot` every second (see
`emulation/DESIGN.md`, "Monitor"). `useMonitor` keeps them; components read computed values
from it and send commands through it (`fault`, `deploy`, `plan`). Nothing is computed from
raw sensor data in the browser.

Adding a panel: write a component that calls `useMonitor()` for what it shows, and place it
in `App.vue`.
