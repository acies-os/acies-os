# Data and Model Weights

Model weights and replay data are **not** in git. Download them separately, place them in the
folders below, and point the `.env` files at them.

## Downloads

| Bundle | Link | Size (approx.) |
| --- | --- | --- |
| Model weights | `<TODO: model weights URL>` | `<TODO>` |
| Replay data | `<TODO: dataset URL>` | ~6 GB (all scenes) |

:::{note}
Access to these links may require an account or a request to the maintainers. Contact
`<TODO: contact>` if a link doesn't work for you.
:::

## Where files go

| Machine | Weights | Data | `.env` variable |
| --- | --- | --- | --- |
| Edge server | any folder, e.g. `~/acies/models` | any folder, e.g. `~/acies/replay` | `MODEL_DIR`, `DATA_DIR` (absolute paths) |
| Each Pi | `/model` | `/data` | `MODEL_DIR=/model` (data path is fixed to `/data`) |

On the edge, `MODEL_DIR` and `DATA_DIR` are host paths that the containers mount as `/model` and
`/data`. Use absolute paths; `~` is not expanded inside `.env`.

## Model weights

Put every file directly in the weights folder (no subfolders). Filenames must match the `.env`
values exactly.

| File | `.env` variable | Used by | Labels |
| --- | --- | --- | --- |
| `Parkland_TransformerV4_vehicle_classification_finetune_ict2024_1.0_multiclass_best.pt` | `VFM_WEIGHT` | `e2e_vehicle` (end-to-end vehicle, "e2e" pipeline) | `VFM_LABELS` |
| `Parkland_ict2024_encoder.pt` | `VFM_ENCODER_WEIGHT` | `vfm_encoder` (shared encoder, "fm" pipeline; includes audio weights) | – |
| `Parkland_ict2024_class_layer.pt` | `VFM_VEHICLE_HEAD_WEIGHT` | `vfm_vehicle` head, ICT site (embed dim 512) | `VFM_LABELS` |
| `Parkland_wp2026_class_layer.pt` | `VFM_VEHICLE_HEAD_WEIGHT` (alternative) | `vfm_vehicle` head, West Point site | WP labels |
| `AuDroK_class_layer.pt` | `VFM_DRONE_HEAD_WEIGHT` | `vfm_drone` head (embed dim 256) | `VFM_DRONE_LABELS` |
| `AuDroK_8k_SwinTransformer_drone_classification_finetune_audrok_1.0_linear_best.pt` | `VFM_DRONE_E2E_WEIGHT` | `e2e_drone` (end-to-end drone) | `VFM_DRONE_LABELS` |
| `spar_2024-03-29-ICT.pt` | `SPAR_WEIGHT` | `edge-spar` (edge only) | `SPAR_LABELS` |
| `diffphys_2024-03-29-ICT_classification.pt` | `DIFFPHYS_WEIGHT` | `diffphys` (optional, `docker/diffphys.yml`) | `DIFFPHYS_LABELS` |

Label lists are order-sensitive and must match the class order each checkpoint was trained with:

```bash
VFM_LABELS=gle350,miata,cx30,mustang
SPAR_LABELS=gle350,miata,cx30,mustang
DIFFPHYS_LABELS=gle350,miata,cx30,mustang
VFM_DRONE_LABELS=cartonic_toy_drone,dji_avata,dji_inspire_2,dji_matrice_300,dji_matrice_30_t,dji_mavic_3e,dji_mini_3,dji_phantom_4,emotion,hp_e616p_1,hpx4,mavic_air_2,potensic_firefly
```

**Which files does each machine need?**

* **Pis:** the five VibroFM/AuDroK files (end-to-end vehicle, encoder, vehicle head(s), drone
  head, end-to-end drone).
* **Edge:** the same files for scenario 2, plus `spar_*.pt` if you run SPAR and `diffphys_*.pt`
  if you run DiffPhys.

Copy weights to the Pis from the edge server:

```bash
just upload-model ~/acies/models/<file>.pt            # one file -> /model on all PI_HOSTS
MODEL_SRC=~/acies/models scripts/update-pis.sh        # the full set listed in the script
```

## Replay data

Replay files are Parquet, one file per run, node, and modality:

```text
<DATA_DIR>/<scene>/run<RUN>_<node>_<modality>.parquet     modality: geo | mic | dis
<DATA_DIR>/<scene>/run<RUN>_<label>_gps.parquet           GPS ground-truth trace of the target vehicle
```

| Scene | Size | Nodes | Runs → target (`GPS_LABEL`) | Site config |
| --- | --- | --- | --- | --- |
| `2024-03-29-ICT` | 1.6 GB | rs1, rs2, rs3, rs5, rs6, rs7, rs8, rs10 | run0 → mustang, run2 → miata, run4 → gle350, run6 → cx30 | `ict.toml` |
| `2024-06-22-ICT` | 2.0 GB | rs1, rs2, rs3, rs5, rs6, rs7, rs8, rs10 | run1 → cx30 + mustang, run3 → miata + mustang, run5 → cx30 + miata (two vehicles; no GPS trace files) | `ict.toml` |
| `2026-04-14-West-Point` | <1 MB | (GPS only) | run0 → kona, run6 → wagoneer | `wp.toml` |
| `audrok` | 126 MB | rs1 | run1 → dji_avata, run2 → potensic_firefly, run3 → cartonic_toy_drone (8 kHz drone audio) | – |
| `replay_audrok_8k_train` | 2.2 GB | rs1 | one folder per drone class (13 classes) | – |

`audrok/` holds **symlinks** into `replay_audrok_8k_train/`. Keep both folders side by side,
or copy with `rsync -aL` to resolve the links, as `scripts/update-pis.sh` does.

Route-to-run mappings used by the UI live in the `[routes]` table of each site config, for example
`packages/controllers/data/ict.toml`.

**Edge (scenarios 2 and 3):** extract everything into `DATA_DIR`:

```text
~/acies/replay/
├── 2024-03-29-ICT/
├── 2024-06-22-ICT/
├── 2026-04-14-West-Point/
├── audrok/
└── replay_audrok_8k_train/
```

**Pis (scenario 3 only):** each Pi needs only the scene it replays and the drone audio:

```bash
SCENE=2024-03-29-ICT RUN=2
for h in $PI_HOSTS; do
  ssh "$h" mkdir -p /data/$SCENE /data/audrok
  rsync -av ~/acies/replay/$SCENE/run${RUN}_${h}_*.parquet "$h:/data/$SCENE/"
  rsync -aL ~/acies/replay/audrok/ "$h:/data/audrok/"
done
```

(This assumes each Pi's SSH alias equals its node name. Otherwise adjust the `run${RUN}_${h}_`
pattern.)

## Data recorded in the field

Live sensors (scenario 1) write to `/data` on each Pi. After an experiment, collect it with:

```bash
for h in $PI_HOSTS; do rsync -av "$h:/data/" ~/acies/collected/$h/; done
```

## Checklist

- [ ] Every filename in the edge `.env` exists in `MODEL_DIR`.
- [ ] Every filename in each Pi `.env` exists in `/model` on that Pi.
- [ ] `DATA_DIR/<SCENE>/run<RUN>_rs1_geo.parquet` exists on the edge (scenario 2).
- [ ] `/data/<SCENE>/` and `/data/audrok/` exist on each Pi (scenario 3).
