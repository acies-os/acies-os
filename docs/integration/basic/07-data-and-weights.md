# 7. Data and Model Weights

## Download

| Item | Link |
| --- | --- |
| Model weights | https://uofi.box.com/s/q4f6my7ujmjfb7lzlh7ren4vlbntog58 |
| Replay data | https://uofi.box.com/s/tk33ye287xz5khbcro3t0srq69vkjmjn |

## Model weights

Vehicle classification with the FM pipeline needs two files:

| File | `.env` variable |
| --- | --- |
| `Parkland_ict2024_encoder.pt` | `VFM_ENCODER_WEIGHT` |
| `Parkland_ict2024_class_layer.pt` | `VFM_VEHICLE_HEAD_WEIGHT` |

The class labels are already set in the template: `VFM_LABELS=gle350,miata,cx30,mustang`.

Put both files in `MODEL_DIR` on the edge server (used by the [server-only test](03-server.md#test-the-server-on-its-own)),
and copy them to `/model/` on every device, from the edge server:

```bash
just upload-model ~/acies/models/Parkland_ict2024_encoder.pt
just upload-model ~/acies/models/Parkland_ict2024_class_layer.pt
```

## Replay data

We use the scene `2024-03-29-ICT`, run `2`. Files are named:

```text
2024-03-29-ICT/run2_<node>_geo.parquet     seismic, one per node
2024-03-29-ICT/run2_<node>_mic.parquet     audio, one per node
2024-03-29-ICT/run2_miata_gps.parquet      GPS trace of the vehicle
```

| Machine | Needs | Location |
| --- | --- | --- |
| Edge server | the GPS file, plus `rs1`'s geo and mic files for the [server-only test](03-server.md#test-the-server-on-its-own) | `<DATA_DIR>/2024-03-29-ICT/` (`DATA_DIR` in the edge `.env`) |
| Each device | its own geo and mic files | `/data/replay/2024-03-29-ICT/` |

Copy each device its own files from the edge server:

```bash
SCENE=2024-03-29-ICT; RUN=2
for h in rs1 rs2 rs3; do
  ssh "$h" mkdir -p /data/replay/$SCENE
  rsync -av ~/acies/replay/$SCENE/run${RUN}_${h}_geo.parquet \
            ~/acies/replay/$SCENE/run${RUN}_${h}_mic.parquet \
            "$h:/data/replay/$SCENE/"
done
```

(This assumes each device's SSH name equals its node name, e.g. `rs1`.)

## Other runs

To replay a different vehicle, change `RUN` (on the edge server and every device) and
`GPS_LABEL` (edge server only), then copy the matching files:

| `RUN` | `GPS_LABEL` |
| --- | --- |
| 0 | `mustang` |
| 2 | `miata` |
| 4 | `gle350` |
| 6 | `cx30` |
