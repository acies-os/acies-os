#!/bin/bash
set -e

export PYTHONPATH=packages/controllers/src
DATA_DIR="/home/jinyang7/data/2026-04-14-West-Point"
ARTIFACTS_DIR="docs/design/artifacts/west_point_2026-04-14"

echo "Generating labels..."
python packages/controllers/scripts/label_ict_ground_truth.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$ARTIFACTS_DIR/labels"

echo "Building assets..."
python packages/controllers/scripts/build_ict_tracker_assets.py \
    --data-dir "$DATA_DIR" \
    --labels-dir "$ARTIFACTS_DIR/labels" \
    --out-dir "$ARTIFACTS_DIR/assets_corridor25" \
    --cache-dir "$ARTIFACTS_DIR/cache" \
    --topology line \
    --max-nearest-sensor-distance-m 25.0

echo "Running runtime eval..."
python packages/controllers/scripts/eval_tracker_runtime_replay.py \
    --data-dir "$DATA_DIR" \
    --labels-dir "$ARTIFACTS_DIR/labels" \
    --asset-dir "$ARTIFACTS_DIR/assets_corridor25" \
    --out-dir "$ARTIFACTS_DIR/runtime_eval_corridor25_fullfit" \
    --cache-dir "$ARTIFACTS_DIR/cache" \
    --max-nearest-sensor-distance-m 25.0 \
    --plot-runs 0 1 2 3 4 5 6 7 8 9 10 \
    --loop-selection best_smoothed

echo "Done!"
