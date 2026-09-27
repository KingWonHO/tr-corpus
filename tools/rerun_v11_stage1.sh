#!/usr/bin/env bash
# Protocol 29, stage 1: every model fit on label v1.1 / adjudicated pools.
# Run from the repository root after native_windows.py has rebuilt W60_native.
#   bash tools/rerun_v11_stage1.sh            # launches everything in the background
# CPU jobs run in parallel; the three GPU jobs are chained (two lanes).
set -u
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/evdatabm
LOG=tr-corpus/results/logs_20260925; mkdir -p "$LOG"
R=tr-corpus/results
PY="uv run --no-sync python"

# ---- CPU lane (trees, rule) -------------------------------------------------
mkdir -p $R/validation_20260925_e3a_trees $R/validation_20260925_e3a_h120
nohup $PY trbench/run_v2.py --arm e3a --models rule,xgboost,lightgbm --windows W60_native --without-age --seeds 0,1,2,3,4 \
  --curves $R/validation_20260925_e3a_trees/traces --out $R/validation_20260925_e3a_trees/e3a_trees_s0to4.csv > $LOG/e3a_trees.log 2>&1 &
nohup $PY trbench/run_v2.py --arm e3a --models xgboost,lightgbm --windows W60_native --without-age --seeds 0,1,2,3,4 --horizon 120 \
  --out $R/validation_20260925_e3a_h120/e3a_trees_h120_s0to4.csv > $LOG/e3a_h120.log 2>&1 &
nohup $PY trbench/run_validation.py --windows W60_native --models rule,xgboost,lightgbm --seeds 0,1,2,3,4 --ages no_age,with_age \
  --out $R/validation_20260925_native > $LOG/native.log 2>&1 &
nohup $PY trbench/run_validation.py --windows W60_native --models xgboost,lightgbm --seeds 0,1,2,3,4 --ages surface_max_only,surface_mean_only \
  --out $R/validation_20260925_common_surface > $LOG/common_surface.log 2>&1 &
nohup $PY trbench/run_validation.py --windows W60_native --models xgboost,lightgbm --seeds 0,1,2,3,4 --ages mask_only,age_only \
  --out $R/validation_20260925_shortcut_controls > $LOG/shortcut_controls.log 2>&1 &
mkdir -p $R/validation_20260925_e3b
nohup $PY trbench/run_v2.py --arm e3b --models rule,xgboost,lightgbm --windows W60_native --without-age --seeds 0,1,2,3,4 \
  --out $R/validation_20260925_e3b/e3b_trees_s0to4.csv > $LOG/e3b_trees.log 2>&1 &
nohup $PY trbench/run_split_sensitivity.py --out $R/validation_20260925_split_sensitivity > $LOG/split_sensitivity.log 2>&1 &

# ---- GPU lane A: internal design, sequence models -------------------------
if [ -z "${SKIP_LANE_A:-}" ]; then
mkdir -p $R/validation_20260925_e3a_seq $R/validation_20260925_e3a_h120_seq
nohup sh -c "uv run --no-sync python trbench/run_v2.py --arm e3a --models gru,mamba,itransformer,convtransformer --windows W60_native --without-age --seeds 0,1,2,3,4 \
    --out $R/validation_20260925_e3a_seq/e3a_seq_s0to4.csv > $LOG/e3a_seq.log 2>&1; \
  uv run --no-sync python trbench/run_v2.py --arm e3a --models gru,mamba,itransformer,convtransformer --windows W60_native --without-age --seeds 0 --horizon 120 \
    --out $R/validation_20260925_e3a_h120_seq/e3a_seq_h120_s0.csv > $LOG/e3a_h120_seq.log 2>&1" > /dev/null 2>&1 &
fi

# ---- GPU lane B: held-out design, sequence models -------------------------
nohup $PY trbench/run_validation.py --windows W60_native --models gru,mamba,itransformer,convtransformer --seeds 0,1,2,3,4 \
  --ages no_age,with_age,surface_max_only,surface_mean_only --out $R/validation_20260925_seq > $LOG/seq.log 2>&1 &

echo "stage 1 launched; logs in $LOG"
