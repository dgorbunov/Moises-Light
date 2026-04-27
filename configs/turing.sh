#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --job-name="moises_light_vocals"
#SBATCH --partition=academic
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Move to submission directory (where sbatch was run).
cd "${SLURM_SUBMIT_DIR}"

# Turing docs recommend loading python/cuda modules for GPU jobs.
module load python
module load cuda/12.2

mkdir -p logs

# ---- Python environment setup ----
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

MUSDB_ROOT="/path/to/musdb18hq"
STEM="vocals"
OUT_DIR="runs/moises_light"
TRAIN_CONFIG="configs/vocals.yaml"

# Preview mode only: use musdb package 7-second preview clips.
if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "Missing training config: ${TRAIN_CONFIG}"
  exit 1
fi

python - <<'PY'
from mamba_light.config import load_config
cfg = load_config("configs/vocals.yaml")
if abs(float(cfg.segment_seconds) - 7.0) > 1e-9:
    raise SystemExit(f"Expected segment_seconds=7.0, got {cfg.segment_seconds}")
print("Config check passed: using 7-second segments.")
PY

# ---- Training on academic partition (single GPU) ----
python scripts/train_lightning.py \
  --config "${TRAIN_CONFIG}" \
  --download-preview \
  --target-stem "${STEM}" \
  --out-dir "${OUT_DIR}"

# ---- One-track validation after training ----
python scripts/validate_checkpoint.py \
  --config "${TRAIN_CONFIG}" \
  --ckpt "${OUT_DIR}/${STEM}/best_legacy.pt" \
  --download-preview \
  --subset test \
  --track-index 0 \
  --save-audio "${OUT_DIR}/${STEM}/validation_estimate.wav" \
  --save-originals