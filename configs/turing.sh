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
module load cuda
# stempeg (required by musdb import path) needs ffmpeg + ffprobe binaries.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  module load ffmpeg >/dev/null 2>&1 || true
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffmpeg and ffprobe are required (musdb -> stempeg)."
  echo "Try checking available modules with: module avail ffmpeg"
  exit 1
fi

mkdir -p logs

# ---- Python environment setup ----
# Rebuild venv each run for deterministic installs on cluster jobs.
if [ -d ".venv" ]; then
  rm -rf .venv
fi
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# Prefer cu129 (for CUDA 12.9 module), fall back to cu128 if unavailable.
TORCH_CUDA_INDEX=""
for cuda_tag in cu129 cu128; do
  candidate_index="https://download.pytorch.org/whl/${cuda_tag}"
  echo "Trying PyTorch index: ${candidate_index}"
  if python -m pip install --index-url "${candidate_index}" torch==2.8.0 torchaudio==2.8.0; then
    TORCH_CUDA_INDEX="${candidate_index}"
    echo "Using PyTorch CUDA index: ${TORCH_CUDA_INDEX}"
    break
  fi
done

if [ -z "${TORCH_CUDA_INDEX}" ]; then
  echo "Failed to install CUDA-enabled torch wheels (tried cu129 and cu128)."
  exit 1
fi

# Install remaining requirements (torch/torchaudio entries are already satisfied).
python -m pip install -r requirements.txt --extra-index-url "${TORCH_CUDA_INDEX}"
python -m pip install -e .

python - <<'PY'
import torch
import musdb
import lightning
print("Torch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device 0:", torch.cuda.get_device_name(0))
print("musdb version:", getattr(musdb, "__version__", "unknown"))
print("lightning version:", getattr(lightning, "__version__", "unknown"))
PY

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
TRAIN_ARGS=(
  --config "${TRAIN_CONFIG}"
  --download-preview
  --target-stem "${STEM}"
  --out-dir "${OUT_DIR}"
)

python scripts/train_lightning.py \
  "${TRAIN_ARGS[@]}"

# ---- One-track validation after training ----
python scripts/validate_checkpoint.py \
  --config "${TRAIN_CONFIG}" \
  --ckpt "${OUT_DIR}/${STEM}/best_legacy.pt" \
  --download-preview \
  --subset test \
  --track-index 0 \
  --save-audio "${OUT_DIR}/${STEM}/validation_estimate.wav" \
  --save-originals