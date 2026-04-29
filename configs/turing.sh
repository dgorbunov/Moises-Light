#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --job-name="moises_light_vocals"
#SBATCH --partition=academic
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- CLI flags (passed through sbatch script args) ----
# Example:
#   sbatch configs/turing.sh --train-config configs/vocals_short.yaml --stem vocals --test-type quick
TRAIN_CONFIG="configs/vocals_short.yaml"
STEM="vocals"
TEST_TYPE="quick"  # quick | full | none
MAX_TEST_TRACKS="" # optional manual override

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --stem)
      STEM="$2"
      shift 2
      ;;
    --test-type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --max-test-tracks)
      MAX_TEST_TRACKS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: sbatch configs/turing.sh [--train-config <path>] [--stem <vocals|drums|bass|other>] [--test-type <quick|full|none>] [--max-test-tracks <N>]"
      exit 1
      ;;
  esac
done

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
# The venv is reused across jobs when the marker matches to avoid recompiling
# mamba-ssm CUDA extensions (~15 min) on every submission. Bump MARKER_CONTENT
# whenever requirements.txt or the torch version changes.
VENV_MARKER=".venv/.installed_marker"
MARKER_CONTENT="torch-2.8.0+mamba-ssm-prebuilt"
TORCH_CUDA_INDEX=""

rebuild_venv=false
if [ ! -d ".venv" ] || [ ! -f "${VENV_MARKER}" ] || [ "$(cat "${VENV_MARKER}" 2>/dev/null)" != "${MARKER_CONTENT}" ]; then
  rebuild_venv=true
fi

if [ "${rebuild_venv}" = "true" ]; then
  echo "Building Python virtual environment (marker '${MARKER_CONTENT}' not found or stale)..."
  rm -rf .venv
  python -m venv .venv
  source .venv/bin/activate

  python -m pip install --upgrade pip

  # Prefer cu129 (for CUDA 12.9 module), fall back to cu128 if unavailable.
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

  # Install all requirements except mamba packages (handled separately below).
  python -m pip install -r requirements.txt --extra-index-url "${TORCH_CUDA_INDEX}"
  python -m pip install -e .

  # Install mamba pre-built wheels matched to CUDA 12 + PyTorch 2.8 + cxx11 ABI TRUE + Python 3.13.
  # These wheels are from the official GitHub releases and avoid any source compilation.
  CAUSAL_CONV1D_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
  MAMBA_SSM_WHL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"

  echo "Installing causal-conv1d pre-built wheel..."
  python -m pip install "${CAUSAL_CONV1D_WHL}"
  echo "Installing mamba-ssm pre-built wheel..."
  python -m pip install "${MAMBA_SSM_WHL}"

  # Verify the import works before writing the marker.
  if python -c "from mamba_ssm import Mamba; print('mamba-ssm import OK')"; then
    echo "${MARKER_CONTENT}" > "${VENV_MARKER}"
    echo "mamba-ssm installed successfully — venv will be reused on future jobs."
  else
    echo "WARNING: mamba-ssm installed but import still fails. Training will use pure-PyTorch fallback."
    echo "${MARKER_CONTENT}-fallback" > "${VENV_MARKER}"
  fi
else
  echo "Reusing existing .venv (marker: ${MARKER_CONTENT})"
  source .venv/bin/activate
fi

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

export TRAIN_CONFIG

if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "Missing training config: ${TRAIN_CONFIG}"
  exit 1
fi

python - <<'PY'
import os
from config import load_config
cfg = load_config(os.environ["TRAIN_CONFIG"])
if abs(float(cfg.segment_seconds) - 7.0) > 1e-9:
    raise SystemExit(f"Expected segment_seconds=7.0, got {cfg.segment_seconds}")
print("Config check passed: using 7-second segments.")
PY

CFG_BASENAME="$(basename "${TRAIN_CONFIG}" .yaml)"
RUN_STAMP="$(date +%m-%d_%H-%M)"
OUT_DIR="runs/${CFG_BASENAME}_${RUN_STAMP}"

TRAIN_ARGS=(
  --config "${TRAIN_CONFIG}"
  --target-stem "${STEM}"
  --out-dir "${OUT_DIR}"
)

python scripts/train.py "${TRAIN_ARGS[@]}"

# ---- Testing after training ----
if [[ "${TEST_TYPE}" != "none" ]]; then
  if [[ -n "${MAX_TEST_TRACKS}" ]]; then
    TEST_TRACK_ARG=(--max-tracks "${MAX_TEST_TRACKS}")
  else
    case "${TEST_TYPE}" in
      quick)
        TEST_TRACK_ARG=(--max-tracks 5)
        ;;
      full)
        TEST_TRACK_ARG=()
        ;;
      *)
        echo "Invalid --test-type '${TEST_TYPE}'. Expected: quick|full|none"
        exit 1
        ;;
    esac
  fi

  python scripts/test.py \
    --config "${TRAIN_CONFIG}" \
    --ckpt "${OUT_DIR}/${STEM}/best_legacy.pt" \
    "${TEST_TRACK_ARG[@]}" \
    --save-json "${OUT_DIR}/${STEM}/test_report.json"
fi