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
MARKER_CONTENT="torch-cu129-uv-mamba-git-v11"

rebuild_venv=false
if [ ! -d ".venv" ] || [ ! -f "${VENV_MARKER}" ] || [ "$(cat "${VENV_MARKER}" 2>/dev/null)" != "${MARKER_CONTENT}" ]; then
  rebuild_venv=true
fi

if [ "${rebuild_venv}" = "true" ]; then
  echo "Building Python virtual environment (marker '${MARKER_CONTENT}' not found or stale)..."
  rm -rf .venv

  # Install uv if not already present (mirrors afrenkai/mamba-glibc-fix approach).
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  uv venv .venv
  source .venv/bin/activate

  # torch from cu129 (no version pin) — uv has its own cache, avoiding the
  # stale cu130 wheel that pip keeps reusing from ~/.cache/pip.
  echo "Installing torch from cu129..."
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129
  echo "Torch installed: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.version.cuda)')"

  uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
  uv pip install -e .

  # Build mamba-ssm from git (exact approach from afrenkai/mamba-glibc-fix).
  export TORCH_CUDA_ARCH_LIST="8.0"
  export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__)+"/lib")'):${LD_LIBRARY_PATH:-}"

  MAMBA_BUILD_DIR="/tmp/mamba-ssm-build-$$"
  git clone --depth 1 https://github.com/state-spaces/mamba.git "${MAMBA_BUILD_DIR}"
  echo "Building mamba-ssm from git clone..."
  # Non-editable install: copies into site-packages so the source dir can be removed.
  MAX_JOBS=4 uv pip install --no-build-isolation --no-cache-dir "${MAMBA_BUILD_DIR}"
  rm -rf "${MAMBA_BUILD_DIR}"

  if python -c "from mamba_ssm import Mamba; print('mamba-ssm import OK')"; then
    echo "${MARKER_CONTENT}" > "${VENV_MARKER}"
    echo "mamba-ssm built successfully — venv will be reused on future jobs."
  else
    echo "WARNING: mamba-ssm CUDA kernels unavailable. Training uses pure-PyTorch Mamba fallback."
    echo "${MARKER_CONTENT}" > "${VENV_MARKER}"
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