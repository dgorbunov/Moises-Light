#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64g
#SBATCH --job-name="moises_test"
#SBATCH --partition=academic
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG=""
CKPT=""
MAX_TRACKS="25"
SAVE_JSON=""
MIXTURE_WAV=""
REFERENCE_WAV=""
SAVE_AUDIO_DIR=""
SAVE_ORIGINALS=""
NO_ADAPT_WEB=""
NORMALIZE_PEAK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --max-tracks)
      MAX_TRACKS="$2"
      shift 2
      ;;
    --save-json)
      SAVE_JSON="$2"
      shift 2
      ;;
    --mixture-wav)
      MIXTURE_WAV="$2"
      shift 2
      ;;
    --reference-wav)
      REFERENCE_WAV="$2"
      shift 2
      ;;
    --save-audio-dir)
      SAVE_AUDIO_DIR="$2"
      shift 2
      ;;
    --save-originals)
      SAVE_ORIGINALS="1"
      shift 1
      ;;
    --no-adapt-web-audio)
      NO_ADAPT_WEB="1"
      shift 1
      ;;
    --normalize-peak)
      NORMALIZE_PEAK="1"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: sbatch configs/turing_test.sh --config <yaml> --ckpt <path> \\"
      echo "       [--max-tracks N|0 for all] [--save-json path] \\"
      echo "       [--mixture-wav path [--reference-wav path] [--save-originals] [--normalize-peak] [--no-adapt-web-audio]] \\"
      echo "       [--save-audio-dir dir [--save-originals]]"
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG}" ]] || [[ -z "${CKPT}" ]]; then
  echo "ERROR: --config and --ckpt are required."
  exit 1
fi

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1

module load python
module load cuda
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  module load ffmpeg >/dev/null 2>&1 || true
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffmpeg and ffprobe are required for musdb."
  exit 1
fi

mkdir -p logs

VENV_MARKER=".venv/.installed_marker"
MARKER_CONTENT="torch-cu129-uv-mamba2-v12"

if [[ ! -d ".venv" ]] || [[ ! -f "${VENV_MARKER}" ]] || [[ "$(cat "${VENV_MARKER}" 2>/dev/null)" != "${MARKER_CONTENT}" ]]; then
  echo "ERROR: Expected training venv with marker '${MARKER_CONTENT}'."
  echo "Run configs/turing_train.sh (or your training sbatch) once so mamba-ssm is built, or align MARKER_CONTENT across scripts."
  exit 1
fi

source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__)+"/lib")'):${LD_LIBRARY_PATH:-}"

TEST_ARGS=(--config "${CONFIG}" --ckpt "${CKPT}")

if [[ -n "${MIXTURE_WAV}" ]]; then
  TEST_ARGS+=(--mixture-wav "${MIXTURE_WAV}")
  [[ -n "${REFERENCE_WAV}" ]] && TEST_ARGS+=(--reference-wav "${REFERENCE_WAV}")
  [[ -n "${NO_ADAPT_WEB}" ]] && TEST_ARGS+=(--no-adapt-web-audio)
  [[ -n "${NORMALIZE_PEAK}" ]] && TEST_ARGS+=(--normalize-peak)
else
  if [[ "${MAX_TRACKS}" != "0" ]]; then
    TEST_ARGS+=(--max-tracks "${MAX_TRACKS}")
  fi
  [[ -n "${SAVE_JSON}" ]] && TEST_ARGS+=(--save-json "${SAVE_JSON}")
  [[ -n "${SAVE_AUDIO_DIR}" ]] && TEST_ARGS+=(--save-audio-dir "${SAVE_AUDIO_DIR}")
fi

[[ -n "${SAVE_ORIGINALS}" ]] && TEST_ARGS+=(--save-originals)

python scripts/test.py "${TEST_ARGS[@]}"
