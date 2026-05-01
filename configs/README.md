# Turing SLURM Configs 

Run from repo root. Logs under **`logs/`**.

## Training

SBATCH defaults are in the script (partition, GPUs, CPUs, mem, time). First job builds **`.venv`** (Torch + deps + Mamba bits); bump **`MARKER_CONTENT`** in the script when you change that stack.

```bash
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals --test-type quick
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals --test-type full
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals --test-type none
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals \
  --out-dir runs/my_run_parent --resume runs/my_run_parent/vocals/checkpoints/last.ckpt --test-type none
```

| Flag | Meaning |
|------|---------|
| **`--train-config`** | YAML for **`scripts/train.py`** |
| **`--stem`** | `vocals` \| `drums` \| `bass` \| `other` |
| **`--test-type`** | **`quick`** (5 tracks after train), **`full`**, **`none`** |
| **`--max-test-tracks`** | Override quick/full track cap |
| **`--out-dir`** | Existing **`runs/...`** parent (resume / reuse dir) |
| **`--resume`** | Lightning **`.ckpt`** |

Match **`trainer.devices`** to SLURM GPUs; YAML batch size is per GPU.

## Testing

Same **`.venv`** marker as train.

```bash
sbatch configs/turing_test.sh --config configs/moises++.yaml --ckpt runs/my_run/vocals/best_legacy.pt --max-tracks 5
sbatch configs/turing_test.sh --config configs/moises++.yaml --ckpt path/to/best_legacy.pt \
  --mixture-wav /path/mix.wav --reference-wav /path/vocals.wav --save-originals
```

Optional with **`--reference-wav`** for metrics compared to reference stem.

## Logging

```bash
squeue -u "$USER"
tail -f logs/moises_train_<JOB_ID>.out
tail -f logs/moises_test_<JOB_ID>.out
```
