# Mamba-Light
WPI CS 541 Final Project

## Baseline: Moises-Light reproduction (MUSDB18)

This repo contains a runnable baseline inspired by **“Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation” (arXiv:2510.06785v1)**.

### What’s implemented
- **Single-stem training** (choose one: `vocals`, `drums`, `bass`, `other`)
- **STFT setup**: window \(6144\), hop \(1024\), truncate to \(2048\) bins
- **Chunking**: train with 75% overlap; eval with 50% overlap-add
- **Augmentations**: polarity inversion, pitch shift, time shift, stereo channel flip (jointly applied to mix+target)
- **Loss**: complex-spectrogram L1 + optional multi-resolution complex STFT MAE
- **Metric**: chunk-level SDR (cSDR) over 1s chunks (median per song, then median over songs)

### Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### MUSDB18 layout
Expected folder layout (either):
- `<MUSDB_ROOT>/train/<track>/{mixture.wav,vocals.wav,drums.wav,bass.wav,other.wav}`
- `<MUSDB_ROOT>/test/<track>/{...}`

If you have a different export, adapt it or symlink to match.

### Train (example)

```bash
python scripts/train.py --musdb-root "/path/to/musdb18hq" --target-stem vocals --out-dir runs/moises_light
```

This writes checkpoints under `runs/moises_light/<stem>/best.pt`.

### Train with PyTorch Lightning (2x GPU DDP)

```bash
python scripts/train_lightning.py --config configs/vocals.yaml
```

This uses:
- `accelerator: gpu`
- `devices: 2`
- `strategy: ddp`
- `precision: 16-mixed`

For quick verification on 2 tracks and 10 epochs:

```bash
python scripts/train_lightning.py --config configs/vocals.yaml --debug
```

Debug metrics are saved to `runs/moises_light/vocals/metrics_debug.json`, and an eval-compatible checkpoint is exported to `runs/moises_light/vocals/best_legacy.pt`.

To validate the training pipeline without downloading full MUSDB first, use musdb's preview clips:

```bash
python scripts/train_lightning.py --download-preview --target-stem vocals --debug
```

You can also set `download_preview: true` in `configs/vocals.yaml`.

To limit epoch size for quick tests, cap sample counts:

```bash
python scripts/train_lightning.py --download-preview --target-stem vocals --max-train-samples 10 --max-val-samples 5
```

Preset configs:

```bash
# quick sanity run (preview clips, tiny sample caps)
python scripts/train_lightning.py --config configs/vocals_short.yaml

# medium run (full MUSDB, capped epoch size)
python scripts/train_lightning.py --config configs/vocals_medium.yaml

# long run (full MUSDB, uncapped sample counts)
python scripts/train_lightning.py --config configs/vocals_long.yaml
```

### Recommended commands

Smoke test on Mac (M3/MPS, quick pipeline validation):

```bash
python scripts/train_lightning.py --download-preview --target-stem vocals --debug
python scripts/validate_checkpoint.py --config configs/vocals.yaml --ckpt runs/moises_light/vocals/best_legacy.pt --download-preview --subset test --track-index 0 --save-audio runs/moises_light/vocals/validation_estimate.wav --save-originals
```

Full training on 2x NVIDIA A30 (DDP + mixed precision):

```bash
python scripts/train_lightning.py --config configs/vocals.yaml --musdb-root "/path/to/musdb18hq" --target-stem vocals
python scripts/validate_checkpoint.py --config configs/vocals.yaml --ckpt runs/moises_light/vocals/best_legacy.pt --subset test --track-index 0 --save-audio runs/moises_light/vocals/validation_estimate.wav --save-originals
```

With `--save-originals`, the script also writes comparison files next to the estimate:
- `validation_estimate_mixture.wav`
- `validation_estimate_reference_vocals.wav`

### Evaluate (example)

Create a minimal config YAML (example `configs/vocals.yaml`) or reuse the saved `config.json` values.
The evaluator uses `cfg.target_stem` to choose the reference stem.

```bash
python scripts/eval.py --config configs/vocals.yaml --ckpt runs/moises_light/vocals/best_legacy.pt
```

### Validation split note
MUSDB18’s 86/14 train/valid split is often stored in external metadata. If you don’t provide it, this repo uses a **deterministic fallback**:
- looks for `musdb_valid.txt` at MUSDB root (one track folder name per line)
- otherwise uses the **last 14 tracks** in sorted order as validation

