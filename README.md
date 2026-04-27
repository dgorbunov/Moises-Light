# Mamba-Light
WPI CS 541 Final Project

## Baseline: Moises-Light reproduction (MUSDB18)

This repo contains a runnable baseline inspired by **“Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation” (arXiv:2510.06785v1)**.

### What’s implemented
- **Single-stem training** (paper trains one model per stem): `vocals`, `drums`, `bass`, `other`
- **STFT setup**: window \(6144\), hop \(1024\), truncate to \(2048\) bins
- **Chunking**: train with 75% overlap; eval with 50% overlap-add
- **Augmentations**: polarity inversion, pitch shift, time shift, stereo channel flip (jointly applied to mix+target)
- **Loss**: complex-spectrogram L1 + optional multi-resolution complex STFT MAE
- **Metric**: chunk-level SDR (cSDR) over 1s chunks (median per song, then median over songs)

### Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
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
python .\scripts\train.py --musdb-root "C:\Users\danie\Documents\Datasets\musdb18hq" --target-stem vocals --out-dir runs\moises_light
```

This writes checkpoints under `runs/moises_light/<stem>/best.pt`.

### Evaluate (example)

Create a minimal config YAML (example `configs/vocals.yaml`) or reuse the saved `config.json` values.
The evaluator uses `cfg.target_stem` to choose the reference stem.

```bash
python .\scripts\eval.py --config configs\vocals.yaml --ckpt runs\moises_light\vocals\best.pt
```

### Validation split note
MUSDB18’s 86/14 train/valid split is often stored in external metadata. If you don’t provide it, this repo uses a **deterministic fallback**:
- looks for `musdb_valid.txt` at MUSDB root (one track folder name per line)
- otherwise uses the **last 14 tracks** in sorted order as validation

