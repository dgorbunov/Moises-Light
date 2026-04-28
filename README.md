# Mamba-Light
WPI CS 541 Final Project

## Baseline: Moises-Light reproduction (MUSDB18-HQ)

This repo contains a runnable baseline inspired by **“Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation” (arXiv:2510.06785v1)**.

### Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Dataset
- Default dataset root in configs: `~/musdb18hq` (expanded automatically)
- Split behavior (deterministic by track name):
  - Train: all tracks in `train/`
  - Validation: first half of tracks in `test/`
  - Test: second half of tracks in `test/`
- Expected layout:
  - `<MUSDB_ROOT>/train/<track>/{mixture.wav,vocals.wav,drums.wav,bass.wav,other.wav}`
  - `<MUSDB_ROOT>/test/<track>/{...}`

### Train

```bash
python scripts/train.py --config configs/vocals_long.yaml
```

Preset configs:

```bash
# quick sanity run (full dataset, tiny sample caps)
python scripts/train.py --config configs/vocals_short.yaml

# medium run (full MUSDB, capped epoch size)
python scripts/train.py --config configs/vocals_medium.yaml

# long run (full MUSDB, uncapped sample counts)
python scripts/train.py --config configs/vocals_long.yaml
```

### Optional: Validate One Track

```bash
python scripts/validate.py --config configs/vocals_short.yaml --ckpt runs/moises_light/vocals/best_legacy.pt --subset val --track-index 0 --save-audio runs/moises_light/vocals/validation_estimate.wav --save-originals
```

### Optional: Analyze Chunk Energy

```bash
python scripts/analyze_chunk_energy.py --config configs/vocals_long.yaml --samples 2000
```

### Test

```bash
python scripts/test.py --config configs/vocals_long.yaml --ckpt runs/moises_light/vocals/best_legacy.pt
```

Optional:

```bash
# quick subset check
python scripts/test.py --config configs/vocals_long.yaml --ckpt runs/moises_light/vocals/best_legacy.pt --max-tracks 5

# save full per-track report
python scripts/test.py --config configs/vocals_long.yaml --ckpt runs/moises_light/vocals/best_legacy.pt --save-json runs/moises_light/vocals/test_report.json
```
