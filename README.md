# Moises-Light++

Band-split source separation inspired by [**Moises-Light**](https://arxiv.org/abs/2510.06785). 

Model configurations:
- Moises-Light: `configs/moises.yaml`
- Moises-Light++:`configs/moises++.yaml`

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Mamba needs **`causal-conv1d`** + CUDA **`mamba-ssm`** — install order is in **`configs/turing_train.sh`**.

## Data

Download MUSDB18-HQ from Zenodo website and set `musdb_root` in YAML (default `~/musdb18hq`).

Layout: `train/<track>/{mixture.wav, vocals.wav, ...}` and the same under `test/`.

Splits: train = all `train/`; val = first half of `test/`; test = second half of `test/`.

## Train

```bash
python scripts/train.py --config configs/moises++.yaml --target-stem vocals --out-dir runs/my_run
```

Artifacts: `runs/<out-dir>/<stem>/` (checkpoints, `config.json`). Resume: `--resume .../vocals/checkpoints/last.ckpt`. Multi-GPU: set `trainer.devices` in YAML; batch size is per GPU.

## Test

```bash
python scripts/test.py --config configs/moises++.yaml --ckpt runs/my_run/vocals/best_legacy.pt
```

- **`--max-tracks N`**, **`--save-json path`** — cap / dump JSON  
- **`--save-audio-dir dir [--save-originals]`** — export WAVs  
- **`--mixture-wav path [--reference-wav path]`** — one-off file; estimate saved as **`{mixture_stem}-{target_stem}.wav`**. Mixture-only: audio is adapted (decode → stereo → config SR → peak-safe clamp). With reference: optional **`--no-adapt-web-audio`**, **`--normalize-peak`**.

## Slurm Training/Testing

All in `configs`!
