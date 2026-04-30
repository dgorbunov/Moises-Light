# Moises-Light++

WPI CS 541 Final Project  

**Authors:** Daniel Gorbunov, Daniel Zhang

Baseline inspired by [**Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation**](https://arxiv.org/abs/2510.06785) (arXiv:2510.06785v1), with optional **Mamba2** paths via **`configs/moises++.yaml`**.

---

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Mamba training** additionally needs **`causal-conv1d`** and a CUDA build of **`mamba-ssm`** (see cluster script in **`configs/turing_train.sh`** for a reproducible install order).

---

## Dataset (MUSDB18-HQ)

- Download **MUSDB18-HQ** (WAV layout) and point **`musdb_root`** in your YAML at the dataset root (default in configs: **`~/musdb18hq`**).
- **Splits** (deterministic by track name):
  - **Train:** all tracks under **`train/`**
  - **Val:** first half of **`test/`** tracks
  - **Test:** second half of **`test/`** tracks
- **Layout:** `<MUSDB_ROOT>/train/<track>/{mixture.wav,vocals.wav,...}` (and similarly for **`test/`**).

---

## Configs

| File | Role |
|------|------|
| **`configs/moises.yaml`** | Band-split stack, no Mamba. |
| **`configs/moises++.yaml`** | **Mamba2** + weight sharing; same audio/STFT defaults. |

Training uses **all train chunks** (`chunks_per_track` × number of train tracks per epoch). **`num_workers`** is commented for local SSD vs NFS tuning.

---

## Train (local)

```bash
python scripts/train.py --config configs/moises++.yaml --target-stem vocals --out-dir runs/my_run
```

Checkpoints and **`config.json`** go under **`runs/<out-dir>/<stem>/`**. Lightning **`TQDMProgressBar`** is forced so logs behave well under **`tail -f`**.

**Resume:**

```bash
python scripts/train.py --config configs/moises++.yaml --target-stem vocals \
  --out-dir runs/my_existing_parent_dir \
  --resume runs/my_existing_parent_dir/vocals/checkpoints/last.ckpt
```

**Multi-GPU:** set **`trainer.devices`** (and **`strategy: "auto"`**) in YAML; batch size is **per GPU**.

---

## Test / inference (`scripts/test.py`)

**MUSDB test split** (metrics + optional JSON):

```bash
python scripts/test.py --config configs/moises++.yaml --ckpt runs/my_run/vocals/best_legacy.pt

python scripts/test.py --config configs/moises++.yaml --ckpt runs/my_run/vocals/best_legacy.pt \
  --max-tracks 5 --save-json runs/my_run/vocals/test_report.json
```

**Export WAVs** (per-track filenames under a directory; **`--save-originals`** adds mixture + reference stem):

```bash
python scripts/test.py --config configs/moises++.yaml --ckpt runs/my_run/vocals/best_legacy.pt \
  --save-audio-dir runs/my_run/vocals/test_wavs --save-originals
```

**Single mixture file** — estimate is written beside the input as **`{mixture_stem}-{target_stem}.wav`**.

Mixtures (including **`--mixture-wav`** / **`--reference-wav`**) use the same **`prepare_waveform_tensor`** path as training in **`audio_io`**: decode when needed, stereo layout, resample to **`sample_rate`**, clamp overs in **[−1, 1]** with mild scaling when peaks exceed **±1**. **MUSDB18-HQ** tracks (**44.1 kHz stereo**, peaks ≤ ~**1**) stay on an inexpensive branch (no resample / clamp copy). JSON includes **`input_adaptation`** (source rate/channels, resampling, scaling metadata).

```bash
python scripts/test.py --config configs/moises++.yaml --ckpt runs/my_run/vocals/best_legacy.pt \
  --mixture-wav path/to/mix.wav --reference-wav path/to/vocals.wav --save-originals
```

---

## Turing cluster (SLURM)

See **`configs/README.md`** for **`turing_train.sh`** and **`turing_test.sh`** (`sbatch`, resume, multi-GPU, logs).

---

## Utilities

```bash
python scripts/analyze_chunk_energy.py --config configs/moises++.yaml --samples 2000
```
