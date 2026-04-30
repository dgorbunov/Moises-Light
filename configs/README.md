# Turing (SLURM) jobs

Submit jobs from the **repository root** (`cd` to `Mamba-Light` first). Scripts assume **`logs/`** exists or create it.

## Training (`turing_train.sh`)

Default batch directives (edit in the script if needed):

- **`#SBATCH --partition=academic`**, **`--gres=gpu:2`**, **`--cpus-per-task=64`**, **`--mem=64g`**, **`--time=0-20:00:00`**
- Logs: **`logs/moises_train_<JOB_ID>.out`** / **`.err`** (job name is **`moises_train`**)

First run builds **`/.venv`** once (Torch cu129, deps, **`causal-conv1d`**, **`mamba-ssm`**). Bump **`MARKER_CONTENT`** in the script when Torch or deps change so workers rebuild consistently.

### Examples

```bash
# Fresh run — YAML presets live under configs/ (moises.yaml vs moises++.yaml)
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals --test-type quick

# Long train + full MUSDB test split after training
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals --test-type full

# Train only (no post-run test.py)
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals --test-type none

# Resume — parent run dir + Lightning checkpoint (usually last.ckpt)
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals \
  --out-dir runs/moises++_05-01_14-30 \
  --resume runs/moises++_05-01_14-30/vocals/checkpoints/last.ckpt \
  --test-type none

# Cap post-training test tracks
sbatch configs/turing_train.sh --train-config configs/moises++.yaml --stem vocals \
  --test-type full --max-test-tracks 10
```

### Flags

| Flag | Meaning |
|------|---------|
| **`--train-config`** | Path to training YAML (passed to **`scripts/train.py --config`**). |
| **`--stem`** | Target stem: **`vocals`**, **`drums`**, **`bass`**, **`other`**. |
| **`--test-type`** | **`quick`** → test **5** tracks after train; **`full`** → all test tracks; **`none`** → skip **`test.py`**. |
| **`--max-test-tracks`** | Override track cap when **`--test-type`** is not **`none`**. |
| **`--out-dir`** | Existing parent **`runs/...`** dir for resume (no new timestamp folder). |
| **`--resume`** | Lightning **`.ckpt`** path (e.g. **`.../checkpoints/last.ckpt`**). |

### Multi-GPU

1. Request GPUs in SLURM (e.g. **`#SBATCH --gres=gpu:2`**).
2. Set **`trainer.devices: 2`** (or **`N`**) in your YAML **`trainer:`** block.

Batch size in YAML is **per GPU**; global batch scales with **`devices`**.

---

## Test-only (`turing_test.sh`)

Separate GPU job for **`scripts/test.py`** (same `.venv` marker as training).

Default: **`--gres=gpu:2`** (edit script if needed), logs **`logs/moises_test_<JOB_ID>.out`**.

### MUSDB test split

```bash
sbatch configs/turing_test.sh \
  --config configs/moises++.yaml \
  --ckpt runs/my_run/vocals/best_legacy.pt \
  --max-tracks 5

# Full split + JSON report
sbatch configs/turing_test.sh \
  --config configs/moises++.yaml \
  --ckpt runs/my_run/vocals/best_legacy.pt \
  --max-tracks 0 \
  --save-json runs/my_run/vocals/test_report.json

# Export per-track WAVs (+ mixture/reference with --save-originals)
sbatch configs/turing_test.sh \
  --config configs/moises++.yaml \
  --ckpt runs/my_run/vocals/best_legacy.pt \
  --save-audio-dir runs/my_run/vocals/test_wavs \
  --save-originals
```

### Arbitrary mixture WAV

Training and **`scripts/test.py`** share **`audio_io.prepare_waveform_tensor`** (also applied in **`MusdbTrainChunkDataset`** / **`MusdbValRandomChunkDataset`**):

- decode with **torchaudio** when possible, else **soundfile**
- **mono → stereo** (duplicate channel); **>2 channels → first two**
- **resample** to the config **`sample_rate`** only when source rate differs
- mild gain + **clamp** when decoded peaks exceed **±1**

**MUSDB18-HQ** WAV (**44.1 kHz stereo**, nominal peaks ≤ ~**1**) matches the fast branch (no resampling or clamp allocation beyond dtype/peek checks).

JSON output includes **`input_adaptation`** (source rate/channels, resampling, **`scaled_down_overs`**). Mixture/reference are truncated to **min length** when lengths disagree slightly.

Estimate path remains **`{mixture_stem}-{target_stem}.wav`** beside the input.

```bash
sbatch configs/turing_test.sh \
  --config configs/moises++.yaml \
  --ckpt runs/my_run/vocals/best_legacy.pt \
  --mixture-wav /path/to/mix.wav \
  --reference-wav /path/to/vocals.wav \
  --save-originals
```

---

## Queue and logs

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Reason

tail -f logs/moises_train_<JOB_ID>.out
tail -f logs/moises_test_<JOB_ID>.out
```

## GPU utilization

```bash
watch -n 1 nvidia-smi
```
