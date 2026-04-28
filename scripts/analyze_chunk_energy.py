from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TrainConfig, load_config
from dataset_utils import looks_like_wav_layout


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate how often sampled MUSDB chunks for a target stem have very low energy."
    )
    p.add_argument("--config", type=str, required=True, help="Config YAML path")
    p.add_argument("--samples", type=int, default=2000, help="Number of random train chunks to sample")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument(
        "--reference-wav",
        type=str,
        default="",
        help="Optional fallback path to a reference stem wav for chunk-energy analysis when MUSDB train split is unavailable.",
    )
    return p.parse_args()


def _sample_train_chunk_stats(cfg: TrainConfig, n_samples: int, seed: int) -> dict[str, object]:
    try:
        import musdb
    except Exception as e:
        raise RuntimeError("musdb package is required.") from e

    rng = random.Random(seed)
    root = Path(cfg.musdb_root).expanduser()
    is_wav = looks_like_wav_layout(root, subset="train")
    train_db = musdb.DB(root=str(root), subsets="train", split="train", is_wav=is_wav)
    tracks = list(train_db.tracks)
    if not tracks:
        raise RuntimeError("No train tracks found.")

    seg = float(cfg.segment_seconds)
    target_rms: list[float] = []
    mix_rms: list[float] = []
    rms_ratio: list[float] = []

    for _ in range(max(1, int(n_samples))):
        tr = rng.choice(tracks)
        tr.chunk_duration = seg
        max_start = max(0.0, float(tr.duration) - seg)
        tr.chunk_start = rng.uniform(0.0, max_start) if max_start > 0.0 else 0.0
        mix = tr.audio
        tgt = tr.targets[cfg.target_stem].audio
        mr = _rms(mix)
        trms = _rms(tgt)
        target_rms.append(trms)
        mix_rms.append(mr)
        rms_ratio.append(float(trms / max(mr, 1e-12)))

    t = np.asarray(target_rms, dtype=np.float64)
    r = np.asarray(rms_ratio, dtype=np.float64)
    thresholds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    return {
        "target_stem": cfg.target_stem,
        "samples": int(n_samples),
        "segment_seconds": float(cfg.segment_seconds),
        "musdb_root": str(root),
        "is_wav_layout": bool(is_wav),
        "target_rms_mean": float(np.mean(t)),
        "target_rms_median": float(np.median(t)),
        "target_rms_p10": float(np.percentile(t, 10)),
        "target_rms_p90": float(np.percentile(t, 90)),
        "ratio_mean": float(np.mean(r)),
        "ratio_median": float(np.median(r)),
        "ratio_p10": float(np.percentile(r, 10)),
        "ratio_p90": float(np.percentile(r, 90)),
        "silent_fraction_by_target_rms": {
            str(th): float(np.mean(t <= th)) for th in thresholds
        },
        "low_fraction_by_target_to_mix_ratio": {
            str(th): float(np.mean(r <= th)) for th in thresholds
        },
    }


def _analyze_reference_wav(cfg: TrainConfig, wav_path: Path) -> dict[str, object]:
    audio, sr = sf.read(wav_path, always_2d=True)
    if int(sr) != int(cfg.sample_rate):
        raise ValueError(f"Expected sample_rate={cfg.sample_rate}, got {sr}")
    x = audio.T.astype(np.float64)  # (C, T)
    seg = int(round(cfg.segment_seconds * cfg.sample_rate))
    if seg <= 0:
        raise ValueError("segment_seconds must be positive")
    n = max(1, x.shape[-1] // seg)
    vals = []
    for i in range(n):
        s = i * seg
        chunk = x[:, s : s + seg]
        if chunk.shape[-1] < seg:
            break
        vals.append(_rms(chunk))
    t = np.asarray(vals, dtype=np.float64)
    thresholds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    return {
        "target_stem": cfg.target_stem,
        "analysis_source": str(wav_path),
        "samples": int(len(vals)),
        "segment_seconds": float(cfg.segment_seconds),
        "target_rms_mean": float(np.mean(t)),
        "target_rms_median": float(np.median(t)),
        "target_rms_p10": float(np.percentile(t, 10)),
        "target_rms_p90": float(np.percentile(t, 90)),
        "silent_fraction_by_target_rms": {
            str(th): float(np.mean(t <= th)) for th in thresholds
        },
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    try:
        stats = _sample_train_chunk_stats(cfg=cfg, n_samples=args.samples, seed=args.seed)
    except RuntimeError as e:
        if "No train tracks found." not in str(e) or not args.reference_wav:
            raise
        stats = _analyze_reference_wav(cfg=cfg, wav_path=Path(args.reference_wav).expanduser())
        stats["note"] = "MUSDB train split unavailable at configured root; reported fallback stats over the provided reference wav."
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
