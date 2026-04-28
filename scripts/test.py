from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TrainConfig, load_config
from dataset_utils import looks_like_wav_layout, split_test_tracks
from infer import load_model_from_ckpt, separate_track
from metrics import chunk_level_sdr
from stft import STFTParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on MUSDB test split.")
    p.add_argument("--config", type=str, required=True, help="Config YAML path")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--max-tracks", type=int, default=0, help="Optional cap on number of test tracks (0 = all)")
    p.add_argument("--save-json", type=str, default="", help="Optional output path for full JSON report")
    return p.parse_args()


@torch.no_grad()
def _eval_track(model: torch.nn.Module, stft_params: STFTParams, cfg: TrainConfig, tr, device: torch.device) -> dict[str, float | int | str]:
    mix = torch.from_numpy(tr.audio.T).float()
    ref = torch.from_numpy(tr.targets[cfg.target_stem].audio.T).float()
    est = separate_track(model=model, mixture=mix, stft_params=stft_params, segment_seconds=cfg.segment_seconds, overlap=cfg.eval_overlap, sample_rate=cfg.sample_rate, device=device)
    sisdr_model = chunk_level_sdr(ref, est, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
    sisdr_mix = chunk_level_sdr(ref, mix, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
    est_rms = float(torch.sqrt(torch.mean(est**2)).item())
    ref_rms = float(torch.sqrt(torch.mean(ref**2)).item())
    mix_rms = float(torch.sqrt(torch.mean(mix**2)).item())
    return {
        "track_name": str(tr.name),
        "song_median_siSDR": float(sisdr_model.song_median_sdr),
        "song_median_siSDR_mixture_baseline": float(sisdr_mix.song_median_sdr),
        "song_median_siSDR_delta_vs_mixture": float(sisdr_model.song_median_sdr - sisdr_mix.song_median_sdr),
        "n_chunks": int(sisdr_model.n_chunks),
        "estimate_rms": est_rms,
        "reference_rms": ref_rms,
        "mixture_rms": mix_rms,
        "estimate_to_reference_rms_ratio": float(est_rms / max(ref_rms, 1e-12)),
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    try:
        import musdb
    except Exception as e:
        raise RuntimeError("musdb package is required for test script.") from e

    root = Path(cfg.musdb_root).expanduser()
    db = musdb.DB(root=str(root), subsets="test", is_wav=looks_like_wav_layout(root, subset="test"))
    tracks = split_test_tracks(list(db.tracks), partition="test")
    if not tracks:
        raise RuntimeError(f"No MUSDB test tracks found at '{root}'.")
    if args.max_tracks > 0:
        tracks = tracks[: args.max_tracks]

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = load_model_from_ckpt(args.ckpt, device=device)
    stft_params = STFTParams(n_fft=cfg.stft.n_fft, hop_length=cfg.stft.hop_length, win_length=cfg.stft.win_length, freq_bins=cfg.stft.freq_bins)
    per_track = [_eval_track(model=model, stft_params=stft_params, cfg=cfg, tr=tr, device=device) for tr in tracks]

    sisdr_vals = np.asarray([r["song_median_siSDR"] for r in per_track], dtype=np.float64)
    mix_vals = np.asarray([r["song_median_siSDR_mixture_baseline"] for r in per_track], dtype=np.float64)
    delta_vals = np.asarray([r["song_median_siSDR_delta_vs_mixture"] for r in per_track], dtype=np.float64)
    ratio_vals = np.asarray([r["estimate_to_reference_rms_ratio"] for r in per_track], dtype=np.float64)
    summary = {
        "target_stem": cfg.target_stem,
        "n_tracks": int(len(per_track)),
        "siSDR_median_over_songs": float(np.median(sisdr_vals)),
        "siSDR_mean_over_songs": float(np.mean(sisdr_vals)),
        "mixture_siSDR_median_over_songs": float(np.median(mix_vals)),
        "delta_siSDR_median_over_songs": float(np.median(delta_vals)),
        "delta_siSDR_mean_over_songs": float(np.mean(delta_vals)),
        "estimate_to_reference_rms_ratio_median": float(np.median(ratio_vals)),
        "estimate_to_reference_rms_ratio_p10": float(np.percentile(ratio_vals, 10)),
        "estimate_to_reference_rms_ratio_p90": float(np.percentile(ratio_vals, 90)),
        "checkpoint": str(args.ckpt),
        "musdb_root": str(root),
    }
    report = {"summary": summary, "per_track": per_track}
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
