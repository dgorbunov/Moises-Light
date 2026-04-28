from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_io import save_audio
from config import TrainConfig, load_config
from dataset_utils import looks_like_wav_layout, split_test_tracks
from infer import load_model_from_ckpt, separate_track
from metrics import chunk_level_sdr
from stft import STFTParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one-track inference sanity check using a trained checkpoint and MUSDB track."
    )
    p.add_argument("--config", type=str, required=True, help="Train/eval config YAML path")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (e.g., best_legacy.pt)")
    p.add_argument("--subset", type=str, default="val", choices=("train", "val", "test"), help="MUSDB subset to sample from")
    p.add_argument("--track-index", type=int, default=0, help="Track index within selected subset")
    p.add_argument("--save-audio", type=str, default="", help="Optional output wav path for separated estimate")
    p.add_argument(
        "--save-originals",
        action="store_true",
        help="When set with --save-audio, also save original mixture and reference stem wav files.",
    )
    return p.parse_args()


def _load_track_audio(
    cfg: TrainConfig,
    subset: str,
    track_index: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    try:
        import musdb
    except Exception as e:
        raise RuntimeError("musdb package is required for validation script.") from e

    root = Path(cfg.musdb_root).expanduser()
    db_kwargs: dict[str, object] = {"root": str(root)}
    is_wav = looks_like_wav_layout(root, subset=subset)

    musdb_subset = "train" if subset == "train" else "test"
    db = musdb.DB(subsets=musdb_subset, is_wav=is_wav, **db_kwargs)
    tracks = list(db.tracks)
    if subset == "val":
        tracks = split_test_tracks(tracks, partition="val")
    elif subset == "test":
        tracks = split_test_tracks(tracks, partition="test")
    if not tracks:
        raise RuntimeError(f"No tracks found for subset='{subset}'.")

    idx = max(0, min(track_index, len(tracks) - 1))
    tr = tracks[idx]
    mix = torch.from_numpy(tr.audio.T).float()
    tgt = torch.from_numpy(tr.targets[cfg.target_stem].audio.T).float()
    return tr.name, mix, tgt


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = load_model_from_ckpt(args.ckpt, device=device)
    stft_params = STFTParams(
        n_fft=cfg.stft.n_fft,
        hop_length=cfg.stft.hop_length,
        win_length=cfg.stft.win_length,
        freq_bins=cfg.stft.freq_bins,
    )

    track_name, mix, ref = _load_track_audio(
        cfg=cfg,
        subset=args.subset,
        track_index=args.track_index,
    )
    est = separate_track(
        model=model,
        mixture=mix,
        stft_params=stft_params,
        segment_seconds=cfg.segment_seconds,
        overlap=cfg.eval_overlap,
        sample_rate=cfg.sample_rate,
        device=device,
    )
    csdr_model = chunk_level_sdr(ref, est, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
    csdr_mix = chunk_level_sdr(ref, mix, sample_rate=cfg.sample_rate, chunk_seconds=1.0)

    est_rms = float(torch.sqrt(torch.mean(est**2)).item())
    ref_rms = float(torch.sqrt(torch.mean(ref**2)).item())
    mix_rms = float(torch.sqrt(torch.mean(mix**2)).item())

    saved_files: dict[str, str] = {}
    if args.save_audio:
        out_est = Path(args.save_audio)
        save_audio(out_est, est, sample_rate=cfg.sample_rate)
        saved_files["estimate"] = str(out_est)

        if args.save_originals:
            out_mix = out_est.with_name(f"{out_est.stem}_mixture{out_est.suffix or '.wav'}")
            out_ref = out_est.with_name(f"{out_est.stem}_reference_{cfg.target_stem}{out_est.suffix or '.wav'}")
            save_audio(out_mix, mix, sample_rate=cfg.sample_rate)
            save_audio(out_ref, ref, sample_rate=cfg.sample_rate)
            saved_files["mixture"] = str(out_mix)
            saved_files["reference"] = str(out_ref)

    result = {
        "track_name": track_name,
        "target_stem": cfg.target_stem,
        "subset": args.subset,
        "track_index": args.track_index,
        "song_median_cSDR": float(csdr_model.song_median_sdr),
        "song_median_cSDR_mixture_baseline": float(csdr_mix.song_median_sdr),
        "song_median_cSDR_delta_vs_mixture": float(csdr_model.song_median_sdr - csdr_mix.song_median_sdr),
        "n_chunks": int(csdr_model.n_chunks),
        "estimate_rms": est_rms,
        "reference_rms": ref_rms,
        "mixture_rms": mix_rms,
        "estimate_to_reference_rms_ratio": float(est_rms / max(ref_rms, 1e-12)),
        "checkpoint": str(args.ckpt),
        "saved_files": saved_files,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
