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
from config import MultiResSTFTConfig, STFTConfig, TrainConfig, TrainerConfig
from dataset_utils import looks_like_wav_layout, split_test_tracks
from infer import load_model_from_ckpt, separate_track
from metrics import chunk_level_sdr
from stft import STFTParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one-track inference sanity check using a trained checkpoint and MUSDB track."
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Top-level run directory, or stem run directory containing config.json and best_legacy.pt",
    )
    p.add_argument("--subset", type=str, default="val", choices=("train", "val", "test"), help="MUSDB subset to sample from")
    p.add_argument("--track-index", type=int, default=0, help="Track index within selected subset")
    p.add_argument("--save-audio", type=str, default="", help="Optional output wav path for separated estimate")
    p.add_argument(
        "--save-originals",
        action="store_true",
        help="When set with --save-audio, also save original mixture and reference stem wav files.",
    )
    return p.parse_args()


def _load_train_config_from_json(path: Path) -> TrainConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "stft" in data and isinstance(data["stft"], dict):
        data["stft"] = STFTConfig(**data["stft"])
    if "multires" in data and isinstance(data["multires"], dict):
        data["multires"] = MultiResSTFTConfig(**data["multires"])
    if "trainer" in data and isinstance(data["trainer"], dict):
        data["trainer"] = TrainerConfig(**data["trainer"])
    return TrainConfig(**data)


def _resolve_stem_run_dir(run_dir: Path) -> Path:
    config_path = run_dir / "config.json"
    if config_path.exists():
        return run_dir

    candidates = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        has_config = (child / "config.json").exists()
        if has_config:
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(sorted(c.name for c in candidates))
        raise RuntimeError(
            f"Multiple stem run directories found under '{run_dir}': {names}. "
            "Pass the specific stem directory to --run-dir."
        )
    raise RuntimeError(
        f"Could not find run artifacts under '{run_dir}'. Expected config.json either directly in this "
        "directory or in exactly one child directory."
    )


def _resolve_checkpoint_path(stem_run_dir: Path) -> Path:
    legacy = stem_run_dir / "best_legacy.pt"
    if legacy.exists():
        return legacy

    ckpt_dir = stem_run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    ckpt_candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpt_candidates:
        return ckpt_candidates[0]
    raise RuntimeError(
        f"No checkpoint found in '{stem_run_dir}'. Expected best_legacy.pt or *.ckpt in '{ckpt_dir}'."
    )


def _resolve_runtime_inputs(args: argparse.Namespace) -> tuple[TrainConfig, Path, Path]:
    if not args.run_dir:
        raise RuntimeError("Pass --run-dir with a top-level or stem run directory.")
    stem_run_dir = _resolve_stem_run_dir(Path(args.run_dir).expanduser())
    cfg = _load_train_config_from_json(stem_run_dir / "config.json")
    ckpt_path = _resolve_checkpoint_path(stem_run_dir)
    return cfg, ckpt_path, stem_run_dir


def _load_track_audio(
    cfg: TrainConfig,
    subset: str,
    track_index: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    try:
        import musdb
    except Exception as e:
        raise RuntimeError("musdb package is required for validation script.") from e

    musdb_subset = "train" if subset == "train" else "test"
    root = Path(cfg.musdb_root).expanduser()
    db_kwargs: dict[str, object] = {"root": str(root)}
    # "val" is a logical split of MUSDB test tracks; filesystem subset is "test".
    is_wav = looks_like_wav_layout(root, subset=musdb_subset)

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
    cfg, ckpt_path, stem_run_dir = _resolve_runtime_inputs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = load_model_from_ckpt(str(ckpt_path), device=device, cfg=cfg)
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
    sisdr_model = chunk_level_sdr(ref, est, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
    sisdr_mix = chunk_level_sdr(ref, mix, sample_rate=cfg.sample_rate, chunk_seconds=1.0)

    est_rms = float(torch.sqrt(torch.mean(est**2)).item())
    ref_rms = float(torch.sqrt(torch.mean(ref**2)).item())
    mix_rms = float(torch.sqrt(torch.mean(mix**2)).item())

    saved_files: dict[str, str] = {}
    out_est: Path | None = None
    if args.save_audio:
        out_est = Path(args.save_audio)
    elif args.run_dir:
        out_est = stem_run_dir / "validation_estimate.wav"

    if out_est is not None:
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
        "song_median_siSDR": float(sisdr_model.song_median_sdr),
        "song_median_siSDR_mixture_baseline": float(sisdr_mix.song_median_sdr),
        "song_median_siSDR_delta_vs_mixture": float(sisdr_model.song_median_sdr - sisdr_mix.song_median_sdr),
        "n_chunks": int(sisdr_model.n_chunks),
        "estimate_rms": est_rms,
        "reference_rms": ref_rms,
        "mixture_rms": mix_rms,
        "estimate_to_reference_rms_ratio": float(est_rms / max(ref_rms, 1e-12)),
        "checkpoint": str(ckpt_path),
        "saved_files": saved_files,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
