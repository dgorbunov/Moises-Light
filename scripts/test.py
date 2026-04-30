from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_io import load_mixture_and_optional_reference_for_test, prepare_waveform_tensor, save_audio
from config import TrainConfig, load_config
from dataset_utils import looks_like_wav_layout, split_test_tracks
from infer import load_model_from_ckpt, separate_track
from metrics import chunk_level_sdr
from stft import STFTParams


def _safe_track_filename(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("_")
    return base[:180] if base else "track"


def _mixture_wav_estimate_path(mix_path: Path, target_stem: str) -> Path:
    suf = mix_path.suffix if mix_path.suffix else ".wav"
    return mix_path.with_name(f"{mix_path.stem}-{target_stem}{suf}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on MUSDB test split or an arbitrary mixture wav.")
    p.add_argument("--config", type=str, required=True, help="Config YAML path")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--max-tracks", type=int, default=0, help="MUSDB mode only: cap number of test tracks (0 = all)")
    p.add_argument("--save-json", type=str, default="", help="Optional output path for full JSON report (MUSDB mode)")
    p.add_argument(
        "--mixture-wav",
        type=str,
        default="",
        help="Instead of MUSDB test split: path to mixture audio (default: resample/stereo/clamp like MUSDB18-HQ).",
    )
    p.add_argument("--reference-wav", type=str, default="", help="With --mixture-wav: optional reference stem wav for SI-SDR metrics")
    p.add_argument(
        "--save-audio-dir",
        type=str,
        default="",
        help="MUSDB mode: directory for per-track exports ({safe_name}_estimate.wav, optionally mixture/reference)",
    )
    p.add_argument(
        "--save-originals",
        action="store_true",
        help="With --save-audio-dir (MUSDB): also save mixture and reference stem wavs per track. "
        "With --mixture-wav: also saves mixture (and reference if --reference-wav) next to the auto-named estimate.",
    )
    return p.parse_args()


def _metrics_row(
    track_name: str,
    cfg: TrainConfig,
    mix: torch.Tensor,
    ref: torch.Tensor,
    est: torch.Tensor,
    saved_files: dict[str, str],
) -> dict[str, float | int | str | dict[str, str]]:
    sisdr_model = chunk_level_sdr(ref, est, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
    sisdr_mix = chunk_level_sdr(ref, mix, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
    est_rms = float(torch.sqrt(torch.mean(est**2)).item())
    ref_rms = float(torch.sqrt(torch.mean(ref**2)).item())
    mix_rms = float(torch.sqrt(torch.mean(mix**2)).item())
    return {
        "track_name": track_name,
        "song_median_siSDR": float(sisdr_model.song_median_sdr),
        "song_median_siSDR_mixture_baseline": float(sisdr_mix.song_median_sdr),
        "song_median_siSDR_delta_vs_mixture": float(sisdr_model.song_median_sdr - sisdr_mix.song_median_sdr),
        "n_chunks": int(sisdr_model.n_chunks),
        "estimate_rms": est_rms,
        "reference_rms": ref_rms,
        "mixture_rms": mix_rms,
        "estimate_to_reference_rms_ratio": float(est_rms / max(ref_rms, 1e-12)),
        "saved_files": saved_files,
    }


@torch.no_grad()
def _eval_musdb_track(
    model: torch.nn.Module,
    stft_params: STFTParams,
    cfg: TrainConfig,
    tr,
    device: torch.device,
    save_dir: Path | None,
    save_originals: bool,
) -> dict[str, float | int | str | dict[str, str]]:
    sr = int(tr.rate) if getattr(tr, "rate", None) is not None else cfg.sample_rate
    mix, _ = prepare_waveform_tensor(torch.from_numpy(tr.audio.T).float(), sr, cfg.sample_rate)
    ref, _ = prepare_waveform_tensor(
        torch.from_numpy(tr.targets[cfg.target_stem].audio.T).float(),
        sr,
        cfg.sample_rate,
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
    saved: dict[str, str] = {}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = save_dir / _safe_track_filename(tr.name)
        est_path = stem.parent / f"{stem.name}_estimate.wav"
        save_audio(est_path, est, sample_rate=cfg.sample_rate)
        saved["estimate"] = str(est_path)
        if save_originals:
            mix_path = stem.parent / f"{stem.name}_mixture.wav"
            ref_path = stem.parent / f"{stem.name}_reference_{cfg.target_stem}.wav"
            save_audio(mix_path, mix, sample_rate=cfg.sample_rate)
            save_audio(ref_path, ref, sample_rate=cfg.sample_rate)
            saved["mixture"] = str(mix_path)
            saved["reference"] = str(ref_path)
    return _metrics_row(str(tr.name), cfg, mix, ref, est, saved)


def _run_mixture_wav(
    cfg: TrainConfig,
    args: argparse.Namespace,
    model: torch.nn.Module,
    stft_params: STFTParams,
    device: torch.device,
) -> dict[str, object]:
    mix_path = Path(args.mixture_wav).expanduser()
    ref_path = Path(args.reference_wav).expanduser() if args.reference_wav else None
    mix, ref, adapt_meta = load_mixture_and_optional_reference_for_test(
        mix_path,
        ref_path,
        target_sample_rate=cfg.sample_rate,
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
    saved: dict[str, str] = {}

    out_est = _mixture_wav_estimate_path(mix_path, cfg.target_stem)
    save_audio(out_est, est, sample_rate=cfg.sample_rate)
    saved["estimate"] = str(out_est)

    if args.save_originals:
        suf = out_est.suffix if out_est.suffix else ".wav"
        mix_out = out_est.with_name(f"{out_est.stem}_mixture{suf}")
        save_audio(mix_out, mix, sample_rate=cfg.sample_rate)
        saved["mixture"] = str(mix_out)
        if ref is not None:
            ref_out = out_est.with_name(f"{out_est.stem}_reference_{cfg.target_stem}{suf}")
            save_audio(ref_out, ref, sample_rate=cfg.sample_rate)
            saved["reference"] = str(ref_out)

    if ref is None:
        summary = {
            "mode": "mixture_wav",
            "mixture_path": str(mix_path),
            "reference_path": "",
            "note": "No --reference-wav; SI-SDR metrics omitted.",
            "saved_files": saved,
            "input_adaptation": adapt_meta,
        }
        print(json.dumps(summary, indent=2))
        return summary

    row = _metrics_row(mix_path.name, cfg, mix, ref, est, saved)
    row["mode"] = "mixture_wav"
    row["mixture_path"] = str(mix_path)
    row["reference_path"] = str(Path(args.reference_wav).expanduser())
    row["input_adaptation"] = adapt_meta
    print(json.dumps(row, indent=2))
    return row


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.mixture_wav and args.save_audio_dir:
        raise RuntimeError("Use either --mixture-wav or --save-audio-dir, not both.")
    if args.reference_wav and not args.mixture_wav:
        raise RuntimeError("--reference-wav is only valid with --mixture-wav.")
    if args.save_originals and not args.mixture_wav and not args.save_audio_dir:
        raise RuntimeError("--save-originals requires --save-audio-dir (MUSDB) or --mixture-wav.")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = load_model_from_ckpt(args.ckpt, device=device, cfg=cfg)
    stft_params = STFTParams(n_fft=cfg.stft.n_fft, hop_length=cfg.stft.hop_length, win_length=cfg.stft.win_length, freq_bins=cfg.stft.freq_bins)

    if args.mixture_wav:
        _run_mixture_wav(cfg, args, model, stft_params, device)
        return

    try:
        import musdb
    except Exception as e:
        raise RuntimeError("musdb package is required for MUSDB test evaluation.") from e

    root = Path(cfg.musdb_root).expanduser()
    db = musdb.DB(root=str(root), subsets="test", is_wav=looks_like_wav_layout(root, subset="test"))
    tracks = split_test_tracks(list(db.tracks), partition="test")
    if not tracks:
        raise RuntimeError(f"No MUSDB test tracks found at '{root}'.")
    if args.max_tracks > 0:
        tracks = tracks[: args.max_tracks]

    save_dir = Path(args.save_audio_dir).expanduser() if args.save_audio_dir else None
    if args.save_originals and save_dir is None:
        raise RuntimeError("--save-originals in MUSDB mode requires --save-audio-dir.")

    per_track = [
        _eval_musdb_track(
            model=model,
            stft_params=stft_params,
            cfg=cfg,
            tr=tr,
            device=device,
            save_dir=save_dir,
            save_originals=args.save_originals,
        )
        for tr in tracks
    ]

    sisdr_vals = np.asarray([float(r["song_median_siSDR"]) for r in per_track], dtype=np.float64)
    mix_vals = np.asarray([float(r["song_median_siSDR_mixture_baseline"]) for r in per_track], dtype=np.float64)
    delta_vals = np.asarray([float(r["song_median_siSDR_delta_vs_mixture"]) for r in per_track], dtype=np.float64)
    ratio_vals = np.asarray([float(r["estimate_to_reference_rms_ratio"]) for r in per_track], dtype=np.float64)
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
