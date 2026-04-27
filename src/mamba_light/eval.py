from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mamba_light.audio_io import load_audio
from mamba_light.config import TrainConfig
from mamba_light.infer import load_model_from_ckpt, separate_track
from mamba_light.metrics import chunk_level_sdr
from mamba_light.musdb import discover_tracks
from mamba_light.stft import STFTParams


@torch.no_grad()
def evaluate(cfg: TrainConfig, ckpt_path: str | Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_ckpt(ckpt_path, device=device)

    stft_params = STFTParams(
        n_fft=cfg.stft.n_fft,
        hop_length=cfg.stft.hop_length,
        win_length=cfg.stft.win_length,
        freq_bins=cfg.stft.freq_bins,
    )

    tracks = discover_tracks(cfg.musdb_root, split="test")
    song_medians: list[float] = []

    for tp in tqdm(tracks, desc="eval(test)"):
        mix = load_audio(tp.mixture, sample_rate=cfg.sample_rate)  # (2, T)
        ref = load_audio(tp.stems[cfg.target_stem], sample_rate=cfg.sample_rate)

        est = separate_track(
            model=model,
            mixture=mix,
            stft_params=stft_params,
            segment_seconds=cfg.segment_seconds,
            overlap=cfg.eval_overlap,
            sample_rate=cfg.sample_rate,
            device=device,
        )

        res = chunk_level_sdr(ref, est, sample_rate=cfg.sample_rate, chunk_seconds=1.0)
        song_medians.append(res.song_median_sdr)

    median_over_songs = float(np.median(song_medians))
    return {
        "target_stem": str(cfg.target_stem),
        "cSDR_median_over_songs": median_over_songs,
        "n_songs": float(len(song_medians)),
    }

