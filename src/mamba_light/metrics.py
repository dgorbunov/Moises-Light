from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mir_eval.separation import bss_eval_sources


def _sdr_1src(ref: np.ndarray, est: np.ndarray) -> float:
    # ref/est: (T,)
    sdr, _, _, _ = bss_eval_sources(ref[None, :], est[None, :])
    return float(sdr[0])


def stereo_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    """
    ref/est: (2, T)
    Returns average SDR over channels.
    """
    r = ref.detach().cpu().numpy()
    e = est.detach().cpu().numpy()
    if r.shape[0] != 2 or e.shape[0] != 2:
        raise ValueError("Expected stereo (2, T)")
    return 0.5 * (_sdr_1src(r[0], e[0]) + _sdr_1src(r[1], e[1]))


@dataclass(frozen=True)
class CSDRResult:
    song_median_sdr: float
    n_chunks: int


def chunk_level_sdr(ref: torch.Tensor, est: torch.Tensor, sample_rate: int, chunk_seconds: float = 1.0) -> CSDRResult:
    """
    Implements paper's cSDR description:
    - compute SDR over 1s chunks
    - take median across chunks for the song
    """
    if ref.shape != est.shape:
        raise ValueError("ref and est must have same shape")
    chunk_len = int(round(chunk_seconds * sample_rate))
    t = ref.shape[-1]
    if t < chunk_len:
        return CSDRResult(song_median_sdr=stereo_sdr(ref, est), n_chunks=1)

    vals: list[float] = []
    for start in range(0, t - chunk_len + 1, chunk_len):
        r = ref[:, start : start + chunk_len]
        e = est[:, start : start + chunk_len]
        vals.append(stereo_sdr(r, e))
    return CSDRResult(song_median_sdr=float(np.median(vals)), n_chunks=len(vals))

