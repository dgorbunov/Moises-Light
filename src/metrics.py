from __future__ import annotations

from dataclasses import dataclass

import torch


def _is_effectively_silent(wav: torch.Tensor, eps: float = 1e-8) -> bool:
    return bool(torch.mean(wav.float().pow(2)).item() <= eps)


def _si_sdr_1src(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Scale-invariant SDR for a single channel, computed in torch.
    """
    if _is_effectively_silent(ref) or _is_effectively_silent(est):
        return float("nan")
    ref_f = ref.float()
    est_f = est.float()
    ref_energy = torch.dot(ref_f, ref_f)
    if ref_energy.abs().item() <= eps:
        return float("nan")
    alpha = torch.dot(est_f, ref_f) / (ref_energy + eps)
    proj = alpha * ref_f
    noise = est_f - proj
    ratio = torch.sum(proj * proj) / (torch.sum(noise * noise) + eps)
    if ratio <= 0:
        return float("nan")
    return float(10.0 * torch.log10(ratio + eps).item())


def stereo_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    r = ref.detach()
    e = est.detach()
    if r.shape[0] != 2 or e.shape[0] != 2:
        raise ValueError("Expected stereo (2, T)")
    if _is_effectively_silent(r.reshape(-1)):
        return float("nan")
    s0 = _si_sdr_1src(r[0], e[0])
    s1 = _si_sdr_1src(r[1], e[1])
    if s0 != s0 and s1 != s1:
        return float("nan")
    if s0 != s0:
        return float(s1)
    if s1 != s1:
        return float(s0)
    return 0.5 * (s0 + s1)


@dataclass(frozen=True)
class CSDRResult:
    song_median_sdr: float
    n_chunks: int


def chunk_level_sdr(ref: torch.Tensor, est: torch.Tensor, sample_rate: int, chunk_seconds: float = 1.0) -> CSDRResult:
    if ref.shape != est.shape:
        raise ValueError("ref and est must have same shape")
    chunk_len = int(round(chunk_seconds * sample_rate))
    t = ref.shape[-1]
    if t < chunk_len:
        return CSDRResult(song_median_sdr=stereo_sdr(ref, est), n_chunks=1)

    vals: list[float] = []
    for start in range(0, t - chunk_len + 1, chunk_len):
        v = stereo_sdr(ref[:, start : start + chunk_len], est[:, start : start + chunk_len])
        if v == v:
            vals.append(v)
    if not vals:
        return CSDRResult(song_median_sdr=float("nan"), n_chunks=0)
    med = torch.tensor(vals, dtype=torch.float32).median().item()
    return CSDRResult(song_median_sdr=float(med), n_chunks=len(vals))
