from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class STFTParams:
    n_fft: int = 6144
    hop_length: int = 1024
    win_length: int = 6144
    freq_bins: int = 2048


def stft(x: torch.Tensor, p: STFTParams) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError("x must be (B, C, T)")
    out_dtype = x.dtype
    x32 = x.float()
    b, c, _ = x32.shape
    window = torch.hann_window(p.win_length, device=x.device, dtype=torch.float32)
    xs = x32.reshape(b * c, -1)
    X = torch.stft(
        xs,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        win_length=p.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    X = X[:, : p.freq_bins, :]
    Xr = torch.view_as_real(X)
    Xr = Xr.permute(0, 3, 1, 2).contiguous()
    Xr = Xr.reshape(b, c * 2, p.freq_bins, Xr.shape[-1])
    return Xr.to(out_dtype)


def istft(X: torch.Tensor, p: STFTParams, length: int) -> torch.Tensor:
    if X.dim() != 4:
        raise ValueError("X must be (B, 2*C, F, TT)")
    out_dtype = X.dtype
    Xf = X.float()
    b, cc2, f, tt = Xf.shape
    if cc2 % 2 != 0:
        raise ValueError("channel dim must be even (real+imag)")
    c = cc2 // 2
    f_full = p.n_fft // 2 + 1
    if f > f_full:
        raise ValueError("freq_bins exceeds full stft bins")

    Xc = Xf.reshape(b * c, 2, f, tt).permute(0, 2, 3, 1).contiguous()
    Xc = torch.view_as_complex(Xc)
    if f < f_full:
        pad = torch.zeros((b * c, f_full - f, tt), device=X.device, dtype=torch.complex64)
        Xc = torch.cat([Xc, pad], dim=1)
    window = torch.hann_window(p.win_length, device=X.device, dtype=torch.float32)
    y = torch.istft(
        Xc,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        win_length=p.win_length,
        window=window,
        center=True,
        length=length,
    )
    return y.reshape(b, c, -1).to(out_dtype)
