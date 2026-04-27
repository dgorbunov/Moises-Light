from __future__ import annotations

import torch


def overlap_add(chunks: torch.Tensor, hop: int) -> torch.Tensor:
    """
    Reconstruct audio from overlapping chunks using simple sum+divide weighting.

    - chunks: (N, C, T)
    - hop: hop size in samples
    returns: (C, out_T)
    """
    if chunks.dim() != 3:
        raise ValueError("chunks must be (N, C, T)")
    n, c, t = chunks.shape
    out_len = hop * (n - 1) + t
    out = torch.zeros((c, out_len), device=chunks.device, dtype=chunks.dtype)
    w = torch.zeros((1, out_len), device=chunks.device, dtype=chunks.dtype)
    for i in range(n):
        start = i * hop
        out[:, start : start + t] += chunks[i]
        w[:, start : start + t] += 1.0
    out = out / w.clamp_min(1.0)
    return out

