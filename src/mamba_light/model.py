from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        b, l, d = x.shape
        t = torch.arange(l, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, D)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)


class RoPEAttentionBlock(nn.Module):
    def __init__(self, dim: int, nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.rope = RoPE(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.norm1(x)
        h = self.rope(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class DualPathRoPE(nn.Module):
    """
    Dual-path transformer over time then frequency.
    Input: (B, C, F, T)
    """

    def __init__(self, channels: int, n_blocks: int = 5, nhead: int = 8) -> None:
        super().__init__()
        self.time_blocks = nn.ModuleList([RoPEAttentionBlock(channels, nhead=nhead) for _ in range(n_blocks)])
        self.freq_blocks = nn.ModuleList([RoPEAttentionBlock(channels, nhead=nhead) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        # time path: treat each frequency bin separately
        xt = x.permute(0, 2, 3, 1).reshape(b * f, t, c)  # (B*F, T, C)
        for blk in self.time_blocks:
            xt = blk(xt)
        x = xt.reshape(b, f, t, c).permute(0, 3, 1, 2).contiguous()

        # freq path: treat each time frame separately
        xf = x.permute(0, 3, 2, 1).reshape(b * t, f, c)  # (B*T, F, C)
        for blk in self.freq_blocks:
            xf = blk(xf)
        x = xf.reshape(b, t, f, c).permute(0, 3, 2, 1).contiguous()
        return x


class SplitConv2d(nn.Module):
    """
    Efficient band-splitting group convolution over (F,T) maps.
    Expects input channels already contain concatenated subbands along channel dim.
    """

    def __init__(self, in_ch: int, out_ch: int, nband: int, k: int = 3) -> None:
        super().__init__()
        if in_ch % nband != 0 or out_ch % nband != 0:
            raise ValueError("in_ch and out_ch must be divisible by nband")
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, groups=nband)
        self.norm = nn.GroupNorm(num_groups=nband, num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Down(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.conv(x)


class MoisesLight(nn.Module):
    """
    Practical reproduction-friendly approximation of Moises-Light:
    - complex spectrogram in (2*C, F, T)
    - split into Nband equal subbands, concat along channel
    - group conv "split modules" throughout encoder/decoder
    - dual-path RoPE bottleneck
    - outputs complex spectrogram estimate (2*C, F, T)
    """

    def __init__(
        self,
        audio_channels: int = 2,  # stereo
        nband: int = 4,
        g: int = 56,
        nrope: int = 5,
        nsplit_enc: int = 3,
        nsplit_dec: int = 1,
        depth: int = 3,
        k_first_last: int = 3,
    ) -> None:
        super().__init__()
        self.audio_channels = audio_channels
        self.nband = nband
        self.depth = depth

        in_ch = 2 * audio_channels * nband  # (real+imag)*stereo * bands
        self.in_proj = SplitConv2d(in_ch, g * nband, nband=nband, k=k_first_last)

        enc: list[nn.Module] = []
        downs: list[nn.Module] = []
        ch = g * nband
        for _ in range(depth):
            blocks = [SplitConv2d(ch, ch, nband=nband, k=3) for _ in range(nsplit_enc)]
            enc.append(nn.Sequential(*blocks))
            downs.append(Down(ch))
        self.encoder = nn.ModuleList(enc)
        self.downs = nn.ModuleList(downs)

        self.bottleneck = nn.Sequential(
            SplitConv2d(ch, ch, nband=nband, k=3),
            DualPathRoPE(channels=ch, n_blocks=nrope, nhead=8),
            SplitConv2d(ch, ch, nband=nband, k=3),
        )

        ups: list[nn.Module] = []
        dec: list[nn.Module] = []
        for _ in range(depth):
            ups.append(Up(ch))
            blocks = [SplitConv2d(ch, ch, nband=nband, k=3) for _ in range(nsplit_dec)]
            dec.append(nn.Sequential(*blocks))
        self.ups = nn.ModuleList(ups)
        self.decoder = nn.ModuleList(dec)

        self.out_proj = SplitConv2d(ch, 2 * audio_channels * nband, nband=nband, k=k_first_last)

    def _split_bands(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*C, F, T) -> (B, 2*C*nband, F/nband, T)
        b, cc2, f, t = x.shape
        if f % self.nband != 0:
            raise ValueError(f"freq bins {f} must be divisible by nband={self.nband}")
        fb = f // self.nband
        xs = x.reshape(b, cc2, self.nband, fb, t)
        xs = xs.permute(0, 2, 1, 3, 4).contiguous().reshape(b, self.nband * cc2, fb, t)
        return xs

    def _merge_bands(self, x: torch.Tensor, f_full: int) -> torch.Tensor:
        # x: (B, 2*C*nband, F/nband, T) -> (B, 2*C, F, T)
        b, ch, fb, t = x.shape
        cc2 = ch // self.nband
        xm = x.reshape(b, self.nband, cc2, fb, t).permute(0, 2, 1, 3, 4).contiguous()
        xm = xm.reshape(b, cc2, f_full, t)
        return xm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, 2*C, F, T)
        b, cc2, f, t = X.shape
        x = self._split_bands(X)  # (B, 2*C*nband, F/nband, T)
        x = self.in_proj(x)

        skips: list[tuple[torch.Tensor, tuple[int, int]]] = []
        for enc, down in zip(self.encoder, self.downs):
            x = enc(x)
            skips.append((x, (x.shape[-2], x.shape[-1])))
            x = down(x)

        x = self.bottleneck(x)

        for up, dec in zip(self.ups, self.decoder):
            skip, hw = skips.pop()
            x = up(x, target_hw=hw)
            x = x + skip
            x = dec(x)

        x = self.out_proj(x)
        y = self._merge_bands(x, f_full=f)
        return y

