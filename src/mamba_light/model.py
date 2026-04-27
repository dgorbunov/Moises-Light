from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        _, seq_len, dim = x.shape
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, D)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape(-1, seq_len, dim)
        return (x * cos) + (x_rot * sin)


class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer with rotary embeddings applied immediately
    before self-attention.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.rope = RoPE(d_model)

    # Keep signature compatible across PyTorch versions.
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x_rope = self.rope(x)
        x = self.self_attn(
            x_rope,
            x_rope,
            x_rope,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)


class BandSplitEncoder(nn.Module):
    def __init__(self, in_channels: int, nband: int, latent_dim: int) -> None:
        super().__init__()
        self.nband = nband
        self.latent_dim = latent_dim
        self.band_proj = nn.Sequential(
            nn.Conv1d(in_channels, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: (B, C_in, F, T) -> z: (B, Bands, T, D)
        b, c, f, t = x.shape
        if f % self.nband != 0:
            raise ValueError(f"Frequency bins {f} must be divisible by nband={self.nband}")
        fb = f // self.nband
        xb = x.view(b, c, self.nband, fb, t)
        xb = xb.permute(0, 2, 4, 1, 3).contiguous().view(b * self.nband * t, c, fb)
        h = self.band_proj(xb)
        h = h.mean(dim=-1)  # pooled over frequency inside each band
        z = h.view(b, self.nband, t, self.latent_dim)
        return z, fb


class BandMergeDecoder(nn.Module):
    def __init__(self, out_channels: int, nband: int, latent_dim: int, freq_bins: int) -> None:
        super().__init__()
        if freq_bins % nband != 0:
            raise ValueError(f"freq_bins={freq_bins} must be divisible by nband={nband}")
        self.out_channels = out_channels
        self.nband = nband
        self.freq_bins = freq_bins
        self.f_per_band = freq_bins // nband
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_channels * self.f_per_band),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, Bands, T, D) -> (B, out_channels, F, T)
        b, bands, t, _ = z.shape
        if bands != self.nband:
            raise ValueError(f"Expected {self.nband} bands, got {bands}")
        y = self.decoder(z).view(b, self.nband, t, self.out_channels, self.f_per_band)
        y = y.permute(0, 3, 1, 4, 2).contiguous()
        return y.view(b, self.out_channels, self.freq_bins, t)


class DualPathRoPEBlock(nn.Module):
    def __init__(self, latent_dim: int, nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        ff_dim = latent_dim * 4
        self.time_layer = RoPETransformerEncoderLayer(latent_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.band_layer = RoPETransformerEncoderLayer(latent_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, Bands, T, D)
        b, bands, t, d = z.shape
        z_time = z.view(b * bands, t, d)
        z_time = self.time_layer(z_time)
        z = z_time.view(b, bands, t, d)

        z_band = z.permute(0, 2, 1, 3).contiguous().view(b * t, bands, d)
        z_band = self.band_layer(z_band)
        z = z_band.view(b, t, bands, d).permute(0, 2, 1, 3).contiguous()
        return z


class MoisesLight(nn.Module):
    """
    Moises-Light base variant:
    - complex STFT input: (B, 2*C, F, T)
    - 4-band split encoder to latent: (B, Bands, T, D)
    - 5 dual-path RoPE Transformer blocks (time/band alternating)
    - symmetric band-merge decoder back to complex STFT
    """

    def __init__(
        self,
        audio_channels: int = 2,
        nband: int = 4,
        g: int = 56,  # kept for compatibility with existing checkpoints/configs
        nrope: int = 5,
        nsplit_enc: int = 3,  # kept for compatibility
        nsplit_dec: int = 1,  # kept for compatibility
        depth: int = 3,  # kept for compatibility
        latent_dim: int = 128,
        freq_bins: int = 2048,
    ) -> None:
        super().__init__()
        del g, nsplit_enc, nsplit_dec, depth
        if latent_dim % 8 != 0:
            raise ValueError("latent_dim must be divisible by 8 for nhead=8")
        self.audio_channels = audio_channels
        self.nband = nband
        self.latent_dim = latent_dim
        self.freq_bins = freq_bins
        self.spec_channels = 2 * audio_channels

        self.encoder = BandSplitEncoder(in_channels=self.spec_channels, nband=nband, latent_dim=latent_dim)
        self.bottleneck = nn.ModuleList([DualPathRoPEBlock(latent_dim=latent_dim, nhead=8) for _ in range(nrope)])
        self.decoder = BandMergeDecoder(
            out_channels=self.spec_channels,
            nband=nband,
            latent_dim=latent_dim,
            freq_bins=freq_bins,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*C, F, T)
        _, c, f, _ = x.shape
        if c != self.spec_channels:
            raise ValueError(f"Expected {self.spec_channels} input channels, got {c}")
        if f != self.freq_bins:
            raise ValueError(f"Expected freq_bins={self.freq_bins}, got {f}")

        z, _ = self.encoder(x)
        for block in self.bottleneck:
            z = z + block(z)
        return self.decoder(z)

