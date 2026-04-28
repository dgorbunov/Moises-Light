from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Rotary positional embedding
# ──────────────────────────────────────────────────────────────────────────────

class RoPE(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, dim = x.shape
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape(-1, seq_len, dim)
        return (x * cos) + (x_rot * sin)


class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
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


# ──────────────────────────────────────────────────────────────────────────────
# Low-level group-conv helper
# ──────────────────────────────────────────────────────────────────────────────

def _split_conv(
    in_ch: int,
    out_ch: int,
    nband: int,
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
) -> nn.Conv2d:
    """
    Group conv2d with `nband` groups so each frequency subband's channels are
    processed independently.  in_ch and out_ch must be divisible by nband.
    """
    assert in_ch % nband == 0, f"in_ch={in_ch} must be divisible by nband={nband}"
    assert out_ch % nband == 0, f"out_ch={out_ch} must be divisible by nband={nband}"
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        groups=nband,
        bias=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SplitConvBlock(nn.Module):
    """
    One "Split Module" from the paper: GroupNorm → group conv K×K → GELU.
    No skip — the outer SplitMergeBlock provides the residual.
    """

    def __init__(self, channels: int, nband: int, K: int = 3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(nband, channels)
        self.conv = _split_conv(channels, channels, nband, kernel_size=(K, K), padding=(K // 2, K // 2))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F_b, T)
        return self.act(self.conv(self.norm(x)))


class TDFBlock(nn.Module):
    """
    Time-Distributed Fully-connected block (TDF).
    Applies shared FC layers over the per-band frequency axis with a 3×3 group-conv skip.
    Output = FC_path(h) + skip_conv(h)  (no outer residual; SplitMergeBlock handles that).
    """

    def __init__(self, channels: int, nband: int, freq_per_band: int, bf: int = 8) -> None:
        super().__init__()
        f_hid = max(1, freq_per_band // bf)
        self.norm = nn.GroupNorm(nband, channels)
        self.fc1 = nn.Linear(freq_per_band, f_hid, bias=True)
        self.fc2 = nn.Linear(f_hid, freq_per_band, bias=True)
        self.skip_conv = _split_conv(channels, channels, nband, kernel_size=(3, 3), padding=(1, 1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F_b, T)
        h = self.norm(x)
        B, C, F, T = h.shape
        # FC path: apply shared linear over frequency dimension
        h_fc = h.permute(0, 1, 3, 2).reshape(B * C * T, F)  # (B·C·T, F_b)
        h_fc = self.act(self.fc1(h_fc))
        h_fc = self.fc2(h_fc).reshape(B, C, T, F).permute(0, 1, 3, 2)  # (B, C, F_b, T)
        # 3×3 conv skip path
        h_skip = self.skip_conv(h)
        return h_fc + h_skip


class SplitMergeBlock(nn.Module):
    """
    Split-and-Merge block (replaces TFC-TDF v3 in Moises-Light):
      [nsplit × SplitConvBlock] → TDFBlock → [nsplit × SplitConvBlock] + outer residual.

    nsplit=3 in the encoder, nsplit=1 in the decoder (asymmetric per paper Section 3.2).
    """

    def __init__(self, channels: int, nband: int, freq_per_band: int, nsplit: int = 3, bf: int = 8) -> None:
        super().__init__()
        self.left = nn.Sequential(*[SplitConvBlock(channels, nband) for _ in range(nsplit)])
        self.tdf = TDFBlock(channels, nband, freq_per_band, bf)
        self.right = nn.Sequential(*[SplitConvBlock(channels, nband) for _ in range(nsplit)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F_b, T) → same shape
        h = self.left(x)
        h = self.tdf(h)
        h = self.right(h)
        return x + h  # outer residual


class DownBlock(nn.Module):
    """
    Encoder downsampling: halves the time dimension and expands channels.
    Frequency dimension is NOT reduced (paper Section 3.1).
    """

    def __init__(self, in_ch: int, out_ch: int, nband: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(nband, in_ch)
        # kernel (1,3) → no freq stride; stride (1,2) → halve time
        self.conv = _split_conv(in_ch, out_ch, nband, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class UpBlock(nn.Module):
    """
    Decoder upsampling: doubles time via nearest interpolation, reduces channels.
    target_t lets the caller pass the exact skip-connection time length to avoid
    off-by-one mismatches from odd input lengths.
    """

    def __init__(self, in_ch: int, out_ch: int, nband: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(nband, in_ch)
        self.conv = _split_conv(in_ch, out_ch, nband, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor, target_t: int | None = None) -> torch.Tensor:
        h = self.norm(x)
        if target_t is not None:
            h = F.interpolate(h, size=(h.shape[-2], target_t), mode="nearest")
        else:
            h = F.interpolate(h, scale_factor=(1.0, 2.0), mode="nearest")
        return self.conv(h)


# ──────────────────────────────────────────────────────────────────────────────
# Dual-path RoPE transformer block (operates on latent band-time sequences)
# ──────────────────────────────────────────────────────────────────────────────

class DualPathRoPEBlock(nn.Module):
    """
    Dual-path RoPE transformer: time-axis attention then band-axis attention.
    Input/output shape: (B, nband, T, latent_dim).
    """

    def __init__(self, latent_dim: int, nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        ff_dim = latent_dim * 4
        self.time_layer = RoPETransformerEncoderLayer(latent_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.band_layer = RoPETransformerEncoderLayer(latent_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, bands, T, d)
        b, bands, t, d = z.shape
        # Time attention: process each band's time sequence independently
        z_time = self.time_layer(z.reshape(b * bands, t, d))
        z = z_time.reshape(b, bands, t, d)
        # Band attention: process each time step's band sequence independently
        z_band = self.band_layer(z.permute(0, 2, 1, 3).reshape(b * t, bands, d))
        return z_band.reshape(b, t, bands, d).permute(0, 2, 1, 3).contiguous()


# ──────────────────────────────────────────────────────────────────────────────
# Bottleneck: spatial features ↔ latent band-time sequence for RoPE blocks
# ──────────────────────────────────────────────────────────────────────────────

class BottleneckRoPE(nn.Module):
    """
    Bridges the spatial bottleneck feature map and the DualPath RoPE transformers.

    Projection path (no skip loss because the U-Net skip connections preserve freq):
      (B, C, F_b, T_b)
        → mean-pool over freq → (B, nband, T_b, ch_per_band)
        → LayerNorm + Linear → (B, nband, T_b, latent_dim)
        → N × DualPathRoPEBlock
        → Linear → (B, nband, T_b, ch_per_band)
        → reshape → (B, C, 1, T_b)
        → add as residual broadcast over freq dim
    """

    def __init__(self, channels: int, nband: int, latent_dim: int, nrope: int) -> None:
        super().__init__()
        if latent_dim % 8 != 0:
            raise ValueError("latent_dim must be divisible by 8 for 8-head attention")
        ch_pb = channels // nband
        self.nband = nband
        self.ch_pb = ch_pb

        self.enc_norm = nn.LayerNorm(ch_pb)
        self.enc_proj = nn.Linear(ch_pb, latent_dim)

        self.blocks = nn.ModuleList([DualPathRoPEBlock(latent_dim=latent_dim) for _ in range(nrope)])
        self.out_norm = nn.LayerNorm(latent_dim)

        self.dec_proj = nn.Linear(latent_dim, ch_pb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F_b, T_b)
        B, C, F, T = x.shape
        ch_pb = self.ch_pb

        # Split into per-band views: (B, nband, ch_pb, F, T)
        xb = x.view(B, self.nband, ch_pb, F, T)

        # Mean-pool over frequency → (B, nband, ch_pb, T) → (B, nband, T, ch_pb)
        z = xb.mean(dim=3).permute(0, 1, 3, 2)

        # Encode to latent
        z = self.enc_proj(self.enc_norm(z))  # (B, nband, T, latent_dim)

        # DualPath RoPE blocks with residual
        for block in self.blocks:
            z = z + block(z)
        z = self.out_norm(z)

        # Decode: (B, nband, T, ch_pb) → (B, C, T)
        residual = self.dec_proj(z).permute(0, 1, 3, 2)  # (B, nband, ch_pb, T)
        residual = residual.reshape(B, C, T)

        # Broadcast residual over freq and add back
        return x + residual.unsqueeze(2)  # (B, C, 1, T) → (B, C, F_b, T_b)


# ──────────────────────────────────────────────────────────────────────────────
# Full Moises-Light model
# ──────────────────────────────────────────────────────────────────────────────

class MoisesLight(nn.Module):
    """
    Moises-Light: lightweight band-split U-Net for music source separation.

    Architecture (paper Fig. 1 + Sections 3.1 & 3.2):
      1. Band split: reshape (B, spec_ch, F, T) → (B, spec_ch·nband, F/nband, T)
      2. Stem: K=3 split conv  spec_ch·nband → G channels
      3. Encoder (depth levels): SplitMergeBlock(nsplit_enc) + time-only DownBlock
         – skip connections stored before each downsample
      4. Bottleneck: SplitMergeBlock(nsplit_enc) + BottleneckRoPE (nrope blocks)
      5. Decoder (depth levels): UpBlock + element-wise multiply skip + SplitMergeBlock(nsplit_dec)
      6. Stem out: K=3 split conv  G → spec_ch·nband channels
      7. Band merge: reshape back to (B, spec_ch, F, T)

    Channel counts grow by G at each encoder level:  G, 2G, 3G, … (depth+1)·G
    Frequency axis is never downsampled (all downsampling is time-only).
    """

    def __init__(
        self,
        audio_channels: int = 2,
        nband: int = 4,
        g: int = 56,
        nrope: int = 5,
        nsplit_enc: int = 3,
        nsplit_dec: int = 1,
        depth: int = 3,
        latent_dim: int = 128,
        freq_bins: int = 2048,
        bf: int = 8,
    ) -> None:
        super().__init__()

        if g % nband != 0:
            raise ValueError(f"g={g} must be divisible by nband={nband}")
        if latent_dim % 8 != 0:
            raise ValueError("latent_dim must be divisible by 8 for 8-head attention")

        spec_ch = 2 * audio_channels        # 4 for stereo
        freq_per_band = freq_bins // nband  # 512
        stem_ch = spec_ch * nband           # 16  (channels after band split)

        # Channel sizes at each depth: [g, 2g, 3g, …, (depth+1)·g]
        dims: list[int] = [g * (i + 1) for i in range(depth + 1)]

        self.nband = nband
        self.spec_ch = spec_ch
        self.freq_per_band = freq_per_band
        self.freq_bins = freq_bins
        self.depth = depth

        # ── Stem encoder ──────────────────────────────────────────────────────
        self.stem_enc = nn.Sequential(
            nn.GroupNorm(nband, stem_ch),
            _split_conv(stem_ch, dims[0], nband, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList([
            SplitMergeBlock(dims[i], nband, freq_per_band, nsplit_enc, bf)
            for i in range(depth)
        ])
        self.down_blocks = nn.ModuleList([
            DownBlock(dims[i], dims[i + 1], nband)
            for i in range(depth)
        ])

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck_sm = SplitMergeBlock(dims[-1], nband, freq_per_band, nsplit_enc, bf)
        self.bottleneck_rope = BottleneckRoPE(dims[-1], nband, latent_dim, nrope)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Level i of the decoder mirrors encoder level (depth-1-i).
        self.up_blocks = nn.ModuleList([
            UpBlock(dims[depth - i], dims[depth - i - 1], nband)
            for i in range(depth)
        ])
        self.dec_blocks = nn.ModuleList([
            SplitMergeBlock(dims[depth - i - 1], nband, freq_per_band, nsplit_dec, bf)
            for i in range(depth)
        ])

        # ── Stem decoder ──────────────────────────────────────────────────────
        self.stem_dec = nn.Sequential(
            nn.GroupNorm(nband, dims[0]),
            _split_conv(dims[0], stem_ch, nband, kernel_size=(3, 3), padding=(1, 1)),
        )

    # ── Band split / merge ────────────────────────────────────────────────────

    def _band_split(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, spec_ch, F, T) → (B, spec_ch·nband, F/nband, T)

        Divides F into nband equal subbands and concatenates them along the channel
        axis so that group convolutions can process each subband independently:
          channels 0 … spec_ch-1        = band-0 (freq 0 … F/nband-1)
          channels spec_ch … 2·spec_ch-1 = band-1 (freq F/nband … 2F/nband-1)
          …
        """
        B, C, F, T = x.shape
        Fb = F // self.nband
        x = x.view(B, C, self.nband, Fb, T)        # (B, C, nband, Fb, T)
        x = x.permute(0, 2, 1, 3, 4)               # (B, nband, C, Fb, T)
        return x.reshape(B, self.nband * C, Fb, T)  # (B, C·nband, Fb, T)

    def _band_merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, spec_ch·nband, F/nband, T) → (B, spec_ch, F, T)"""
        B, Cnb, Fb, T = x.shape
        C = Cnb // self.nband
        x = x.reshape(B, self.nband, C, Fb, T)     # (B, nband, C, Fb, T)
        x = x.permute(0, 2, 1, 3, 4)               # (B, C, nband, Fb, T)
        return x.reshape(B, C, self.nband * Fb, T)  # (B, spec_ch, F, T)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, spec_ch, freq_bins, T)  —  complex spectrogram with real/imag interleaved
        returns same shape as x.
        """
        if x.shape[1] != self.spec_ch:
            raise ValueError(f"Expected {self.spec_ch} input channels, got {x.shape[1]}")
        if x.shape[2] != self.freq_bins:
            raise ValueError(f"Expected freq_bins={self.freq_bins}, got {x.shape[2]}")

        # 1. Band split
        h = self._band_split(x)         # (B, spec_ch·nband, Fb, T)

        # 2. Stem
        h = self.stem_enc(h)            # (B, G, Fb, T)

        # 3. Encoder — save skip BEFORE each downsample
        skips: list[torch.Tensor] = []
        for enc_block, down_block in zip(self.enc_blocks, self.down_blocks):
            h = enc_block(h)
            skips.append(h)             # save at dims[i], full time resolution
            h = down_block(h)           # halve time, expand channels

        # 4. Bottleneck
        h = self.bottleneck_sm(h)
        h = self.bottleneck_rope(h)

        # 5. Decoder — element-wise multiply with encoder skips (DTTNet-style)
        for up_block, dec_block, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            h = up_block(h, target_t=skip.shape[-1])  # restore exact time length
            h = h * skip                               # element-wise skip connection
            h = dec_block(h)

        # 6. Stem out
        h = self.stem_dec(h)            # (B, spec_ch·nband, Fb, T)

        # 7. Band merge
        return self._band_merge(h)      # (B, spec_ch, freq_bins, T)
