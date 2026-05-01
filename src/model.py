from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Rotary positional embedding

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


# Group-conv helper

def _split_conv(
    in_ch: int,
    out_ch: int,
    nband: int,
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
) -> nn.Conv2d:
    """Group conv with `nband` groups over subband channels."""
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


# Core building blocks

class SplitConvBlock(nn.Module):
    """GroupNorm -> group conv -> GELU."""

    def __init__(self, channels: int, nband: int, K: int = 3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(nband, channels)
        self.conv = _split_conv(channels, channels, nband, kernel_size=(K, K), padding=(K // 2, K // 2))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, C, F_b, T)
        return self.act(self.conv(self.norm(x)))


class SharedBandBlock(nn.Module):
    """SplitConvBlock variant with shared high-frequency weights."""

    def __init__(self, channels: int, nband: int, share_from_band: int = 2, K: int = 3) -> None:
        super().__init__()
        ch_pb = channels // nband
        self.nband = nband
        self.ch_pb = ch_pb
        self.share_from = share_from_band
        # Independent blocks for low-frequency bands.
        self.low_blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(1, ch_pb),
                nn.Conv2d(ch_pb, ch_pb, K, padding=K // 2),
                nn.GELU(),
            )
            for _ in range(share_from_band)
        ])
        # Shared block for high-frequency bands.
        self.high_block = nn.Sequential(
            nn.GroupNorm(1, ch_pb),
            nn.Conv2d(ch_pb, ch_pb, K, padding=K // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, C, Fb, T)
        B, C, Fb, T = x.shape
        bands = x.reshape(B, self.nband, self.ch_pb, Fb, T)
        out = []
        for i in range(self.nband):
            b_i = bands[:, i] # (B, ch_pb, Fb, T)
            if i < self.share_from:
                out.append(self.low_blocks[i](b_i))
            else:
                out.append(self.high_block(b_i))
        return torch.stack(out, dim=1).reshape(B, C, Fb, T)


class TDFBlock(nn.Module):
    """Time-distributed FC over frequency with conv skip branch."""

    def __init__(self, channels: int, nband: int, freq_per_band: int, bf: int = 8) -> None:
        super().__init__()
        f_hid = max(1, freq_per_band // bf)
        self.norm = nn.GroupNorm(nband, channels)
        self.fc1 = nn.Linear(freq_per_band, f_hid, bias=True)
        self.fc2 = nn.Linear(f_hid, freq_per_band, bias=True)
        self.skip_conv = _split_conv(channels, channels, nband, kernel_size=(3, 3), padding=(1, 1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, C, F_b, T)
        h = self.norm(x)
        B, C, F, T = h.shape
        # FC path over frequency dimension.
        h_fc = h.permute(0, 1, 3, 2).reshape(B * C * T, F) # (B·C·T, F_b)
        h_fc = self.act(self.fc1(h_fc))
        h_fc = self.fc2(h_fc).reshape(B, C, T, F).permute(0, 1, 3, 2) # (B, C, F_b, T)
        # 3x3 conv skip path.
        h_skip = self.skip_conv(h)
        return h_fc + h_skip


class SplitMergeBlock(nn.Module):
    """[conv blocks] -> TDF -> [conv blocks] with outer residual."""

    def __init__(
        self,
        channels: int,
        nband: int,
        freq_per_band: int,
        nsplit: int = 3,
        bf: int = 8,
        use_weight_sharing: bool = False,
    ) -> None:
        super().__init__()

        def _make_conv_block() -> nn.Module:
            if use_weight_sharing:
                return SharedBandBlock(channels, nband)
            return SplitConvBlock(channels, nband)

        self.left = nn.Sequential(*[_make_conv_block() for _ in range(nsplit)])
        self.tdf = TDFBlock(channels, nband, freq_per_band, bf)
        self.right = nn.Sequential(*[_make_conv_block() for _ in range(nsplit)])

    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, C, F_b, T), same shape
        h = self.left(x)
        h = self.tdf(h)
        h = self.right(h)
        return x + h # outer residual


class DownBlock(nn.Module):
    """Encoder downsampling over time only."""

    def __init__(self, in_ch: int, out_ch: int, nband: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(nband, in_ch)
        # Keep frequency size, downsample time.
        self.conv = _split_conv(in_ch, out_ch, nband, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class UpBlock(nn.Module):
    """Decoder upsampling over time with optional target length."""

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


# Dual-path RoPE block

class DualPathRoPEBlock(nn.Module):
    """Dual-path RoPE attention over time then bands."""

    def __init__(self, latent_dim: int, nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        ff_dim = latent_dim * 4
        self.time_layer = RoPETransformerEncoderLayer(latent_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.band_layer = RoPETransformerEncoderLayer(latent_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor: # (B, bands, T, d)
        b, bands, t, d = z.shape
        # Time attention per band.
        z_time = self.time_layer(z.reshape(b * bands, t, d))
        z = z_time.reshape(b, bands, t, d)
        # Band attention per time step.
        z_band = self.band_layer(z.permute(0, 2, 1, 3).reshape(b * t, bands, d))
        return z_band.reshape(b, t, bands, d).permute(0, 2, 1, 3).contiguous()


from mamba_ssm import Mamba2


# Dual-path Mamba block

class DualPathMambaBlock(nn.Module):
    """Dual-path Mamba2 block over time then bands."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(latent_dim)
        self.band_norm = nn.LayerNorm(latent_dim)
        # Use chunk_size compatible with both sequence lengths.
        self.time_mamba = Mamba2(d_model=latent_dim, d_state=64, d_conv=4, expand=2, chunk_size=2)
        self.band_mamba = Mamba2(d_model=latent_dim, d_state=64, d_conv=4, expand=2, chunk_size=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor: # (B, bands, T, d)
        b, bands, t, d = z.shape
        z_t = self.time_mamba(self.time_norm(z.reshape(b * bands, t, d)))
        z = z + z_t.reshape(b, bands, t, d)
        z_b = self.band_mamba(self.band_norm(z.permute(0, 2, 1, 3).reshape(b * t, bands, d)))
        return z + z_b.reshape(b, t, bands, d).permute(0, 2, 1, 3).contiguous()


# Bottleneck projection between spatial and sequence views

class BottleneckRoPE(nn.Module):
    """Project bottleneck features to sequence blocks and back."""

    def __init__(
        self,
        channels: int,
        nband: int,
        latent_dim: int,
        nrope: int,
        use_mamba: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim % 8 != 0:
            raise ValueError("latent_dim must be divisible by 8 for 8-head attention")
        ch_pb = channels // nband
        self.nband = nband
        self.ch_pb = ch_pb

        self.enc_norm = nn.LayerNorm(ch_pb)
        self.enc_proj = nn.Linear(ch_pb, latent_dim)

        if use_mamba:
            self.blocks: nn.ModuleList = nn.ModuleList(
                [DualPathMambaBlock(latent_dim=latent_dim) for _ in range(nrope)]
            )
        else:
            self.blocks = nn.ModuleList(
                [DualPathRoPEBlock(latent_dim=latent_dim) for _ in range(nrope)]
            )

        self.out_norm = nn.LayerNorm(latent_dim)
        self.dec_proj = nn.Linear(latent_dim, ch_pb)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, C, F_b, T_b)
        B, C, F, T = x.shape
        ch_pb = self.ch_pb

        # Split into per-band views.
        xb = x.view(B, self.nband, ch_pb, F, T)

        # Pool over frequency.
        z = xb.mean(dim=3).permute(0, 1, 3, 2)

        # Encode to latent.
        z = self.enc_proj(self.enc_norm(z)) # (B, nband, T, latent_dim)

        # Apply dual-path blocks.
        for block in self.blocks:
            z = z + block(z)
        z = self.out_norm(z)

        # Decode to channel space.
        residual = self.dec_proj(z).permute(0, 1, 3, 2) # (B, nband, ch_pb, T)
        residual = residual.reshape(B, C, T)

        # Broadcast residual over frequency and add back.
        return x + residual.unsqueeze(2) # (B, C, 1, T) to (B, C, F_b, T_b)


# Full Moises-Light model

class MoisesLight(nn.Module):
    """Band-split U-Net for music source separation."""

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
        use_weight_sharing: bool = False,
        use_mamba: bool = False,
    ) -> None:
        super().__init__()

        if g % nband != 0:
            raise ValueError(f"g={g} must be divisible by nband={nband}")
        if latent_dim % 8 != 0:
            raise ValueError("latent_dim must be divisible by 8 for 8-head attention")
        if freq_bins % nband != 0:
            raise ValueError(f"freq_bins={freq_bins} must be divisible by nband={nband}")

        spec_ch = 2 * audio_channels # 4 for stereo
        freq_per_band = freq_bins // nband # e.g. 768 for freq_bins=3072, nband=4
        stem_ch = spec_ch * nband # 16 (channels after band split)

        # Channel sizes at each depth.
        dims: list[int] = [g * (i + 1) for i in range(depth + 1)]

        self.nband = nband
        self.spec_ch = spec_ch
        self.freq_per_band = freq_per_band
        self.freq_bins = freq_bins
        self.depth = depth

        # Stem encoder.
        self.stem_enc = nn.Sequential(
            nn.GroupNorm(nband, stem_ch),
            _split_conv(stem_ch, dims[0], nband, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
        )

        # Encoder.
        self.enc_blocks = nn.ModuleList([
            SplitMergeBlock(dims[i], nband, freq_per_band, nsplit_enc, bf, use_weight_sharing)
            for i in range(depth)
        ])
        self.down_blocks = nn.ModuleList([
            DownBlock(dims[i], dims[i + 1], nband)
            for i in range(depth)
        ])

        # Bottleneck.
        self.bottleneck_sm = SplitMergeBlock(dims[-1], nband, freq_per_band, nsplit_enc, bf, use_weight_sharing)
        self.bottleneck_rope = BottleneckRoPE(dims[-1], nband, latent_dim, nrope, use_mamba)

        # Decoder.
        # Decoder level i mirrors encoder level (depth-1-i).
        self.up_blocks = nn.ModuleList([
            UpBlock(dims[depth - i], dims[depth - i - 1], nband)
            for i in range(depth)
        ])
        self.dec_blocks = nn.ModuleList([
            SplitMergeBlock(dims[depth - i - 1], nband, freq_per_band, nsplit_dec, bf, use_weight_sharing)
            for i in range(depth)
        ])

        # Stem decoder.
        self.stem_dec = nn.Sequential(
            nn.GroupNorm(nband, dims[0]),
            _split_conv(dims[0], stem_ch, nband, kernel_size=(3, 3), padding=(1, 1)),
        )

    # Band split / merge.

    def _band_split(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, C, F, T) to band-grouped channels."""
        B, C, F, T = x.shape
        Fb = F // self.nband
        x = x.view(B, C, self.nband, Fb, T) # (B, C, nband, Fb, T)
        x = x.permute(0, 2, 1, 3, 4) # (B, nband, C, Fb, T)
        return x.reshape(B, self.nband * C, Fb, T) # (B, C·nband, Fb, T)

    def _band_merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, spec_ch·nband, F/nband, T) to (B, spec_ch, F, T)."""
        B, Cnb, Fb, T = x.shape
        C = Cnb // self.nband
        x = x.reshape(B, self.nband, C, Fb, T) # (B, nband, C, Fb, T)
        x = x.permute(0, 2, 1, 3, 4) # (B, C, nband, Fb, T)
        return x.reshape(B, C, self.nband * Fb, T) # (B, spec_ch, F, T)

    # Forward pass.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on interleaved real/imag spectrograms."""
        if x.shape[1] != self.spec_ch:
            raise ValueError(f"Expected {self.spec_ch} input channels, got {x.shape[1]}")
        if x.shape[2] != self.freq_bins:
            raise ValueError(f"Expected freq_bins={self.freq_bins}, got {x.shape[2]}")

        # Band split.
        h = self._band_split(x) # (B, spec_ch·nband, Fb, T)

        # Stem.
        h = self.stem_enc(h) # (B, G, Fb, T)

        # Encoder and skip capture.
        skips: list[torch.Tensor] = []
        for enc_block, down_block in zip(self.enc_blocks, self.down_blocks):
            h = enc_block(h)
            skips.append(h) # save at dims[i], full time resolution
            h = down_block(h) # halve time, expand channels

        # Bottleneck.
        h = self.bottleneck_sm(h)
        h = self.bottleneck_rope(h)

        # Decoder with element-wise skip fusion.
        for up_block, dec_block, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            h = up_block(h, target_t=skip.shape[-1]) # restore exact time length
            h = h * skip # element-wise skip connection
            h = dec_block(h)

        # Stem output.
        h = self.stem_dec(h) # (B, spec_ch·nband, Fb, T)

        # Band merge.
        return self._band_merge(h) # (B, spec_ch, freq_bins, T)
