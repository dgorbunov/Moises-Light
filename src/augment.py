from __future__ import annotations

import torch
from torch import nn

try:
    from torch_audiomentations import PitchShift, PolarityInversion, Shift
except Exception:
    PitchShift = None
    PolarityInversion = None
    Shift = None


class ChannelFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 2:
            return x
        if torch.rand(()) < self.p:
            return x[:, [1, 0], :]
        return x


class JointAugment(nn.Module):
    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.flip = ChannelFlip(p=0.5)
        self.has_ta = PitchShift is not None
        if self.has_ta:
            self.shift = Shift(min_shift=-0.25, max_shift=0.25, shift_unit="fraction", p=0.5, output_type="tensor")
            self.pitch = PitchShift(sample_rate=self.sample_rate, min_transpose_semitones=-2.0, max_transpose_semitones=2.0, p=0.3, output_type="tensor")
            self.pol = PolarityInversion(p=0.5, output_type="tensor")
        else:
            self.shift = None
            self.pitch = None
            self.pol = None

    def forward(self, mixture: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixture = self.flip(mixture)
        target = self.flip(target)
        x = torch.cat([mixture, target], dim=1)
        if self.has_ta:
            x = self.pol(x)
            x = self.shift(x)
            x = self.pitch(x)
        return x[:, :2, :], x[:, 2:, :]
