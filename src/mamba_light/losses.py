from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mamba_light.stft import STFTParams, istft, stft


@dataclass(frozen=True)
class MultiResParams:
    fft_sizes: tuple[int, ...] = (1024, 2048, 4096)
    hop_sizes: tuple[int, ...] = (256, 512, 1024)
    win_lengths: tuple[int, ...] = (1024, 2048, 4096)


def _stft_complex_mae(wav: torch.Tensor, wav_ref: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    # wav: (B, C, T)
    # Use float32 for STFT (AMP-friendly; avoids cuFFT half constraints on some n_fft sizes).
    window = torch.hann_window(win, device=wav.device, dtype=torch.float32)
    b, c, _ = wav.shape
    X = torch.stft(
        wav.float().reshape(b * c, -1),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    Y = torch.stft(
        wav_ref.float().reshape(b * c, -1),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    Xr = torch.view_as_real(X)  # (B*C, F, TT, 2)
    Yr = torch.view_as_real(Y)
    return torch.mean(torch.abs(Xr - Yr))


class MoisesLoss(nn.Module):
    def __init__(self, stft_params: STFTParams, multires: MultiResParams | None = None) -> None:
        super().__init__()
        self.stft_params = stft_params
        self.multires = multires

    def forward(
        self,
        pred_spec: torch.Tensor,
        tgt_spec: torch.Tensor,
        tgt_wav: torch.Tensor,
    ) -> torch.Tensor:
        # primary L1 in complex-spectrogram space
        loss = torch.mean(torch.abs(pred_spec - tgt_spec))
        if self.multires is None:
            return loss

        # reconstruct waveform from predicted spectrogram using the main STFT params,
        # then compute multi-resolution complex MAE vs target waveform.
        pred_wav = istft(pred_spec, self.stft_params, length=tgt_wav.shape[-1])
        mr = self.multires
        mr_loss = 0.0
        for n_fft, hop, win in zip(mr.fft_sizes, mr.hop_sizes, mr.win_lengths):
            mr_loss = mr_loss + _stft_complex_mae(pred_wav, tgt_wav, n_fft=n_fft, hop=hop, win=win)
        mr_loss = mr_loss / float(len(mr.fft_sizes))
        return loss + mr_loss


def target_spectrogram(wav: torch.Tensor, stft_params: STFTParams) -> torch.Tensor:
    return stft(wav, stft_params)

