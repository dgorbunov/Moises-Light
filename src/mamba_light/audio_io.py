from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


def load_audio(path: str | Path, sample_rate: int = 44100) -> torch.Tensor:
    audio, sr = sf.read(str(path), always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"Expected sample_rate={sample_rate}, got {sr} for {path}")
    # soundfile returns (T, C)
    x = torch.from_numpy(audio).float().transpose(0, 1).contiguous()  # (C, T)
    return x


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int = 44100) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = audio.detach().cpu()
    if x.dim() != 2:
        raise ValueError("audio must be (C, T)")
    sf.write(str(p), x.transpose(0, 1).numpy(), samplerate=sample_rate)

