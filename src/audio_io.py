from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 44100) -> torch.Tensor:
    """Strict loader for files already matching training format (e.g. MUSDB18-HQ WAV)."""
    audio, sr = sf.read(str(path), always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"Expected sample_rate={sample_rate}, got {sr} for {path}")
    return torch.from_numpy(audio).float().transpose(0, 1).contiguous()


def decode_audio_file(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode arbitrary audio to ``(wav, sr)`` with ``wav`` shape ``(C, T)`` float32."""
    path = Path(path)
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        audio, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(audio.T).float()
    return wav.contiguous(), int(sr)


def prepare_waveform_tensor(wav: torch.Tensor, source_sr: int, target_sr: int) -> tuple[torch.Tensor, bool]:
    """Bring decoded PCM toward training tensors: stereo ``(C, T)``, ``target_sr``, samples in ``[-1, 1]``.

    Steps (when needed): mono→duplicate stereo or take first two channels, resample, scale overs,
    clamp.

    **MUSDB18-HQ-style data** (already stereo at ``target_sr`` with peaks ≤ ~1) hits a cheap path:
    only dtype alignment and scalar peak checks — no resampling or clamp allocations.

    Returns ``(tensor, scaled_down_overs)`` where the latter is True if samples above ±1 were
    attenuated before clamping.
    """
    if wav.dim() != 2:
        raise ValueError(f"Expected waveform (C, T), got shape {tuple(wav.shape)}")
    wav = wav.float()
    c = wav.shape[0]
    if source_sr == target_sr and c == 2:
        peak = wav.abs().max()
        if peak <= 1.0 + 1e-6:
            return wav, False

    if c == 1:
        wav = wav.repeat(2, 1)
    elif c >= 2:
        wav = wav[:2].contiguous()
    else:
        raise ValueError("Audio has zero channels.")

    if source_sr != target_sr:
        wav = torchaudio.functional.resample(wav, source_sr, target_sr)

    peak = float(wav.abs().max().clamp_min(1e-12))
    scaled = False
    if peak > 1.0:
        wav = wav * (0.999 / peak)
        scaled = True
    return wav.clamp(-1.0, 1.0), scaled


def load_mixture_and_optional_reference_for_test(
    mixture_path: Path,
    reference_path: Path | None,
    *,
    target_sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Decode mixture (+ optional reference), apply :func:`prepare_waveform_tensor`, trim to common length."""

    def _prep_one(p: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        raw, sr = decode_audio_file(p)
        peak_in = float(raw.abs().max())
        out, scaled = prepare_waveform_tensor(raw, sr, target_sample_rate)
        meta: dict[str, Any] = {
            "path": str(p),
            "source_sample_rate": sr,
            "source_channels": int(raw.shape[0]),
            "target_sample_rate": target_sample_rate,
            "resampled": sr != target_sample_rate,
            "peak_abs_before_adapt": peak_in,
            "peak_abs_after_adapt": float(out.abs().max()),
            "scaled_down_overs": scaled,
        }
        return out, meta

    summary: dict[str, Any] = {}
    mix, mix_meta = _prep_one(mixture_path)
    summary["mixture"] = mix_meta

    ref: torch.Tensor | None = None
    if reference_path is not None:
        ref, ref_meta = _prep_one(reference_path)
        summary["reference"] = ref_meta

    if ref is not None and mix.shape[-1] != ref.shape[-1]:
        n = min(mix.shape[-1], ref.shape[-1])
        summary["trimmed_to_samples"] = n
        mix = mix[..., :n]
        ref = ref[..., :n]

    return mix, ref, summary


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int = 44100) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = audio.detach().cpu()
    if x.dim() != 2:
        raise ValueError("audio must be (C, T)")
    sf.write(str(p), x.transpose(0, 1).numpy(), samplerate=sample_rate)
