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


def _load_waveform_any_format(path: Path) -> tuple[torch.Tensor, int]:
    """Return ``(wav, sr)`` with ``wav`` shape ``(C, T)`` float32."""
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        audio, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(audio.T).float()
    return wav.contiguous(), int(sr)


def _to_target_channels_stereo(wav: torch.Tensor, target_channels: int) -> torch.Tensor:
    c = wav.shape[0]
    if c == 0:
        raise ValueError("Audio has zero channels.")
    if target_channels != 2:
        raise ValueError("Only target_channels=2 is supported (MUSDB mixtures/stems are stereo).")
    if c == 1:
        return wav.repeat(2, 1)
    if c >= 2:
        return wav[:2].contiguous()
    raise RuntimeError(f"Unexpected channel count {c}")


def load_audio_adapted_for_inference(
    path: str | Path,
    *,
    target_sample_rate: int,
    target_channels: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode arbitrary audio and shape it like **MUSDB18-HQ** training inputs (used for non-dataset WAVs).

    Stereo layout, resample to ``target_sample_rate``, clamp to [-1, 1] with mild scaling when peaks exceed ±1.
    """
    path = Path(path)
    wav, sr = _load_waveform_any_format(path)
    meta: dict[str, Any] = {
        "path": str(path),
        "source_sample_rate": sr,
        "source_channels": int(wav.shape[0]),
        "target_sample_rate": target_sample_rate,
        "resampled": sr != target_sample_rate,
    }

    wav = _to_target_channels_stereo(wav, target_channels)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)

    peak = float(wav.abs().max().clamp_min(1e-12))
    meta["peak_abs_before_scale"] = peak
    if peak > 1.0:
        wav = wav * (0.999 / peak)
        meta["scaled_down_overs"] = True
    else:
        meta["scaled_down_overs"] = False
    wav = wav.clamp(-1.0, 1.0)
    meta["peak_abs_after_adapt"] = float(wav.abs().max())
    return wav, meta


def load_mixture_and_optional_reference_for_test(
    mixture_path: Path,
    reference_path: Path | None,
    *,
    target_sample_rate: int,
    adapt_web_audio: bool,
    normalize_peak: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Load mixture (+ optional reference) for ``--mixture-wav`` when a reference stem is provided."""
    summary: dict[str, Any] = {"adapt_web_audio": adapt_web_audio, "normalize_peak": normalize_peak}

    if not adapt_web_audio:
        mix = load_audio(mixture_path, target_sample_rate)
        ref = load_audio(reference_path, target_sample_rate) if reference_path else None
        summary["mixture"] = {"mode": "strict"}
        if reference_path:
            summary["reference"] = {"mode": "strict"}
    else:
        mix, mix_meta = load_audio_adapted_for_inference(mixture_path, target_sample_rate=target_sample_rate)
        summary["mixture"] = mix_meta
        ref = None
        if reference_path:
            ref, ref_meta = load_audio_adapted_for_inference(reference_path, target_sample_rate=target_sample_rate)
            summary["reference"] = ref_meta

    if ref is not None and mix.shape[-1] != ref.shape[-1]:
        n = min(mix.shape[-1], ref.shape[-1])
        summary["trimmed_to_samples"] = n
        mix = mix[..., :n]
        ref = ref[..., :n]

    if normalize_peak:
        peak = mix.abs().max().clamp_min(1e-12)
        gain = 0.99 / peak
        summary["peak_normalize_gain"] = float(gain)
        mix = mix * gain
        if ref is not None:
            ref = ref * gain

    return mix, ref, summary


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int = 44100) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = audio.detach().cpu()
    if x.dim() != 2:
        raise ValueError("audio must be (C, T)")
    sf.write(str(p), x.transpose(0, 1).numpy(), samplerate=sample_rate)
