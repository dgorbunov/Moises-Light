from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from mamba_light.audio_io import load_audio


STEMS = ("vocals", "drums", "bass", "other")

_STEMPEG_STEM_INDEX = {"drums": 1, "bass": 2, "other": 3, "vocals": 4}  # common MUSDB ordering w/ mixture at 0


@dataclass(frozen=True)
class TrackPaths:
    name: str
    mixture: Path
    stems: dict[str, Path]
    is_stem_mp4: bool = False


def _infer_layout(root: Path) -> tuple[Path, Path, Path | None]:
    """
    Supports common MUSDB18 layouts:
    - <root>/train/<track>/mixture.wav + stems/*.wav
    - <root>/valid/<track>/... (optional)
    - <root>/test/<track>/...
    or the "musdb18" layout:
    - <root>/MUSDB18/train/<track>/...
    """
    if (root / "train").exists() and (root / "test").exists():
        valid = root / "valid"
        return root / "train", root / "test", (valid if valid.exists() else None)
    if (root / "MUSDB18" / "train").exists():
        valid = root / "MUSDB18" / "valid"
        return root / "MUSDB18" / "train", root / "MUSDB18" / "test", (valid if valid.exists() else None)
    raise FileNotFoundError(
        f"Could not find MUSDB18 layout under {root}. Expected train/test folders."
    )


def discover_tracks(root: str | Path, split: str) -> list[TrackPaths]:
    train_dir, test_dir, valid_dir = _infer_layout(Path(root))
    if split in ("train",):
        base = train_dir
    elif split in ("valid", "val"):
        base = valid_dir if valid_dir is not None else train_dir
    else:
        base = test_dir
    if not base.exists():
        raise FileNotFoundError(base)

    tracks: list[TrackPaths] = []
    # Case A: directory-per-track layout with WAV/FLAC
    for track_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        mix = track_dir / "mixture.wav"
        if not mix.exists():
            alt = track_dir / "mixture.flac"
            if alt.exists():
                mix = alt
            else:
                continue
        stems: dict[str, Path] = {}
        for s in STEMS:
            cand = track_dir / f"{s}.wav"
            if not cand.exists():
                cand2 = track_dir / "stems" / f"{s}.wav"
                if cand2.exists():
                    cand = cand2
            if not cand.exists():
                raise FileNotFoundError(f"Missing stem {s} for {track_dir}")
            stems[s] = cand
        tracks.append(TrackPaths(name=track_dir.name, mixture=mix, stems=stems, is_stem_mp4=False))

    # Case B: original MUSDB layout with *.stem.mp4 files
    if not tracks:
        mp4s = sorted([p for p in base.iterdir() if p.is_file() and p.name.endswith(".stem.mp4")])
        for mp4 in mp4s:
            stems = {s: mp4 for s in STEMS}
            name = mp4.name.replace(".stem.mp4", "")
            tracks.append(TrackPaths(name=name, mixture=mp4, stems=stems, is_stem_mp4=True))

    if not tracks:
        raise FileNotFoundError(f"No tracks found under {base}")

    # If valid is a separate folder, we're done (no need to infer split).
    if split in ("valid", "val") and valid_dir is not None:
        return tracks

    # Otherwise infer a valid split from the train folder if needed.
    if split in ("train", "valid", "val") and valid_dir is None:
        full_train = tracks
        valid_names = _load_valid_names(Path(root), full_train)
        if split in ("train",):
            return [t for t in full_train if t.name not in valid_names]
        return [t for t in full_train if t.name in valid_names]

    return tracks


def _load_valid_names(root: Path, tracks: list[TrackPaths]) -> set[str]:
    """
    If a file exists at <root>/musdb_valid.txt (one track folder name per line),
    use that. Otherwise use the last 14 tracks in sorted order (deterministic).
    """
    p = root / "musdb_valid.txt"
    if p.exists():
        names = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
        return names
    names_sorted = [t.name for t in tracks]
    return set(names_sorted[-14:])


def iter_segment_starts(num_samples: int, seg_samples: int, overlap: float) -> Iterator[int]:
    if seg_samples <= 0:
        raise ValueError("seg_samples must be positive")
    if num_samples <= seg_samples:
        yield 0
        return
    hop = max(1, int(seg_samples * (1.0 - overlap)))
    for start in range(0, num_samples - seg_samples + 1, hop):
        yield start


class MusdbSegmentDataset(Dataset):
    """
    Returns (mixture, target) segments as (C, T).
    One epoch iterates over all precomputed segments once (paper behavior).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        sample_rate: int,
        segment_seconds: float,
        overlap: float,
        target_stem: str,
    ) -> None:
        if target_stem not in STEMS:
            raise ValueError(f"target_stem must be one of {STEMS}")
        self.tracks = discover_tracks(root, split=split)
        self.sample_rate = sample_rate
        self.seg_samples = int(round(segment_seconds * sample_rate))
        self.overlap = overlap
        self.target_stem = target_stem

        # Build index of all segments
        self._index: list[tuple[int, int]] = []
        self._track_num_samples: list[int] = []
        for ti, tp in enumerate(self.tracks):
            mix = _load_mixture(tp, sample_rate=sample_rate)
            n = mix.shape[-1]
            self._track_num_samples.append(n)
            for start in iter_segment_starts(n, self.seg_samples, overlap=overlap):
                self._index.append((ti, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track_idx, start = self._index[idx]
        tp = self.tracks[track_idx]
        mix = _load_mixture_segment(tp, sample_rate=self.sample_rate, start_sample=start, num_samples=self.seg_samples)
        tgt = _load_target_segment(
            tp,
            stem=self.target_stem,
            sample_rate=self.sample_rate,
            start_sample=start,
            num_samples=self.seg_samples,
        )

        # Ensure fixed length
        if mix.shape[-1] < self.seg_samples:
            mix = torch.nn.functional.pad(mix, (0, self.seg_samples - mix.shape[-1]))
        if tgt.shape[-1] < self.seg_samples:
            tgt = torch.nn.functional.pad(tgt, (0, self.seg_samples - tgt.shape[-1]))
        return mix, tgt


def _read_stem_mp4(path: Path, sample_rate: int, stem_id: int | None, start_sec: float | None, duration_sec: float | None) -> torch.Tensor:
    try:
        import stempeg
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This MUSDB layout uses *.stem.mp4. Install 'stempeg' and ensure ffmpeg is available."
        ) from e

    audio, sr = stempeg.read_stems(
        str(path),
        stem_id=stem_id,
        start=start_sec,
        duration=duration_sec,
        sample_rate=sample_rate,
    )
    if sr != sample_rate:
        raise ValueError(f"Expected sample_rate={sample_rate}, got {sr} for {path}")

    # stempeg returns either (stems, samples, channels) or (samples, channels) depending on stem_id
    if audio.ndim == 3:
        # if multiple stems, take first one returned
        audio = audio[0]
    # now (samples, channels)
    x = torch.from_numpy(audio).float().transpose(0, 1).contiguous()  # (C, T)
    return x


def _load_mixture(tp: TrackPaths, sample_rate: int) -> torch.Tensor:
    if not tp.is_stem_mp4:
        return load_audio(tp.mixture, sample_rate=sample_rate)
    # mixture is commonly stem 0
    return _read_stem_mp4(tp.mixture, sample_rate=sample_rate, stem_id=0, start_sec=None, duration_sec=None)


def _load_mixture_segment(tp: TrackPaths, sample_rate: int, start_sample: int, num_samples: int) -> torch.Tensor:
    if not tp.is_stem_mp4:
        mix = load_audio(tp.mixture, sample_rate=sample_rate)
        return mix[:, start_sample : start_sample + num_samples]
    start_sec = float(start_sample) / float(sample_rate)
    duration_sec = float(num_samples) / float(sample_rate)
    return _read_stem_mp4(tp.mixture, sample_rate=sample_rate, stem_id=0, start_sec=start_sec, duration_sec=duration_sec)


def _load_target_segment(tp: TrackPaths, stem: str, sample_rate: int, start_sample: int, num_samples: int) -> torch.Tensor:
    if not tp.is_stem_mp4:
        tgt = load_audio(tp.stems[stem], sample_rate=sample_rate)
        return tgt[:, start_sample : start_sample + num_samples]
    stem_id = _STEMPEG_STEM_INDEX.get(stem)
    if stem_id is None:
        raise ValueError(f"Unknown stem: {stem}")
    start_sec = float(start_sample) / float(sample_rate)
    duration_sec = float(num_samples) / float(sample_rate)
    return _read_stem_mp4(tp.mixture, sample_rate=sample_rate, stem_id=stem_id, start_sec=start_sec, duration_sec=duration_sec)

