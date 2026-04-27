from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from mamba_light.audio_io import load_audio


STEMS = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class TrackPaths:
    name: str
    mixture: Path
    stems: dict[str, Path]


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
    for track_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        mix = track_dir / "mixture.wav"
        if not mix.exists():
            # alternative name used in some exports
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
        tracks.append(TrackPaths(name=track_dir.name, mixture=mix, stems=stems))
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


class MusdbSegmentDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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
            mix = load_audio(tp.mixture, sample_rate=sample_rate)
            n = mix.shape[-1]
            self._track_num_samples.append(n)
            for start in iter_segment_starts(n, self.seg_samples, overlap=overlap):
                self._index.append((ti, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track_idx, start = self._index[idx]
        tp = self.tracks[track_idx]
        mix = load_audio(tp.mixture, sample_rate=self.sample_rate)
        tgt = load_audio(tp.stems[self.target_stem], sample_rate=self.sample_rate)

        end = start + self.seg_samples
        mix_seg = mix[:, start:end]
        tgt_seg = tgt[:, start:end]

        # pad last segment if needed (should be rare due to indexing)
        if mix_seg.shape[-1] < self.seg_samples:
            pad = self.seg_samples - mix_seg.shape[-1]
            mix_seg = torch.nn.functional.pad(mix_seg, (0, pad))
            tgt_seg = torch.nn.functional.pad(tgt_seg, (0, pad))
        return mix_seg, tgt_seg

