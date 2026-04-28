from __future__ import annotations

from pathlib import Path


def looks_like_wav_layout(root: Path, subset: str = "train") -> bool:
    subset_dir = root / subset
    if not subset_dir.exists():
        subset_dir = root / "MUSDB18" / subset
    if not subset_dir.exists():
        return False
    for track_dir in sorted([p for p in subset_dir.iterdir() if p.is_dir()])[:5]:
        if (track_dir / "mixture.wav").exists():
            return True
    return False


def split_test_tracks(tracks: list, partition: str) -> list:
    """
    Deterministic split for MUSDB test subset when no explicit split file exists.
    - val: first half
    - test: second half
    - all: full list
    """
    if partition == "all":
        return tracks
    if partition not in ("val", "test"):
        raise ValueError("partition must be one of: val, test, all")
    ordered = sorted(tracks, key=lambda t: str(getattr(t, "name", "")))
    mid = len(ordered) // 2
    if partition == "val":
        return ordered[:mid]
    return ordered[mid:]
