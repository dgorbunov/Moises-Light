from __future__ import annotations

import random
import warnings
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


def _looks_like_wav_layout(root: Path) -> bool:
    train_dir = root / "train"
    if not train_dir.exists():
        train_dir = root / "MUSDB18" / "train"
    if not train_dir.exists():
        return False
    for track_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()])[:5]:
        if (track_dir / "mixture.wav").exists():
            return True
    return False


class MusdbTrainChunkDataset(Dataset):
    """
    Uses musdb DB tracks and samples random 7-second chunks.
    """

    def __init__(
        self,
        tracks: list,
        target_stem: str,
        sample_rate: int,
        segment_seconds: float = 7.0,
        chunks_per_track: int = 8,
    ) -> None:
        self.tracks = tracks
        self.target_stem = target_stem
        self.sample_rate = sample_rate
        self.segment_seconds = float(segment_seconds)
        self.chunks_per_track = int(chunks_per_track)
        self._length = max(1, len(tracks) * self.chunks_per_track)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track = self.tracks[idx % len(self.tracks)]
        track.chunk_duration = self.segment_seconds
        max_start = max(0.0, float(track.duration) - self.segment_seconds)
        track.chunk_start = random.uniform(0.0, max_start) if max_start > 0.0 else 0.0

        mix = torch.from_numpy(track.audio.T).float()  # (2, T)
        tgt = torch.from_numpy(track.targets[self.target_stem].audio.T).float()
        return mix, tgt


class MusdbValFirstChunkDataset(Dataset):
    """
    Validation on deterministic first 7 seconds for each test track.
    """

    def __init__(self, tracks: list, target_stem: str, segment_seconds: float = 7.0) -> None:
        self.tracks = tracks
        self.target_stem = target_stem
        self.segment_seconds = float(segment_seconds)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track = self.tracks[idx]
        track.chunk_start = 0.0
        track.chunk_duration = self.segment_seconds
        mix = torch.from_numpy(track.audio.T).float()
        tgt = torch.from_numpy(track.targets[self.target_stem].audio.T).float()
        return mix, tgt


class MusdbLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        musdb_root: str,
        target_stem: str,
        download_preview: bool = False,
        sample_rate: int = 44100,
        segment_seconds: float = 7.0,
        batch_size: int = 2,
        num_workers: int = 4,
        debug: bool = False,
        debug_num_tracks: int = 2,
        chunks_per_track: int = 8,
    ) -> None:
        super().__init__()
        self.musdb_root = musdb_root
        self.target_stem = target_stem
        self.download_preview = bool(download_preview)
        self.sample_rate = sample_rate
        self.segment_seconds = float(segment_seconds)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.debug_num_tracks = debug_num_tracks
        self.chunks_per_track = chunks_per_track

        self._train_ds: MusdbTrainChunkDataset | None = None
        self._val_ds: MusdbValFirstChunkDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        try:
            import musdb
        except Exception as e:
            raise RuntimeError(
                "musdb package is required for LightningDataModule. "
                "Install with `pip install musdb`."
            ) from e

        if self.download_preview:
            # musdb can download short 7-second preview excerpts for quick experiments.
            # Use root only if it looks user-specified (not placeholder/empty).
            db_kwargs: dict[str, object] = {"download": True}
            if self.musdb_root and self.musdb_root != "/path/to/musdb18hq":
                db_kwargs["root"] = self.musdb_root
            train_db = musdb.DB(subsets="train", split="train", **db_kwargs)
            val_db = musdb.DB(subsets="test", **db_kwargs)
        else:
            root = Path(self.musdb_root)
            is_wav = _looks_like_wav_layout(root)
            if not is_wav:
                warnings.warn(
                    "WAV MUSDB layout not detected. Falling back to STEM decoding (is_wav=False), "
                    "which is slower and requires ffmpeg/stempeg.",
                    stacklevel=2,
                )
            train_db = musdb.DB(root=str(root), subsets="train", split="train", is_wav=is_wav)
            val_db = musdb.DB(root=str(root), subsets="test", is_wav=is_wav)

        train_tracks = list(train_db.tracks)
        val_tracks = list(val_db.tracks)
        if self.debug:
            train_tracks = train_tracks[: self.debug_num_tracks]
            val_tracks = val_tracks[: self.debug_num_tracks]

        self._train_ds = MusdbTrainChunkDataset(
            tracks=train_tracks,
            target_stem=self.target_stem,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            chunks_per_track=self.chunks_per_track,
        )
        self._val_ds = MusdbValFirstChunkDataset(
            tracks=val_tracks,
            target_stem=self.target_stem,
            segment_seconds=self.segment_seconds,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("DataModule.setup() must run before requesting dataloaders.")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("DataModule.setup() must run before requesting dataloaders.")
        return DataLoader(
            self._val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
            persistent_workers=self.num_workers > 1,
            drop_last=False,
        )
