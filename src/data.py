from __future__ import annotations

import random
import warnings
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from dataset_utils import looks_like_wav_layout, split_test_tracks


class MusdbTrainChunkDataset(Dataset):
    def __init__(
        self,
        tracks: list,
        target_stem: str,
        sample_rate: int,
        segment_seconds: float = 7.0,
        chunks_per_track: int = 100,
        max_samples: int = 0,
        min_target_rms_ratio: float = 0.15,
    ) -> None:
        self.tracks = tracks
        self.target_stem = target_stem
        self.sample_rate = sample_rate
        self.segment_seconds = float(segment_seconds)
        self.chunks_per_track = int(chunks_per_track)
        self.min_target_rms_ratio = float(min_target_rms_ratio)
        base_length = max(1, len(tracks) * self.chunks_per_track)
        self._length = min(base_length, max_samples) if max_samples and max_samples > 0 else base_length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track = self.tracks[idx % len(self.tracks)]
        track.chunk_duration = self.segment_seconds
        max_start = max(0.0, float(track.duration) - self.segment_seconds)
        # Retry to find a chunk with sufficient target-stem energy.
        for _ in range(8):
            track.chunk_start = random.uniform(0.0, max_start) if max_start > 0.0 else 0.0
            mixture = torch.from_numpy(track.audio.T).float()
            target = torch.from_numpy(track.targets[self.target_stem].audio.T).float()
            mix_rms = float(mixture.pow(2).mean().sqrt())
            tgt_rms = float(target.pow(2).mean().sqrt())
            if mix_rms < 1e-6 or tgt_rms / (mix_rms + 1e-9) >= self.min_target_rms_ratio:
                return mixture, target
        return mixture, target


class MusdbValRandomChunkDataset(Dataset):
    def __init__(self, tracks: list, target_stem: str, segment_seconds: float = 7.0, max_samples: int = 0) -> None:
        self.tracks = tracks
        self.target_stem = target_stem
        self.segment_seconds = float(segment_seconds)
        self.max_samples = max_samples

    def __len__(self) -> int:
        n = len(self.tracks)
        return min(n, self.max_samples) if self.max_samples and self.max_samples > 0 else n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track = self.tracks[idx]
        track.chunk_duration = self.segment_seconds
        max_start = max(0.0, float(track.duration) - self.segment_seconds)
        track.chunk_start = random.uniform(0.0, max_start) if max_start > 0.0 else 0.0
        return torch.from_numpy(track.audio.T).float(), torch.from_numpy(track.targets[self.target_stem].audio.T).float()


class MusdbDataModule(L.LightningDataModule):
    def __init__(
        self,
        musdb_root: str,
        target_stem: str,
        sample_rate: int = 44100,
        segment_seconds: float = 7.0,
        batch_size: int = 2,
        num_workers: int = 4,
        debug: bool = False,
        debug_num_tracks: int = 2,
        chunks_per_track: int = 8,
        max_val_samples: int = 0,
    ) -> None:
        super().__init__()
        self.musdb_root = musdb_root
        self.target_stem = target_stem
        self.sample_rate = sample_rate
        self.segment_seconds = float(segment_seconds)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.debug_num_tracks = debug_num_tracks
        self.chunks_per_track = chunks_per_track
        self.max_val_samples = max_val_samples
        self._train_ds: MusdbTrainChunkDataset | None = None
        self._val_ds: MusdbValRandomChunkDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        try:
            import musdb
        except Exception as e:
            raise RuntimeError("musdb package is required for DataModule. Install with `pip install musdb`.") from e
        root = Path(self.musdb_root).expanduser()
        is_wav = looks_like_wav_layout(root, subset="train")
        if not is_wav:
            warnings.warn("WAV MUSDB layout not detected. Falling back to STEM decoding (is_wav=False).", stacklevel=2)
        train_db = musdb.DB(root=str(root), subsets="train", split="train", is_wav=is_wav)
        val_db = musdb.DB(root=str(root), subsets="test", is_wav=is_wav)
        train_tracks = list(train_db.tracks)
        val_tracks = split_test_tracks(list(val_db.tracks), partition="val")
        if self.debug:
            train_tracks = train_tracks[: self.debug_num_tracks]
            val_tracks = val_tracks[: self.debug_num_tracks]
        self._train_ds = MusdbTrainChunkDataset(
            train_tracks,
            self.target_stem,
            self.sample_rate,
            self.segment_seconds,
            self.chunks_per_track,
        )
        self._val_ds = MusdbValRandomChunkDataset(val_tracks, self.target_stem, self.segment_seconds, self.max_val_samples)

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("setup() must run before requesting dataloaders.")
        # Disable persistent workers for more reliable loader transitions.
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("setup() must run before requesting dataloaders.")
        # Keep validation batch size aligned with training.
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )
