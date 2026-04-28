from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    accelerator: str = "gpu"
    devices: int = 1
    strategy: str = "auto"
    precision: str = "32-true"
    max_epochs: int = 300
    log_every_n_steps: int = 20
    deterministic: bool = False


@dataclass(frozen=True)
class MultiResSTFTConfig:
    fft_sizes: tuple[int, ...] = (1024, 2048, 4096)
    hop_sizes: tuple[int, ...] = (256, 512, 1024)
    win_lengths: tuple[int, ...] = (1024, 2048, 4096)


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 6144
    hop_length: int = 1024
    win_length: int = 6144
    freq_bins: int = 2048


@dataclass(frozen=True)
class TrainConfig:
    musdb_root: str = "~/musdb18hq"
    max_train_samples: int = 0
    max_val_samples: int = 0
    sample_rate: int = 44100
    segment_seconds: float = 7.0
    train_overlap: float = 0.75
    eval_overlap: float = 0.50

    batch_size: int = 2
    num_workers: int = 4

    lr: float = 2e-4
    lr_patience_epochs: int = 20
    lr_factor: float = 0.9
    early_stop_patience_epochs: int = 50
    metrics_every_n_epochs: int = 5

    seed: int = 1337

    target_stem: str = "vocals"

    use_multires_loss: bool = True
    multires: MultiResSTFTConfig = MultiResSTFTConfig()
    stft: STFTConfig = STFTConfig()

    nband: int = 4
    g: int = 56
    nrope: int = 5
    nsplit_enc: int = 3
    nsplit_dec: int = 1
    latent_dim: int = 128

    out_dir: str = "runs/moises_light"
    trainer: TrainerConfig = TrainerConfig()
    debug: bool = False
    debug_num_tracks: int = 2
    debug_epochs: int = 10


def load_config(path: str | Path) -> TrainConfig:
    import yaml

    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "stft" in data:
        data["stft"] = STFTConfig(**data["stft"])
    if "multires" in data:
        data["multires"] = MultiResSTFTConfig(**data["multires"])
    if "trainer" in data:
        data["trainer"] = TrainerConfig(**data["trainer"])
    return TrainConfig(**data)
