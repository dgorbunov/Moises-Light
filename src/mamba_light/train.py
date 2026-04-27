from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mamba_light.augment import JointAugment
from mamba_light.config import TrainConfig
from mamba_light.losses import MoisesLoss, MultiResParams, target_spectrogram
from mamba_light.model import MoisesLight
from mamba_light.musdb import MusdbSegmentDataset
from mamba_light.stft import STFTParams


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_ckpt(path: Path, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int, best_val: float, h: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "hparams": h,
        },
        str(path),
    )


def train(cfg: TrainConfig) -> None:
    _seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stft_params = STFTParams(
        n_fft=cfg.stft.n_fft,
        hop_length=cfg.stft.hop_length,
        win_length=cfg.stft.win_length,
        freq_bins=cfg.stft.freq_bins,
    )
    multires = None
    if cfg.use_multires_loss:
        multires = MultiResParams(
            fft_sizes=cfg.multires.fft_sizes,
            hop_sizes=cfg.multires.hop_sizes,
            win_lengths=cfg.multires.win_lengths,
        )

    ds_train = MusdbSegmentDataset(
        root=cfg.musdb_root,
        split="train",
        sample_rate=cfg.sample_rate,
        segment_seconds=cfg.segment_seconds,
        overlap=cfg.train_overlap,
        target_stem=cfg.target_stem,
    )
    ds_val = MusdbSegmentDataset(
        root=cfg.musdb_root,
        split="valid",
        sample_rate=cfg.sample_rate,
        segment_seconds=cfg.segment_seconds,
        overlap=cfg.train_overlap,
        target_stem=cfg.target_stem,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    model = MoisesLight(
        audio_channels=2,
        nband=cfg.nband,
        g=cfg.g,
        nrope=cfg.nrope,
        nsplit_enc=cfg.nsplit_enc,
        nsplit_dec=cfg.nsplit_dec,
        depth=3,
    ).to(device)

    aug = JointAugment(sample_rate=cfg.sample_rate)
    loss_fn = MoisesLoss(stft_params=stft_params, multires=multires)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    out_dir = Path(cfg.out_dir) / cfg.target_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    lr_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(dl_train, desc=f"train e{epoch}", leave=False)
        run_loss = 0.0
        for mixture, target in pbar:
            mixture = mixture.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mixture, target = aug(mixture, target)

            tgt_spec = target_spectrogram(target, stft_params)

            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                mix_spec = target_spectrogram(mixture, stft_params)
                pred_spec = model(mix_spec)
                loss = loss_fn(pred_spec, tgt_spec, tgt_wav=target)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            run_loss += float(loss.detach().cpu())
            pbar.set_postfix(loss=run_loss / max(1, pbar.n + 1))

        # validation: use spectrogram loss proxy (fast, stable)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mixture, target in tqdm(dl_val, desc=f"val e{epoch}", leave=False):
                mixture = mixture.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                tgt_spec = target_spectrogram(target, stft_params)
                mix_spec = target_spectrogram(mixture, stft_params)
                pred_spec = model(mix_spec)
                val_loss += float(torch.mean(torch.abs(pred_spec - tgt_spec)).detach().cpu())
        val_loss /= max(1, len(dl_val))

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            lr_no_improve = 0
            _save_ckpt(out_dir / "best.pt", model, optim, epoch=epoch, best_val=best_val, h={
                "nband": cfg.nband,
                "g": cfg.g,
                "nrope": cfg.nrope,
                "nsplit_enc": cfg.nsplit_enc,
                "nsplit_dec": cfg.nsplit_dec,
                "depth": 3,
                "target_stem": cfg.target_stem,
                "stft": asdict(cfg.stft),
                "sample_rate": cfg.sample_rate,
                "segment_seconds": cfg.segment_seconds,
            })
        else:
            epochs_no_improve += 1
            lr_no_improve += 1

        # LR schedule (paper): *0.9 if no val loss improvement for 20 epochs
        if lr_no_improve >= cfg.lr_patience_epochs:
            for pg in optim.param_groups:
                pg["lr"] = pg["lr"] * cfg.lr_factor
            lr_no_improve = 0

        # early stop (paper): no val loss improvement for 50 epochs
        if epochs_no_improve >= cfg.early_stop_patience_epochs:
            break

        # periodic checkpoint
        if (epoch + 1) % 10 == 0:
            _save_ckpt(out_dir / f"epoch_{epoch:04d}.pt", model, optim, epoch=epoch, best_val=best_val, h={})

    (out_dir / "train_summary.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val}, indent=2),
        encoding="utf-8",
    )

