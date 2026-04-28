from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TrainConfig, load_config
from data import MusdbDataModule
from module import MoisesModule


class DebugMetricsCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.history: dict[str, list[float]] = {"val_siSDR": [], "val_siSDR_mix": [], "val_siSDR_delta": [], "val_loss": []}

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        for key in self.history:
            v = trainer.callback_metrics.get(key)
            if v is not None:
                self.history[key].append(float(v.detach().cpu()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--musdb-root", type=str, default="", help="MUSDB18 root folder")
    p.add_argument("--target-stem", type=str, default="", help="vocals|drums|bass|other")
    p.add_argument("--out-dir", type=str, default="", help="output dir for checkpoints/logs")
    p.add_argument("--max-train-samples", type=int, default=-1, help="Cap number of train samples per epoch")
    p.add_argument("--max-val-samples", type=int, default=-1, help="Cap number of validation samples per epoch")
    p.add_argument("--debug", action="store_true", help="Enable debug mode (2 tracks, 10 epochs)")
    p.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume from")
    return p.parse_args()


def _build_cfg(args: argparse.Namespace) -> TrainConfig:
    cfg = load_config(args.config) if args.config else TrainConfig(musdb_root=args.musdb_root or "~/musdb18hq")
    if args.target_stem:
        cfg = TrainConfig(**{**cfg.__dict__, "target_stem": args.target_stem})
    if args.out_dir:
        cfg = TrainConfig(**{**cfg.__dict__, "out_dir": args.out_dir})
    if args.max_train_samples >= 0:
        cfg = TrainConfig(**{**cfg.__dict__, "max_train_samples": args.max_train_samples})
    if args.max_val_samples >= 0:
        cfg = TrainConfig(**{**cfg.__dict__, "max_val_samples": args.max_val_samples})
    if args.debug and not cfg.debug:
        cfg = TrainConfig(**{**cfg.__dict__, "debug": True})
    return cfg


def _write_debug_report(run_dir: Path, debug_cb: DebugMetricsCallback) -> None:
    (run_dir / "metrics_debug.json").write_text(json.dumps(debug_cb.history, indent=2), encoding="utf-8")


def _export_legacy_checkpoint(trainer: L.Trainer, cfg: TrainConfig, run_dir: Path) -> None:
    if trainer.global_rank != 0:
        return
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt:
        return
    raw = torch.load(best_ckpt, map_location="cpu")
    state = raw["state_dict"]
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    payload = {
        "model": model_state,
        "hparams": {
            "nband": cfg.nband,
            "g": cfg.g,
            "nrope": cfg.nrope,
            "nsplit_enc": cfg.nsplit_enc,
            "nsplit_dec": cfg.nsplit_dec,
            "depth": 3,
            "latent_dim": cfg.latent_dim,
            "target_stem": cfg.target_stem,
            "stft": asdict(cfg.stft),
            "sample_rate": cfg.sample_rate,
            "segment_seconds": cfg.segment_seconds,
        },
        "epoch": int(raw.get("epoch", 0)),
        "best_val": float(trainer.checkpoint_callback.best_model_score.detach().cpu()) if trainer.checkpoint_callback.best_model_score is not None else None,
    }
    torch.save(payload, run_dir / "best_legacy.pt")


def _resolve_runtime(cfg: TrainConfig) -> dict[str, object]:
    accelerator = cfg.trainer.accelerator
    devices = cfg.trainer.devices
    strategy = cfg.trainer.strategy
    precision = cfg.trainer.precision
    if accelerator == "gpu":
        if torch.cuda.is_available():
            avail = torch.cuda.device_count()
            use_devices = max(1, min(int(devices), avail))
            if int(devices) > avail:
                warnings.warn(f"Requested devices={devices}, but only {avail} visible; using {use_devices}.", stacklevel=2)
            use_strategy = "auto" if (use_devices == 1 and str(strategy).startswith("ddp")) else strategy
            return {"accelerator": accelerator, "devices": use_devices, "strategy": use_strategy, "precision": precision}
        if torch.backends.mps.is_available():
            return {"accelerator": "mps", "devices": 1, "strategy": "auto", "precision": "32-true"}
        return {"accelerator": "cpu", "devices": 1, "strategy": "auto", "precision": "32-true"}
    if accelerator in ("mps", "cpu"):
        return {"accelerator": accelerator, "devices": 1, "strategy": "auto", "precision": "32-true"}
    return {"accelerator": accelerator, "devices": devices, "strategy": strategy, "precision": precision}


def main() -> None:
    args = _parse_args()
    cfg = _build_cfg(args)
    L.seed_everything(cfg.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    run_dir = Path(cfg.out_dir) / cfg.target_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    datamodule = MusdbDataModule(
        musdb_root=cfg.musdb_root,
        target_stem=cfg.target_stem,
        sample_rate=cfg.sample_rate,
        segment_seconds=cfg.segment_seconds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        debug=cfg.debug,
        debug_num_tracks=cfg.debug_num_tracks,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
    )
    module = MoisesModule(cfg=cfg)
    debug_cb = DebugMetricsCallback()
    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_siSDR:.3f}-{val_loss:.4f}",
        monitor="val_siSDR",
        mode="max",
        save_top_k=1,
        every_n_epochs=max(1, int(cfg.metrics_every_n_epochs)),
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.early_stop_patience_epochs)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    runtime = _resolve_runtime(cfg)
    trainer = L.Trainer(
        accelerator=runtime["accelerator"],
        devices=runtime["devices"],
        strategy=runtime["strategy"],
        precision=runtime["precision"],
        max_epochs=cfg.debug_epochs if cfg.debug else cfg.trainer.max_epochs,
        default_root_dir=str(run_dir),
        callbacks=[ckpt_cb, early_stop, lr_monitor, debug_cb],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        enable_progress_bar=True,
    )
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=args.resume or None)
    _export_legacy_checkpoint(trainer=trainer, cfg=cfg, run_dir=run_dir)
    if cfg.debug:
        _write_debug_report(run_dir=run_dir, debug_cb=debug_cb)


if __name__ == "__main__":
    main()

