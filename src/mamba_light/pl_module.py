from __future__ import annotations

from dataclasses import asdict

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mamba_light.augment import JointAugment
from mamba_light.config import TrainConfig
from mamba_light.losses import MoisesLoss, MultiResParams, target_spectrogram
from mamba_light.metrics import chunk_level_sdr
from mamba_light.model import MoisesLight
from mamba_light.stft import STFTParams, istft


class MoisesLightningModule(L.LightningModule):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(asdict(cfg))

        self.stft_params = STFTParams(
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

        self.model = MoisesLight(
            audio_channels=2,
            nband=cfg.nband,
            g=cfg.g,
            nrope=cfg.nrope,
            nsplit_enc=cfg.nsplit_enc,
            nsplit_dec=cfg.nsplit_dec,
            depth=3,
            latent_dim=cfg.latent_dim,
            freq_bins=cfg.stft.freq_bins,
        )
        self.loss_fn = MoisesLoss(stft_params=self.stft_params, multires=multires)
        self.augment = JointAugment(sample_rate=cfg.sample_rate)
        self.val_csdr: list[float] = []

    def forward(self, mix_spec: torch.Tensor) -> torch.Tensor:
        return self.model(mix_spec)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mixture, target = batch
        mixture, target = self.augment(mixture, target)
        tgt_spec = target_spectrogram(target, self.stft_params)
        mix_spec = target_spectrogram(mixture, self.stft_params)
        pred_spec = self.model(mix_spec)
        loss = self.loss_fn(pred_spec, tgt_spec, tgt_wav=target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        interval = max(1, int(self.cfg.trainer.log_every_n_steps))
        if (batch_idx + 1) % interval == 0:
            # Emit newline-based logs for train steps (helpful in SLURM logs).
            self.print(
                f"train_step={self.global_step} batch_idx={batch_idx} loss={float(loss.detach().cpu()):.6f}"
            )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mixture, target = batch
        tgt_spec = target_spectrogram(target, self.stft_params)
        mix_spec = target_spectrogram(mixture, self.stft_params)
        pred_spec = self.model(mix_spec)
        val_loss = torch.mean(torch.abs(pred_spec - tgt_spec))
        # Log both names for readability and checkpoint compatibility.
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            pred_wav = istft(pred_spec, self.stft_params, length=target.shape[-1]).detach().cpu()
            tgt_wav = target.detach().cpu()
            for i in range(pred_wav.shape[0]):
                csdr = chunk_level_sdr(tgt_wav[i], pred_wav[i], sample_rate=self.cfg.sample_rate, chunk_seconds=1.0)
                m = float(csdr.song_median_sdr)
                if m == m:  # skip NaN (e.g. silent reference stem)
                    self.val_csdr.append(m)
        return val_loss

    def on_validation_epoch_end(self) -> None:
        if self.val_csdr:
            median = float(torch.tensor(self.val_csdr, dtype=torch.float32).median().item())
            self.log("val/cSDR", median, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            self.val_csdr.clear()

    def configure_optimizers(self) -> dict:
        optim = AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.cfg.lr_factor,
            patience=self.cfg.lr_patience_epochs,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "interval": "epoch", "frequency": 1},
        }
