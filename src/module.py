from __future__ import annotations

from dataclasses import asdict

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from augment import JointAugment
from config import TrainConfig
from losses import MoisesLoss, MultiResParams, target_spectrogram
from metrics import chunk_level_sdr
from model import MoisesLight
from stft import STFTParams, istft


class MoisesModule(L.LightningModule):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(asdict(cfg))
        self.stft_params = STFTParams(n_fft=cfg.stft.n_fft, hop_length=cfg.stft.hop_length, win_length=cfg.stft.win_length, freq_bins=cfg.stft.freq_bins)
        multires = MultiResParams(cfg.multires.fft_sizes, cfg.multires.hop_sizes, cfg.multires.win_lengths) if cfg.use_multires_loss else None
        self.model = MoisesLight(audio_channels=2, nband=cfg.nband, g=cfg.g, nrope=cfg.nrope, nsplit_enc=cfg.nsplit_enc, nsplit_dec=cfg.nsplit_dec, depth=3, latent_dim=cfg.latent_dim, freq_bins=cfg.stft.freq_bins)
        self.loss_fn = MoisesLoss(stft_params=self.stft_params, multires=multires)
        self.augment = JointAugment(sample_rate=cfg.sample_rate)
        self.val_csdr_model: list[float] = []
        self.val_csdr_mix: list[float] = []
        self.val_est_rms: list[float] = []
        self.val_tgt_rms: list[float] = []

    def forward(self, mix_spec: torch.Tensor) -> torch.Tensor:
        return self.model(mix_spec)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mixture, target = self.augment(*batch)
        tgt_spec = target_spectrogram(target, self.stft_params)
        pred_spec = self.model(target_spectrogram(mixture, self.stft_params))
        loss = self.loss_fn(pred_spec, tgt_spec, tgt_wav=target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        interval = max(1, int(self.cfg.trainer.log_every_n_steps))
        if (batch_idx + 1) % interval == 0:
            self.print(f"train_step={self.global_step} batch_idx={batch_idx} loss={float(loss.detach().cpu()):.6f}")
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        mixture, target = batch
        tgt_spec = target_spectrogram(target, self.stft_params)
        pred_spec = self.model(target_spectrogram(mixture, self.stft_params))
        val_loss = torch.mean(torch.abs(pred_spec - tgt_spec))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        with torch.no_grad():
            pred_wav = istft(pred_spec, self.stft_params, length=target.shape[-1]).detach().cpu()
            tgt_wav = target.detach().cpu()
            mix_wav = mixture.detach().cpu()
            for i in range(pred_wav.shape[0]):
                csdr_model = chunk_level_sdr(tgt_wav[i], pred_wav[i], sample_rate=self.cfg.sample_rate, chunk_seconds=1.0)
                csdr_mix = chunk_level_sdr(tgt_wav[i], mix_wav[i], sample_rate=self.cfg.sample_rate, chunk_seconds=1.0)
                if csdr_model.song_median_sdr == csdr_model.song_median_sdr:
                    self.val_csdr_model.append(float(csdr_model.song_median_sdr))
                if csdr_mix.song_median_sdr == csdr_mix.song_median_sdr:
                    self.val_csdr_mix.append(float(csdr_mix.song_median_sdr))
                self.val_est_rms.append(float(torch.sqrt(torch.mean(pred_wav[i] ** 2)).item()))
                self.val_tgt_rms.append(float(torch.sqrt(torch.mean(tgt_wav[i] ** 2)).item()))
        return val_loss

    def on_validation_epoch_end(self) -> None:
        if self.val_csdr_model:
            model_med = float(torch.tensor(self.val_csdr_model, dtype=torch.float32).median().item())
            self.log("val/cSDR", model_med, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            self.log("val_cSDR", model_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        if self.val_csdr_mix:
            mix_med = float(torch.tensor(self.val_csdr_mix, dtype=torch.float32).median().item())
            self.log("val/cSDR_mix", mix_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val_cSDR_mix", mix_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            if self.val_csdr_model:
                delta = model_med - mix_med
                self.log("val/cSDR_delta", delta, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                self.log("val_cSDR_delta", delta, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        if self.val_est_rms and self.val_tgt_rms:
            est_med = float(torch.tensor(self.val_est_rms, dtype=torch.float32).median().item())
            tgt_med = float(torch.tensor(self.val_tgt_rms, dtype=torch.float32).median().item())
            self.log("val/est_rms", est_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val/tgt_rms", tgt_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val/est_to_tgt_rms", est_med / max(tgt_med, 1e-12), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.val_csdr_model.clear()
        self.val_csdr_mix.clear()
        self.val_est_rms.clear()
        self.val_tgt_rms.clear()

    def configure_optimizers(self) -> dict:
        optim = AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = ReduceLROnPlateau(optim, mode="min", factor=self.cfg.lr_factor, patience=self.cfg.lr_patience_epochs)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}
