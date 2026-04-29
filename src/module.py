from __future__ import annotations

from dataclasses import asdict
import os
import time

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
        self.metrics_every_n_epochs = max(1, int(cfg.metrics_every_n_epochs))
        multires = MultiResParams(cfg.multires.fft_sizes, cfg.multires.hop_sizes, cfg.multires.win_lengths) if cfg.use_multires_loss else None
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
            use_weight_sharing=cfg.use_weight_sharing,
            use_mamba=cfg.use_mamba,
        )
        self.loss_fn = MoisesLoss(stft_params=self.stft_params, multires=multires)
        self.augment = JointAugment(sample_rate=cfg.sample_rate)
        self.val_sisdr_model: list[float] = []
        self.val_sisdr_mix: list[float] = []
        self.val_est_rms: list[float] = []
        self.val_tgt_rms: list[float] = []

    def forward(self, mix_spec: torch.Tensor) -> torch.Tensor:
        return self.model(mix_spec)

    def on_fit_start(self) -> None:
        """Mirror real train + val paths so JIT sees fwd/bwd + MoisesLoss + eval mode.

        A forward-only warmup completes instantly when kernels are cached, but the first real
        training_step still spends minutes compiling backward kernels (Mamba2 bwd Triton),
        istft/multires grads inside MoisesLoss, and audiomentations — none of which ran here.
        Validation runs in eval() without augment — separate graph from train.
        """
        if not self.cfg.use_mamba or not torch.cuda.is_available():
            return
        self.print(
            "Mamba2 warmup: train fwd+bwd + eval fwd (MoisesLoss & augment included). "
            "First run can take several minutes...",
            flush=True,
        )
        seg_len = int(self.cfg.sample_rate * self.cfg.segment_seconds)
        bs = self.cfg.batch_size

        # Train path — matches training_step (Lightning wraps autocast for bf16-mixed too).
        self.train()
        wav_m = torch.randn(bs, 2, seg_len, device=self.device)
        wav_t = torch.randn(bs, 2, seg_len, device=self.device)
        self.zero_grad(set_to_none=True)
        mixture, target = self.augment(wav_m, wav_t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tgt_spec = target_spectrogram(target, self.stft_params)
            pred_spec = self.model(target_spectrogram(mixture, self.stft_params))
            loss = self.loss_fn(pred_spec, tgt_spec, tgt_wav=target)
        loss.backward()
        self.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Val path — matches validation_step (eval(), no augment).
        self.eval()
        with torch.no_grad():
            wav = torch.randn(bs, 2, seg_len, device=self.device)
            mixture_v, target_v = wav, wav.clone()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tgt_spec_v = target_spectrogram(target_v, self.stft_params)
                pred_spec_v = self.model(target_spectrogram(mixture_v, self.stft_params))
                _ = torch.mean(torch.abs(pred_spec_v - tgt_spec_v))
        self.train()
        torch.cuda.synchronize()
        self.print("Mamba2 warmup done.", flush=True)

    def on_train_epoch_start(self) -> None:
        # Avoid printing every epoch — tqdm/Rich progress bar uses \\r on one line; extra prints
        # break the bar (looks like "missing progress bar"). Enable via env when debugging:
        #   MAMBA_LIGHT_DEBUG_PRINTS=1 sbatch configs/turing.sh ...
        if self.cfg.use_mamba and os.environ.get("MAMBA_LIGHT_DEBUG_PRINTS"):
            self.print(f"on_train_epoch_start epoch={self.current_epoch}", flush=True)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        dbg = os.environ.get("MAMBA_LIGHT_DEBUG_PRINTS")
        if dbg and batch_idx == 0:
            self.print(f"training_step batch 0 arrived global_step={self.global_step}", flush=True)
        t0 = time.perf_counter()
        mixture, target = self.augment(*batch)
        tgt_spec = target_spectrogram(target, self.stft_params)
        pred_spec = self.model(target_spectrogram(mixture, self.stft_params))
        loss = self.loss_fn(pred_spec, tgt_spec, tgt_wav=target)
        if dbg and batch_idx == 0:
            torch.cuda.synchronize()
            self.print(f"first batch fwd+bwd took {time.perf_counter() - t0:.2f}s", flush=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        interval = max(1, int(self.cfg.trainer.log_every_n_steps))
        if dbg and (batch_idx + 1) % interval == 0:
            self.print(f"train_step={self.global_step} batch_idx={batch_idx} loss={float(loss.detach().cpu()):.6f}", flush=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        if self.cfg.use_mamba and os.environ.get("MAMBA_LIGHT_DEBUG_PRINTS"):
            self.print(f"on_validation_epoch_start epoch={self.current_epoch}", flush=True)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        dbg = os.environ.get("MAMBA_LIGHT_DEBUG_PRINTS")
        if dbg and batch_idx == 0:
            self.print(f"validation epoch {self.current_epoch}: batch 0 entering...", flush=True)
        elif dbg and batch_idx % 5 == 0:
            self.print(f"val batch_idx={batch_idx}", flush=True)
        mixture, target = batch
        tgt_spec = target_spectrogram(target, self.stft_params)
        pred_spec = self.model(target_spectrogram(mixture, self.stft_params))
        val_loss = torch.mean(torch.abs(pred_spec - tgt_spec))
        if os.environ.get("MAMBA_LIGHT_DEBUG_PRINTS") and batch_idx == 0:
            torch.cuda.synchronize()
            self.print(f"val batch 0 forward complete loss_mean={float(val_loss.detach().cpu()):.6f}", flush=True)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if (self.current_epoch + 1) % self.metrics_every_n_epochs != 0:
            return val_loss
        with torch.no_grad():
            pred_wav = istft(pred_spec, self.stft_params, length=target.shape[-1]).detach()
            tgt_wav = target.detach()
            mix_wav = mixture.detach()
            for i in range(pred_wav.shape[0]):
                sisdr_model = chunk_level_sdr(tgt_wav[i], pred_wav[i], sample_rate=self.cfg.sample_rate, chunk_seconds=1.0)
                sisdr_mix = chunk_level_sdr(tgt_wav[i], mix_wav[i], sample_rate=self.cfg.sample_rate, chunk_seconds=1.0)
                if sisdr_model.song_median_sdr == sisdr_model.song_median_sdr:
                    self.val_sisdr_model.append(float(sisdr_model.song_median_sdr))
                if sisdr_mix.song_median_sdr == sisdr_mix.song_median_sdr:
                    self.val_sisdr_mix.append(float(sisdr_mix.song_median_sdr))
                self.val_est_rms.append(float(torch.sqrt(torch.mean(pred_wav[i] ** 2)).item()))
                self.val_tgt_rms.append(float(torch.sqrt(torch.mean(tgt_wav[i] ** 2)).item()))
        return val_loss

    def on_validation_epoch_end(self) -> None:
        if self.val_sisdr_model:
            model_med = float(torch.tensor(self.val_sisdr_model, dtype=torch.float32).median().item())
            self.log("val/siSDR", model_med, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            self.log("val_siSDR", model_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        if self.val_sisdr_mix:
            mix_med = float(torch.tensor(self.val_sisdr_mix, dtype=torch.float32).median().item())
            self.log("val/siSDR_mix", mix_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val_siSDR_mix", mix_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            if self.val_sisdr_model:
                delta = model_med - mix_med
                self.log("val/siSDR_delta", delta, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
                self.log("val_siSDR_delta", delta, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        if self.val_est_rms and self.val_tgt_rms:
            est_med = float(torch.tensor(self.val_est_rms, dtype=torch.float32).median().item())
            tgt_med = float(torch.tensor(self.val_tgt_rms, dtype=torch.float32).median().item())
            self.log("val/est_rms", est_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val/tgt_rms", tgt_med, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val/est_to_tgt_rms", est_med / max(tgt_med, 1e-12), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.val_sisdr_model.clear()
        self.val_sisdr_mix.clear()
        self.val_est_rms.clear()
        self.val_tgt_rms.clear()

    def configure_optimizers(self) -> dict:
        optim = AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = ReduceLROnPlateau(optim, mode="min", factor=self.cfg.lr_factor, patience=self.cfg.lr_patience_epochs)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}
