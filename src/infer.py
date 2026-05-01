from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from model import MoisesLight
from overlap_add import overlap_add
from stft import STFTParams, istft, stft

if TYPE_CHECKING:
    from config import TrainConfig


@torch.no_grad()
def separate_track(
    model: MoisesLight,
    mixture: torch.Tensor,
    stft_params: STFTParams,
    segment_seconds: float,
    overlap: float,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    seg_samples = int(round(segment_seconds * sample_rate))
    hop = max(1, int(round(seg_samples * (1.0 - overlap))))
    t = mixture.shape[-1]
    starts = list(range(0, max(1, t - seg_samples + 1), hop))
    if starts[-1] + seg_samples < t:
        starts.append(t - seg_samples)

    chunks: list[torch.Tensor] = []
    for s in starts:
        x = mixture[:, s : s + seg_samples]
        if x.shape[-1] < seg_samples:
            x = torch.nn.functional.pad(x, (0, seg_samples - x.shape[-1]))
        X = stft(x[None, ...].to(device), stft_params)
        Y = model(X)
        y = istft(Y, stft_params, length=seg_samples)[0].cpu()
        chunks.append(y)
    return overlap_add(torch.stack(chunks, dim=0), hop=hop)[:, :t]


def load_model_from_ckpt(ckpt_path: str | Path, device: torch.device, cfg: TrainConfig | None = None) -> MoisesLight:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "hparams" in ckpt and "model" in ckpt:
        # Exported inference checkpoint.
        h = ckpt["hparams"]
        stft_h = h.get("stft") or {}
        freq_bins = int(stft_h.get("freq_bins", h.get("freq_bins", 2048))) if isinstance(stft_h, dict) else int(h.get("freq_bins", 2048))
        model = MoisesLight(
            audio_channels=2,
            nband=h["nband"],
            g=h["g"],
            nrope=h["nrope"],
            nsplit_enc=h["nsplit_enc"],
            nsplit_dec=h["nsplit_dec"],
            depth=h["depth"],
            latent_dim=h.get("latent_dim", 128),
            freq_bins=freq_bins,
            bf=h.get("bf", 8),
            use_weight_sharing=h.get("use_weight_sharing", False),
            use_mamba=h.get("use_mamba", False),
        )
        model.load_state_dict(ckpt["model"])
    elif "state_dict" in ckpt:
        # Lightning training checkpoint (.ckpt).
        if cfg is None:
            raise RuntimeError("Loading Lightning .ckpt requires cfg for model hyperparameters.")
        model = MoisesLight(
            audio_channels=2,
            nband=cfg.nband,
            g=cfg.g,
            nrope=cfg.nrope,
            nsplit_enc=cfg.nsplit_enc,
            nsplit_dec=cfg.nsplit_dec,
            depth=3,
            latent_dim=cfg.latent_dim,
            freq_bins=cfg.stft.freq_bins,
            bf=getattr(cfg, "bf", 8),
            use_weight_sharing=cfg.use_weight_sharing,
            use_mamba=cfg.use_mamba,
        )
        model_state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        if not model_state:
            raise RuntimeError(f"No model weights found in Lightning checkpoint: '{ckpt_path}'")
        model.load_state_dict(model_state)
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: '{ckpt_path}'")
    model.to(device)
    return model
