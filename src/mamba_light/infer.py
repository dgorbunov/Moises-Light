from __future__ import annotations

from pathlib import Path

import torch

from mamba_light.model import MoisesLight
from mamba_light.overlap_add import overlap_add
from mamba_light.stft import STFTParams, istft, stft


@torch.no_grad()
def separate_track(
    model: MoisesLight,
    mixture: torch.Tensor,  # (2, T)
    stft_params: STFTParams,
    segment_seconds: float,
    overlap: float,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    seg_samples = int(round(segment_seconds * sample_rate))
    hop = int(round(seg_samples * (1.0 - overlap)))
    hop = max(1, hop)
    t = mixture.shape[-1]

    starts = list(range(0, max(1, t - seg_samples + 1), hop))
    if starts[-1] + seg_samples < t:
        starts.append(t - seg_samples)

    chunks: list[torch.Tensor] = []
    for s in starts:
        x = mixture[:, s : s + seg_samples]
        if x.shape[-1] < seg_samples:
            x = torch.nn.functional.pad(x, (0, seg_samples - x.shape[-1]))
        xb = x[None, ...].to(device)
        X = stft(xb, stft_params)
        Y = model(X)
        y = istft(Y, stft_params, length=seg_samples)[0].cpu()
        chunks.append(y)

    y_full = overlap_add(torch.stack(chunks, dim=0), hop=hop)
    return y_full[:, :t]


def load_model_from_ckpt(ckpt_path: str | Path, device: torch.device) -> MoisesLight:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    h = ckpt["hparams"]
    model = MoisesLight(
        audio_channels=2,
        nband=h["nband"],
        g=h["g"],
        nrope=h["nrope"],
        nsplit_enc=h["nsplit_enc"],
        nsplit_dec=h["nsplit_dec"],
        depth=h["depth"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model

