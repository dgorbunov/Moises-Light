from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mamba_light.config import TrainConfig, load_config
from mamba_light.eval import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config path used for eval")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    args = p.parse_args()

    cfg: TrainConfig = load_config(args.config)
    metrics = evaluate(cfg, ckpt_path=args.ckpt)
    print(metrics)


if __name__ == "__main__":
    main()

