from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mamba_light.config import TrainConfig, load_config
from mamba_light.train import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--musdb-root", type=str, default="", help="MUSDB18 root folder")
    p.add_argument("--target-stem", type=str, default="", help="vocals|drums|bass|other")
    p.add_argument("--out-dir", type=str, default="", help="output dir for checkpoints/logs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.config:
        cfg = load_config(args.config)
    else:
        if not args.musdb_root:
            raise SystemExit("Provide --config or --musdb-root")
        cfg = TrainConfig(musdb_root=args.musdb_root)

    if args.target_stem:
        cfg = TrainConfig(**{**cfg.__dict__, "target_stem": args.target_stem})
    if args.out_dir:
        cfg = TrainConfig(**{**cfg.__dict__, "out_dir": args.out_dir})

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()

