#!/usr/bin/env python
"""Train and evaluate ShapeNet or launch the existing interactive drawing demo."""
import argparse
from pathlib import Path

from shape_analyzer.config import cfg
from shape_analyzer.train import train

__all__ = ["train", "interactive_demo"]


def interactive_demo(weight_path):
    from shape_analyzer.demo import interactive_demo as draw
    return draw(weight_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--demo", action="store_true", help="One offline epoch on generated geometry")
    size.add_argument("--quick", action="store_true", help="Small real Quick, Draw! benchmark")
    parser.add_argument("--work-dir", type=Path, help="Directory for generated splits and checkpoints")
    parser.add_argument("--cache-dir", type=Path, help="Directory for the five official bitmap files")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--seed", type=int, default=cfg.RNG_SEED)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or (args.epochs is not None and args.epochs < 1):
        parser.error("Seed must be nonnegative and epochs must be positive.")
    cfg.RNG_SEED = args.seed
    sources = None
    source_name = "Google Quick, Draw!"
    if args.quick or args.demo:
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cfg.OUT_RES, cfg.BATCH_SIZE, cfg.WORKERS, cfg.TRAIN_FRAC = 64, 32, 0, 0.8
        cfg.PRECISION = "32-true"
        if args.demo:
            from shape_analyzer.fixture import geometry_sources
            cfg.N_PER_SHAPE, cfg.HOLD_QD_PER_SHAPE, cfg.MAX_EPOCHS = 16, 4, 1
            sources = geometry_sources(24, args.seed)
            source_name = "Generated geometric smoke fixture"
        else:
            cfg.N_PER_SHAPE, cfg.HOLD_QD_PER_SHAPE, cfg.MAX_EPOCHS = 128, 40, 4
    directory = args.work_dir
    if directory is None and (args.quick or args.demo):
        directory = Path("shape-run-demo" if args.demo else "shape-run-quick")
    if directory is not None:
        cfg.DATA_PATH = directory / "shapes_data.npz"
        cfg.HOLD_PATH = directory / "qd_holdout.npz"
        cfg.CKPT_DIR = directory / "checkpoints"
    if args.cache_dir:
        cfg.CACHE_DIR = args.cache_dir
    if args.epochs is not None:
        cfg.MAX_EPOCHS = args.epochs
    out = train(sources=sources, source_name=source_name, accelerator=args.device,
                progress=not args.no_progress)
    print(f"Weights: {out}")
    print(f"Held-out metrics: {out.parent / 'evaluation.json'}")


if __name__ == "__main__":
    main()
