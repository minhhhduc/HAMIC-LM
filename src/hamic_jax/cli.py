"""Command line entry point for the JAX backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["main"]


def _doctor(_: argparse.Namespace) -> int:
    import jax

    from hamic_jax import __version__

    devices = jax.devices()
    payload = {
        "version": __version__,
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "hf_key": bool(os.environ.get("HF_KEY")),
        "python": sys.version.split()[0],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _params(args: argparse.Namespace) -> int:
    from hamic_jax.params import count_parameters, parameter_shapes
    from hamic_jax.presets import load_preset

    preset = load_preset(args.preset)
    payload = {
        "preset": preset.to_dict(),
        "parameters": count_parameters(preset),
        "leaves": len(parameter_shapes(preset)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train(args: argparse.Namespace) -> int:
    import jax

    from hamic_jax.checkpoint import load_checkpoint, save_checkpoint
    from hamic_jax.data import TokenizedCorpus
    from hamic_jax.presets import load_training_config
    from hamic_jax.train import Trainer

    config, preset = load_training_config(args.config)
    corpus = TokenizedCorpus(config.tokenized)
    trainer = Trainer(config, preset, corpus)
    if args.resume:
        params, optimizer_state, step, cursor, _ = load_checkpoint(
            args.resume, trainer.optimizer
        )
        trainer.restore(params, optimizer_state)
        trainer.step = step
        trainer.batches.restore(cursor)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limit = args.steps if args.steps is not None else config.total_steps
    metadata = {
        "run_id": config.run_id,
        "identity": config.identity(),
        "preset": preset.to_dict(),
    }
    for _ in range(limit):
        loss = trainer.train_step()
        if trainer.step % args.log_every == 0:
            print(json.dumps({"step": trainer.step, "loss": float(loss)}), flush=True)
        if args.checkpoint_every and trainer.step % args.checkpoint_every == 0:
            target = out / "checkpoints" / f"step-{trainer.step:09d}"
            save_checkpoint(
                target,
                trainer.params,
                trainer.optimizer_state,
                trainer.step,
                trainer.cursor(),
                metadata,
            )
    if limit:
        jax.block_until_ready(loss)
    if args.checkpoint_every:
        target = out / "checkpoints" / f"step-{trainer.step:09d}"
        if not target.exists():
            save_checkpoint(
                target,
                trainer.params,
                trainer.optimizer_state,
                trainer.step,
                trainer.cursor(),
                metadata,
            )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    from hamic_jax.checkpoint import load_checkpoint
    from hamic_jax.data import TokenizedCorpus
    from hamic_jax.presets import load_training_config
    from hamic_jax.train import evaluate

    config, preset = load_training_config(args.config)
    params, _, step, _, _ = load_checkpoint(args.checkpoint)
    corpus = TokenizedCorpus(config.tokenized)
    value = evaluate(
        config,
        preset,
        corpus,
        {name: np.asarray(array) for name, array in params.items()},
        sequences=args.sequences,
        batch_size=args.batch_size,
    )
    print(json.dumps({"checkpoint": args.checkpoint, "step": step, "loss": value}))
    return 0


def _export(args: argparse.Namespace) -> int:
    import numpy as np

    from hamic_jax.checkpoint import load_checkpoint
    from hamic_jax.export import export_pretrained
    from hamic_jax.presets import load_preset

    preset = load_preset(args.preset)
    params, _, step, _, _ = load_checkpoint(args.checkpoint)
    export_pretrained(
        preset,
        {name: np.asarray(value) for name, value in params.items()},
        args.out,
        tokenizer_dir=args.tokenizer,
        lineage={"checkpoint": args.checkpoint, "step": step},
    )
    print(args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``hamic-jax`` argument parser."""
    parser = argparse.ArgumentParser(prog="hamic-jax")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="print JAX device and credential state")
    doctor.set_defaults(command=_doctor)
    params = sub.add_parser("params", help="print a preset's exact parameter count")
    params.add_argument("--preset", required=True)
    params.set_defaults(command=_params)
    train = sub.add_parser("train", help="train or resume a run")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--resume", default=None)
    train.add_argument("--checkpoint-every", type=int, default=0)
    train.add_argument("--log-every", type=int, default=10)
    train.set_defaults(command=_train)
    evaluate = sub.add_parser("evaluate", help="evaluate a committed checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--sequences", type=int, default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.set_defaults(command=_evaluate)
    export = sub.add_parser("export", help="write a Transformers model directory")
    export.add_argument("--preset", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--tokenizer", default=None)
    export.set_defaults(command=_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.command(args))


if __name__ == "__main__":
    raise SystemExit(main())
