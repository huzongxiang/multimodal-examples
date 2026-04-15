#!/usr/bin/env python3

"""Demo-sized training entrypoint tuned for a 12 GB GPU."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent

sys.path.append(str(PROJECT_ROOT))

from tld.configs import DataConfig, DenoiserConfig, ModelConfig, TrainConfig
from tld.train import main as train_main


PROFILES = {
    "minimum": {
        "embed_dim": 128,
        "n_layers": 3,
        "batch_size": 16,
        "n_epoch": 20,
        "save_and_eval_every_iters": 157,
    },
    "demo": {
        "embed_dim": 192,
        "n_layers": 4,
        "batch_size": 16,
        "n_epoch": 20,
        "save_and_eval_every_iters": 157,
    },
    "formal": {
        "embed_dim": 768,
        "n_layers": 12,
        "batch_size": 64,
        "n_epoch": 5,
        "save_and_eval_every_iters": 3000,
    },
}


DATA_LAYOUTS = {
    "default": ("latents.npy", "text_emb.npy", "val_emb.npy"),
    "mj": ("mj_latents.npy", "mj_text_emb.npy", "val_encs.npy"),
    "image256": ("image_latents256.npy", "orig_text_encodings256.npy", "val_encs.npy"),
}


def resolve_data_paths(data_dir: Path, layout: str) -> tuple[Path, Path, Path]:
    if layout != "auto":
        names = DATA_LAYOUTS[layout]
        return tuple(data_dir / name for name in names)

    for names in DATA_LAYOUTS.values():
        paths = tuple(data_dir / name for name in names)
        if all(path.exists() for path in paths):
            return paths

    tried = [", ".join(names) for names in DATA_LAYOUTS.values()]
    raise FileNotFoundError(
        f"Could not find a supported data layout in {data_dir}. Tried: {tried}"
    )


def build_config(data_dir: Path, profile: str, layout: str) -> ModelConfig:
    profile_cfg = PROFILES[profile]
    latent_path, text_emb_path, val_path = resolve_data_paths(data_dir, layout)
    denoiser_config = DenoiserConfig(
        image_size=32,
        patch_size=2,
        embed_dim=profile_cfg["embed_dim"],
        n_layers=profile_cfg["n_layers"],
        n_channels=4,
        text_emb_size=768,
        noise_embed_dims=256,
        dropout=0,
        mlp_multiplier=4,
    )
    train_config = TrainConfig(
        batch_size=profile_cfg["batch_size"],
        lr=3e-4,
        n_epoch=profile_cfg["n_epoch"],
        save_and_eval_every_iters=profile_cfg["save_and_eval_every_iters"],
        compile=False,
        use_wandb=False,
        save_model=True,
        model_name=str(DEMO_ROOT / "checkpoints" / f"tld_{profile}_4070ti_demo.pth"),
    )
    if profile == "formal":
        train_config.early_stopping_patience = 0
        train_config.save_reconstruction_every_iters = 1000
        train_config.checkpoint_every_iters = 3000
        train_config.sample_every_epochs = 0
        train_config.recon_every_epochs = 0
        train_config.weight_every_epochs = 0
    data_config = DataConfig(
        latent_path=str(latent_path),
        text_emb_path=str(text_emb_path),
        val_path=str(val_path),
    )
    return ModelConfig(
        data_config=data_config,
        denoiser_config=denoiser_config,
        train_config=train_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing latents.npy/text_emb.npy/val_emb.npy.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="demo",
        help="minimum is the smallest viable run; demo is the better default.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", *sorted(DATA_LAYOUTS)],
        default="auto",
        help="Which file naming layout to use inside --data-dir.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override the profile batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the profile epoch count.")
    parser.add_argument("--embed-dim", type=int, default=None, help="Override the denoiser embed dim.")
    parser.add_argument("--layers", type=int, default=None, help="Override the denoiser layer count.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="How often to run sampling during training, in optimizer steps.",
    )
    parser.add_argument(
        "--save-recon-every",
        type=int,
        default=None,
        help="How often to save decoded train target/prediction grids, in optimizer steps.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="How often to save versioned training checkpoints, in optimizer steps.",
    )
    parser.add_argument(
        "--sample-every-epochs",
        type=int,
        default=None,
        help="How often to save generated sample grids, in epochs.",
    )
    parser.add_argument(
        "--recon-every-epochs",
        type=int,
        default=None,
        help="How often to save reconstruction grids, in epochs.",
    )
    parser.add_argument(
        "--weight-every-epochs",
        type=int,
        default=None,
        help="How often to save EMA-only weight snapshots, in epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics and image outputs.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop if epoch mean loss does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="Minimum epoch mean loss improvement needed to reset patience.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="transformer-latent-diffusion",
        help="W&B project name when --use-wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument("--run", action="store_true", help="Actually start training instead of only printing the config.")
    return parser.parse_args()


def resolve_demo_path(path: Path) -> Path:
    return path if path.is_absolute() else (DEMO_ROOT / path)


def main() -> None:
    args = parse_args()
    config = build_config(args.data_dir, args.profile, args.layout)

    if args.batch_size is not None:
        config.train_config.batch_size = args.batch_size
    if args.epochs is not None:
        config.train_config.n_epoch = args.epochs
    if args.embed_dim is not None:
        config.denoiser_config.embed_dim = args.embed_dim
    if args.layers is not None:
        config.denoiser_config.n_layers = args.layers
    if args.save_every is not None:
        config.train_config.save_and_eval_every_iters = args.save_every
    if args.save_recon_every is not None:
        config.train_config.save_reconstruction_every_iters = args.save_recon_every
    if args.checkpoint_every is not None:
        config.train_config.checkpoint_every_iters = args.checkpoint_every
    if args.sample_every_epochs is not None:
        config.train_config.sample_every_epochs = args.sample_every_epochs
    if args.recon_every_epochs is not None:
        config.train_config.recon_every_epochs = args.recon_every_epochs
    if args.weight_every_epochs is not None:
        config.train_config.weight_every_epochs = args.weight_every_epochs
    if args.output_dir is not None:
        config.train_config.output_dir = str(resolve_demo_path(args.output_dir))
    if args.early_stop_patience is not None:
        config.train_config.early_stopping_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        config.train_config.early_stopping_min_delta = args.early_stop_min_delta
    if args.use_wandb:
        config.train_config.use_wandb = True
        config.train_config.wandb_project = args.wandb_project
        config.train_config.wandb_entity = args.wandb_entity
        if args.wandb_run_name is not None:
            config.train_config.wandb_run_name = args.wandb_run_name

    if args.output_dir is None:
        auto_output_dir = DEMO_ROOT / "outputs" / (
            f"{args.data_dir.name}_{args.profile}_e{config.train_config.n_epoch}_bs{config.train_config.batch_size}"
        )
        config.train_config.output_dir = str(auto_output_dir)
    else:
        auto_output_dir = resolve_demo_path(args.output_dir)
        config.train_config.output_dir = str(auto_output_dir)

    if not config.train_config.best_model_name:
        config.train_config.best_model_name = str(Path(config.train_config.output_dir) / "best_model.pth")
    config.train_config.model_name = str(Path(config.train_config.output_dir) / "last_checkpoint.pth")
    if not config.train_config.wandb_run_name:
        config.train_config.wandb_run_name = auto_output_dir.name

    print("Resolved config:")
    print(asdict(config))

    if not args.run:
        print("Pass --run to start training with accelerate.")
        return

    (DEMO_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_main(config)


if __name__ == "__main__":
    main()
