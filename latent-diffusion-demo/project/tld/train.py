#!/usr/bin/env python3

import copy
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL.Image import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator
from tld.configs import ModelConfig
from tld.reporting import plot_training_metrics, write_training_summary


def eval_gen(diffuser: DiffusionGenerator, labels: Tensor, img_size: int, num_imgs: int = 16) -> Image:
    class_guidance = 4.5
    seed = 10
    repeats = max(1, int(np.ceil(num_imgs / len(labels))))
    labels = torch.repeat_interleave(labels, repeats, dim=0)[:num_imgs]
    out, _ = diffuser.generate(
        labels=labels,
        num_imgs=num_imgs,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=img_size
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg_{class_guidance}_seed_{seed}.png")

    return out


def append_metrics(metrics_path: str, record: dict) -> None:
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_full_state_dict(
    model_ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int | None = None,
    best_epoch_loss: float | None = None,
) -> dict:
    payload = {
        "model_ema": model_ema.state_dict(),
        "opt_state": optimizer.state_dict(),
        "global_step": global_step,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if best_epoch_loss is not None:
        payload["best_epoch_loss"] = best_epoch_loss
    return payload


@torch.no_grad()
def save_reconstruction_grid(
    vae: AutoencoderKL,
    target_latents: Tensor,
    pred_latents: Tensor,
    out_path: str,
    scale_factor: float,
    nrow: int = 4,
) -> None:
    vae_dtype = next(vae.parameters()).dtype
    n_img = min(len(target_latents), nrow)
    target = vae.decode((target_latents[:n_img] * scale_factor).to(vae_dtype))[0].cpu()
    pred = vae.decode((pred_latents[:n_img] * scale_factor).to(vae_dtype))[0].cpu()
    grid = torch.cat([target, pred], dim=0)
    to_pil((vutils.make_grid((grid + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1)).save(out_path)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


to_pil = torchvision.transforms.ToPILImage()


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(inner_value) for inner_value in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value



def main(config: ModelConfig) -> None:
    """main train loop to be used with accelerate"""
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config
    wandb = None
    output_dir = train_config.output_dir
    samples_dir = os.path.join(output_dir, "samples")
    recon_dir = os.path.join(output_dir, "recon")
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    best_model_path = train_config.best_model_name or os.path.join(output_dir, "best_model.pth")

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)
    if accelerator.is_main_process:
        open(metrics_path, "w", encoding="utf-8").close()

    accelerator.print("Loading Data:")
    latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
    train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
    emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    dataset = TensorDataset(latent_train_data, train_label_embeddings)
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name, torch_dtype=config.vae_cfg.vae_dtype)

    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)

    model = Denoiser(**asdict(denoiser_config))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    if train_config.compile:
        accelerator.print("Compiling model:")
        model = torch.compile(model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        import wandb
        wandb.restore(
            train_config.model_name, run_path=f"apapiu/cifar_diffusion/runs/{train_config.run_id}", replace=True
        )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, vae, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        import wandb
        wandb_kwargs = {}
        if train_config.wandb_entity:
            wandb_kwargs["entity"] = train_config.wandb_entity
        if train_config.wandb_run_name:
            wandb_kwargs["name"] = train_config.wandb_run_name
        accelerator.init_trackers(
            project_name=train_config.wandb_project,
            config=make_json_safe(asdict(config)),
            init_kwargs={"wandb": wandb_kwargs},
        )

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))
    best_epoch_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    last_x = None
    last_pred = None

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}")
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        epoch_start = time.time()
        if accelerator.is_main_process and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for x, y in tqdm(train_loader):
            x = x / config.vae_cfg.vae_scale_factor

            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            )
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0  # OR replacement_vector

            should_run_eval = (
                train_config.save_and_eval_every_iters > 0
                and global_step > 0
                and global_step % train_config.save_and_eval_every_iters == 0
            )
            if should_run_eval:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    out = eval_gen(
                        diffuser=diffuser,
                        labels=emb_val,
                        img_size=denoiser_config.image_size,
                        num_imgs=train_config.eval_num_imgs,
                    )
                    sample_path = os.path.join(samples_dir, f"sample_step_{global_step:06d}.png")
                    out.save(sample_path)
                    out.save("img.jpg")
                    if train_config.use_wandb:
                        accelerator.log({"samples/grid": wandb.Image(sample_path)}, step=global_step)

                    if train_config.save_model:
                        accelerator.save(
                            build_full_state_dict(
                                model_ema=ema_model,
                                optimizer=optimizer,
                                global_step=global_step,
                            ),
                            train_config.model_name,
                        )
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                pred = model(x_noisy, noise_level.view(-1, 1), label)
                loss = loss_fn(pred, x)
                loss_value = loss.item()
                accelerator.log({"train_loss": loss.item()}, step=global_step)
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)
                    epoch_loss_sum += loss_value
                    epoch_step_count += 1
                    last_x = x.detach().cpu()
                    last_pred = pred.detach().cpu()

                    step_record = {
                        "kind": "step",
                        "epoch": i,
                        "global_step": global_step,
                        "loss": loss_value,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    if torch.cuda.is_available():
                        step_record["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
                        step_record["cuda_max_allocated_mb"] = round(
                            torch.cuda.max_memory_allocated() / 1024 / 1024, 2
                        )
                    append_metrics(metrics_path, step_record)

                    should_save_recon = (
                        train_config.save_reconstruction_every_iters > 0
                        and global_step > 0
                        and global_step % train_config.save_reconstruction_every_iters == 0
                    )
                    if should_save_recon:
                        recon_path = os.path.join(recon_dir, f"recon_step_{global_step:06d}.png")
                        save_reconstruction_grid(
                            vae=vae,
                            target_latents=x.detach(),
                            pred_latents=pred.detach(),
                            out_path=recon_path,
                            scale_factor=config.vae_cfg.vae_scale_factor,
                            nrow=train_config.reconstruction_n_examples,
                        )
                        if train_config.use_wandb:
                            accelerator.log({"reconstructions/grid": wandb.Image(recon_path)}, step=global_step)

                    should_save_checkpoint = (
                        train_config.save_model
                        and train_config.checkpoint_every_iters > 0
                        and global_step > 0
                        and global_step % train_config.checkpoint_every_iters == 0
                    )
                    if should_save_checkpoint:
                        checkpoint_path = os.path.join(weights_dir, f"checkpoint_step_{global_step:06d}.pth")
                        accelerator.save(
                            build_full_state_dict(
                                model_ema=ema_model,
                                optimizer=optimizer,
                                global_step=global_step,
                                epoch=i,
                                best_epoch_loss=best_epoch_loss,
                            ),
                            checkpoint_path,
                        )
                        if train_config.use_wandb:
                            wandb.save(checkpoint_path)

            global_step += 1

        if accelerator.is_main_process and epoch_step_count > 0:
            epoch_mean_loss = epoch_loss_sum / epoch_step_count
            improved = epoch_mean_loss < (best_epoch_loss - train_config.early_stopping_min_delta)
            if improved:
                best_epoch_loss = epoch_mean_loss
                best_epoch = i
                epochs_without_improvement = 0
                if train_config.save_model and train_config.save_best_model:
                    accelerator.save(
                        build_full_state_dict(
                            model_ema=ema_model,
                            optimizer=optimizer,
                            global_step=global_step,
                            epoch=i,
                            best_epoch_loss=best_epoch_loss,
                        ),
                        best_model_path,
                    )
            else:
                epochs_without_improvement += 1

            epoch_record = {
                "kind": "epoch",
                "epoch": i,
                "global_step": global_step,
                "mean_loss": epoch_mean_loss,
                "epoch_seconds": time.time() - epoch_start,
                "steps": epoch_step_count,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_mean_loss": best_epoch_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }
            if torch.cuda.is_available():
                epoch_record["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
            append_metrics(metrics_path, epoch_record)
            if train_config.use_wandb:
                accelerator.log(
                    {
                        "epoch_mean_loss": epoch_mean_loss,
                        "epoch_seconds": epoch_record["epoch_seconds"],
                        "epochs_without_improvement": epochs_without_improvement,
                        "best_epoch_loss": best_epoch_loss,
                    },
                    step=global_step,
                )

        should_stop = False
        if accelerator.is_main_process and train_config.early_stopping_patience > 0:
            should_stop = epochs_without_improvement >= train_config.early_stopping_patience
            if should_stop:
                append_metrics(
                    metrics_path,
                    {
                        "kind": "early_stop",
                        "epoch": i,
                        "global_step": global_step,
                        "best_epoch": best_epoch,
                        "best_mean_loss": best_epoch_loss,
                        "patience": train_config.early_stopping_patience,
                    },
                )
                accelerator.print(f"Early stopping triggered at epoch {i}. Best epoch was {best_epoch}.")

        is_last_epoch = i == train_config.n_epoch or should_stop
        should_sample_epoch = train_config.sample_every_epochs > 0 and (
            i % train_config.sample_every_epochs == 0 or is_last_epoch
        )
        should_recon_epoch = train_config.recon_every_epochs > 0 and (
            i % train_config.recon_every_epochs == 0 or is_last_epoch
        )
        should_save_weight_epoch = train_config.weight_every_epochs > 0 and (
            i % train_config.weight_every_epochs == 0 or is_last_epoch
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if should_sample_epoch:
                out = eval_gen(
                    diffuser=diffuser,
                    labels=emb_val,
                    img_size=denoiser_config.image_size,
                    num_imgs=train_config.eval_num_imgs,
                )
                sample_path = os.path.join(samples_dir, f"sample_epoch_{i:04d}_step_{global_step:06d}.png")
                out.save(sample_path)
                if train_config.use_wandb:
                    accelerator.log({"samples/grid": wandb.Image(sample_path)}, step=global_step)

            if should_recon_epoch and last_x is not None and last_pred is not None:
                recon_path = os.path.join(recon_dir, f"recon_epoch_{i:04d}_step_{global_step:06d}.png")
                save_reconstruction_grid(
                    vae=vae,
                    target_latents=last_x.to(accelerator.device),
                    pred_latents=last_pred.to(accelerator.device),
                    out_path=recon_path,
                    scale_factor=config.vae_cfg.vae_scale_factor,
                    nrow=train_config.reconstruction_n_examples,
                )
                if train_config.use_wandb:
                    accelerator.log({"reconstructions/grid": wandb.Image(recon_path)}, step=global_step)

            if train_config.save_model and should_save_weight_epoch:
                weight_path = os.path.join(weights_dir, f"ema_epoch_{i:04d}.pth")
                accelerator.save(
                    {
                        "model_ema": ema_model.state_dict(),
                        "epoch": i,
                        "global_step": global_step,
                    },
                    weight_path,
                )

        stop_tensor = torch.tensor([1 if should_stop else 0], device=accelerator.device)
        if accelerator.gather(stop_tensor).max().item() > 0:
            break

    if accelerator.is_main_process:
        if train_config.save_model:
            accelerator.save(
                build_full_state_dict(
                    model_ema=ema_model,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=i,
                    best_epoch_loss=best_epoch_loss,
                ),
                train_config.model_name,
            )
        write_training_summary(metrics_path, output_dir, best_model_path if os.path.exists(best_model_path) else None)
        if train_config.plot_metrics_at_end:
            plot_training_metrics(metrics_path, output_dir)

    accelerator.end_training()


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
