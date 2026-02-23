import os
import sys
import random
import datetime
import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler
import wandb
import yaml
from rich.console import Console
from rich.text import Text
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5Dataset
from ModelUniPhy import UniPhyModel

import warnings

warnings.filterwarnings("ignore")


class SpeedColumn(ProgressColumn):
    def render(self, task):
        if task.speed is None:
            return Text("0.00 it/s", style="progress.data.speed")
        return Text(f"{task.speed:.2f} it/s", style="progress.data.speed")


def setup_logging(log_path, rank):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_path, "train_metrics.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lat_weights(H, W, device):
    lat = torch.linspace(-90, 90, H, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, H, 1)


def validate_model_grid_cfg(cfg):
    model_cfg = cfg["model"]
    in_ch = int(model_cfg["in_channels"])
    out_ch = int(model_cfg["out_channels"])
    embed_dim = int(model_cfg["embed_dim"])
    depth = int(model_cfg["depth"])
    H = int(model_cfg["img_height"])
    W = int(model_cfg["img_width"])
    ph, pw = map(int, model_cfg["patch_size"])

    if in_ch != out_ch:
        raise ValueError(
            f"in_channels({in_ch}) 与 out_channels({out_ch}) 必须一致"
        )
    if in_ch < 70:
        raise ValueError(
            f"当前通道数为 {in_ch}，建议 >=70（13 层高空配置通常至少 70-84 通道）"
        )
    if H < 181 or W < 360:
        raise ValueError(
            f"网格过小：H={H}, W={W}，建议至少 H>=181, W>=360"
        )
    if embed_dim < 256:
        raise ValueError(f"embed_dim={embed_dim} 过小，建议 >=256")
    if depth < 6:
        raise ValueError(f"depth={depth} 过小，建议 >=6")

    h_patches = (H + ph - 1) // ph
    w_patches = (W + pw - 1) // pw
    if h_patches < 24 or w_patches < 48:
        raise ValueError(
            f"patch 后网格过小：h_patches={h_patches}, w_patches={w_patches}，"
            "请减小 patch_size 或增大图像分辨率"
        )


def compute_crps(pred_ensemble, target):
    M = pred_ensemble.shape[0]
    target_exp = target.unsqueeze(0)
    mae = (pred_ensemble - target_exp).abs().mean()
    if M > 1:
        total_diff = torch.tensor(0.0, device=target.device)
        for i in range(M):
            for j in range(i + 1, M):
                total_diff = total_diff + (
                    pred_ensemble[i] - pred_ensemble[j]
                ).abs().mean()
        loss = mae - total_diff / (M * M * 2)
    else:
        loss = mae
    return loss


def train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx,
               lat_weights):
    device = next(model.parameters()).device
    data, dt_data = batch
    data = data.to(device, non_blocking=True).float()
    dt_data = dt_data.to(device, non_blocking=True).float()

    B, T, C, H, W = data.shape
    exp_c = int(cfg["model"]["in_channels"])
    exp_h = int(cfg["model"]["img_height"])
    exp_w = int(cfg["model"]["img_width"])
    if C != exp_c:
        raise ValueError(f"训练数据通道数 C={C}，但模型配置为 C={exp_c}")
    if H != exp_h or W != exp_w:
        raise ValueError(
            f"训练数据网格为 HxW={H}x{W}，但模型配置为 {exp_h}x{exp_w}"
        )
    ensemble_size = cfg["model"]["ensemble_size"]

    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_input = dt_data[:, 1:]

    if ensemble_size > 1:
        member_idx = torch.randint(0, ensemble_size, (B,), device=device)
    else:
        member_idx = None

    out, z_latent = model(x_input, dt_input, member_idx=member_idx,
                          return_latent=True)

    out_real = out.real.contiguous() if out.is_complex() else out.contiguous()

    l1_loss = (out_real - x_target).abs().mean()
    mse_loss = ((out_real - x_target) ** 2 * lat_weights).mean()

    if ensemble_size > 1:
        ensemble_preds = [out_real]
        infer_model = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            z_det = z_latent.detach()
            for _ in range(ensemble_size - 1):
                rand_idx = torch.randint(
                    0, ensemble_size, (B,), device=device,
                )
                out_ens = infer_model.decoder(z_det, member_idx=rand_idx)
                ens_real = (
                    out_ens.real.contiguous()
                    if out_ens.is_complex()
                    else out_ens.contiguous()
                )
                ensemble_preds.append(ens_real)
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        crps_loss = compute_crps(ensemble_stack, x_target)
        ensemble_std = ensemble_stack.std(dim=0).mean()
        loss = l1_loss + crps_loss
    else:
        crps_loss = torch.tensor(0.0, device=device)
        ensemble_std = torch.tensor(0.0, device=device)
        loss = l1_loss

    basis_reg = torch.tensor(0.0, device=device)
    base_model = model.module if hasattr(model, "module") else model
    for block in base_model.blocks:
        W, W_inv = block.prop.basis.get_matrix(torch.complex64)
        eye = torch.eye(W.shape[0], device=device, dtype=W.dtype)
        basis_reg = basis_reg + (W @ W_inv - eye).abs().mean()
    loss = loss + 0.01 * basis_reg

    loss_scaled = loss / grad_accum_steps
    loss_scaled.backward()

    grad_norm = 0.0
    if (batch_idx + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"],
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        rmse = torch.sqrt(mse_loss)

    metrics = {
        "loss": loss.item(),
        "l1_loss": l1_loss.item(),
        "crps_loss": crps_loss.item(),
        "mse": mse_loss.item(),
        "rmse": rmse.item(),
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_std.item(),
    }
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg,
                    path):
    state_dict = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )
    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "global_step": global_step,
        "cfg": cfg,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    clean_state = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(clean_state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


def flush_remaining_grads(model, optimizer, cfg, batch_idx, grad_accum_steps):
    if (batch_idx + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"],
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def train(cfg):
    validate_model_grid_cfg(cfg)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    set_seed(42 + rank)

    if rank == 0:
        log_dir = cfg["logging"]["log_path"]
        os.makedirs(log_dir, exist_ok=True)
        console = Console()
    else:
        devnull = open(os.devnull, "w")
        console = Console(file=devnull)

    logger = setup_logging(cfg["logging"]["log_path"], rank)

    if rank == 0:
        logger.info(f"训练启动，使用 {world_size}GPU")

    if cfg["train"]["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if rank == 0 and cfg["logging"]["use_wandb"]:
        run_name = cfg["logging"]["wandb_run_name"] or (
            f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=run_name,
            config=cfg,
        )

    model = UniPhyModel(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        expand=cfg["model"]["expand"],
        depth=cfg["model"]["depth"],
        patch_size=tuple(cfg["model"]["patch_size"]),
        img_height=cfg["model"]["img_height"],
        img_width=cfg["model"]["img_width"],
        dt_ref=cfg["model"]["dt_ref"],
        sde_mode=cfg["model"]["sde_mode"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
        ensemble_size=cfg["model"]["ensemble_size"],
    ).cuda()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_dataset = ERA5Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["data"]["sample_k"],
        look_ahead=cfg["data"]["look_ahead"],
        is_train=True,
        dt_ref=cfg["model"]["dt_ref"],
        sampling_mode=cfg["data"]["sampling_mode"],
        expected_channels=cfg["model"]["in_channels"],
        min_img_height=cfg["model"]["img_height"],
        min_img_width=cfg["model"]["img_width"],
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )

    basis_params = []
    other_params = []
    basis_keys = {"w_re", "w_im", "w_inv_re", "w_inv_im"}
    for name, p in model.named_parameters():
        if any(name.endswith("." + k) for k in basis_keys):
            basis_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "weight_decay": cfg["train"]["weight_decay"]},
        {"params": basis_params, "weight_decay": 0.0},
    ], lr=float(cfg["train"]["lr"]))

    grad_accum_steps = cfg["train"]["grad_accum_steps"]
    epochs = cfg["train"]["epochs"]

    scheduler = None
    if cfg["train"]["use_scheduler"]:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg["train"]["lr"]),
            steps_per_epoch=len(train_loader) // grad_accum_steps,
            epochs=epochs,
        )

    start_epoch = 0
    global_step = 0

    if cfg["logging"]["ckpt"]:
        start_epoch, global_step = load_checkpoint(
            cfg["logging"]["ckpt"], model, optimizer, scheduler,
        )
        if rank == 0:
            logger.info(f"已从检查点恢复： {cfg['logging']['ckpt']}")

    log_every = cfg["logging"]["log_every"]
    save_interval = max(
        1, int(len(train_loader) * cfg["logging"]["ckpt_step"]),
    )

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    H = cfg["model"]["img_height"]
    W = cfg["model"]["img_width"]
    lat_weights = build_lat_weights(H, W, torch.device(f"cuda:{local_rank}"))

    progress = None
    if rank == 0:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            SpeedColumn(),
            console=console,
        )
        progress.start()
        task_id = progress.add_task(
            "训练中",
            total=len(train_loader) * epochs,
            completed=global_step,
        )

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step(
                model, batch, optimizer, cfg, grad_accum_steps, batch_idx,
                lat_weights,
            )
            global_step += 1

            if (
                scheduler is not None
                and (batch_idx + 1) % grad_accum_steps == 0
            ):
                scheduler.step()

            if rank == 0:
                progress.update(task_id, advance=1)

                if (batch_idx + 1) % log_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_msg = (
                        f"[E{epoch + 1:03d} B{batch_idx + 1:04d}] "
                        f"损失: {metrics['loss']:.4f} | "
                        f"L1: {metrics['l1_loss']:.4f} | "
                        f"CRPS: {metrics['crps_loss']:.4f} | "
                        f"RMSE: {metrics['rmse']:.4f} | "
                        f"梯度: {metrics['grad_norm']:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)

                if cfg["logging"]["use_wandb"]:
                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/l1_loss": metrics["l1_loss"],
                            "train/crps_loss": metrics["crps_loss"],
                            "train/mse": metrics["mse"],
                            "train/rmse": metrics["rmse"],
                            "train/ensemble_std": metrics["ensemble_std"],
                            "train/grad_norm": metrics["grad_norm"],
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                if (
                    save_interval > 0
                    and (batch_idx + 1) % save_interval == 0
                ):
                    ckpt_path = os.path.join(
                        cfg["logging"]["ckpt_dir"],
                        f"ckpt_e{epoch + 1}_s{global_step}.pt",
                    )
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, global_step, cfg, ckpt_path,
                    )
                    logger.info(f"已保存检查点： {ckpt_path}")

        flush_remaining_grads(
            model, optimizer, cfg, batch_idx, grad_accum_steps,
        )

        if rank == 0:
            epoch_path = os.path.join(
                cfg["logging"]["ckpt_dir"], f"ckpt_epoch{epoch + 1}.pt",
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, cfg,
                epoch_path,
            )
            logger.info(f"第 {epoch + 1} 轮完成，检查点已保存。")

        dist.barrier()

    if rank == 0:
        final_path = os.path.join(
            cfg["logging"]["ckpt_dir"], "ckpt_final.pt",
        )
        save_checkpoint(
            model, optimizer, scheduler, epochs, global_step, cfg, final_path,
        )
        progress.console.print(
            f"[bold green]训练中 Completed. Final checkpoint: {final_path}"
        )
        if cfg["logging"]["use_wandb"]:
            wandb.finish()
        progress.stop()

    train_dataset.cleanup()
    dist.destroy_process_group()


def main():
    with open("train.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
    
