import argparse
import os
from pathlib import Path
from typing import Union

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from ms_model.data.evimo_dataset import EvimoSegDataset
from ms_model.models import build_model
from ms_model.training.losses import build_loss


def build_dataset(paths: list[str], data_cfg: dict, root: Path) -> EvimoSegDataset:
    return EvimoSegDataset(
        npz_paths=[root / p for p in paths],
        seq_len=data_cfg["seq_len"],
        nb_time_bins=data_cfg["nb_time_bins"],
        patch_size=data_cfg["patch_size"],
        mask_thicken_radius=data_cfg.get("mask_thicken_radius", 0),
        mask_motion=data_cfg.get("mask_motion", "all"),
        motion_thresh_m=data_cfg.get("motion_thresh_m", 0.02),
        motion_window_s=data_cfg.get("motion_window_s", 0.2),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    for voxel_seq, mask_seq in loader:
        voxel_seq = voxel_seq.to(device)
        mask_seq = mask_seq.to(device).unsqueeze(2)  # (B,T,1,Hp,Wp)

        with torch.set_grad_enabled(train_mode):
            logits = model(voxel_seq)
            loss = loss_fn(logits, mask_seq)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            pred = (logits.detach() > 0).float()
            gt   = (mask_seq > 0.5).float()
            intersection = (pred * gt).sum()
            union        = (pred + gt).clamp(max=1).sum()
            total_iou   += (intersection / union.clamp(min=1)).item()

        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / n_batches, "iou": total_iou / n_batches}


def load_resume_checkpoint(checkpoint_dir: Path) -> Union[dict, None]:
    """Charge last.pt si present et au format de reprise (dict avec model/optimizer/epoch/...).
    Retourne None si rien a reprendre (pas de fichier, ou ancien format sans ces infos)."""
    last_ckpt_path = checkpoint_dir / "last.pt"
    if not last_ckpt_path.exists():
        return None

    ckpt = torch.load(last_ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "epoch" in ckpt and "optimizer" in ckpt):
        print(f"[resume] {last_ckpt_path} existe mais dans un ancien format (pas de reprise possible).")
        return None
    return ckpt


def confirm_overwrite_dir(checkpoint_dir: Path) -> None:
    """Si le dossier de checkpoints existe deja (et n'est pas repris), demande confirmation."""
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        try:
            answer = input(f"Le dossier {checkpoint_dir} existe deja et contient des fichiers "
                            f"(non repris). Continuer et ecraser ? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes", "o", "oui"):
            raise SystemExit("Run annule.")


def set_seed(seed: int) -> None:
    """Seed torch / numpy / random, pour que `--seeds 1-5` produise 5 runs distincts."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str, data_root_override: str = None, wandb_name_override: str = None,
         run_dir: str = None, seed: int = None):
    config = yaml.safe_load(Path(config_path).read_text())
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]
    data_cfg = config["data"]

    if data_root_override:
        data_cfg["root"] = data_root_override
        print(f"[data] root override: {data_root_override}", flush=True)

    if seed is not None:
        set_seed(seed)
        print(f"[seed] torch/numpy/random seedés à {seed}", flush=True)

    # Avec --run-dir (jobs/train_ms.sh), les checkpoints vivent dans le dossier du run et
    # ne dépendent plus du nom wandb : c'est ce qui rend la reprise fiable même en mode
    # offline ou disabled (cf. INTEGRITY_CHECK.md, point 5).
    fixed_name = wandb_name_override or wandb_cfg.get("name") or None
    if run_dir:
        checkpoint_dir = Path(run_dir) / "checkpoints"
        if seed is not None and not fixed_name:
            fixed_name = f"{config.get('exp', Path(config_path).stem)}_s{seed}"
    else:
        checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / (fixed_name or "")

    resume_ckpt = None
    if run_dir or fixed_name:
        resume_ckpt = load_resume_checkpoint(checkpoint_dir)
        if resume_ckpt is not None:
            print(f"[resume] reprise depuis {checkpoint_dir} a l'epoch "
                  f"{resume_ckpt['epoch'] + 1} (best_val_loss={resume_ckpt['best_val_loss']:.4f})", flush=True)
        elif not run_dir:
            confirm_overwrite_dir(checkpoint_dir)

    root = Path(data_cfg.get("root", "."))

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")

    print("== Chargement du dataset d'entrainement ==", flush=True)
    train_set = build_dataset(data_cfg["train_sequences"], data_cfg, root)
    print("== Chargement du dataset de validation ==", flush=True)
    val_set = build_dataset(data_cfg["val_sequences"], data_cfg, root)

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True,
                               num_workers=train_cfg.get("num_workers", 0))
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg.get("num_workers", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        model_name,
        in_channels=data_cfg["nb_time_bins"],
        patch_size=data_cfg["patch_size"],
        **model_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    # ⚠️ Construit mais JAMAIS appelé : dans le code du stage, `scheduler.step()` ne se
    # trouvait que dans la branche EV-IMO (supprimée), donc tous les runs MS ConvLSTM ont
    # tourné à LR constant malgré `use_scheduler: true`. Comportement laissé tel quel pour
    # ne pas changer la référence en douce ; à trancher avec les hyperparamètres de T7.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    ) if train_cfg.get("use_scheduler", False) else None
    loss_fn = build_loss(
        train_cfg.get("loss", "bce"),
        pos_weight=train_cfg.get("loss_pos_weight"),
        focal_gamma=train_cfg.get("loss_focal_gamma", 2.0),
        bce_weight=train_cfg.get("loss_bce_weight", 0.5),
    ).to(device)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_iou = 0.0
    wandb_run_id = None
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_val_loss"]
        best_val_iou = resume_ckpt.get("best_val_iou", 0.0)
        wandb_run_id = resume_ckpt.get("wandb_run_id")

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        name=fixed_name,
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
        config=config,
    )

    if not run_dir:
        # Sans --run-dir on garde le comportement historique (dossier = nom du run wandb).
        checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / run.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if run_dir:
        # `url` n'existe pas en mode offline selon la version de wandb : on n'échoue pas
        # pour si peu, l'id suffit à retrouver le run après un `wandb sync`.
        try:
            url = run.url or "(hors ligne)"
        except Exception:
            url = "(hors ligne)"
        (Path(run_dir) / "wandb.txt").write_text(
            f"id: {run.id}\nname: {run.name}\nmode: {os.environ.get('WANDB_MODE', 'online')}\n"
            f"url: {url}\n")

    if start_epoch >= train_cfg["epochs"]:
        print(f"[resume] entrainement deja termine ({start_epoch}/{train_cfg['epochs']} epochs), rien a faire.")
        wandb.finish()
        return

    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics   = run_epoch(model, val_loader, loss_fn, device)

        wandb.log({
            "epoch":      epoch,
            "train/loss": train_metrics["loss"],
            "train/iou":  train_metrics["iou"],
            "val/loss":   val_metrics["loss"],
            "val/iou":    val_metrics["iou"],
        })
        print(f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['iou']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f}", flush=True)

        val_loss = val_metrics["loss"]
        val_iou = val_metrics["iou"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": config,
            }, checkpoint_dir / "best.pt")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "model": model.state_dict(),
                "config": config,
            }, checkpoint_dir / "best_iou.pt")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_iou": best_val_iou,
            "wandb_run_id": run.id,
            "config": config,
        }, checkpoint_dir / "last.pt")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None,
                        help="Ecrase data.root du yaml (utile sur cluster)")
    parser.add_argument("--wandb-name", default=None,
                        help="Ecrase wandb.name du yaml")
    parser.add_argument("--run-dir", default=None,
                        help="Dossier du run (jobs/train_ms.sh) : les checkpoints vont dans "
                             "<run-dir>/checkpoints, indépendamment du nom wandb")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch/numpy/random")
    args = parser.parse_args()
    main(args.config, data_root_override=args.data_root, wandb_name_override=args.wandb_name,
         run_dir=args.run_dir, seed=args.seed)
