"""M2 — entraînement COUPLÉ DEVO + modèle de motion segmentation (voir PLAN.md).

Pipeline par pas :
    images (voxels) --MS--> masque dynamique dense (B,n,Hp,Wp)
                     --eVONet(dyn_mask=...)--> poses estimées (BA pondérée par 1-masque)
    loss = pose + flow (+ scores DEVO) [+ λ·mask si GT dynamique fourni]
    loss.backward() -> le gradient de la pose-loss remonte JUSQU'AU modèle de masque.

Stratégie (cf. PLAN.md §6) : DEVO gelé d'abord (seul le masque apprend à ne pas casser la
VO), puis dégel progressif pour affiner conjointement. Sauvegarde SÉPARÉE des poids DEVO et
MS (+ un checkpoint combiné pour reprise).

Ce script réutilise le data reader `tartan_evs` de DEVO pour être immédiatement runnable et
valider le flux de gradient. Le vrai gain M2 (« couplé > découplé ») se mesure en pointant
`--datapath` sur des séquences DYNAMIQUES avec pose GT (EVIMO) — et, pour la variante
supervisée, en fournissant un masque GT par le dataset (voir `--help` / TODO gt_dyn_mask).

Mono-GPU (fine-tuning). Lancer depuis la racine du repo, env conda `madevo`.
"""
import os
import random
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from devo.data_readers.factory import dataset_factory
from utils.wandb_logger import RunLogger, mask_image, patches_image, voxel_image
from devo.lietorch import SE3
from devo.enet import eVONet, _sample_patch_scores
from devo.selector import SelectionMethod


def seed_everything(seed):
    """Seed torch/numpy/python + retourne un worker_init_fn qui seed chaque worker DataLoader."""
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    def _wif(worker_id):
        s = seed + worker_id
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
    return _wif


def kabsch_umeyama(A, B):
    """Échelle optimale (SIM3) minimisant le RMSD — copie de train.py (pour la pose-loss)."""
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1) ** 2)
    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)
    c = VarA / torch.trace(torch.diag(D))
    return c


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def load_devo_weights(net, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    if 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        sd = OrderedDict((k.replace('module.', ''), v) for k, v in ckpt.items()
                         if 'update.lmbda' not in k)
        net.load_state_dict(sd, strict=False)


def build_ms_model(ms_weights, ms_config, device):
    """Charge ou initialise le modèle de motion segmentation (convlstm, …).

    ms_weights=None → initialisation aléatoire (scratch training).
    """
    import yaml
    from ms_model.models import build_model
    cfg = yaml.safe_load(open(ms_config))
    data_cfg = cfg["data"]
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(
        model_name,
        in_channels=data_cfg["nb_time_bins"],
        patch_size=data_cfg["patch_size"],
        **model_cfg,
    ).to(device)
    if ms_weights is not None:
        import torch
        ckpt = torch.load(ms_weights, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        print(f"[ms_model] poids chargés depuis {ms_weights}")
    else:
        print("[ms_model] initialisation ALÉATOIRE (scratch training)")
    return model, data_cfg


def compute_pose_flow_loss(traj, args, so):
    """Loss pose + flow + scores — reprise fidèle de la boucle de train.py (mono-GPU)."""
    loss = 0.0
    pose_loss_val = 0.0
    flow_loss_val = 0.0
    for i, data in enumerate(traj):
        (v, x, y, P1, P2, kl, scores, v_full, x_full, y_full, ba_weights, kk, dij) = data
        valid = (v > 0.5).reshape(-1)
        e = (x - y).norm(dim=-1)
        ef = e.reshape(-1, args.P ** 2)[valid].min(dim=-1).values
        flow_loss = ef.mean()

        scores_loss = torch.as_tensor(0.0, device=e.device)
        if i == (len(traj) - 1):
            valid_full = (v_full >= 0.5).reshape(-1)
            kkf = kk[valid_full]
            e_full = (x_full - y_full).norm(dim=-1)
            e_full = e_full.reshape(-1, args.P ** 2)[valid_full].min(dim=-1).values
            scores_loss = ((-0.5 * (ba_weights.view(-1, 2)[valid_full].mean(dim=-1)).log() + 1)
                           * scores.view(-1)[kkf] * e_full).mean()
            s = torch.max(scores, torch.as_tensor(1e-6, device=scores.device))
            scores_loss = scores_loss + (-s.log()).mean()

        N = P1.shape[1]
        ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        ii = ii.reshape(-1).to(e.device); jj = jj.reshape(-1).to(e.device)
        k = ii != jj; ii = ii[k]; jj = jj[k]

        P1 = P1.inv(); P2 = P2.inv()
        t1 = P1.matrix()[..., :3, 3]; t2 = P2.matrix()[..., :3, 3]
        s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
        P1 = P1.scale(s.view(1, 1))
        dP = P1[:, ii].inv() * P1[:, jj]
        dG = P2[:, ii].inv() * P2[:, jj]
        e1 = (dP * dG.inv()).log()
        tr = e1[..., 0:3].norm(dim=-1); ro = e1[..., 3:6].norm(dim=-1)
        pose_loss = tr.mean() + ro.mean()

        loss = loss + args.flow_weight * flow_loss + args.scores_weight * scores_loss
        if not so and i >= 2:
            loss = loss + args.pose_weight * pose_loss
        pose_loss_val = float(pose_loss); flow_loss_val = float(flow_loss)
    return loss, pose_loss_val, flow_loss_val


def save_coupled(outdir, net, ms, optimizer, steps, tag):
    os.makedirs(outdir, exist_ok=True)
    torch.save({'model_state_dict': net.state_dict()}, os.path.join(outdir, f"devo_{tag}.pth"))
    torch.save({'model': ms.state_dict()}, os.path.join(outdir, f"ms_{tag}.pt"))
    torch.save({'devo': net.state_dict(), 'ms': ms.state_dict(),
                'optimizer': optimizer.state_dict(), 'steps': steps},
               os.path.join(outdir, f"coupled_{tag}.pt"))
    print(f"[save] {outdir}/(devo|ms|coupled)_{tag} @ step {steps}")


def _patch_scores(dyn_mask, patches, ix):
    """Score dynamique par patch sélectionné, détaché — None si indisponible."""
    if patches is None or ix is None:
        return None
    with torch.no_grad():
        return _sample_patch_scores(dyn_mask, patches, ix).view(-1).detach()


def log_coupled_step(logger, step, *, loss, pose_l, flow_l, mask_l, soft_prior,
                     dyn_mask, gt_dyn, unfrozen, dyn_patch_scores, selfsup_stats):
    """Métriques d'un pas couplé, préfixées par domaine.

    `ba/omega_*` : ω est le poids que la BA applique à une arête, soit (1 - score
    dynamique). On le sépare en « statique » et « dynamique » selon le GT quand il existe
    (M2) et selon le seuil 0,5 sinon : c'est la mesure qui dit si le masque écarte vraiment
    les patches sur objets mobiles, et non s'il baisse tout le monde uniformément.
    """
    metrics = {
        "devo/loss":        float(loss),
        "devo/pose_loss":   float(pose_l),
        "devo/flow_loss":   float(flow_l),
        "devo/mask_loss":   float(mask_l),
        "devo/soft_prior":  float(soft_prior),
        "coupling/mask_mean": float(dyn_mask.mean()),
        "coupling/mask_std":  float(dyn_mask.std()),
        "coupling/devo_unfrozen": int(unfrozen),
    }
    for key, value in selfsup_stats.items():
        metrics[f"coupling/selfsup_{key}"] = value

    if dyn_patch_scores is not None and dyn_patch_scores.numel():
        scores = dyn_patch_scores
        omega = (1.0 - scores).clamp(0.0, 1.0)
        rejected = scores > 0.5
        metrics["coupling/patches_rejected_frac"] = float(rejected.float().mean())
        metrics["coupling/patches_total"] = int(scores.numel())
        metrics["ba/omega_mean"] = float(omega.mean())
        metrics["ba/omega_min"] = float(omega.min())

        is_dynamic = rejected
        if gt_dyn is not None:
            # Avec le GT on sépare sur la vérité plutôt que sur la prédiction.
            with torch.no_grad():
                gt = gt_dyn
                if gt.shape[-2:] != dyn_mask.shape[-2:]:
                    gt = F.interpolate(gt, size=dyn_mask.shape[-2:], mode="nearest")
                metrics["coupling/gt_pos_frac"] = float((gt > 0.5).float().mean())
                pred_bin = (dyn_mask > 0.5).float()
                gt_bin = (gt > 0.5).float()
                inter = (pred_bin * gt_bin).sum()
                union = (pred_bin + gt_bin).clamp(max=1).sum()
                metrics["coupling/iou_vs_gt"] = float(inter / union.clamp(min=1))
        if is_dynamic.any():
            metrics["ba/omega_mean_dynamic"] = float(omega[is_dynamic].mean())
        if (~is_dynamic).any():
            metrics["ba/omega_mean_static"] = float(omega[~is_dynamic].mean())

    logger.log(metrics, step=step)
    if dyn_patch_scores is not None and dyn_patch_scores.numel():
        logger.log_histogram("ba/omega_hist", 1.0 - dyn_patch_scores, step=step)
    logger.log_histogram("coupling/mask_hist", dyn_mask, step=step)


def log_coupled_images(logger, step, images, dyn_mask, gt_dyn, dyn_patch_scores, patches):
    """Voxels, masque prédit, masque GT et patches sélectionnés, à la même étape."""
    frame = 0
    panels = {
        "voxels":      voxel_image(images[0, frame]),
        "masque_pred": mask_image(dyn_mask[0, frame]),
    }
    if gt_dyn is not None:
        panels["masque_gt"] = mask_image(gt_dyn[0, frame])
    if patches is not None and dyn_patch_scores is not None:
        p = patches.shape[-1]
        xy = patches[0, :, :2, p // 2, p // 2].detach().cpu()        # (K, 2) en unités Hp/Wp
        panels["patches"] = patches_image(dyn_mask[0, frame], xy, dyn_patch_scores, radius=1)
    logger.log_images(panels, step=step, prefix="coupling/")


def build_dataset(args):
    """Construit le dataset d'entraînement (tartan_evs statique, ou EVIMO dynamique pour M2)."""
    if args.dataset == "evimo":
        import yaml
        from ms_model.data.evimo_clip_dataset import EvimoClipDataset
        seqs = args.evimo_seqs
        if not seqs:  # à défaut, réutilise le split train du yaml MS
            data_cfg = yaml.safe_load(open(args.ms_config))["data"]
            seqs = data_cfg["train_sequences"]
        # Les séquences du yaml MS sont relatives à la racine de la caméra, p. ex.
        # « imo/train/scene10_dyn_train_00_000000 » : --datapath doit donc valoir
        # datasets/evimo2/samsung_mono, et non datasets/evimo2 (erreur des anciens
        # scripts Slurm, qui cherchaient les séquences un niveau trop haut).
        npz_paths = [os.path.join(args.datapath, s) for s in seqs]
        missing = [p for p in npz_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(npz_paths)} séquences EVIMO introuvables sous "
                f"--datapath={args.datapath} (p. ex. {missing[0]}). Pour EVIMO2, la racine "
                f"est datasets/evimo2/samsung_mono."
            )
        return EvimoClipDataset(
            npz_paths, n_frames=args.n_frames, nb_time_bins=args.nb_time_bins,
            patch_size=args.patch_size, depth_scale=args.depth_scale,
            provide_gt_mask=args.provide_gt_mask, mask_thicken_radius=args.mask_thicken_radius,
            mask_motion=args.mask_motion, motion_thresh_m=args.motion_thresh_m,
            motion_window_s=args.motion_window_s)
    return dataset_factory(['tartan_evs'], datapath=args.datapath, n_frames=args.n_frames,
                           fgraph_pickle=args.fgraph_pickle, train_split=args.train_split,
                           val_split=args.val_split, strict_split=False, sample=True,
                           return_fname=True, scale=args.scale)


def train_coupled(args):
    device = torch.device("cuda")

    worker_init = seed_everything(args.seed)
    if args.seed is not None:
        print(f"[seed] torch/numpy/random/cuda seedés à {args.seed}, DataLoader workers seedés en cascade", flush=True)

    db = build_dataset(args)
    g = torch.Generator(); g.manual_seed(args.seed) if args.seed is not None else None
    loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=4,
                        worker_init_fn=worker_init,
                        generator=g if args.seed is not None else None)

    net = eVONet(dim_inet=args.dim_inet, dim_fnet=args.dim_fnet, dim=args.dim,
                 patch_selector=args.patch_selector.lower(), norm=args.norm)
    load_devo_weights(net, args.devo_weights)
    net.train().to(device)
    args.P = net.P

    ms, _ = build_ms_model(args.ms_weights, args.ms_config, device)
    ms.train()

    # Phase 1 : DEVO gelé, seul le masque apprend.
    set_requires_grad(net, False)
    unfrozen = False
    optimizer = torch.optim.AdamW(ms.parameters(), lr=args.ms_lr, weight_decay=1e-4)

    # Les diagnostics par patch (ω, seuil selfsup) ne sont calculés par eVONet que pour le
    # sélecteur « scorer ».
    want_diag = args.patch_selector.lower() == "scorer"

    logger = RunLogger(
        "train_coupled", vars(args),
        exp=args.exp or os.path.basename(os.path.normpath(args.outdir)),
        seed=args.seed, run_dir=args.run_dir or None,
        launch_id=os.environ.get("LAUNCH_ID"),
        resume_id=args.wandb_id or None,
        images_every=args.images_every,
    )

    total_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['devo'])
        ms.load_state_dict(ckpt['ms'])
        total_steps = int(ckpt['steps'])
        # L'optimiseur n'a pas les mêmes groupes de paramètres avant et après le dégel :
        # il faut reconstruire le bon AVANT de recharger son état.
        if total_steps >= args.freeze_devo_steps:
            set_requires_grad(net, True)
            optimizer = torch.optim.AdamW(
                [{"params": ms.parameters(), "lr": args.ms_lr},
                 {"params": net.parameters(), "lr": args.devo_lr}], weight_decay=1e-4)
            unfrozen = True
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"[resume] reprise de {args.resume} au step {total_steps}"
              f"{' (DEVO déjà dégelé)' if unfrozen else ''}", flush=True)
        if total_steps >= args.steps:
            print(f"[resume] déjà terminé ({total_steps}/{args.steps} steps), rien à faire.")
            return
    while total_steps < args.steps:
        for blob in loader:
            blob.pop()  # scene_id (str)
            gt_dyn = blob.pop().to(device).float() if args.provide_gt_mask else None  # (B,n,Hp,Wp)
            images, poses, disps, intrinsics = [x.to(device).float() for x in blob]

            if (not unfrozen) and total_steps >= args.freeze_devo_steps:
                set_requires_grad(net, True)
                optimizer = torch.optim.AdamW(
                    [{"params": ms.parameters(), "lr": args.ms_lr},
                     {"params": net.parameters(), "lr": args.devo_lr}], weight_decay=1e-4)
                unfrozen = True
                print(f"[schedule] DEVO dégelé au step {total_steps} (lr={args.devo_lr})")

            optimizer.zero_grad()
            poses_w2c = SE3(poses).inv()

            # masque dynamique dense (B, n, Hp, Wp) dans [0,1], différentiable
            dyn_logits = ms(images)                       # (B, n, 1, Hp, Wp)
            dyn_mask = torch.sigmoid(dyn_logits)[:, :, 0]  # (B, n, Hp, Wp)

            # `return_selfsup` n'ajoute que des diagnostics DÉTACHÉS (résidu par patch,
            # patches, index de frame) : ni la loss ni le graphe ne changent. On les
            # demande donc aussi en M2, pour pouvoir loguer ω et le seuil médiane + k·MAD
            # dans les deux jalons.
            res_patch = patches_out = ix_out = None
            if args.selfsup or want_diag:
                traj, res_patch, patches_out, ix_out = net(
                    images, poses_w2c, disps, intrinsics, M=1024, STEPS=args.iters,
                    structure_only=False, patches_per_image=args.patches_per_image,
                    dyn_mask=dyn_mask, return_selfsup=True)
            else:
                traj = net(images, poses_w2c, disps, intrinsics, M=1024, STEPS=args.iters,
                           structure_only=False, patches_per_image=args.patches_per_image,
                           dyn_mask=dyn_mask)

            loss, pose_l, flow_l = compute_pose_flow_loss(traj, args, so=False)

            mask_l = 0.0
            selfsup_stats = {}
            if args.selfsup:
                # M3 auto-supervisé : cible = patches à résidu anormalement haut (incohérents
                # au mouvement rigide) = dynamiques. AUCUN GT. Seuil robuste médiane + k·MAD.
                dyn_patch = _sample_patch_scores(dyn_mask, patches_out, ix_out).view(-1)  # (K,) diff.
                res = res_patch.view(-1)                                                   # (K,) détaché
                med = res.median()
                mad = (res - med).abs().median() + 1e-6
                selfsup_stats = {"median": float(med), "mad": float(mad),
                                 "threshold": float(med + args.selfsup_k * mad)}
                target = (res > (med + args.selfsup_k * mad)).float().detach()
                selfsup_stats["pos_frac"] = float(target.mean())
                mask_l = F.binary_cross_entropy(dyn_patch.clamp(1e-6, 1 - 1e-6), target)
                loss = loss + args.mask_weight * mask_l
            elif gt_dyn is not None and args.mask_weight > 0:
                # M2 supervisé : garde le masque calibré vs GT dynamique
                gt = gt_dyn
                if gt.shape[-2:] != dyn_mask.shape[-2:]:
                    gt = F.interpolate(gt, size=dyn_mask.shape[-2:], mode="nearest")
                mask_l = F.binary_cross_entropy(dyn_mask.clamp(1e-6, 1 - 1e-6), gt)
                loss = loss + args.mask_weight * mask_l
            # régularisation optionnelle : évite un masque qui sature
            if args.mask_l1 > 0:
                loss = loss + args.mask_l1 * dyn_mask.mean()
            # prior doux : pénalise mask_mean trop loin de la cible (évite collapse 0 ou 1)
            soft_prior = 0.0
            if args.soft_prior_beta > 0:
                soft_prior = args.soft_prior_beta * (dyn_mask.mean() - args.soft_prior_target) ** 2
                loss = loss + soft_prior

            if torch.isnan(loss):
                print(f"[warn] nan loss @ {total_steps}, skip"); optimizer.zero_grad(); total_steps += 1; continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(ms.parameters()) + list(net.parameters()) if p.requires_grad], args.clip)
            optimizer.step()

            if total_steps % args.log_every == 0:
                print(f"step {total_steps:6d} | loss {float(loss):.4f} | pose {pose_l:.4f} "
                      f"| flow {flow_l:.4f} | mask_l {float(mask_l):.4f} "
                      f"| mask_mean {float(dyn_mask.mean()):.4f} "
                      f"| {'joint' if unfrozen else 'devo-frozen'}")
                log_coupled_step(logger, total_steps, loss=loss, pose_l=pose_l,
                                 flow_l=flow_l, mask_l=mask_l, soft_prior=soft_prior,
                                 dyn_mask=dyn_mask, gt_dyn=gt_dyn, unfrozen=unfrozen,
                                 dyn_patch_scores=_patch_scores(dyn_mask, patches_out, ix_out),
                                 selfsup_stats=selfsup_stats)
            if logger.should_log_images(total_steps):
                log_coupled_images(logger, total_steps, images, dyn_mask, gt_dyn,
                                   _patch_scores(dyn_mask, patches_out, ix_out),
                                   patches_out)
            if total_steps > 0 and total_steps % args.save_every == 0:
                save_coupled(args.outdir, net, ms, optimizer, total_steps, tag=f"step{total_steps}")

            total_steps += 1
            if total_steps >= args.steps:
                break

    save_coupled(args.outdir, net, ms, optimizer, total_steps, tag="final")
    logger.summary({"coupling/final_step": total_steps})
    logger.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # modèles
    p.add_argument('--devo_weights', required=True, help='checkpoint DEVO pré-entraîné (.pth)')
    p.add_argument('--ms_weights', default=None, help='checkpoint modèle MS (best.pt) — None = scratch')
    p.add_argument('--ms_config', required=True, help='yaml entraînement MS')
    p.add_argument('--outdir', default='results_coupled/m2')
    p.add_argument('--exp', default=None,
                   help='nom d\'expérience pour wandb (group) ; défaut : nom du dossier de sortie')
    p.add_argument('--run-dir', dest='run_dir', default=None,
                   help='dossier du run, où écrire wandb.txt (posé par jobs/train_coupled.sh)')
    p.add_argument('--wandb_id', default=None,
                   help='id du run wandb à reprendre (reprise d\'un run interrompu)')
    p.add_argument('--images_every', type=int, default=500,
                   help='envoie voxels / masques / patches sur wandb tous les N steps (0 = jamais)')
    p.add_argument('--resume', default=None,
                   help='coupled_step<K>.pt depuis lequel reprendre (modèles + optimiseur + '
                        'compteur de steps) ; jobs/train_coupled.sh le remplit tout seul')
    # données
    p.add_argument('--dataset', default="tartan_evs", choices=["tartan_evs", "evimo"],
                   help='tartan_evs (statique, valide la plomberie) ou evimo (dynamique, run M2 réel)')
    p.add_argument('--datapath', required=True,
                   help='racine des données : tartan, ou racine de la caméra EVIMO '
                        '(EVIMO2 : datasets/evimo2/samsung_mono)')
    p.add_argument('--n_frames', type=int, default=15)
    p.add_argument('--fgraph_pickle', default="")
    p.add_argument('--train_split', default="")
    p.add_argument('--val_split', default="")
    p.add_argument('--scale', type=float, default=1.0)
    # spécifiques EVIMO
    p.add_argument('--evimo_seqs', nargs='*', default=None,
                   help='npz relatifs à --datapath (défaut: train_sequences du yaml MS)')
    p.add_argument('--nb_time_bins', type=int, default=5)
    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--depth_scale', type=float, default=1000.0, help='depth EVIMO en mm -> m')
    p.add_argument('--provide_gt_mask', action='store_true', help='loss masque supervisée (M2 sup.)')
    p.add_argument('--mask_thicken_radius', type=int, default=3)
    p.add_argument('--mask_motion', default="all", choices=["all", "sequence", "frame"],
                   help="masque GT = tous les objets annotés (all) ou seulement ceux qui bougent")
    p.add_argument('--motion_thresh_m', type=float, default=0.02,
                   help="seuil de mouvement en mètres (modes sequence/frame)")
    p.add_argument('--motion_window_s', type=float, default=0.2,
                   help="fenêtre temporelle du mode frame, en secondes")
    p.add_argument('--mask_weight', type=float, default=1.0, help='poids loss masque (supervisée ou auto-sup.)')
    # M3 auto-supervisé (sans GT) : masque supervisé par le résidu de la DBA
    p.add_argument('--selfsup', action='store_true',
                   help='M3 : supervise le masque par le résidu DBA (incohérence rigide), sans GT')
    p.add_argument('--selfsup_k', type=float, default=3.0,
                   help='seuil robuste médiane + k·MAD sur le résidu pour étiqueter dynamique')
    # archi DEVO
    p.add_argument('--dim_inet', type=int, default=384)
    p.add_argument('--dim_fnet', type=int, default=128)
    p.add_argument('--dim', type=int, default=32)
    p.add_argument('--patch_selector', default="scorer")
    p.add_argument('--norm', default="std2")
    p.add_argument('--patches_per_image', type=int, default=80)
    p.add_argument('--iters', type=int, default=18)
    # optim / schedule
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--freeze_devo_steps', type=int, default=5000, help='pas avant dégel de DEVO')
    p.add_argument('--ms_lr', type=float, default=1e-4)
    p.add_argument('--devo_lr', type=float, default=2e-5, help='lr DEVO après dégel (faible)')
    p.add_argument('--clip', type=float, default=10.0)
    p.add_argument('--mask_l1', type=float, default=0.0, help='régul. sparsité du masque')
    p.add_argument('--soft_prior_beta', type=float, default=0.0,
                   help='prior doux anti-collapse : β·(mask_mean − target)² (recommandé: 0.5)')
    p.add_argument('--soft_prior_target', type=float, default=0.1,
                   help='cible de mask_mean pour le prior doux (défaut: 0.1)')
    p.add_argument('--pose_weight', type=float, default=10.0)
    p.add_argument('--flow_weight', type=float, default=0.1)
    p.add_argument('--scores_weight', type=float, default=0.05)
    # logging
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every', type=int, default=2000)
    # reproductibilité
    p.add_argument('--seed', type=int, default=None,
                   help='seed torch/numpy/random/DataLoader (défaut None = pas de seeding explicite)')
    args = p.parse_args()

    train_coupled(args)
