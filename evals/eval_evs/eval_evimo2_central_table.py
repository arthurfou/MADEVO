"""M4 EVIMO2 — tableau central (vanilla / oracle / M1 / M2 / M3).

Identique à eval_evimo_central_table.py mais importe depuis eval_evimo2_evs
et eval_evimo2_m1_decoupled (résolution 480×640, _resolve_npz EVIMO2).
"""
import os
import csv
import argparse

import torch
from devo.config import cfg
from utils.eval_utils import assert_eval_config, mean_scene_ate
from utils.wandb_logger import RunLogger

from eval_evimo2_evs import evaluate
from eval_evimo2_m1_decoupled import make_oracle_factory, make_learned_factory


def log_to_wandb(args, rows, per_scene):
    """Envoie la table (et les ATE par scène) comme `wandb.Table`."""
    logger = RunLogger("eval", vars(args), exp=args.exp, seed=args.seed,
                       run_dir=args.run_dir or None,
                       launch_id=os.environ.get("LAUNCH_ID"))
    base = next((a for n, a in rows if n == "DEVO vanilla" and a is not None), None)

    table_rows = []
    summary = {}
    for name, ate in rows:
        delta = None if (ate is None or not base) else 100.0 * (base - ate) / base
        table_rows.append([name, ate, delta])
        if ate is not None:
            summary[f"eval/ate/{_slug(name)}"] = ate
            if delta is not None:
                summary[f"eval/delta_pct/{_slug(name)}"] = delta
    logger.log_table("eval/table", ["configuration", "ate_mean", "delta_pct_vs_vanilla"],
                     table_rows)

    scenes = sorted({s for res in per_scene.values() for s in res if s not in ("AUC", "AVG")})
    logger.log_table(
        "eval/ate_par_scene", ["scene"] + list(per_scene),
        [[scene] + [per_scene[m].get(scene) for m in per_scene] for scene in scenes])

    logger.summary(summary)
    logger.finish()


def _slug(name):
    return (name.replace("DEVO + ", "").replace("DEVO ", "")
                .replace(" ", "_").replace("(", "").replace(")", "").lower())


def run_row(name, cfg, args, devo_weights, factory, common):
    print(f"\n=== [table] {name} (DEVO={os.path.basename(devo_weights)}) ===")
    args.expname = "central_" + name.replace(" ", "_").replace("(", "").replace(")", "")
    res, _ = evaluate(cfg, args, devo_weights, dyn_mask_provider_factory=factory, **common)
    return mean_scene_ate(res), res


def write_table(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    md = os.path.join(outdir, "central_table.md")
    csvp = os.path.join(outdir, "central_table.csv")

    base = next((a for n, a in rows if n == "DEVO vanilla" and a is not None), None)
    with open(md, "w") as f:
        f.write("| Configuration | ATE moyen | Δ vs vanilla |\n|---|---|---|\n")
        for name, ate in rows:
            if ate is None:
                f.write(f"| {name} | — | — |\n")
            else:
                delta = "" if base is None else f"{100*(base-ate)/base:+.1f}%"
                f.write(f"| {name} | {ate:.4f} | {delta} |\n")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "ate_mean", "delta_pct_vs_vanilla"])
        for name, ate in rows:
            delta = "" if (ate is None or base is None) else f"{100*(base-ate)/base:.2f}"
            w.writerow([name, "" if ate is None else f"{ate:.6f}", delta])
    print(f"\n[table] écrit -> {md}\n[table] écrit -> {csvp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', default="config/eval_evimo.yaml")
    p.add_argument('--datapath', required=True, help='EVIMO2 eval_preprocessed/')
    p.add_argument('--val_split', default="splits/evimo2/evimo2_val.txt")
    p.add_argument('--mask_root', required=True, help='racine npz EVIMO2')
    p.add_argument('--ms_config', required=True, help='yaml archi MS (convlstm_v4_evimo2.yaml)')
    p.add_argument('--weights', required=True, help='DEVO vanilla (.pth)')
    p.add_argument('--devo_coupled_sup', default=None)
    p.add_argument('--devo_coupled_selfsup', default=None)
    p.add_argument('--ms_decoupled', default=None, help='M1 checkpoint')
    p.add_argument('--ms_coupled_sup', default=None, help='M2 checkpoint')
    p.add_argument('--ms_coupled_selfsup', default=None, help='M3 checkpoint')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--mask_motion', default="all", choices=["all", "sequence", "frame"],
                   help="masque oracle = tous les objets annotés (all) ou seulement ceux qui bougent")
    p.add_argument('--motion_thresh_m', type=float, default=0.02)
    p.add_argument('--motion_window_s', type=float, default=0.2)
    p.add_argument('--thicken_radius', type=int, default=2)
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--outdir', default="results/central_table_evimo2")
    p.add_argument('--expname', default="central_evimo2")
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--exp', default='eval_evimo2',
                   help="nom d'expérience pour wandb (group)")
    p.add_argument('--run-dir', dest='run_dir', default=None,
                   help='dossier du run, où écrire wandb.txt (posé par jobs/eval_table.sh)')
    args = p.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    torch.manual_seed(args.seed)
    import numpy as np, random
    np.random.seed(args.seed); random.seed(args.seed)
    print(f"[central_table_evimo2] seed = {args.seed}", flush=True)
    args.plot = False
    args.save_trajectory = True
    common = dict(datapath=args.datapath, split_file=args.val_split,
                  trials=args.trials, plot=False, save=True, stride=args.stride)

    def learned(ms_weights):
        return make_learned_factory(args.mask_root, ms_weights, args.ms_config, threshold=args.threshold)

    rows = []
    per_scene = {}

    def add(name, devo_weights, factory):
        ate, res = run_row(name, cfg, args, devo_weights, factory, common)
        rows.append((name, ate))
        per_scene[name] = res
        return ate

    add("DEVO vanilla", args.weights, None)
    add("DEVO + oracle GT", args.weights,
        make_oracle_factory(args.mask_root, thicken_radius=args.thicken_radius,
                            mask_motion=args.mask_motion, motion_thresh_m=args.motion_thresh_m,
                            motion_window_s=args.motion_window_s))
    if args.ms_decoupled:
        add("DEVO + appris découplé (M1)", args.weights, learned(args.ms_decoupled))
    else:
        rows.append(("DEVO + appris découplé (M1)", None))
    if args.ms_coupled_sup:
        add("DEVO + couplé supervisé (M2)", args.devo_coupled_sup or args.weights,
            learned(args.ms_coupled_sup))
    else:
        rows.append(("DEVO + couplé supervisé (M2)", None))
    if args.ms_coupled_selfsup:
        add("DEVO + couplé auto-sup (M3)", args.devo_coupled_selfsup or args.weights,
            learned(args.ms_coupled_selfsup))
    else:
        rows.append(("DEVO + couplé auto-sup (M3)", None))

    print("\n==================== TABLEAU CENTRAL EVIMO2 ====================")
    for name, ate in rows:
        print(f"  {name:32s}: {'—' if ate is None else f'{ate:.4f}'}")
    write_table(rows, args.outdir)
    log_to_wandb(args, rows, per_scene)
