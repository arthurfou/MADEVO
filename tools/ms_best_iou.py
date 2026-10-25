"""Meilleure IoU de validation de runs MS, lue dans leur `checkpoints/last.pt`.

Sert au balayage de `loss_pos_weight` (configs/ms/v4_evimo2_holdout.yaml) : une ligne par
run, triée par IoU décroissante.

Usage :
    python tools/ms_best_iou.py runs/v4_evimo2_pw*
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml


def collect(roots):
    rows = []
    for root in map(Path, roots):
        for last in sorted(root.glob("**/checkpoints/last.pt")):
            run_dir = last.parent.parent
            ckpt = torch.load(last, map_location="cpu", weights_only=False)
            cfg_path = run_dir / "config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else ckpt.get("config", {})
            rows.append({
                "run": str(run_dir),
                "pos_weight": cfg.get("training", {}).get("loss_pos_weight"),
                "epochs": f"{ckpt['epoch'] + 1}/{cfg.get('training', {}).get('epochs', '?')}",
                "best_val_iou": float(ckpt.get("best_val_iou", float("nan"))),
                "done": (run_dir / "DONE").exists(),
            })
    return sorted(rows, key=lambda r: -r["best_val_iou"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="dossiers de runs (cherche **/checkpoints/last.pt)")
    rows = collect(p.parse_args(argv).roots)
    if not rows:
        sys.exit("aucun checkpoints/last.pt trouvé")
    print("| run | pos_weight | epochs | meilleure IoU val | fini |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['pos_weight']} | {r['epochs']} | {r['best_val_iou']:.4f} | "
              f"{'oui' if r['done'] else 'non'} |")


if __name__ == "__main__":
    main()
