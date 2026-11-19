"""Statistiques du score « dynamique » prédit par le modèle MS (M1) sur un dataset cross.

Sert à comprendre pourquoi M1 aide ou dégrade une scène : M1 multiplie la score map DEVO par
(1 - score). Si le modèle voit du mouvement partout (score élevé sur une large part de l'image),
DEVO perd ses patches — c'est l'hypothèse testée sur EDS.

Usage :
    python tools/ms_score_stats.py --dataset eds --datapath $MADEVO_DATASETS/eds \
        --ms runs/v4_evimo2/<camp>/seed1/checkpoints/best_iou.pt --arch configs/ms/v4_evimo2.yaml \
        [--scenes a,b] [--device cuda]

Sortie : une ligne par scène — score moyen, part des cellules > 0,5, part des frames où plus de
30 % de l'image dépasse 0,5, et la part moyenne de la « masse » de score map retirée si la score
map était uniforme (= score moyen).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ms_dataset_factory import make_ms_factory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--datapath", required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    scenes = a.scenes.split(",") if a.scenes else sorted(
        d for d in os.listdir(a.datapath) if os.path.isdir(os.path.join(a.datapath, d)) and d not in ("raw", "bin"))
    fac = make_ms_factory(a.dataset, a.ms, a.arch, device=a.device)
    print(f"{'scène':26s} {'frames':>6s} {'score moy':>9s} {'>0.5':>6s} {'frames>30%':>10s} {'p95 frame':>9s}")
    for sc in scenes:
        try:
            prov = fac(sc, os.path.join(a.datapath, sc))
        except Exception as e:  # scène non prétraitée, etc. : on le dit, on continue
            print(f"{sc:26s} ERREUR {type(e).__name__}: {e}")
            continue
        s = prov.scores  # (N, Hp, Wp) dans [0, 1]
        per_frame = (s > 0.5).float().mean(dim=(1, 2)).cpu().numpy()
        print(f"{sc:26s} {s.shape[0]:6d} {s.mean().item():9.3f} {(s > 0.5).float().mean().item():6.3f} "
              f"{(per_frame > 0.3).mean():10.3f} {np.quantile(per_frame, 0.95):9.3f}", flush=True)
        del prov
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
