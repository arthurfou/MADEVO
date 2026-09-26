#!/usr/bin/env python3
"""Statistiques du filtrage de mouvement sur les masques GT EVIMO.

Pour chaque séquence : objets annotés, objets mobiles, et part de patches positifs dans les
trois modes (`all`, `sequence`, `frame`). Sert à choisir le mode et le seuil avant
d'entraîner, et à documenter l'ablation.

Usage :
    python scripts/mask_motion_stats.py --data-root ../datasets/evimo2/samsung_mono/imo/eval
    python scripts/mask_motion_stats.py --data-root ... --thresh 0.01 --window 0.1
"""

import argparse
from pathlib import Path

import numpy as np

from ms_model.data.evimo_dataset import downsample_mask
from ms_model.io.loaders import load_evimo_mask
from ms_model.motion import (DEFAULT_THRESH_M, DEFAULT_WINDOW_S, filter_masks_to_moving,
                             moving_object_ids, MASK_ID_SCALE)


def find_sequences(root: Path) -> list[Path]:
    dirs = sorted(p.parent for p in root.rglob("dataset_info.npz"))
    npz = sorted(p for p in root.rglob("*.npz") if not p.name.startswith("dataset_"))
    return dirs + npz


def positive_share(masks: np.ndarray, patch_size: int, stride: int) -> float:
    pos = tot = 0
    for m in masks[::stride]:
        d = downsample_mask(m, patch_size) > 0
        pos += d.sum().item()
        tot += d.numel()
    return pos / max(tot, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--thresh", type=float, default=DEFAULT_THRESH_M, help="seuil de mouvement (m)")
    p.add_argument("--window", type=float, default=DEFAULT_WINDOW_S, help="fenêtre du mode frame (s)")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=4, help="1 frame sur N (accélère)")
    args = p.parse_args()

    sequences = find_sequences(Path(args.data_root))
    if not sequences:
        raise SystemExit(f"Aucune séquence EVIMO sous {args.data_root}")

    print(f"seuil = {args.thresh * 100:.1f} cm, fenêtre = {args.window:.2f} s, "
          f"patch = {args.patch_size}, 1 frame sur {args.stride}\n")
    header = f"{'séquence':32s} {'objets':>22s} {'mobiles':>16s} {'patches+ all':>12s} {'sequence':>9s} {'frame':>7s}"
    print(header); print("-" * len(header))
    tot = {"all": [], "sequence": [], "frame": []}
    for seq in sequences:
        fm = load_evimo_mask(seq)
        ids = sorted({int(v) // MASK_ID_SCALE for v in np.unique(fm.masks) if v > 0})
        moving = sorted(moving_object_ids(seq, args.thresh))
        shares = {}
        for mode in ("all", "sequence", "frame"):
            f = filter_masks_to_moving(fm, seq, mode, args.thresh, args.window)
            shares[mode] = positive_share(f.masks, args.patch_size, args.stride)
            tot[mode].append(shares[mode])
        print(f"{seq.name[:32]:32s} {str(ids):>22s} {str(moving):>16s} "
              f"{shares['all']:11.1%} {shares['sequence']:8.1%} {shares['frame']:6.1%}")
    print("-" * len(header))
    print(f"{'moyenne':32s} {'':>22s} {'':>16s} "
          f"{np.mean(tot['all']):11.1%} {np.mean(tot['sequence']):8.1%} {np.mean(tot['frame']):6.1%}")


if __name__ == "__main__":
    main()
