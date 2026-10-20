#!/usr/bin/env python3
"""Figure du README : objets annotés par EVIMO2 contre objets réellement mobiles (T7).

Trois panneaux pour une frame d'éval : les events autour de l'instant t, le masque GT brut
(`mask != 0`, tous les objets suivis par VICON) et la cible retenue (`mask_motion: frame`).

Usage :
    python scripts/plot_mask_definition.py \\
        --seq datasets/evimo2/samsung_mono/imo/eval/scene13_dyn_test_00_000000 --frame 50 \\
        --out assets/mask_definition.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ms_model.io.loaders import load_events, load_evimo_mask  # noqa: E402
from ms_model.motion import filter_masks_to_moving  # noqa: E402

STATIC = "#2a78d6"    # annoté mais immobile
MOVING = "#eb6834"    # mobile à l'instant t
SURFACE = "#fcfcfb"
INK = "#2b2b29"


def event_image(ea, t0, t1, H, W):
    sel = (ea.t >= t0) & (ea.t < t1)
    img = np.zeros((H, W))
    np.add.at(img, (ea.y[sel].astype(int), ea.x[sel].astype(int)), 1.0)
    return 1.0 - np.clip(img / max(np.percentile(img[img > 0], 95), 1.0), 0, 1)


def overlay(gray, layers):
    rgb = np.repeat(gray[..., None], 3, axis=2) * 0.85 + 0.15
    for mask, color in layers:
        c = np.array(matplotlib.colors.to_rgb(color))
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * c
    return rgb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", required=True)
    p.add_argument("--frame", type=int, required=True, help="indice parmi les frames avec GT")
    p.add_argument("--thresh", type=float, default=0.01)
    p.add_argument("--window", type=float, default=0.2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fm = load_evimo_mask(args.seq)
    moving = filter_masks_to_moving(fm, args.seq, "frame", args.thresh, args.window).masks[args.frame] > 0
    annotated = fm.masks[args.frame] > 0
    i, H, W = args.frame, *annotated.shape
    ea = load_events(args.seq)
    gray = event_image(ea, fm.ts[i], fm.ts[min(i + 1, len(fm.ts) - 1)], H, W)

    panels = [
        ("Events", overlay(gray, [])),
        ("EVIMO2 GT mask (mask ≠ 0)", overlay(gray, [(annotated & ~moving, STATIC), (moving, MOVING)])),
        ("Dynamic target (moving at t)", overlay(gray, [(moving, MOVING)])),
    ]
    plt.rcParams.update({"font.size": 10, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, color=INK, fontsize=11, pad=6)
        ax.axis("off")
    fig.legend(handles=[Patch(color=STATIC, label="annotated object, static at t"),
                        Patch(color=MOVING, label="annotated object, moving at t")],
               loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.03)
    fig.savefig(args.out, dpi=110, facecolor=SURFACE)
    print(f"→ {args.out}  (annotés : {annotated.mean():.0%} des pixels, mobiles : {moving.mean():.1%})")


if __name__ == "__main__":
    main()
