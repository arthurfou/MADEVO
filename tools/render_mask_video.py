#!/usr/bin/env python
"""Vidéo d'une séquence EVIMO2 : events (rouge = +, bleu = −) et masques en jaune.

    python tools/render_mask_video.py <séquence EVIMO2> -o out.mp4
    python tools/render_mask_video.py <séquence> -o out.mp4 --ms-run runs/v4_evimo2/<id>/seed1

Panneau de gauche : le masque GT tel que l'entraînement le voit (définition `mask_motion`,
dilatation comprise, lues dans la config du run) — jaune = objet en mouvement, gris = objet
suivi par VICON mais immobile à cet instant. Panneau de droite (avec --ms-run) : le masque
prédit par le MS (`best_iou.pt` par défaut), binarisé à --threshold. Une frame vidéo = un
intervalle entre deux masques GT, comme pour l'entraînement.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ms_model.io.loaders import load_events, load_evimo_mask  # noqa: E402
from ms_model.masking import thicken_mask  # noqa: E402
from ms_model.motion import filter_masks_to_moving  # noqa: E402
from ms_model.viz.mask import render_mask_overlay  # noqa: E402
from ms_model.viz.render import render_event_frame  # noqa: E402
from ms_model.viz.video import frames_to_video  # noqa: E402

YELLOW = (255, 220, 0)
GREY = (150, 150, 150)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq", type=Path, help="dossier de séquence EVIMO2 (samsung_mono/imo/<split>/<seq>)")
    p.add_argument("-o", "--out", type=Path, required=True, help="vidéo de sortie (.mp4)")
    p.add_argument("--ms-run", type=Path, help="dossier de run MS (config.yaml + checkpoints/)")
    p.add_argument("--ckpt", default="best_iou.pt", help="checkpoint du run (défaut : best_iou.pt)")
    p.add_argument("--config", type=Path, help="config MS pour la définition du masque GT "
                   "(défaut : celle du run, sinon configs/ms/v4_evimo2.yaml)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.45, help="opacité du masque")
    p.add_argument("--max-frames", type=int, default=0, help="0 = toute la séquence")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from ms_model.inference import load_config
    cfg_path = args.config or (args.ms_run / "config.yaml" if args.ms_run else
                               REPO_ROOT / "configs/ms/v4_evimo2.yaml")
    data_cfg = load_config(cfg_path)["data"]

    ea = load_events(args.seq)
    fm_raw = load_evimo_mask(args.seq)
    fm = filter_masks_to_moving(fm_raw, args.seq, data_cfg.get("mask_motion", "all"),
                                data_cfg.get("motion_thresh_m", 0.01),
                                data_cfg.get("motion_window_s", 0.2))
    if data_cfg.get("mask_thicken_radius", 0) > 0:
        fm = thicken_mask(fm, radius=data_cfg["mask_thicken_radius"])

    pred = None
    if args.ms_run:
        from ms_model.inference import Predictor
        predictor = Predictor(args.ms_run / "checkpoints" / args.ckpt, args.ms_run / "config.yaml",
                              device=args.device, threshold=args.threshold)
        pred = predictor.predict_sequence(args.seq)["pred_masks"]

    n = len(fm.ts) - 1
    if args.max_frames:
        n = min(n, args.max_frames)
    frames, pos_gt, pos_pred = [], 0, 0
    for i in range(n):
        ev = render_event_frame(ea, fm.ts[i], fm.ts[i + 1])
        moving, tracked = fm.masks[i] != 0, fm_raw.masks[i] != 0
        gt = render_mask_overlay(ev, tracked & ~moving, color=GREY, alpha=args.alpha * 0.6)
        gt = _label(render_mask_overlay(gt, moving, color=YELLOW, alpha=args.alpha),
                    f"GT ({data_cfg.get('mask_motion', 'all')})  {i + 1}/{n}  t={fm.ts[i]:.2f}s")
        pos_gt += moving.mean()
        panels = [gt]
        if pred is not None:
            panels.append(_label(render_mask_overlay(ev, pred[i], color=YELLOW, alpha=args.alpha),
                                 f"MS {args.ckpt} (seuil {args.threshold})"))
            pos_pred += pred[i].mean()
        frames.append(np.concatenate(panels, axis=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames_to_video(frames, args.out, fps=args.fps)
    # OpenCV n'encode qu'en MPEG-4 Part 2 (~4x plus lourd) : H.264 si ffmpeg est là.
    if shutil.which("ffmpeg"):
        tmp = args.out.with_suffix(".h264.mp4")
        ok = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(args.out), "-c:v", "libx264",
                             "-crf", "24", "-pix_fmt", "yuv420p", str(tmp)]).returncode == 0
        if ok:
            tmp.replace(args.out)
    msg = f"{args.out} : {n} frames, {100 * pos_gt / n:.2f} % de pixels GT en mouvement"
    if pred is not None:
        msg += f", {100 * pos_pred / n:.2f} % prédits"
    print(msg)


if __name__ == "__main__":
    main()
