#!/usr/bin/env python3
"""Statistiques du filtrage de mouvement sur les masques GT EVIMO (choix du mode, T7).

Mesure, sur un split, ce que la loss du MS verra selon le mode `mask_motion`, le seuil de
mouvement et la fenêtre du mode `frame` :

1. l'écart au repos de chaque objet annoté (cm), sur la séquence et par fenêtre glissante —
   pour placer le seuil entre le bruit VICON des objets statiques et le mouvement réel ;
2. pour chaque couple (seuil, fenêtre), la part de cellules positives à la résolution du MS
   (H/patch × W/patch, après dilatation de `--thicken` px comme à l'entraînement), la part
   de frames sans aucun positif, et le `pos_weight` équilibrant (négatifs / positifs).

Les parts sont **poolées** sur toutes les frames du split (c'est ce que voit la loss), pas
moyennées par séquence.

Usage :
    python tools/mask_motion_stats.py --data-root datasets/evimo2/samsung_mono/imo/train
    python tools/mask_motion_stats.py --data-root ... --thresh 0.005 0.01 0.02 \\
        --window 0.1 0.2 0.5 --out stats.md
"""

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

from ms_model.data.evimo_dataset import downsample_mask
from ms_model.io.loaders import load_evimo_mask
from ms_model.motion import (MASK_ID_SCALE, _spread, _world_tracks, moving_object_ids,
                             moving_object_ids_at)


def find_sequences(root: Path) -> list[Path]:
    dirs = sorted(p.parent for p in root.rglob("dataset_info.npz"))
    npz = sorted(p for p in root.rglob("*.npz") if not p.name.startswith("dataset_"))
    return dirs + npz


def object_cells(mask: np.ndarray, thicken: int, patch_size: int) -> dict[int, np.ndarray]:
    """{id objet: cellules (H/p, W/p) qu'il occupe après dilatation}.

    Dilater puis max-pooler l'union des objets gardés = union des cellules de chaque objet :
    on calcule donc une fois par objet, et chaque mode n'est plus qu'une union.
    """
    ids = mask // MASK_ID_SCALE
    kernel = np.ones((2 * thicken + 1, 2 * thicken + 1), np.uint8) if thicken > 0 else None
    out = {}
    for oid in np.unique(ids[ids > 0]):
        m = (ids == oid).astype(np.uint8)
        if kernel is not None:
            m = cv2.dilate(m, kernel)
        out[int(oid)] = downsample_mask(m, patch_size).numpy().astype(bool)
    return out


def windowed_spreads(seq: Path, ts: np.ndarray, window_s: float) -> dict[int, np.ndarray]:
    """{id objet: écart au repos (m) dans la fenêtre centrée sur chaque instant de ts}."""
    out = {}
    half = window_s / 2.0
    for oid, (obj_ts, pos) in _world_tracks(str(seq)).items():
        lo = np.searchsorted(obj_ts, ts - half, side="left")
        hi = np.searchsorted(obj_ts, ts + half, side="right")
        out[oid] = np.array([_spread(pos[a:b]) if b - a >= 2 else 0.0 for a, b in zip(lo, hi)])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--thresh", type=float, nargs="+", default=[0.005, 0.01, 0.02], help="seuils de mouvement (m)")
    p.add_argument("--window", type=float, nargs="+", default=[0.1, 0.2, 0.5], help="fenêtres du mode frame (s)")
    p.add_argument("--thicken", type=int, default=3, help="dilatation des masques (px), comme mask_thicken_radius")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=2, help="1 frame sur N (accélère)")
    p.add_argument("--out", help="écrit aussi le rapport markdown dans ce fichier")
    args = p.parse_args()

    sequences = find_sequences(Path(args.data_root))
    if not sequences:
        raise SystemExit(f"Aucune séquence EVIMO sous {args.data_root}")

    lines: list[str] = []
    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    combos = list(itertools.product(args.thresh, args.window))
    # compteurs poolés : cellules positives, cellules totales, frames sans positif
    pos = {"all": 0, **{("sequence", t): 0 for t in args.thresh}, **{("frame", t, w): 0 for t, w in combos}}
    empty = dict.fromkeys(pos, 0)
    cells = frames = 0
    obj_rows = []

    emit(f"# Filtrage de mouvement des masques GT — `{args.data_root}`\n")
    emit(f"{len(sequences)} séquences, dilatation {args.thicken} px, patch {args.patch_size}, "
         f"1 frame sur {args.stride}.\n")

    for seq in sequences:
        fm = load_evimo_mask(seq)
        masks, ts = fm.masks[::args.stride], fm.ts[::args.stride]
        seq_moving = {t: moving_object_ids(seq, t) for t in args.thresh}
        frame_moving = {(t, w): moving_object_ids_at(seq, ts, t, w) for t, w in combos}
        spreads = {w: windowed_spreads(seq, ts, w) for w in args.window}

        annotated = set()
        for i, m in enumerate(masks):
            occ = object_cells(m, args.thicken, args.patch_size)
            annotated |= occ.keys()
            cells += next(iter(occ.values())).size if occ else (m.shape[0] // args.patch_size) * (m.shape[1] // args.patch_size)
            frames += 1

            def count(key, keep):
                sel = [occ[o] for o in occ if o in keep]
                n = int(np.logical_or.reduce(sel).sum()) if sel else 0
                pos[key] += n
                empty[key] += n == 0

            count("all", occ.keys())
            for t in args.thresh:
                count(("sequence", t), seq_moving[t])
            for t, w in combos:
                count(("frame", t, w), frame_moving[(t, w)][i])

        tracks = _world_tracks(str(seq))
        for oid in sorted(tracks):
            if oid not in annotated:
                continue  # suivi par VICON mais jamais visible dans le masque
            row = {"seq": seq.name.replace("_000000", ""), "oid": oid,
                   "spread_seq": _spread(tracks[oid][1])}
            for w in args.window:
                s = spreads[w][oid]
                row[f"p50_{w}"], row[f"p95_{w}"], row[f"max_{w}"] = np.percentile(s, 50), np.percentile(s, 95), s.max()
            obj_rows.append(row)
        print(f"  {seq.name} : {len(masks)} frames, objets {sorted(annotated)}, "
              f"mobiles (seq, {args.thresh[-1] * 100:g} cm) {sorted(seq_moving[args.thresh[-1]] & annotated)}",
              file=sys.stderr, flush=True)

    # 1. Écart au repos par objet
    emit("## Écart au repos par objet (cm)\n")
    emit("`séquence` = écart max à la position médiane sur toute la séquence (critère du mode "
         "`sequence`) ; `p50/p95/max (w)` = distribution de l'écart dans une fenêtre glissante "
         "de w secondes (critère du mode `frame`).\n")
    head = ["séquence", "obj", "séquence (cm)"] + [f"{k} {w:g}s" for w in args.window for k in ("p50", "p95", "max")]
    emit("| " + " | ".join(head) + " |")
    emit("|" + "---|" * len(head))
    for r in sorted(obj_rows, key=lambda r: r["spread_seq"]):
        vals = [r["seq"], str(r["oid"]), f"{r['spread_seq'] * 100:.2f}"]
        vals += [f"{r[f'{k}_{w}'] * 100:.2f}" for w in args.window for k in ("p50", "p95", "max")]
        emit("| " + " | ".join(vals) + " |")

    # 2. Parts de positifs
    def row(label, key):
        share = pos[key] / max(cells, 1)
        pw = (cells - pos[key]) / pos[key] if pos[key] else float("inf")
        return f"| {label} | {share:.2%} | {empty[key] / max(frames, 1):.1%} | {pw:.1f} |"

    emit(f"\n## Part de cellules positives ({frames} frames, {cells} cellules)\n")
    emit("| mode | positifs | frames sans positif | pos_weight équilibrant (neg/pos) |")
    emit("|---|---|---|---|")
    emit(row("all", "all"))
    for t in args.thresh:
        emit(row(f"sequence, {t * 100:g} cm", ("sequence", t)))
    for t, w in combos:
        emit(row(f"frame, {t * 100:g} cm, {w:g} s", ("frame", t, w)))

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\n→ {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
