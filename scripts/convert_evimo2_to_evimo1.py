"""Convertit les séquences EVIMO2 (samsung_mono) au format EVIMO1 attendu par ms_model.

Format EVIMO2 (répertoire par séquence) :
    dataset_events_t.npy     float32 (N,)       timestamps en secondes
    dataset_events_xy.npy    uint16  (N, 2)     [x, y]
    dataset_events_p.npy     uint8   (N,)        polarity 0/1
    dataset_info.npz         → K (3,3), D (4,), index (N_frames,), meta dict
    dataset_mask.npz         → mask_0000000000 .. mask_XXXXXXXXX  (480,640) uint16
    dataset_depth.npz        → depth_0000000000 ..               (480,640) uint16 mm

Format EVIMO1-compatible (un seul .npz par séquence) :
    events  (N, 4) float32   [t_sec, x, y, p]
    index   (N_frames,) int64   event index par frame
    mask    (N_frames, 480, 640) uint16
    depth   (N_frames, 480, 640) uint16   mm
    K       (3, 3) float64
    meta    pickle dict → frames (liste de dicts: ts, cam.pos.q/t)

Usage:
    python scripts/convert_evimo2_to_evimo1.py \\
        --src  datasets/evimo2/samsung_mono \\
        --dst  datasets/evimo2/evimo1_format \\
        [--split imo]   # sous-répertoire dans samsung_mono/ (imo, sfm, …)

Chaque séquence srcdir/imo/{train,eval}/{seq_name}/ produit :
    dstdir/imo/{train,eval}/{seq_name}.npz
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from ms_model.io.loaders import _evimo2_gt_frames, load_evimo_mask


def convert_sequence(src_dir: str, dst_path: str, verbose: bool = True) -> None:
    t0 = time.time()

    # ── Events ───────────────────────────────────────────────────────────────
    ev_t  = np.load(os.path.join(src_dir, "dataset_events_t.npy"))   # float32 (N,)
    ev_xy = np.load(os.path.join(src_dir, "dataset_events_xy.npy"))  # uint16  (N,2)
    ev_p  = np.load(os.path.join(src_dir, "dataset_events_p.npy"))   # uint8   (N,)

    events = np.empty((len(ev_t), 4), dtype=np.float32)
    events[:, 0] = ev_t                        # t en secondes
    events[:, 1] = ev_xy[:, 0].astype(np.float32)  # x
    events[:, 2] = ev_xy[:, 1].astype(np.float32)  # y
    events[:, 3] = ev_p.astype(np.float32)    # polarity

    # ── Info (K, meta) ───────────────────────────────────────────────────────
    fi   = np.load(os.path.join(src_dir, "dataset_info.npz"), allow_pickle=True)
    K    = fi["K"].astype(np.float64)           # (3,3)

    # ── Masques, depth et frames : alignés sur les frames possédant une GT ──
    # EVIMO2 laisse des trous (tracking VICON perdu) : la frame reste dans meta["frames"]
    # mais n'a ni masque ni depth. Le i-ème masque va donc avec la i-ème frame **avec GT**,
    # pas avec frames[i] — l'ancienne version de ce script faisait ce second appariement et
    # datait faux 33 % des masques des séquences d'éval (cf. INTEGRITY_CHECK.md, bug 6).
    frames = _evimo2_gt_frames(Path(src_dir))
    fm = load_evimo_mask(src_dir)               # même alignement que l'entraînement
    mask_arr = fm.masks.astype(np.uint16)
    N_frames = len(frames)

    fd = np.load(os.path.join(src_dir, "dataset_depth.npz"), allow_pickle=True)
    depth_keys = sorted(fd.files, key=lambda k: int(k.split("_")[-1]))
    if len(depth_keys) != N_frames:
        raise ValueError(f"{src_dir}: {len(depth_keys)} depth pour {N_frames} frames avec GT")
    depth_arr = np.stack([fd[k] for k in depth_keys]).astype(np.uint16)

    # `index` d'EVIMO2 est une table de lookup sur une grille de 10 ms, pas un index par
    # frame : on le recalcule (premier event de chaque frame), comme en EVIMO1.
    idx = np.searchsorted(ev_t, fm.ts, side="left").astype(np.int64)

    meta_out = {"frames": frames}

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.savez_compressed(
        dst_path,
        events=events,
        index=idx,
        mask=mask_arr,
        depth=depth_arr,
        K=K,
        meta=np.array(meta_out, dtype=object),
    )

    elapsed = time.time() - t0
    size_mb = os.path.getsize(dst_path + ".npz") / 1e6 if os.path.exists(dst_path + ".npz") else 0
    if verbose:
        print(f"  -> {dst_path}.npz  "
              f"({N_frames} frames, {len(ev_t)//1e6:.1f}M events, {size_mb:.0f} MB, {elapsed:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Racine samsung_mono/ extraite du tar (ex: datasets/evimo2/samsung_mono)")
    p.add_argument("--dst", required=True,
                   help="Répertoire de sortie (ex: datasets/evimo2/evimo1_format)")
    p.add_argument("--split", default=None,
                   help="Filtrer un sous-split : 'imo', 'sfm', 'imo/train', 'imo/eval', …")
    args = p.parse_args()

    # Découverte automatique des répertoires séquences
    seq_dirs = []
    for dirpath, dirnames, filenames in os.walk(args.src):
        if "dataset_info.npz" in filenames:
            seq_dirs.append(dirpath)
    seq_dirs.sort()

    if args.split:
        seq_dirs = [d for d in seq_dirs if args.split in d]

    if not seq_dirs:
        print(f"Aucune séquence trouvée sous {args.src} (split={args.split})", file=sys.stderr)
        sys.exit(1)

    print(f"Conversion de {len(seq_dirs)} séquences -> {args.dst}")
    ok, errors = 0, []
    for src_dir in seq_dirs:
        # chemin relatif par rapport à args.src
        rel = os.path.relpath(src_dir, args.src)
        dst_path = os.path.join(args.dst, rel)
        print(f"[{rel}]")
        try:
            convert_sequence(src_dir, dst_path)
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append((rel, str(e)))

    print(f"\nDone: {ok}/{len(seq_dirs)} ok" + (f", {len(errors)} erreurs" if errors else ""))
    for r, e in errors:
        print(f"  ERREUR {r}: {e}")


if __name__ == "__main__":
    main()
