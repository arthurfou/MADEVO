#!/usr/bin/env python
"""Vérifie sur CPU que le masque appris et DEVO regardent les mêmes pixels au même instant.

Sur les datasets cross-dataset, DEVO voxelise les events avec ses itérateurs
(`utils/load_utils.py` : rectification, fenêtre temporelle propre au dataset) et le modèle MS
avec les siens (`utils/ms_dataset_factory.py`). Si les deux divergent (base temporelle,
rectification, résolution), le masque est appliqué à côté des objets et la table mesure du
bruit. Ce script compare, frame par frame :

- **temps** : écart entre l'instant de chaque voxel DEVO et la frame du provider qui lui sera
  servie (plus proche voisin, comme `TimestampMaskProvider`) ;
- **espace** : corrélation entre les densités d'events des deux voxels à la résolution de la
  score map (H/4 × W/4), et décalage (dx, dy) qui la maximise — (0, 0) si aligné.

Aucun GPU : `.cuda()` est neutralisé pour appeler les vrais itérateurs DEVO sur CPU.

    python tools/check_cross_alignment.py --dataset mvsec --scene datasets/mvsec/indoor_flying4_data
    python tools/check_cross_alignment.py --dataset eds --scene datasets/eds/00_peanuts_dark --frames 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Les itérateurs DEVO finissent par `voxel.cuda()` : sur CPU, on rend l'identité.
torch.Tensor.cuda = lambda self, *args, **kwargs: self  # noqa: E731

# Mêmes arguments que les runners de evals/eval_evs/ (cf. evals/eval_table.py)
ITERATORS = {
    "rpg":    ("rpg_evs_iterator", dict(side="left", dT_ms=None, H=180, W=240)),
    "mvsec":  ("mvsec_evs_iterator", dict(side="left", H=260, W=346)),
    "eds":    ("eds_evs_iterator", dict(calib1=False, H=480, W=640)),
    "vector": ("vector_evs_iterator", dict(side="left", H=480, W=640)),
    "tumvie": ("tumvie_evs_iterator", dict(camID=2, dT_ms=25, H=720, W=1280)),
}


def density(voxel: torch.Tensor, patch: int = 4) -> np.ndarray:
    """|voxel| sommé sur les bins, max-poolé à la résolution de la score map."""
    d = voxel.abs().sum(0, keepdim=True)[None].float()
    hp, wp = d.shape[-2] // patch, d.shape[-1] // patch
    return torch.nn.functional.adaptive_max_pool2d(d, (hp, wp))[0, 0].numpy()


def best_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 4) -> tuple[float, int, int]:
    """(corrélation max, dx, dy) entre deux cartes, b décalée de (dx, dy) cellules."""
    best = (-1.0, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            bb = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            c = np.corrcoef(a.ravel(), bb.ravel())[0, 1]
            if np.isfinite(c) and c > best[0]:
                best = (float(c), dx, dy)
    return best


def _limit_h5_events(n: int) -> None:
    """Remplace le chargeur h5 du provider par une version limitée aux n premiers events.

    Les events sont triés par temps : on vérifie donc l'alignement sur le début de la
    séquence. Sans effet sur MVSEC et RPG, qui ont leurs propres chargeurs.
    """
    import h5py
    import ms_model.io.loaders as loaders
    from ms_model.io.events import EventArray

    def load_head(h5_path, H=260, W=346):
        with h5py.File(str(h5_path), "r") as f:
            prefix = "events/" if "events/x" in f else ""
            m = min(n, f[f"{prefix}t"].shape[0])
            t_offset = int(f["t_offset"][()]) if "t_offset" in f else 0
            ev = np.empty((m, 4), dtype=np.float64)
            for col, key in enumerate("txyp"):
                ev[:, col] = f[f"{prefix}{key}"][:m]
        ev[:, 0] = (ev[:, 0] + t_offset) / 1e6
        return EventArray(events=ev, H=H, W=W)

    loaders.load_events_prophesee_h5 = load_head


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(ITERATORS))
    p.add_argument("--scene", required=True, help="dossier de la scène prétraitée")
    p.add_argument("--frames", type=int, default=30, help="nombre de voxels DEVO comparés")
    p.add_argument("--skip", type=int, default=20, help="voxels ignorés en début de séquence")
    p.add_argument("--min-corr", type=float, default=0.5)
    p.add_argument("--max-events", type=int, default=60_000_000,
                   help="h5 (EDS, VECtor, TUM-VIE) : ne charge que les N premiers events, pour "
                        "tenir en RAM sur une petite machine (0 = tout)")
    args = p.parse_args()

    if args.max_events:
        _limit_h5_events(args.max_events)

    import utils.load_utils as lu
    from ms_model.representations.voxel import make_voxel_sequence
    from utils.ms_dataset_factory import _LOADERS, rectify_events

    name, kwargs = ITERATORS[args.dataset]
    scene = str(Path(args.scene))

    # côté MS : exactement ce que fait la factory (chargement, rectification, frames)
    ea, frame_ts_s, rmap_path = _LOADERS[args.dataset](scene, kwargs.get("side", "left"), kwargs.get("camID", 2))
    ea = rectify_events(ea, rmap_path)
    print(f"[MS] {len(ea)} events, t ∈ [{ea.t.min():.3f}, {ea.t.max():.3f}] s ; "
          f"{len(frame_ts_s)} frames, t ∈ [{frame_ts_s[0]:.3f}, {frame_ts_s[-1]:.3f}] s ; "
          f"rectification : {rmap_path if rmap_path and Path(rmap_path).exists() else 'aucune'}")
    overlap = min(ea.t.max(), frame_ts_s[-1]) - max(ea.t.min(), frame_ts_s[0])
    if overlap <= 0:
        print("✗ base temporelle : events et frames ne se recouvrent pas")
        return 1

    # côté DEVO : le vrai itérateur, sur CPU
    it = getattr(lu, name)(scene, **kwargs)
    rows = []
    for k, (voxel, _, ts_us) in enumerate(it):
        if k < args.skip:
            continue
        if len(rows) >= args.frames:
            break
        t = float(ts_us) / 1e6
        i = int(np.argmin(np.abs(frame_ts_s - t)))           # frame servie par le provider
        if i + 1 >= len(frame_ts_s) or frame_ts_s[i + 1] > ea.t.max():
            break                                            # au-delà des events chargés
        ms_vox = make_voxel_sequence(ea, frame_ts_s[i:i + 2], nb_of_time_bins=5)[0]
        if ms_vox.shape[-2:] != voxel.shape[-2:]:
            print(f"✗ résolution : MS {tuple(ms_vox.shape[-2:])} ≠ DEVO {tuple(voxel.shape[-2:])}")
            return 1
        corr, dx, dy = best_shift(density(voxel), density(ms_vox))
        corr0 = np.corrcoef(density(voxel).ravel(), density(ms_vox).ravel())[0, 1]
        rows.append((t, (frame_ts_s[i] - t) * 1e3, float(corr0), corr, dx, dy))

    if not rows:
        print("✗ aucun voxel DEVO comparé")
        return 1
    a = np.array(rows)
    dt, c0, shifts = a[:, 1], a[:, 2], a[:, 4:6]
    frame_ms = np.median(np.diff(frame_ts_s)) * 1e3
    n_shift = int((np.abs(shifts).sum(1) > 0).sum())
    print(f"[DEVO] {len(rows)} voxels comparés ({args.dataset}, {Path(scene).name})")
    print(f"  temps  : écart frame servie − voxel DEVO = {np.median(dt):+.1f} ms en médiane "
          f"(max {np.abs(dt).max():.1f} ms ; période des frames {frame_ms:.1f} ms)")
    print(f"  espace : corrélation des densités sans décalage = {np.median(c0):.2f} en médiane "
          f"(min {c0.min():.2f}) ; meilleur décalage ≠ (0, 0) pour {n_shift}/{len(rows)} voxels")

    ok = True
    if np.abs(dt).max() > frame_ms:
        print("✗ temps : le provider sert une frame éloignée de plus d'une période")
        ok = False
    if np.median(c0) < args.min_corr:
        print(f"✗ espace : corrélation médiane < {args.min_corr}")
        ok = False
    if n_shift > len(rows) // 2:
        print("✗ espace : décalage systématique entre les deux voxelisations")
        ok = False
    print("→ aligné" if ok else "→ NON aligné")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
