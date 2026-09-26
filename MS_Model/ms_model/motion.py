"""Sélection des objets réellement mobiles dans les annotations EVIMO.

EVIMO segmente **tous** les objets suivis par VICON, décor immobile compris : sur les
séquences EVIMO2, 35 à 95 % des pixels annotés appartiennent à des objets qui ne bougent pas
(mesures dans `INTEGRITY_CHECK.md`). Prendre `mask != 0` comme « dynamique » revient donc à
apprendre à reconnaître des objets annotés plutôt que du mouvement.

Ce module reconstruit la position **monde** de chaque objet à partir des poses caméra et
objet de la meta (les poses objet sont exprimées dans le repère caméra), puis filtre les
masques. Rien n'est écrit dans le dataset : le tri se fait au chargement.

Modes (`mask_motion`) :
- `all`      : aucun filtrage (comportement historique, sert d'ablation) ;
- `sequence` : on garde les objets qui bougent quelque part dans la séquence ;
- `frame`    : on garde, à chaque frame, les objets qui bougent **autour de cet instant**
               (fenêtre glissante) — attrape les objets lents et exclut un objet reposé.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np

from ms_model.io.masks import FrameMasks

MODES = ("all", "sequence", "frame")

# Clés de meta["frames"][i] qui ne sont pas des objets.
NON_OBJ_KEYS = {"cam", "gt_frame", "classical_frame", "id", "ts"}

# Valeur du masque pour l'objet d'id n : n × MASK_ID_SCALE.
MASK_ID_SCALE = 1000

DEFAULT_THRESH_M = 0.02
DEFAULT_WINDOW_S = 0.2


def _meta_frames(seq: Path) -> list:
    """Frames de la meta, que la séquence soit un répertoire EVIMO2 ou un .npz EVIMO1."""
    if seq.is_dir():
        return np.load(seq / "dataset_info.npz", allow_pickle=True)["meta"].item()["frames"]
    with np.load(seq, allow_pickle=True) as data:
        return data["meta"].item()["frames"]


def _t(pos) -> np.ndarray:
    return np.array([pos["t"]["x"], pos["t"]["y"], pos["t"]["z"]], dtype=np.float64)


def _rotate(pos, v: np.ndarray) -> np.ndarray:
    """Applique la rotation du quaternion de `pos` au vecteur v (formule de Rodrigues)."""
    q = pos["q"]
    u = np.array([q["x"], q["y"], q["z"]], dtype=np.float64)
    w = float(q["w"])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


@lru_cache(maxsize=64)
def _world_tracks(seq_str: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{id objet: (ts (K,), positions monde (K, 3))} — poses objet ramenées dans le repère monde."""
    frames = [f for f in _meta_frames(Path(seq_str)) if "cam" in f]
    tracks: dict[int, tuple[list, list]] = {}
    for f in frames:
        for key in f.keys() - NON_OBJ_KEYS:
            ts, pos = tracks.setdefault(int(key), ([], []))
            ts.append(float(f["ts"]))
            pos.append(_rotate(f["cam"]["pos"], _t(f[key]["pos"])) + _t(f["cam"]["pos"]))
    return {oid: (np.asarray(ts), np.asarray(pos)) for oid, (ts, pos) in tracks.items()}


def _spread(positions: np.ndarray) -> float:
    """Écart maximal à la position médiane (m) — 0 si moins de 2 échantillons."""
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(positions - np.median(positions, axis=0), axis=1).max())


def moving_object_ids(
    seq: Union[str, Path], thresh_m: float = DEFAULT_THRESH_M, verbose: bool = False
) -> set[int]:
    """Ids des objets qui bougent quelque part dans la séquence."""
    moving = set()
    for oid, (_, pos) in sorted(_world_tracks(str(seq)).items()):
        spread = _spread(pos)
        if verbose:
            print(f"  obj {oid:>3} : écart max au repos = {spread * 100:6.2f} cm"
                  f"  → {'MOBILE' if spread > thresh_m else 'statique'}")
        if spread > thresh_m:
            moving.add(oid)
    return moving


def moving_object_ids_at(
    seq: Union[str, Path],
    ts: np.ndarray,
    thresh_m: float = DEFAULT_THRESH_M,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[set[int]]:
    """Pour chaque instant de `ts`, ids des objets qui bougent dans la fenêtre centrée dessus."""
    tracks = _world_tracks(str(seq))
    out: list[set[int]] = [set() for _ in ts]
    half = window_s / 2.0
    for oid, (obj_ts, pos) in tracks.items():
        for i, t in enumerate(np.asarray(ts, dtype=np.float64)):
            sel = np.abs(obj_ts - t) <= half
            if sel.sum() >= 2 and _spread(pos[sel]) > thresh_m:
                out[i].add(oid)
    return out


def filter_masks_to_moving(
    fm: FrameMasks,
    seq: Union[str, Path],
    mode: str = "all",
    thresh_m: float = DEFAULT_THRESH_M,
    window_s: float = DEFAULT_WINDOW_S,
) -> FrameMasks:
    """Met à zéro, dans les masques, les objets qui ne bougent pas (voir les modes du module).

    Les timestamps de `fm` servent à aligner le filtrage par frame ; `fm` n'est pas modifié.
    """
    if mode not in MODES:
        raise ValueError(f"mask_motion inconnu : {mode!r} (attendu : {', '.join(MODES)})")
    if mode == "all":
        return fm

    ids = fm.masks // MASK_ID_SCALE
    if mode == "sequence":
        keep = np.isin(ids, list(moving_object_ids(seq, thresh_m)))
    else:
        per_frame = moving_object_ids_at(seq, fm.ts, thresh_m, window_s)
        keep = np.stack([np.isin(ids[i], list(mv)) for i, mv in enumerate(per_frame)])

    return FrameMasks(masks=np.where(keep, fm.masks, 0), ts=fm.ts)
