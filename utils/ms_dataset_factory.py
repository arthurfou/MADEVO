"""Factories pour créer un LearnedDynMaskProvider sur n'importe quel dataset DEVO.

Pour les datasets sans objets dynamiques GT (tous sauf EVIMO), on applique le modèle MS
en inférence pure (mode M1/M2/M3). Le modèle donnerait idéalement des scores proches de 0
sur des scènes statiques, laissant DEVO se comporter comme vanilla.

Géométrie : DEVO voxelise ces datasets en coordonnées **rectifiées** (`rectify_map[y, x]`,
cf. `utils/load_utils.py`). Le provider applique la même carte avant de voxeliser, sinon la
score map du MS et celle de DEVO ne désignent pas les mêmes pixels (jusqu'à plusieurs
dizaines de pixels au bord des optiques fisheye de TUM-VIE et MVSEC).

Usage :
    from utils.ms_dataset_factory import make_ms_factory

    factory = make_ms_factory("rpg", ms_weights="path/to/best.pt",
                              ms_config="path/to/config.yaml", side="left")
    # factory(scene, datapath_val) -> LearnedDynMaskProvider
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _frame_ts_s(tss_txt_path: str) -> np.ndarray:
    """Charge tss_imgs_us (µs) et les convertit en secondes."""
    return np.sort(np.loadtxt(tss_txt_path).astype(np.float64)) / 1e6


def _read_rectify_map(path: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return np.array(f["rectify_map"])  # (H, W, 2) : (x, y) rectifiés


def rectify_events(ea, rmap_path: Optional[str], inplace: bool = False, chunk: int = 20_000_000):
    """EventArray aux coordonnées rectifiées, comme les itérateurs DEVO (`rectify_map[y, x]`).

    Sans carte (`None` ou fichier absent), l'EventArray est rendu tel quel. Les events qui
    tombent hors du capteur après rectification sont écartés par la voxelisation.
    `inplace=True` réécrit `ea.events` sans copie (une scène EDS fait des dizaines de Go) ;
    la carte est appliquée par tranches de `chunk` events pour borner les temporaires.
    """
    if rmap_path is None or not os.path.exists(rmap_path):
        return ea
    from ms_model.io.events import EventArray
    rmap = _read_rectify_map(rmap_path)
    assert rmap.shape[:2] == (ea.H, ea.W), \
        f"carte de rectification {rmap.shape[:2]} ≠ capteur {(ea.H, ea.W)} ({rmap_path})"
    events = ea.events if inplace else ea.events.copy()
    for a in range(0, len(events), chunk):
        ev = events[a:a + chunk]
        rect = rmap[ev[:, 2].astype(np.int64), ev[:, 1].astype(np.int64)]
        ev[:, 1], ev[:, 2] = rect[:, 0], rect[:, 1]
    return EventArray(events=events, H=ea.H, W=ea.W, frame_ts=ea.frame_ts)


class H5EventWindows:
    """Source d'events fenêtrée sur un h5 Prophesee/DSEC (`events/{t,x,y,p}` + `ms_to_idx`).

    TUM-VIE compte de 0,6 à 4,5 milliards d'events par scène (jusqu'à ~140 Go en EventArray
    float64) : au lieu de tout charger, `window(t0, t1)` ne lit que les events de
    [t0 − 1 ms, t1 + 1 ms] grâce à l'index `ms_to_idx`, les convertit exactement comme
    `load_events_prophesee_h5` (t en s = (t_us + t_offset) / 1e6, float64) et les rectifie.
    La fenêtre est un sur-ensemble de [t0, t1) : `make_voxel_sequence` refait la sélection
    exacte, d'où des voxels identiques à ceux du chargement complet.
    """

    def __init__(self, h5_path: str, H: int, W: int, rmap_path: Optional[str] = None):
        import h5py
        try:
            import hdf5plugin  # noqa: F401  (h5 compressés en Blosc/zstd)
        except ImportError:
            pass
        self.h5 = h5py.File(str(h5_path), "r")
        pre = "events/" if "events/x" in self.h5 else ""
        self.d = {k: self.h5[pre + k] for k in "txyp"}
        self.ms_to_idx = np.asarray(self.h5["ms_to_idx"], dtype=np.int64)
        self.t_offset = int(self.h5["t_offset"][()]) if "t_offset" in self.h5 else 0
        self.n = int(self.d["t"].shape[0])
        self.H, self.W, self.rmap_path = H, W, rmap_path

    def __len__(self) -> int:
        return self.n

    def _idx(self, ms: int) -> int:
        if ms <= 0:
            return 0
        return int(self.ms_to_idx[ms]) if ms < len(self.ms_to_idx) else self.n

    def window(self, t0_s: float, t1_s: float):
        from ms_model.io.events import EventArray
        ms0 = int(np.floor((t0_s * 1e6 - self.t_offset) / 1000)) - 1
        ms1 = int(np.ceil((t1_s * 1e6 - self.t_offset) / 1000)) + 1
        a, b = self._idx(ms0), self._idx(ms1)
        events = np.empty((b - a, 4), dtype=np.float64)
        events[:, 0] = self.d["t"][a:b]
        events[:, 1] = self.d["x"][a:b]
        events[:, 2] = self.d["y"][a:b]
        events[:, 3] = self.d["p"][a:b]
        events[:, 0] += self.t_offset
        events[:, 0] /= 1e6
        ea = EventArray(events=events, H=self.H, W=self.W)
        return rectify_events(ea, self.rmap_path, inplace=True)


def _make_provider(ea, frame_ts_s, ms_weights, ms_config, threshold, device):
    from ms_model.oracle import LearnedDynMaskProvider
    return LearnedDynMaskProvider.from_event_array(
        ea, frame_ts_s, ms_weights, ms_config,
        device=device, threshold=threshold, ts_offset_s=0.0,
    )


# ---------------------------------------------------------------------------
# Loaders par dataset
# ---------------------------------------------------------------------------

def _load_rpg(scenedir: str, side: str = "left") -> tuple:
    from ms_model.io.loaders import load_events_rpg
    # Résolution variable : 180×240 (Davis240C) ou 260×346 (DVS346)
    evs_path = os.path.join(scenedir, f"evs_{side}.txt")
    tss_path = os.path.join(scenedir, f"tss_imgs_us_{side}.txt")
    H, W = (260, 346) if "simulation_3planes" in scenedir else (180, 240)
    ea = load_events_rpg(evs_path, H=H, W=W)
    rmap = None if "simulation_3planes" in scenedir else os.path.join(scenedir, f"rectify_map_{side}.h5")
    return ea, _frame_ts_s(tss_path), rmap


def _load_fpv(scenedir: str) -> tuple:
    from ms_model.io.loaders import load_events_fpv
    tss_path = os.path.join(scenedir, "images_timestamps_us.txt")
    ea = load_events_fpv(scenedir, H=260, W=346)
    return ea, _frame_ts_s(tss_path), os.path.join(scenedir, "rectify_map.h5")


def _load_hku(scenedir: str, side: str = "left") -> tuple:
    from ms_model.io.loaders import load_events_prophesee_h5
    h5_path = os.path.join(scenedir, f"evs_{side}.h5")
    tss_path = os.path.join(scenedir, f"tss_imgs_us_{side}.txt")
    ea = load_events_prophesee_h5(h5_path, H=260, W=346)
    return ea, _frame_ts_s(tss_path), os.path.join(scenedir, f"rectify_map_{side}.h5")


def _load_eds(scenedir: str) -> tuple:
    from ms_model.io.loaders import load_events_prophesee_h5
    h5_path = glob.glob(os.path.join(scenedir, "events.h5"))[0]
    tss_path = os.path.join(scenedir, "images_timestamps_us.txt")
    ea = load_events_prophesee_h5(h5_path, H=480, W=640)
    return ea, _frame_ts_s(tss_path), os.path.join(scenedir, "rectify_map_calib0.h5")


def _load_mvsec(scenedir: str, side: str = "left") -> tuple:
    from ms_model.io.loaders import load_events_mvsec
    h5_files = glob.glob(os.path.join(scenedir, "*_data.hdf5"))
    assert len(h5_files) == 1, f"Expected 1 MVSEC hdf5 in {scenedir}, found {h5_files}"
    tss_path = os.path.join(scenedir, f"tss_imgs_us_{side}.txt")
    ea = load_events_mvsec(h5_files[0], side=side, H=260, W=346)
    return ea, _frame_ts_s(tss_path), os.path.join(scenedir, f"rectify_map_{side}.h5")


def _load_vector(scenedir: str, side: str = "left") -> tuple:
    from ms_model.io.loaders import load_events_prophesee_h5
    seq = os.path.basename(scenedir)
    h5_path = os.path.join(scenedir, f"{seq}1.synced.{side}_event.hdf5")
    tss_path = os.path.join(scenedir, f"tss_imgs_us_{side}.txt")
    ea = load_events_prophesee_h5(h5_path, H=480, W=640)
    return ea, _frame_ts_s(tss_path), os.path.join(scenedir, f"rectify_map_{side}.h5")


def _windows_eds(scenedir: str) -> tuple:
    h5_path = glob.glob(os.path.join(scenedir, "events.h5"))[0]
    tss_path = os.path.join(scenedir, "images_timestamps_us.txt")
    return H5EventWindows(h5_path, 480, 640, os.path.join(scenedir, "rectify_map_calib0.h5")), _frame_ts_s(tss_path)


def _windows_tumvie(scenedir: str, camID: int = 2) -> tuple:
    side = "left" if camID == 2 else "right"
    h5_files = glob.glob(os.path.join(scenedir, f"*events_{side}.h5"))
    assert len(h5_files) == 1, f"Expected 1 TUM-VIE h5 in {scenedir}, found {h5_files}"
    tss_path = os.path.join(scenedir, f"{side}_images_undistorted", f"image_timestamps_{side}.txt")
    rmap = os.path.join(scenedir, f"rectify_map_{side}.h5")
    return H5EventWindows(h5_files[0], 720, 1280, rmap), _frame_ts_s(tss_path)


def _windows_vector(scenedir: str, side: str = "left") -> tuple:
    seq = os.path.basename(scenedir)
    h5_path = os.path.join(scenedir, f"{seq}1.synced.{side}_event.hdf5")
    tss_path = os.path.join(scenedir, f"tss_imgs_us_{side}.txt")
    rmap = os.path.join(scenedir, f"rectify_map_{side}.h5")
    return H5EventWindows(h5_path, 480, 640, rmap), _frame_ts_s(tss_path)


def _load_tumvie(scenedir: str, camID: int = 2) -> tuple:
    from ms_model.io.loaders import load_events_prophesee_h5
    side = "left" if camID == 2 else "right"
    h5_files = glob.glob(os.path.join(scenedir, f"*events_{side}.h5"))
    assert len(h5_files) == 1, f"Expected 1 TUM-VIE h5 in {scenedir}, found {h5_files}"
    tss_path = os.path.join(scenedir, f"{side}_images_undistorted", f"image_timestamps_{side}.txt")
    ea = load_events_prophesee_h5(h5_files[0], H=720, W=1280)
    return ea, _frame_ts_s(tss_path), os.path.join(scenedir, f"rectify_map_{side}.h5")


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

_LOADERS = {
    "rpg": lambda sd, side, camID: _load_rpg(sd, side),
    "fpv": lambda sd, side, camID: _load_fpv(sd),
    "hku": lambda sd, side, camID: _load_hku(sd, side),
    "eds": lambda sd, side, camID: _load_eds(sd),
    "mvsec": lambda sd, side, camID: _load_mvsec(sd, side),
    "vector": lambda sd, side, camID: _load_vector(sd, side),
    "tumvie": lambda sd, side, camID: _load_tumvie(sd, camID),
}

# Datasets lus par fenêtres (H5EventWindows) par la fabrique : trop d'events pour la RAM.
# `_LOADERS` (chargement complet) reste utilisé par tools/check_cross_alignment.py.
_WINDOWED = {
    "eds": lambda sd, side, camID: _windows_eds(sd),
    "tumvie": lambda sd, side, camID: _windows_tumvie(sd, camID),
    "vector": lambda sd, side, camID: _windows_vector(sd, side),
}


def make_ms_factory(
    dataset: str,
    ms_weights: str,
    ms_config: str,
    side: str = "left",
    camID: int = 2,
    threshold: Optional[float] = None,
    device: str = "cuda",
):
    """Retourne une factory `(scene, datapath_val) -> LearnedDynMaskProvider`.

    Args:
        dataset: 'rpg' | 'fpv' | 'hku' | 'eds' | 'mvsec' | 'vector' | 'tumvie'
        ms_weights: chemin vers le checkpoint MS (best.pt ou ms_final.pt).
        ms_config: chemin vers le yaml de config MS.
        side: 'left' | 'right' (pour datasets stéréo).
        camID: 2 | 3 (TUM-VIE seulement).
        threshold: None => score continu ; float => masque binaire.
        device: 'cuda' | 'cpu'.
    """
    assert dataset in _LOADERS, f"Dataset inconnu : '{dataset}'. Choix : {list(_LOADERS)}"
    loader = _LOADERS[dataset]

    def factory(scene: str, datapath_val: str):
        print(f"[MS-factory/{dataset}] Chargement events pour '{scene}'…")
        # Pas de repli silencieux : un provider None ferait passer DEVO vanilla pour la
        # méthode apprise dans la table.
        if dataset in _WINDOWED:
            ea, frame_ts_s = _WINDOWED[dataset](datapath_val, side, camID)
        else:
            ea, frame_ts_s, rmap_path = loader(datapath_val, side, camID)
            ea = rectify_events(ea, rmap_path, inplace=True)
        if len(ea) == 0:
            raise RuntimeError(f"[MS-factory/{dataset}] aucun event dans '{scene}' ({datapath_val})")
        return _make_provider(ea, frame_ts_s, ms_weights, ms_config, threshold, device)

    return factory
