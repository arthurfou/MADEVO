"""Tests CPU des voxel grids et de leur cache disque (T8).

Le cache (`ensure_voxel_cache`) est construit à la demande, versionné (`.v2`) depuis que les
caches v1 bâtis sur des timestamps décalés ont été trouvés (problème 5), et écrit de façon
atomique (fichier temporaire puis renommage).
"""
import numpy as np
import pytest
import torch

from conftest import make_evimo1_npz
from ms_model.data import evimo_dataset
from ms_model.data.evimo_dataset import VOXEL_CACHE_VERSION, ensure_voxel_cache, voxel_cache_path
from ms_model.io.events import EventArray
from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.representations.voxel import make_voxel_sequence, to_voxel_grid


# ----------------------------------------------------------------- voxel grid

def test_voxel_grid_conserve_la_polarite_totale():
    rng = np.random.default_rng(0)
    n, H, W = 500, 20, 30
    xs = rng.uniform(1, W - 2, n)   # loin des bords : rien n'est tronqué
    ys = rng.uniform(1, H - 2, n)
    ts = np.sort(rng.uniform(0, 1, n))
    ps = rng.integers(0, 2, n)
    grid = to_voxel_grid(xs, ys, ts, ps.copy(), H=H, W=W, nb_of_time_bins=5)
    assert grid.shape == (5, H, W)
    # polarité 0 → −1 ; l'interpolation trilinéaire répartit le poids sans le perdre
    assert grid.sum().item() == pytest.approx(float(np.where(ps == 0, -1, 1).sum()), abs=1e-3)


def test_voxel_grid_event_entier_tombe_dans_une_cellule():
    xs, ys = np.array([3.0, 7.0]), np.array([2.0, 5.0])
    ts, ps = np.array([0.0, 1.0]), np.array([1, 0])
    grid = to_voxel_grid(xs, ys, ts, ps, H=8, W=10, nb_of_time_bins=3)
    assert grid[0, 2, 3].item() == pytest.approx(1.0)
    assert grid[2, 5, 7].item() == pytest.approx(-1.0)
    assert torch.count_nonzero(grid).item() == 2


def test_voxel_sequence_un_grid_par_intervalle_et_zero_si_vide():
    events = np.array([[0.1, 1, 1, 1], [0.2, 2, 2, 0], [0.9, 3, 3, 1]], dtype=np.float64)
    ea = EventArray(events=events, H=6, W=8)
    seq = make_voxel_sequence(ea, np.array([0.0, 0.5, 0.8, 1.0]), nb_of_time_bins=4)
    assert seq.shape == (3, 4, 6, 8)
    assert torch.count_nonzero(seq[0]) > 0
    assert torch.count_nonzero(seq[1]) == 0, "aucun event dans [0,5 ; 0,8)"
    assert torch.count_nonzero(seq[2]) == 0, "un seul event : en dessous du minimum de 2"


# ---------------------------------------------------------------------- cache

def test_nom_du_cache_versionne(tmp_path):
    assert VOXEL_CACHE_VERSION == 2
    assert voxel_cache_path(tmp_path / "seq.npz", 5).name == "seq.voxels_bins5.v2.npy"
    assert voxel_cache_path(tmp_path / "scene_000000", 3).name == "scene_000000.voxels_bins3.v2.npy"


def test_cache_construit_sur_les_frames_gt(tmp_path):
    npz = make_evimo1_npz(tmp_path / "seq.npz", H=12, W=16, n_frames=5)
    path = ensure_voxel_cache(npz, 5)
    assert path == voxel_cache_path(npz, 5) and path.exists()
    cached = np.load(path)
    expected = make_voxel_sequence(load_events(npz), load_evimo_mask(npz).ts, nb_of_time_bins=5)
    assert cached.shape == (4, 5, 12, 16)
    np.testing.assert_allclose(cached, expected.numpy())
    assert not list(tmp_path.glob("*.tmp")), "aucun fichier temporaire ne doit rester"


def test_cache_reutilise_sans_recalcul(tmp_path, monkeypatch):
    npz = make_evimo1_npz(tmp_path / "seq.npz", H=12, W=16)
    first = ensure_voxel_cache(npz, 5)
    mtime = first.stat().st_mtime_ns

    def interdit(*a, **k):
        raise AssertionError("le cache existant aurait dû être relu")
    monkeypatch.setattr(evimo_dataset, "make_voxel_sequence", interdit)
    assert ensure_voxel_cache(npz, 5) == first
    assert first.stat().st_mtime_ns == mtime


def test_cache_v1_ignore(tmp_path):
    """Un cache v1 (timestamps décalés, problème 5) ne doit jamais être relu."""
    npz = make_evimo1_npz(tmp_path / "seq.npz", H=12, W=16)
    v1 = tmp_path / "seq.voxels_bins5.npy"
    np.save(v1, np.full((4, 5, 12, 16), 99.0, dtype=np.float32))
    path = ensure_voxel_cache(npz, 5)
    assert path != v1
    assert np.load(path).max() < 99.0


def test_cache_ecriture_atomique(tmp_path, monkeypatch):
    """Une écriture interrompue ne laisse pas de cache tronqué qui serait relu ensuite."""
    npz = make_evimo1_npz(tmp_path / "seq.npz", H=12, W=16)

    def save_interrompu(f, arr):
        f.write(b"\x93NUMPY tronqu")
        raise KeyboardInterrupt
    monkeypatch.setattr(evimo_dataset.np, "save", save_interrompu)
    with pytest.raises(KeyboardInterrupt):
        ensure_voxel_cache(npz, 5)
    assert not voxel_cache_path(npz, 5).exists()

    monkeypatch.undo()
    path = ensure_voxel_cache(npz, 5)
    assert np.load(path).shape == (5, 5, 12, 16)


def test_cache_overwrite_reconstruit(tmp_path):
    npz = make_evimo1_npz(tmp_path / "seq.npz", H=12, W=16)
    path = ensure_voxel_cache(npz, 5)
    np.save(path, np.zeros((1, 1, 1, 1), dtype=np.float32))
    assert np.load(ensure_voxel_cache(npz, 5)).shape == (1, 1, 1, 1)
    assert np.load(ensure_voxel_cache(npz, 5, overwrite=True)).shape == (5, 5, 12, 16)
