"""Tests CPU des loaders (T8) : résolution, alignement des masques EVIMO2, conversion npz.

Remplace les trois tests du stage, qui lisaient des séquences EVIMO1 d'un dossier disparu
et étaient toujours sautés. Tout tourne sur les fixtures synthétiques de `conftest.py`.
"""
import sys

import numpy as np
import pytest

from conftest import (EVIMO2_H, EVIMO2_W, MOVING_ID, REPO_ROOT, STATIC_ID,
                      make_evimo1_npz)
from ms_model.io.loaders import (EVIMO_H, EVIMO_W, load_events, load_evimo_mask,
                                 load_evimo_npz, load_npy_canonical)
from ms_model.motion import filter_masks_to_moving

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from convert_evimo2_to_evimo1 import convert_sequence  # noqa: E402


# ----------------------------------------------------------- résolution (bug 1)

@pytest.mark.parametrize("hw", [(EVIMO2_H, EVIMO2_W), (EVIMO_H, EVIMO_W), (24, 32)])
def test_load_evimo_npz_lit_la_resolution_dans_le_masque(tmp_path, hw):
    """Bug 1 du stage : tout npz était lu en 260×346, y compris les évals EVIMO2 (480×640)."""
    ea = load_evimo_npz(make_evimo1_npz(tmp_path / "s.npz", H=hw[0], W=hw[1], n_frames=2))
    assert (ea.H, ea.W) == hw


def test_load_evimo_npz_sans_masque_retombe_sur_evimo1(tmp_path):
    ea = load_evimo_npz(make_evimo1_npz(tmp_path / "s.npz", with_mask=False))
    assert (ea.H, ea.W) == (EVIMO_H, EVIMO_W)


def test_load_evimo_npz_format_et_frame_ts(tmp_path):
    path = make_evimo1_npz(tmp_path / "s.npz")
    ea = load_evimo_npz(path)
    with np.load(path) as data:
        events, index = data["events"], data["index"]
    assert ea.events.shape == events.shape and ea.events.shape[1] == 4
    assert np.all(np.diff(ea.t) >= 0)
    assert set(np.unique(ea.p)) <= {0.0, 1.0}
    np.testing.assert_array_equal(ea.frame_ts, events[np.clip(index, 0, len(events) - 1), 0])


def test_load_npy_canonical_convertit_les_microsecondes(tmp_path):
    raw = np.array([[1_000_000, 3, 4, 1], [2_500_000, 5, 6, 0]], dtype=np.float64)
    np.save(tmp_path / "evs.npy", raw)
    np.savetxt(tmp_path / "tss_imgs_us.txt", [1_000_000, 3_000_000])
    ea = load_npy_canonical(tmp_path / "evs.npy", frame_ts_path=tmp_path / "tss_imgs_us.txt")
    np.testing.assert_allclose(ea.t, [1.0, 2.5])
    np.testing.assert_allclose(ea.x, [3, 5])
    np.testing.assert_allclose(ea.frame_ts, [1.0, 3.0])
    assert (ea.H, ea.W) == (EVIMO_H, EVIMO_W)


# ------------------------------------------- EVIMO2 : alignement sur les frames GT

def test_masques_evimo2_alignes_sur_les_frames_avec_gt(evimo2_seq):
    """Problème 5 : le i-ème masque va avec la i-ème frame **possédant une GT**."""
    fm = load_evimo_mask(evimo2_seq["path"])
    np.testing.assert_array_equal(fm.ts, evimo2_seq["gt_ts"])
    np.testing.assert_array_equal(fm.masks, evimo2_seq["masks"])
    # l'appariement naïf (toutes frames confondues) donnerait d'autres instants
    assert not np.array_equal(fm.ts, evimo2_seq["all_ts"][:len(fm.ts)])


def test_evimo2_nombre_de_masques_incoherent_echoue(evimo2_seq_rw):
    path = evimo2_seq_rw["path"]
    with np.load(path / "dataset_mask.npz") as f:
        kept = {k: f[k] for k in sorted(f.files)[:-1]}
    np.savez_compressed(path / "dataset_mask.npz", **kept)
    with pytest.raises(ValueError, match="alignement impossible"):
        load_evimo_mask(path)


def test_events_evimo2_en_480x640_et_bornes_gt(evimo2_seq):
    ea = load_events(evimo2_seq["path"])
    assert (ea.H, ea.W) == (EVIMO2_H, EVIMO2_W)
    np.testing.assert_array_equal(ea.frame_ts, evimo2_seq["gt_ts"])
    assert ea.events.shape[1] == 4 and np.all(np.diff(ea.t) >= 0)


# --------------------------------- conversion EVIMO2 → npz des évals (bug 6)

@pytest.fixture(scope="module")
def converted(evimo2_seq, tmp_path_factory):
    dst = tmp_path_factory.mktemp("conv") / "scene_test_000000"
    convert_sequence(str(evimo2_seq["path"]), str(dst), verbose=False)
    return dst.with_suffix(".npz")


def test_conversion_garde_l_alignement_du_dossier_natif(evimo2_seq, converted):
    """Bug 6 : la conversion appariait le i-ème masque à frames[i], trous compris."""
    native, npz = load_evimo_mask(evimo2_seq["path"]), load_evimo_mask(converted)
    np.testing.assert_array_equal(npz.ts, native.ts)
    np.testing.assert_array_equal(npz.masks, native.masks)


@pytest.mark.parametrize("mode", ["sequence", "frame"])
def test_conversion_donne_le_meme_filtre_de_mouvement(evimo2_seq, converted, mode):
    """L'oracle des évals lit le npz, l'entraînement lit le dossier : même masque filtré."""
    native, npz = load_evimo_mask(evimo2_seq["path"]), load_evimo_mask(converted)
    a = filter_masks_to_moving(native, evimo2_seq["path"], mode, 0.01, 0.2)
    b = filter_masks_to_moving(npz, converted, mode, 0.01, 0.2)
    np.testing.assert_array_equal(a.masks, b.masks)


def test_conversion_index_par_frame_et_resolution(evimo2_seq, converted):
    ea = load_evimo_npz(converted)
    assert (ea.H, ea.W) == (EVIMO2_H, EVIMO2_W)
    # `index` = premier event de chaque frame GT (et non la table de 10 ms d'EVIMO2)
    t = np.load(evimo2_seq["path"] / "dataset_events_t.npy")
    first = np.searchsorted(t, evimo2_seq["gt_ts"], side="left")
    with np.load(converted) as data:
        np.testing.assert_array_equal(data["index"], first)
        assert len(data["depth"]) == len(data["mask"]) == len(evimo2_seq["gt_ts"])


def test_ids_objets_presents_dans_les_masques(evimo2_seq):
    ids = {int(v) // 1000 for v in np.unique(load_evimo_mask(evimo2_seq["path"]).masks) if v}
    assert ids == {STATIC_ID, MOVING_ID}
