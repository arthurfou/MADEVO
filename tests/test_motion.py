"""Tests CPU du filtre de mouvement des masques GT (`ms_model/motion.py`, T7).

Trajectoires synthétiques (`conftest.evimo2_long`) : la caméra bouge, STATIC_ID est fixe
dans le monde (donc mobile dans le repère caméra), MOVING_ID est immobile, se déplace de
5 cm entre 0,3 et 0,4 s, puis s'arrête.
"""
import numpy as np
import pytest

from conftest import MOVING_ID, STATIC_ID
from ms_model.io.loaders import load_evimo_mask
from ms_model.io.masks import FrameMasks
from ms_model.motion import (MODES, _world_tracks, filter_masks_to_moving,
                             moving_object_ids, moving_object_ids_at)


def test_positions_monde_reconstruites_malgre_la_camera_mobile(evimo2_long):
    """Les poses objet sont dans le repère caméra : sans les ramener dans le monde, le
    décor paraîtrait mobile dès que la caméra bouge."""
    ts, pos = _world_tracks(str(evimo2_long["path"]))[STATIC_ID]
    np.testing.assert_allclose(pos, np.tile([0.2, 0.0, 1.5], (len(pos), 1)), atol=1e-9)
    assert len(ts) == len(evimo2_long["gt_ts"]), "les trous de tracking n'ont pas de pose"


def test_mode_sequence_garde_les_objets_mobiles_seulement(evimo2_long):
    assert moving_object_ids(evimo2_long["path"], thresh_m=0.01) == {MOVING_ID}
    # 5 cm de déplacement : l'écart maximal à la médiane ne dépasse pas 5 cm
    assert moving_object_ids(evimo2_long["path"], thresh_m=0.06) == set()


def test_mode_frame_ne_marque_l_objet_que_pendant_son_mouvement(evimo2_long):
    ts = evimo2_long["gt_ts"]
    rel = ts - 0.5
    per_frame = moving_object_ids_at(evimo2_long["path"], ts, thresh_m=0.01, window_s=0.2)
    moving = np.array([MOVING_ID in s for s in per_frame])
    assert not any(STATIC_ID in s for s in per_frame)
    # au cœur du mouvement : mobile ; bien avant / bien après (hors fenêtre) : au repos
    assert moving[(rel > 0.33) & (rel < 0.37)].all()
    assert not moving[rel < 0.18].any()
    assert not moving[rel > 0.52].any()


def test_fenetre_plus_large_etend_l_etiquette(evimo2_long):
    ts = evimo2_long["gt_ts"]
    n = {w: sum(MOVING_ID in s for s in moving_object_ids_at(evimo2_long["path"], ts, 0.01, w))
         for w in (0.1, 0.2, 0.5)}
    assert 0 < n[0.1] <= n[0.2] <= n[0.5]


@pytest.mark.parametrize("mode", MODES)
def test_filter_masks_par_mode(evimo2_long, mode):
    fm = load_evimo_mask(evimo2_long["path"])
    out = filter_masks_to_moving(fm, evimo2_long["path"], mode, 0.01, 0.2)
    ids_per_frame = [{int(v) // 1000 for v in np.unique(m) if v} for m in out.masks]
    np.testing.assert_array_equal(out.ts, fm.ts)
    if mode == "all":
        assert out is fm
    elif mode == "sequence":
        assert all(ids == {MOVING_ID} for ids in ids_per_frame)
    else:
        assert any(ids == {MOVING_ID} for ids in ids_per_frame)
        assert any(ids == set() for ids in ids_per_frame)
        assert all(STATIC_ID not in ids for ids in ids_per_frame)


def test_filter_ne_modifie_pas_l_entree(evimo2_long):
    fm = load_evimo_mask(evimo2_long["path"])
    before = fm.masks.copy()
    filter_masks_to_moving(fm, evimo2_long["path"], "frame", 0.01, 0.2)
    np.testing.assert_array_equal(fm.masks, before)


def test_mode_inconnu_refuse(evimo2_long):
    fm = FrameMasks(masks=np.zeros((1, 4, 4), dtype=np.uint16), ts=np.zeros(1))
    with pytest.raises(ValueError, match="mask_motion"):
        filter_masks_to_moving(fm, evimo2_long["path"], "moving_only")
