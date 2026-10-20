"""Tests CPU des datasets d'entraînement sur une séquence EVIMO2 synthétique à trous (T8).

Ce qui est vérifié, c'est l'**alignement** : voxels, masques, poses et profondeurs d'un même
indice doivent décrire le même instant, malgré les trous de tracking (problème 5).
"""
import numpy as np
import pytest
import torch

from conftest import MOVING_ID, STATIC_ID, rotmat
from ms_model.data.evimo_clip_dataset import EvimoClipDataset
from ms_model.data.evimo_dataset import EvimoSegDataset, downsample_mask, voxel_cache_path
from ms_model.io.loaders import load_events
from ms_model.representations.voxel import make_voxel_sequence


def _gt_frame_indices(seq):
    """Indice global (trous compris) de chaque frame avec GT."""
    return [int(np.argmin(np.abs(seq["all_ts"] - t))) for t in seq["gt_ts"]]


def test_downsample_mask_max_pool():
    m = np.zeros((8, 8), dtype=np.uint16)
    m[5, 2] = 7000
    out = downsample_mask(m, 4)
    assert out.shape == (2, 2)
    assert out.tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_seg_dataset_voxels_et_masques_alignes(evimo2_seq_rw):
    seq = evimo2_seq_rw
    ds = EvimoSegDataset([seq["path"]], seq_len=3, nb_time_bins=5, patch_size=4)
    n_gt = len(seq["gt_ts"])
    assert len(ds) == (n_gt - 1) // 3
    voxels, masks = ds[1]
    assert voxels.shape == (3, 5, 480, 640) and masks.shape == (3, 120, 160)
    # le voxel i couvre [ts_i, ts_{i+1}) des frames GT, le masque i est celui de ts_i
    expected = make_voxel_sequence(load_events(seq["path"]), seq["gt_ts"], nb_of_time_bins=5)
    torch.testing.assert_close(voxels, expected[3:6])
    for j in range(3):
        torch.testing.assert_close(masks[j], downsample_mask(seq["masks"][3 + j], 4))
    assert voxel_cache_path(seq["path"], 5).exists()


@pytest.mark.parametrize("mode, ids", [("all", {STATIC_ID, MOVING_ID}), ("sequence", {MOVING_ID})])
def test_seg_dataset_applique_le_filtre_de_mouvement(evimo2_seq_rw, mode, ids):
    seq = evimo2_seq_rw
    ds = EvimoSegDataset([seq["path"]], seq_len=3, mask_motion=mode, motion_thresh_m=0.01)
    _, masks = ds[0]
    rows = {"static": masks[:, 25:35, 50:65], "moving": masks[:, 75:85, 100:110]}
    assert bool(rows["static"].any()) == (STATIC_ID in ids)
    assert bool(rows["moving"].all()) == (MOVING_ID in ids)


@pytest.fixture
def clip_ds(evimo2_seq_rw):
    return EvimoClipDataset([evimo2_seq_rw["path"]], n_frames=4, provide_gt_mask=True,
                            mask_motion="all")


def test_clip_dataset_poses_alignees_sur_les_frames_gt(evimo2_seq_rw, clip_ds):
    seq = evimo2_seq_rw
    images, poses, disps, intrinsics, gt_dyn, scene = clip_ds[2]
    assert images.shape == (4, 5, 480, 640) and poses.shape == (4, 7)
    assert intrinsics.shape == (4, 4) and gt_dyn.shape == (4, 120, 160)
    for j in range(4):
        cam_t, cam_q = seq["gt_cam"][2 + j]
        np.testing.assert_allclose(poses[j, :3].numpy(), cam_t, atol=1e-6)
        np.testing.assert_allclose(poses[j, 3:].numpy(), cam_q, atol=1e-6)


def test_clip_dataset_depth_et_masque_de_la_meme_frame(evimo2_seq_rw, clip_ds):
    """La depth de la frame k vaut 1500 + k mm (k = indice global) : on retrouve k."""
    seq = evimo2_seq_rw
    _, _, disps, _, gt_dyn, _ = clip_ds[2]
    k = _gt_frame_indices(seq)[2:6]
    np.testing.assert_allclose(disps[:, 0, 0].numpy(), [1000.0 / (1500 + i) for i in k], rtol=1e-5)
    for j in range(4):
        torch.testing.assert_close(gt_dyn[j], downsample_mask(seq["masks"][2 + j], 4))


def test_clip_dataset_voxels_du_meme_cache_que_le_ms(evimo2_seq_rw, clip_ds):
    seq = evimo2_seq_rw
    images = clip_ds[0][0]
    expected = make_voxel_sequence(load_events(seq["path"]), seq["gt_ts"], nb_of_time_bins=5)
    torch.testing.assert_close(images, expected[:4])


def test_clip_dataset_rotation_des_poses_coherente(evimo2_seq_rw, clip_ds):
    """Quaternion normé et convention [qx, qy, qz, qw] (DEVO l'attend dans cet ordre)."""
    _, poses, *_ = clip_ds[3]
    q = poses[:, 3:].numpy()
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
    cam_t, cam_q = evimo2_seq_rw["gt_cam"][3]
    np.testing.assert_allclose(rotmat(q[0]), rotmat(cam_q), atol=1e-6)
