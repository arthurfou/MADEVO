"""Tests CPU du logger wandb commun (T5).

On vérifie les fabriques d'images (pures, sans wandb) et le fait que le logger ne fait
jamais échouer un entraînement quand wandb est absent ou désactivé.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.wandb_logger import (  # noqa: E402
    JOB_TYPES, RunLogger, mask_image, patches_image, voxel_image,
)


# --------------------------------------------------------------------------- images

def test_voxel_image_somme_les_bins_et_code_la_polarite():
    voxel = torch.zeros(5, 8, 10)
    voxel[0, 2, 3] = 1.0      # polarité positive -> rouge
    voxel[1, 5, 7] = -1.0     # polarité négative -> bleu
    img = voxel_image(voxel)

    assert img.shape == (8, 10, 3) and img.dtype == np.uint8
    assert img[2, 3, 0] == 255 and img[2, 3, 2] == 0
    assert img[5, 7, 2] == 255 and img[5, 7, 0] == 0
    assert img[0, 0].sum() == 0, "une cellule sans event reste noire"


def test_voxel_image_accepte_une_image_2d():
    assert voxel_image(np.zeros((4, 6))).shape == (4, 6, 3)


def test_voxel_image_refuse_une_forme_inattendue():
    with pytest.raises(ValueError):
        voxel_image(np.zeros((2, 3, 4, 5)))


def test_mask_image_monotone_de_noir_a_jaune():
    img = mask_image(torch.tensor([[0.0, 0.5, 1.0]]))
    assert img.shape == (1, 3, 3)
    assert img[0, 0].sum() == 0
    assert tuple(img[0, 2]) == (255, 255, 0)
    assert img[0, 1, 0] == pytest.approx(127, abs=2)


def test_mask_image_reduit_les_dimensions_superflues():
    """Le masque arrive souvent en (1, H, W) ou (B, 1, H, W) selon l'appelant."""
    assert mask_image(torch.rand(1, 4, 5)).shape == (4, 5, 3)
    assert mask_image(torch.rand(1, 1, 4, 5)).shape == (4, 5, 3)


def test_patches_image_colore_selon_le_score_dynamique():
    voxel = torch.zeros(5, 20, 20)
    xy = torch.tensor([[5.0, 5.0], [15.0, 15.0]])
    img = patches_image(voxel, xy, dyn_score=torch.tensor([0.0, 1.0]), radius=0)
    assert tuple(img[5, 5]) == (0, 255, 0), "statique = vert (gardé par la BA)"
    assert tuple(img[15, 15]) == (255, 0, 0), "dynamique = rouge (écarté)"


def test_patches_image_ignore_les_patches_hors_cadre():
    img = patches_image(torch.zeros(5, 10, 10), torch.tensor([[-3.0, 2.0], [99.0, 99.0]]))
    assert img.shape == (10, 10, 3)
    assert img.sum() == 0


# ---------------------------------------------------------------------------- logger

def test_logger_desactive_ne_leve_jamais(tmp_path, monkeypatch):
    """Sans wandb importable, le logger doit être un no-op silencieux.

    C'est la garantie qui compte : un problème de logging ne doit jamais tuer un
    entraînement de plusieurs heures.
    """
    class _Bloque:
        def find_spec(self, name, path=None, target=None):
            if name == "wandb":
                raise ImportError("wandb indisponible (test)")
            return None

    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    sys.meta_path.insert(0, _Bloque())
    try:
        logger = RunLogger("train_ms", {"a": 1}, exp="exp", seed=1, run_dir=tmp_path)
        assert not logger.enabled and logger.id is None
        logger.log({"ms/loss": 1.0}, step=0)
        logger.log_histogram("h", torch.rand(10), step=0)
        logger.log_images({"img": np.zeros((2, 2, 3), np.uint8)}, step=0)
        logger.log_table("t", ["a"], [[1]])
        logger.summary({"x": 1})
        assert logger.should_log_images(0) is False
        logger.finish()
    finally:
        sys.meta_path.pop(0)


def test_job_type_inconnu_refuse():
    with pytest.raises(ValueError):
        RunLogger("entrainement", {}, exp="exp")


def test_les_trois_types_de_jobs_sont_declares():
    assert set(JOB_TYPES) == {"train_ms", "train_coupled", "eval"}


def test_should_log_images_respecte_la_periode():
    logger = RunLogger.__new__(RunLogger)
    logger.enabled, logger.images_every = True, 500
    assert logger.should_log_images(0) and logger.should_log_images(1000)
    assert not logger.should_log_images(499)

    logger.images_every = 0
    assert not logger.should_log_images(0), "0 = jamais d'images"
