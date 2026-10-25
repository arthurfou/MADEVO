"""Sauvegardes de visualisation (25/10) : score maps et patches des évals, résidus des
couplés, et le split du balayage de `loss_pos_weight`. Sans GPU : DEVO est simulé."""

import sys

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT
from utils.eval_dump import RecordingProvider, recording_factory
from utils.run_utils import resolve_config

sys.path.insert(0, str(REPO_ROOT / "tools"))
import cfg as cfg_tool  # noqa: E402


class FakeSlam:
    """Imite ce que RecordingProvider lit de DEVO : n, counter, patches_ (N, M, 3, P, P)."""

    def __init__(self, n_buffer=8, M=4, P=3):
        self.n = self.counter = 0
        self.patches_ = torch.zeros(n_buffer, M, 3, P, P)
        self.M = M

    def __call__(self, frame_xy, skip=False, keyframe_removed=False):
        if skip:
            return  # DEVO sort avant d'écrire quoi que ce soit (voxel vide)
        self.patches_[self.n, :, 0] = torch.as_tensor(frame_xy[:, 0])[:, None, None]
        self.patches_[self.n, :, 1] = torch.as_tensor(frame_xy[:, 1])[:, None, None]
        self.counter += 1
        self.n += 1
        if keyframe_removed:  # retrait d'une keyframe ancienne : recopie les cases < n
            k = self.n - 2
            for i in range(k, self.n - 1):
                self.patches_[i] = self.patches_[i + 1]
            self.n -= 1


def _run(provider, slam, frames):
    for t, (xy, kw) in enumerate(frames):
        score = provider(t * 1000)
        provider.before_frame(slam)
        slam(xy, **kw)
        provider.after_frame(slam)
    return provider.finish()


def test_provider_enregistre_scores_et_patches_sans_changer_le_score(tmp_path):
    dense = torch.zeros(1, 1, 6, 8)
    dense[..., 2, 3] = 1.0
    inner = lambda t: dense
    rec = RecordingProvider(inner, tmp_path / "m1" / "scene.npz")
    assert rec(0) is dense, "le score renvoyé à DEVO doit être exactement celui du provider"

    xy = np.array([[3, 2], [0, 0], [7, 5], [1, 4]], dtype=np.float32)
    slam = FakeSlam()
    out = _run(RecordingProvider(inner, tmp_path / "m1" / "scene.npz"), slam,
               [(xy, {}), (xy, {"skip": True}), (xy + 0, {}), (xy, {"keyframe_removed": True})])
    d = np.load(out)
    assert d["ts_us"].tolist() == [0, 2000, 3000], "la frame sautée n'est pas enregistrée"
    assert d["scores"].shape == (3, 6, 8) and d["scores"].dtype == np.uint8
    assert d["scores"][0, 2, 3] == 255
    np.testing.assert_allclose(d["patch_xy"][2], xy)  # même après retrait d'une keyframe
    assert d["patch_dyn"][:, 0].tolist() == [255] * 3 and d["patch_dyn"][:, 1].tolist() == [0] * 3


def test_vanilla_enregistre_les_patches_sans_score(tmp_path):
    make = recording_factory(None, tmp_path, "vanilla")
    rec = make("imo/eval/scene13_dyn_test_00", "/unused")
    assert rec(0) is None, "vanilla : DEVO doit recevoir None, donc rester strictement vanilla"
    xy = np.zeros((4, 2), np.float32)
    out = _run(make("imo/eval/scene13_dyn_test_00", "/unused"), FakeSlam(), [(xy, {})])
    assert out == tmp_path / "vanilla" / "imo__eval__scene13_dyn_test_00.npz"
    assert "scores" not in np.load(out).files


def test_keep_scores_false_ne_garde_que_les_patches(tmp_path):
    rec = RecordingProvider(lambda t: torch.ones(6, 8), tmp_path / "x.npz", keep_scores=False)
    out = _run(rec, FakeSlam(), [(np.zeros((4, 2), np.float32), {})])
    d = np.load(out)
    assert "scores" not in d.files and d["patch_dyn"].tolist() == [[255] * 4]


@pytest.mark.parametrize("name", ["evimo2", "evimo2_all", "rpg", "mvsec", "eds", "vector", "tumvie"])
def test_les_evals_ne_sauvent_rien_par_defaut(name):
    assert resolve_config(REPO_ROOT / f"configs/eval/{name}.yaml")["save_viz"] is False


# ----------------------------------------------------- couplés : résidus et GT de diagnostic

def _coupled_argv(name, tmp_path):
    path = REPO_ROOT / "configs/coupled" / name
    return cfg_tool.argv_coupled(resolve_config(path), path, tmp_path, seed=1)


def _coupled_argv_set(name, tmp_path, **sets):
    path = REPO_ROOT / "configs/coupled" / name
    cfg = resolve_config(path)
    cfg.update(sets)
    return cfg_tool.argv_coupled(cfg, path, tmp_path, seed=1)


def test_couples_ne_sauvent_les_residus_que_sous_save_viz(tmp_path):
    argv = _coupled_argv("m3_evimo2_scratch.yaml", tmp_path)
    assert argv[argv.index("--dump_every") + 1] == "0"
    argv = _coupled_argv_set("m3_evimo2_scratch.yaml", tmp_path, save_viz=True)
    assert argv[argv.index("--dump_every") + 1] == "500"


def test_m3_charge_le_gt_en_diagnostic_seulement(tmp_path):
    argv = _coupled_argv("m3_evimo2_scratch.yaml", tmp_path)
    assert "--diag_gt_mask" in argv and "--provide_gt_mask" not in argv
    assert argv[argv.index("--soft_prior_target") + 1] == "0.02"


def test_m2_garde_sa_supervision(tmp_path):
    argv = _coupled_argv("m2_evimo2_scratch.yaml", tmp_path)
    assert "--provide_gt_mask" in argv and "--diag_gt_mask" not in argv


def test_la_loss_de_m2_ne_lit_le_gt_que_sous_provide_gt_mask():
    """Le GT de diagnostic de M3 ne doit jamais tomber dans la branche supervisée."""
    src = (REPO_ROOT / "train_coupled.py").read_text()
    assert "elif args.provide_gt_mask and gt_dyn is not None and args.mask_weight > 0:" in src


# --------------------------------------------------------- balayage de loss_pos_weight

def test_split_du_balayage_reste_dans_le_train():
    ref = resolve_config(REPO_ROOT / "configs/ms/v4_evimo2.yaml")["data"]
    sweep = resolve_config(REPO_ROOT / "configs/ms/v4_evimo2_holdout.yaml")["data"]
    train, val = set(sweep["train_sequences"]), set(sweep["val_sequences"])
    assert len(val) == 3 and not train & val
    assert train | val == set(ref["train_sequences"]), "les 21 séquences train, rien d'autre"
    assert not val & set(ref["val_sequences"]), "jamais le split d'éval des tables"
    assert {k: sweep[k] for k in ("mask_motion", "mask_thicken_radius")} == \
           {k: ref[k] for k in ("mask_motion", "mask_thicken_radius")}
