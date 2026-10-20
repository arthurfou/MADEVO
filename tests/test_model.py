"""Tests CPU du modèle MS (`convlstm_v4`) et des fournisseurs de masque des évals (T8)."""
import numpy as np
import pytest
import torch
import yaml

from conftest import REPO_ROOT, make_evimo1_npz
from ms_model.inference import load_config, load_model_from_checkpoint
from ms_model.io.loaders import load_evimo_mask
from ms_model.io.masks import FrameMasks
from ms_model.models import build_model
from ms_model.models.base import MODEL_REGISTRY
from ms_model.oracle import LearnedDynMaskProvider, OracleDynMaskProvider, TimestampMaskProvider


def _small_model(**kw):
    torch.manual_seed(0)
    return build_model("convlstm_v4", in_channels=5, hidden_dim=16, num_groups=4, **kw).eval()


# ------------------------------------------------------------------- modèle

def test_seul_convlstm_v4_est_enregistre():
    assert sorted(MODEL_REGISTRY) == ["convlstm_v4"]
    with pytest.raises(ValueError, match="inconnu"):
        build_model("convlstm_v2")


@pytest.mark.parametrize("hw", [(64, 80), (60, 84)])
def test_forward_cpu_sort_a_la_resolution_de_la_score_map(hw):
    x = torch.randn(2, 3, 5, *hw)
    with torch.no_grad():
        out = _small_model()(x)
    assert out.shape == (2, 3, 1, hw[0] // 4, hw[1] // 4)
    assert torch.isfinite(out).all()


def test_forward_causal():
    """ConvLSTM causal : la sortie à t ne dépend pas des entrées futures."""
    model = _small_model()
    x = torch.randn(1, 4, 5, 32, 40)
    y = x.clone()
    y[:, 2:] = torch.randn_like(y[:, 2:])
    with torch.no_grad():
        a, b = model(x), model(y)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_forward_backward_cpu():
    model = _small_model().train()
    out = model(torch.randn(1, 2, 5, 32, 40))
    out.mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_groupes_incompatibles_refuses():
    with pytest.raises(ValueError, match="divisible"):
        build_model("convlstm_v4", hidden_dim=20, num_groups=8)


def test_configs_de_campagne_construisent_le_modele():
    for name, hidden in (("v4_evimo2", 96), ("v4large_evimo2", 128), ("v4_evimo2_all", 96)):
        cfg = load_config(REPO_ROOT / "configs/ms" / f"{name}.yaml")
        model_cfg = dict(cfg["model"])
        model = build_model(model_cfg.pop("name"), in_channels=cfg["data"]["nb_time_bins"],
                            patch_size=cfg["data"]["patch_size"], **model_cfg)
        assert model.cell.hidden_dim == hidden, name


# -------------------------------------------------------- fournisseurs de masque

def test_timestamp_provider_plus_proche_voisin_et_offset():
    scores = torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 2, 2) / 2
    p = TimestampMaskProvider(np.array([1.0, 2.0, 3.0]), scores, device="cpu", ts_offset_s=100.0)
    assert p(101.4e6).shape == (1, 1, 2, 2)
    assert p(101.4e6).max().item() == 0.0        # → ts 1.0
    assert p(102.6e6).max().item() == 1.0        # → ts 3.0
    assert p(torch.tensor(150e6)).max().item() == 1.0, "au-delà de la fin : dernier score"


def test_timestamp_provider_trie_les_timestamps():
    scores = torch.tensor([1.0, 0.0]).view(2, 1, 1)
    p = TimestampMaskProvider(np.array([2.0, 1.0]), scores, device="cpu")
    assert p(1.0e6).item() == 0.0 and p(2.0e6).item() == 1.0


def test_oracle_depuis_npz_filtre_et_dilate(evimo2_seq, tmp_path_factory):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from convert_evimo2_to_evimo1 import convert_sequence
    dst = tmp_path_factory.mktemp("oracle") / "scene_test_000000"
    convert_sequence(str(evimo2_seq["path"]), str(dst), verbose=False)
    npz = dst.with_suffix(".npz")

    full = OracleDynMaskProvider.from_evimo_npz(npz, device="cpu", mask_motion="all")
    moving = OracleDynMaskProvider.from_evimo_npz(npz, device="cpu", mask_motion="sequence",
                                                  motion_thresh_m=0.01)
    thick = OracleDynMaskProvider.from_evimo_npz(npz, device="cpu", mask_motion="sequence",
                                                 motion_thresh_m=0.01, thicken_radius=4)
    t_us = evimo2_seq["gt_ts"][1] * 1e6
    assert full(t_us).shape == (1, 1, 120, 160)
    assert moving(t_us).sum() < full(t_us).sum(), "le décor statique doit disparaître"
    assert thick(t_us).sum() > moving(t_us).sum()
    np.testing.assert_array_equal(moving.ts, load_evimo_mask(evimo2_seq["path"]).ts)


def test_provider_appris_cpu_bout_en_bout(tmp_path):
    """Checkpoint + config d'entraînement → un score dans [0, 1] par frame, sur CPU."""
    cfg = {"data": {"nb_time_bins": 5, "patch_size": 4, "seq_len": 2},
           "model": {"name": "convlstm_v4", "hidden_dim": 16, "num_groups": 4}}
    (tmp_path / "ms.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(_small_model().state_dict(), tmp_path / "best_iou.pt")
    npz = make_evimo1_npz(tmp_path / "seq.npz", H=32, W=40, n_frames=6)

    model, data_cfg = load_model_from_checkpoint(tmp_path / "best_iou.pt", tmp_path / "ms.yaml",
                                                 torch.device("cpu"))
    assert data_cfg["seq_len"] == 2 and not model.training
    p = LearnedDynMaskProvider(npz, tmp_path / "best_iou.pt", tmp_path / "ms.yaml", device="cpu")
    assert len(p.ts) == 5 and p.scores.shape == (5, 8, 10)
    assert 0.0 <= p.scores.min() and p.scores.max() <= 1.0
    binary = LearnedDynMaskProvider(npz, tmp_path / "best_iou.pt", tmp_path / "ms.yaml",
                                    device="cpu", threshold=0.5)
    assert set(binary.scores.unique().tolist()) <= {0.0, 1.0}


def test_load_config_suit_extends():
    """Les configs d'ablation (`*_all.yaml`) héritent de leur parent par `extends:`."""
    cfg = load_config(REPO_ROOT / "configs/ms/v4_evimo2_all.yaml")
    assert cfg["model"]["name"] == "convlstm_v4" and cfg["data"]["mask_motion"] == "all"
    raw = yaml.safe_load((REPO_ROOT / "configs/ms/v4_evimo2_all.yaml").read_text())
    assert "model" not in raw, "sans résolution de `extends`, la config serait incomplète"


def test_frame_masks_longueur():
    assert len(FrameMasks(masks=np.zeros((3, 2, 2)), ts=np.zeros(3))) == 3
