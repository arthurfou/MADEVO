"""Tests CPU de la validation cross-dataset et des outils de campagne.

- les masques appris voient les events **rectifiés**, comme DEVO ;
- aucune erreur de chargement ne se transforme en DEVO vanilla déguisé ;
- les configs d'éval se valident sans GPU (dry-run) ;
- les options passées aux runners existent bien dans leur signature ;
- non-régression, séquences d'entraînement additionnelles, plan de campagne.
"""
import ast
import csv
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

from conftest import REPO_ROOT
from ms_model.io.events import EventArray
from ms_model.training.train import training_sequences
from utils.ms_dataset_factory import H5EventWindows, make_ms_factory, rectify_events
from utils.run_utils import (EVAL_DATASETS, ORACLE_DATASETS, eval_config_errors, eval_manifest,
                             resolve_config, resolve_eval_sources)

sys.path.insert(0, str(REPO_ROOT / "tools"))
import check_nonregression  # noqa: E402


# ------------------------------------------------------------- rectification

def _rmap(path, H, W, shift=(3.0, -2.0)):
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    rmap = np.stack([xs + shift[0], ys + shift[1]], axis=-1)
    with h5py.File(path, "w") as f:
        f.create_dataset("rectify_map", data=rmap)
    return path


def test_rectify_events_applique_la_carte_comme_devo(tmp_path):
    events = np.array([[0.0, 5, 7, 1], [0.1, 10, 2, 0]], dtype=np.float64)
    ea = EventArray(events=events, H=12, W=16, frame_ts=np.array([0.0, 0.1]))
    out = rectify_events(ea, str(_rmap(tmp_path / "r.h5", 12, 16)))
    # DEVO : rect = rectify_map[y, x] → (x', y')
    np.testing.assert_allclose(out.x, [8.0, 13.0])
    np.testing.assert_allclose(out.y, [5.0, 0.0])
    np.testing.assert_array_equal(out.t, ea.t)
    np.testing.assert_array_equal(ea.x, [5, 10]), "l'entrée ne doit pas être modifiée"


def test_rectify_events_sans_carte_rend_l_entree(tmp_path):
    ea = EventArray(events=np.zeros((1, 4)), H=4, W=4)
    assert rectify_events(ea, None) is ea
    assert rectify_events(ea, str(tmp_path / "absente.h5")) is ea


def test_rectify_events_refuse_une_carte_d_une_autre_resolution(tmp_path):
    ea = EventArray(events=np.zeros((1, 4)), H=12, W=16)
    with pytest.raises(AssertionError, match="rectification"):
        rectify_events(ea, str(_rmap(tmp_path / "r.h5", 10, 16)))


def _prophesee_h5(path, t_us, x, y, p, t_offset):
    """h5 au format DSEC/TUM-VIE : events/{t,x,y,p} + ms_to_idx (1er event à t >= ms·1000)."""
    ms_to_idx = np.searchsorted(t_us, np.arange(0, t_us[-1] // 1000 + 2) * 1000, side="left")
    with h5py.File(path, "w") as f:
        for k, v in dict(t=t_us, x=x, y=y, p=p).items():
            f.create_dataset(f"events/{k}", data=v)
        f.create_dataset("ms_to_idx", data=ms_to_idx.astype(np.uint64))
        f.create_dataset("t_offset", data=np.int64(t_offset))
    return path


def test_lecture_fenetree_donne_les_memes_voxels_que_le_chargement_complet(tmp_path):
    """TUM-VIE ne tient pas en RAM : la fabrique lit les events par fenêtres (`ms_to_idx`).
    Les voxels doivent être identiques à ceux du chargement complet, rectification comprise,
    y compris avec des events pile sur les bornes de frames."""
    from ms_model.io.loaders import load_events_prophesee_h5
    from ms_model.representations.voxel import make_voxel_sequence
    rng = np.random.default_rng(0)
    H, W, n, off = 12, 16, 50_000, 1_600_000_000_000_000
    t_us = np.sort(rng.integers(0, 2_000_000, n)).astype(np.int64)
    frames_us = np.unique(np.concatenate([np.linspace(100_000, 1_900_000, 37).astype(np.int64),
                                          t_us[[500, 20_000]]]))     # bornes pile sur des events
    t_us[1000:1010] = t_us[1000]
    h5 = _prophesee_h5(tmp_path / "ev.h5", t_us, rng.integers(0, W, n).astype(np.uint16),
                       rng.integers(0, H, n).astype(np.uint16), rng.integers(0, 2, n).astype(np.uint8), off)
    rm = str(_rmap(tmp_path / "r.h5", H, W, shift=(1.0, 0.0)))
    fts = (frames_us + off) / 1e6
    full = rectify_events(load_events_prophesee_h5(h5, H=H, W=W), rm)
    win = H5EventWindows(str(h5), H, W, rm)
    assert len(win) == n
    for a in range(0, len(fts) - 1, 8):
        f = fts[a:a + 9]
        np.testing.assert_array_equal(make_voxel_sequence(win.window(f[0], f[-1]), f).numpy(),
                                      make_voxel_sequence(full, f).numpy())


def test_voxels_par_dichotomie_identiques_au_masque_par_frame():
    """make_voxel_sequence borne chaque frame par searchsorted (O(log N)) au lieu d'un masque
    sur tout le flux (O(N), des heures sur EDS) : même ensemble d'events, mêmes voxels."""
    from ms_model.representations.voxel import make_voxel_sequence, to_voxel_grid
    import torch
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1, 20_000)); t[100:110] = t[100]
    ev = np.stack([t, rng.integers(0, 16, t.size), rng.integers(0, 12, t.size),
                   rng.integers(0, 2, t.size)], 1).astype(np.float64)
    ea = EventArray(events=ev, H=12, W=16)
    fts = np.sort(np.concatenate([[-0.1, t[100], 1.1], rng.uniform(0, 1, 20)]))
    old = []
    for i in range(1, len(fts)):
        m = (ea.t >= fts[i - 1]) & (ea.t < fts[i])
        old.append(torch.zeros(5, 12, 16) if m.sum() < 2
                   else to_voxel_grid(ea.x[m], ea.y[m], ea.t[m], ea.p[m], H=12, W=16))
    assert torch.equal(make_voxel_sequence(ea, fts), torch.stack(old))


def test_factory_ne_retombe_jamais_sur_vanilla(tmp_path):
    """Avant : toute erreur de chargement donnait un provider None, donc DEVO vanilla
    présenté comme M1/M2/M3 dans la table."""
    factory = make_ms_factory("mvsec", "absent.pt", "absent.yaml", device="cpu")
    with pytest.raises(Exception):
        factory("indoor_flying1_data", str(tmp_path / "indoor_flying1_data"))


# ------------------------------------------------------ configs d'éval cross

CROSS = [d for d in EVAL_DATASETS if d not in ORACLE_DATASETS]


@pytest.mark.parametrize("dataset", CROSS)
def test_config_cross_dataset_valide(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("MADEVO_RUNS", str(tmp_path))
    cfg = resolve_config(REPO_ROOT / "configs/eval" / f"{dataset}.yaml")
    assert cfg["dataset"]["name"] == dataset
    assert "oracle" not in cfg["methods"], "pas de masque GT de mouvement hors EVIMO2"
    assert (REPO_ROOT / cfg["dataset"]["split"]).exists()
    assert (REPO_ROOT / cfg["devo"]["config"]).exists()
    manifest = eval_manifest(cfg, 1, {}, None)
    assert eval_config_errors(cfg, manifest) == []


@pytest.mark.parametrize("dataset", CROSS)
def test_oracle_refuse_hors_evimo2(dataset):
    cfg = resolve_config(REPO_ROOT / "configs/eval" / f"{dataset}.yaml")
    cfg["methods"] = ["vanilla", "oracle"]
    errors = eval_config_errors(cfg, eval_manifest(cfg, 1, {}, None))
    assert any("pas d'oracle" in e for e in errors)


def test_dataset_inconnu_refuse():
    cfg = resolve_config(REPO_ROOT / "configs/eval/evimo2.yaml")
    cfg["dataset"]["name"] = "kitti"
    cfg["methods"] = ["vanilla"]
    assert any("inconnu" in e for e in eval_config_errors(cfg, eval_manifest(cfg, 1, {}, None)))


def _evaluate_params(runner: str) -> set:
    tree = ast.parse((REPO_ROOT / "evals/eval_evs" / f"{runner}.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    return {a.arg for a in fn.args.args}


def test_options_des_runners_existent_dans_leur_signature():
    """`evals/eval_table.py` passe ces options à `evaluate(...)` : une faute de frappe ne
    se verrait qu'au bout d'une heure de GPU."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_table", REPO_ROOT / "evals/eval_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = {"datapath", "split_file", "trials", "plot", "save", "stride",
              "dyn_mask_provider_factory"}
    for name, spec_ in module.DATASETS.items():
        params = _evaluate_params(spec_["runner"])
        missing = (common | set(spec_["runner_args"])) - params
        assert not missing, f"{name} : {missing} absent(s) de evaluate()"


def test_split_tumvie_public_sans_sequence_fantome():
    seqs = (REPO_ROOT / "splits/tumvie/tumvie_val_public.txt").read_text().split()
    assert "mocap-1d-trans-calibB" not in seqs and len(seqs) == 12


# ---------------------------------------------------------- non-régression

def _write_tables(d, mean, per_scene):
    d.mkdir(parents=True)
    with open(d / "central_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "ate_mean", "delta_pct_vs_vanilla"])
        w.writerow(["DEVO vanilla", f"{mean:.6f}", ""])
    with open(d / "per_scene.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "vanilla"])
        for s, v in per_scene.items():
            w.writerow([f"rpg_evs/{s}", f"{v:.6f}"])


def test_non_regression_evimo2(tmp_path):
    refs = yaml.safe_load((REPO_ROOT / "configs/nonregression.yaml").read_text())
    _write_tables(tmp_path / "ok", 2.70, {})     # une passe haute mais dans 2 σ
    _write_tables(tmp_path / "ko", 3.40, {})
    assert all(r[4] for r in check_nonregression.check(tmp_path / "ok", refs["evimo2"]))
    assert not all(r[4] for r in check_nonregression.check(tmp_path / "ko", refs["evimo2"]))
    # sur 4 seeds la tolérance se resserre : 2,70 de moyenne n'est plus acceptable
    seeds = []
    for i in range(4):
        _write_tables(tmp_path / f"s{i}", 2.70, {})
        seeds.append(tmp_path / f"s{i}")
    assert not all(r[4] for r in check_nonregression.check(seeds, refs["evimo2"]))


def test_non_regression_rpg_par_scene(tmp_path):
    refs = yaml.safe_load((REPO_ROOT / "configs/nonregression.yaml").read_text())["rpg"]
    good = {s: v * 1.05 for s, v in refs["vanilla_per_scene"].items()}
    _write_tables(tmp_path / "ok", 0.9, good)
    assert all(r[4] for r in check_nonregression.check(tmp_path / "ok", refs))
    bad = dict(good, reader=refs["vanilla_per_scene"]["reader"] * 2)
    _write_tables(tmp_path / "ko", 0.9, bad)
    failed = [r[0] for r in check_nonregression.check(tmp_path / "ko", refs) if not r[4]]
    assert "reader" in failed


# ------------------------------------------- séquences additionnelles (SfM)

def test_training_sequences_ajoute_les_extras():
    assert training_sequences({"train_sequences": ["a"]}) == ["a"]
    assert training_sequences({"train_sequences": ["a"], "extra_train_sequences": None}) == ["a"]
    assert training_sequences({"train_sequences": ["a"], "extra_train_sequences": ["b"]}) == ["a", "b"]


# ------------------------------------------------------------- campagne

def test_plan_de_campagne_se_resout(tmp_path):
    out = subprocess.run(
        ["bash", str(REPO_ROOT / "launch/campaign_evimo2.sh"), "--launcher", "local", "--cross",
         "--dry-run"], cwd=REPO_ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "MADEVO_LOGS": str(tmp_path)})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "configs du plan se résolvent" in out.stdout + out.stderr
    for cfg in ("configs/ms/v4_evimo2_all.yaml", "configs/coupled/m3_evimo2_all.yaml",
                "configs/eval/tumvie.yaml"):
        assert cfg in out.stdout
