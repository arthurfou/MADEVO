"""Tests CPU de la couche configs / dossiers de runs (T4).

Tout tourne sans GPU et sans dataset : ce sont les briques que `jobs/` et `launch/`
utilisent pour résoudre une config, nommer un dossier de run et retrouver un checkpoint.
"""
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import run_utils  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fusion et surcharges

def test_deep_merge_garde_les_cles_non_touchees():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = run_utils.deep_merge(base, {"a": {"y": 20}, "c": 4})
    assert out == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}, "la base ne doit pas être modifiée"


def test_overrides_typees_comme_du_yaml():
    cfg = {"train": {"steps": 1}, "methods": [], "ms": {"init": "x"}, "flag": False}
    out = run_utils.apply_overrides(cfg, [
        "train.steps=40000", "methods=[vanilla,oracle]", "ms.init=null", "flag=true"])
    assert out["train"]["steps"] == 40000 and isinstance(out["train"]["steps"], int)
    assert out["methods"] == ["vanilla", "oracle"]
    assert out["ms"]["init"] is None
    assert out["flag"] is True


def test_override_dune_cle_inconnue_echoue():
    """C'est ce qui attrape les fautes de frappe dans les --set du catalogue."""
    with pytest.raises(KeyError):
        run_utils.apply_overrides({"train": {"steps": 1}}, ["train.step=2"])
    with pytest.raises(KeyError):
        run_utils.apply_overrides({"train": {"steps": 1}}, ["trian.steps=2"])


def test_override_prefixe_plus_cree_la_cle():
    out = run_utils.apply_overrides({"a": 1}, ["+b.c=2"])
    assert out["b"]["c"] == 2


def test_override_mal_formee_echoue():
    with pytest.raises(ValueError):
        run_utils.apply_overrides({}, ["train.steps"])


def test_expand_env_et_defaut(monkeypatch):
    monkeypatch.setenv("MADEVO_TEST_VAR", "/data")
    assert run_utils.expand_env({"p": "${MADEVO_TEST_VAR}/x"}) == {"p": "/data/x"}
    assert run_utils.expand_env("${MADEVO_ABSENT:-repli}") == "repli"
    with pytest.raises(KeyError):
        run_utils.expand_env("${MADEVO_ABSENT_SANS_DEFAUT}")


# ---------------------------------------------------------------------- héritage extends

def test_extends_fusionne_la_base(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(
        {"train": {"steps": 40000, "clip": 10.0}, "loss": {"mode": "supervised"}}))
    (tmp_path / "enfant.yaml").write_text(
        "extends: base.yaml\nexp: test\nloss:\n  mode: selfsup\n")
    cfg = run_utils.resolve_config(tmp_path / "enfant.yaml")
    assert cfg["exp"] == "test"
    assert cfg["loss"]["mode"] == "selfsup"
    assert cfg["train"] == {"steps": 40000, "clip": 10.0}, "la base doit être héritée"
    assert "extends" not in cfg


# ------------------------------------------------------------------- dossiers de runs

def test_run_dir_sans_niveau_de_type(monkeypatch, tmp_path):
    monkeypatch.setenv("MADEVO_RUNS", str(tmp_path))
    assert run_utils.run_dir("m3_evimo2", "2026-11-04_0910", 3) == \
        tmp_path / "m3_evimo2" / "2026-11-04_0910" / "seed3"


def test_eval_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MADEVO_EVALS", str(tmp_path))
    assert run_utils.eval_dir("evimo2", "evimo2_scratch", "2026-11-07_1100", 2) == \
        tmp_path / "evimo2" / "evimo2_scratch" / "2026-11-07_1100" / "seed2"


@pytest.mark.parametrize("bad", ["2026-11-04", "latest", "2026-11-04_091", ""])
def test_launch_id_invalide_refuse(bad):
    with pytest.raises(ValueError):
        run_utils.run_dir("exp", bad, 1)


# ------------------------------------------------------------------- résolution @latest

def _make_run(root, exp, launch_id, seed, done=True):
    d = root / exp / launch_id / f"seed{seed}"
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    if done:
        (d / "DONE").write_text("2026-11-04T09:10:00\n")
    return d


def test_at_latest_prend_le_dernier_lancement_termine(monkeypatch, tmp_path):
    monkeypatch.setenv("MADEVO_RUNS", str(tmp_path))
    _make_run(tmp_path, "v4_evimo2", "2026-11-01_0900", 1)
    attendu = _make_run(tmp_path, "v4_evimo2", "2026-11-03_1200", 1)
    _make_run(tmp_path, "v4_evimo2", "2026-11-05_1500", 1, done=False)  # pas DONE
    assert run_utils.resolve_ref("runs/v4_evimo2/@latest", 1) == attendu
    # le préfixe runs/ est optionnel
    assert run_utils.resolve_ref("v4_evimo2/@latest", 1) == attendu


def test_at_latest_est_par_seed(monkeypatch, tmp_path):
    """Un run couplé doit partir du MS de LA MÊME seed, pas du dernier tous seeds confondus."""
    monkeypatch.setenv("MADEVO_RUNS", str(tmp_path))
    seed1 = _make_run(tmp_path, "v4_evimo2", "2026-11-01_0900", 1)
    _make_run(tmp_path, "v4_evimo2", "2026-11-03_1200", 2)
    assert run_utils.resolve_ref("v4_evimo2/@latest", 1) == seed1
    with pytest.raises(FileNotFoundError):
        run_utils.resolve_ref("v4_evimo2/@latest", 5)


def test_ref_nulle_signifie_from_scratch():
    assert run_utils.resolve_ref(None, 1) is None
    assert run_utils.resolve_ref("null", 1) is None


def test_ref_explicite_ne_demande_pas_done(monkeypatch, tmp_path):
    monkeypatch.setenv("MADEVO_RUNS", str(tmp_path))
    assert run_utils.resolve_ref("v4_evimo2/2026-11-09_0800", 4) == \
        tmp_path / "v4_evimo2" / "2026-11-09_0800" / "seed4"


# ------------------------------------------------------------------------- reprise

def test_dernier_checkpoint_couple_est_le_plus_avance(tmp_path):
    for step in (2000, 10000, 4000):
        (tmp_path / f"coupled_step{step}.pt").write_text("x")
    assert run_utils.latest_coupled_checkpoint(tmp_path).name == "coupled_step10000.pt"


def test_pas_de_checkpoint_a_reprendre(tmp_path):
    assert run_utils.latest_coupled_checkpoint(tmp_path) is None
    assert run_utils.latest_coupled_checkpoint(tmp_path / "absent") is None


# ------------------------------------------------------ les configs du repo sont valides

ALL_CONFIGS = sorted(p for p in (REPO_ROOT / "configs").rglob("*.yaml")
                     if not p.name.startswith("_"))


def test_il_y_a_bien_des_configs():
    assert ALL_CONFIGS, "aucune config trouvée sous configs/"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_chaque_config_se_resout(path):
    cfg = run_utils.resolve_config(path)
    assert cfg, f"{path} résout à vide"
    if path.parent.name in ("ms", "coupled"):
        assert "exp" in cfg, "une config d'entraînement doit porter un `exp`"
        run_utils.run_dir(cfg["exp"], "2026-11-04_0910", 1)  # nommage valide


@pytest.mark.parametrize("path", [p for p in ALL_CONFIGS if p.parent.name == "coupled"],
                         ids=lambda p: p.stem)
def test_budget_couple_conforme_a_la_decision(path):
    """40 000 steps, dégel de DEVO à 10 000 : décision arrêtée pour le papier."""
    cfg = run_utils.resolve_config(path)
    assert cfg["train"]["steps"] == 40000
    assert cfg["train"]["freeze_devo_steps"] == 10000
    assert cfg["loss"]["mode"] in ("supervised", "selfsup")
    assert cfg["ms"]["ckpt"] == "best_iou.pt"


@pytest.mark.parametrize("path", [p for p in ALL_CONFIGS if p.parent.name == "ms"],
                         ids=lambda p: p.stem)
def test_configs_ms_sur_v4_et_mode_de_masque_connu(path):
    cfg = run_utils.resolve_config(path)
    assert cfg["model"]["name"] == "convlstm_v4", "seul convlstm_v4 est retenu"
    assert cfg["data"]["mask_motion"] in ("all", "sequence", "frame")


def test_aucun_chemin_absolu_dans_les_configs():
    for path in ALL_CONFIGS:
        text = path.read_text()
        assert "/home/" not in text, f"{path} contient un chemin personnel"


def test_manifest_dune_reference_non_satisfaite_sort_en_code_3(monkeypatch, tmp_path):
    """Code 3 = « l'entraînement évalué n'a pas encore tourné pour cette seed ».

    C'est le contrat que `jobs/_common.sh` et les lanceurs lisent pour distinguer un
    `--dry-run` légitimement incomplet d'une vraie erreur de configuration (code 1).
    """
    import subprocess
    env = {**os.environ, "MADEVO_RUNS": str(tmp_path / "runs")}
    out = subprocess.run(
        [sys.executable, "tools/cfg.py", "manifest", "configs/eval/evimo2.yaml",
         "--seed", "1", "--out", str(tmp_path / "manifest.yaml")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 3, out.stderr
    assert "non résolue" in out.stderr
