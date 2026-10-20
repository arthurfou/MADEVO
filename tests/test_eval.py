"""Tests CPU de l'éval unifiée (T6).

Sans GPU ni dataset : sélection des méthodes, provenance des checkpoints, format de la
table, et surtout `mean_scene_ate`, dont la version buggée du stage sous-estimait l'ATE de
toutes les tables centrales (cf. INTEGRITY_CHECK.md, bug 2).
"""
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import mean_scene_ate as _mean_scene_ate  # noqa: E402
from utils.run_utils import (eval_manifest, mask_definition_mismatches, resolve_config,  # noqa: E402
                             resolve_eval_sources)


# ------------------------------------------------------------------ mean_scene_ate

def _mean_scene_ate_reference(d):
    """Réimplémentation indépendante, pour ne pas tester la fonction par elle-même."""
    vals = [float(v) for k, v in d.items() if k not in ("AUC", "AVG")]
    return sum(vals) / len(vals)


def test_mean_scene_ate_ignore_auc_et_avg():
    """Le bug du stage : `(somme des scènes + AUC + AVG) / (n + 2)`."""
    results = {
        "evimo2/scene13_dyn_test_00": 2.0,
        "evimo2/scene13_dyn_test_05": 3.0,
        "evimo2/scene14_dyn_test_03": 4.0,
        "AUC": 0.5,     # agrégat, PAS un ATE
        "AVG": 100.0,   # agrégat, PAS un ATE
    }
    assert _mean_scene_ate(results) == pytest.approx(3.0)
    assert _mean_scene_ate(results) == pytest.approx(_mean_scene_ate_reference(results))

    buggee = sum(float(v) for v in results.values()) / len(results)
    assert buggee != pytest.approx(3.0), "la formule buggée doit bien donner autre chose"


def test_mean_scene_ate_sans_agregats():
    assert _mean_scene_ate({"a": 1.0, "b": 2.0}) == pytest.approx(1.5)


def test_mean_scene_ate_dict_vide_est_nan():
    import math
    assert math.isnan(_mean_scene_ate({}))
    assert math.isnan(_mean_scene_ate({"AUC": 1.0, "AVG": 2.0}))


# ------------------------------------------------------------ provenance / manifest

@pytest.fixture
def faux_runs(tmp_path, monkeypatch):
    """Arborescence de runs terminés, pour 3 seeds."""
    monkeypatch.setenv("MADEVO_RUNS", str(tmp_path / "runs"))
    root = tmp_path / "runs"
    for exp, files in (("v4_evimo2", ["best_iou.pt"]),
                       ("m2_evimo2", ["devo_final.pth", "ms_final.pt"]),
                       ("m3_evimo2", ["devo_final.pth", "ms_final.pt"])):
        for seed in (1, 2, 3):
            d = root / exp / "2026-11-01_0900" / f"seed{seed}" / "checkpoints"
            d.mkdir(parents=True)
            for f in files:
                (d / f).write_text("x")
            (d.parent / "DONE").write_text("ok")
    return root


def _eval_cfg(**surcharges):
    cfg = resolve_config(REPO_ROOT / "configs/eval/evimo2.yaml")
    cfg.update(surcharges)
    return cfg


def test_sources_resout_les_checkpoints_de_la_bonne_seed(faux_runs):
    sources, ms_arch = resolve_eval_sources(_eval_cfg(), seed=2)
    assert ms_arch == "configs/ms/v4_evimo2.yaml"
    assert set(sources) == {"m1", "m2", "m3"}
    assert "seed2" in str(sources["m1"]["ms"])
    assert sources["m1"]["ms"].name == "best_iou.pt", "le défaut doit rester best_iou.pt"
    assert sources["m2"]["devo"].name == "devo_final.pth"
    assert sources["m2"]["ms"].name == "ms_final.pt"
    assert sources["m1"]["devo"] is None, "M1 ne change pas les poids DEVO"


def test_methods_pilote_les_checkpoints_charges(faux_runs):
    sources, _ = resolve_eval_sources(_eval_cfg(methods=["vanilla", "oracle", "m1"]), seed=1)
    assert set(sources) == {"m1"}, "ni M2 ni M3 ne doivent être chargés"

    sources, ms_arch = resolve_eval_sources(
        _eval_cfg(methods=["vanilla", "oracle"], sources="none"), seed=1)
    assert sources == {} and ms_arch is None


def test_sources_none_avec_methode_apprise_echoue(faux_runs):
    with pytest.raises(ValueError, match="sources"):
        resolve_eval_sources(_eval_cfg(sources="none"), seed=1)


def test_manifest_dit_quel_fichier_exact_a_ete_charge(faux_runs, tmp_path):
    cfg = _eval_cfg()
    sources, ms_arch = resolve_eval_sources(cfg, seed=3)
    manifest = eval_manifest(cfg, 3, sources, ms_arch)

    assert manifest["seed"] == 3
    assert manifest["dataset"] == "evimo2"
    assert manifest["methods"] == ["vanilla", "oracle", "m1", "m2", "m3"]
    assert all(info["exists"] for info in manifest["checkpoints"].values())
    assert "seed3" in manifest["checkpoints"]["m3"]["files"]["ms"]
    # le mode de masque de l'oracle doit être tracé : sans lui la comparaison à
    # l'entraînement n'a pas de sens (T7)
    assert "mask_motion" in manifest["oracle"]
    yaml.safe_dump(manifest)  # doit être sérialisable tel quel


def test_manifest_signale_un_checkpoint_manquant(faux_runs):
    cfg = _eval_cfg()
    sources, ms_arch = resolve_eval_sources(cfg, seed=1)
    sources["m2"]["ms"].unlink()
    manifest = eval_manifest(cfg, 1, sources, ms_arch)
    assert manifest["checkpoints"]["m2"]["exists"] is False
    assert manifest["checkpoints"]["m1"]["exists"] is True


def _ecrit_config_run(run_dir, **data):
    (Path(run_dir) / "config.yaml").write_text(yaml.safe_dump({"data": data}))


def test_manifest_trace_le_masque_d_entrainement(faux_runs):
    cfg = _eval_cfg()
    sources, ms_arch = resolve_eval_sources(cfg, seed=1)
    _ecrit_config_run(sources["m1"]["run_dir"], mask_motion="frame",
                      motion_thresh_m=0.01, motion_window_s=0.2)
    manifest = eval_manifest(cfg, 1, sources, ms_arch)
    assert manifest["checkpoints"]["m1"]["mask"]["mask_motion"] == "frame"
    assert manifest["checkpoints"]["m2"]["mask"] is None, "run sans config.yaml : non vérifiable"


def test_oracle_et_entrainement_doivent_avoir_le_meme_masque(faux_runs):
    cfg = _eval_cfg()
    cfg["oracle"] = {**cfg["oracle"], "mask_motion": "frame",
                     "motion_thresh_m": 0.01, "motion_window_s": 0.2}
    sources, ms_arch = resolve_eval_sources(cfg, seed=1)
    _ecrit_config_run(sources["m1"]["run_dir"], mask_motion="frame",
                      motion_thresh_m=0.01, motion_window_s=0.2)
    _ecrit_config_run(sources["m2"]["run_dir"], mask_motion="all",
                      motion_thresh_m=0.01, motion_window_s=0.2)
    _ecrit_config_run(sources["m3"]["run_dir"], mask_motion="frame",
                      motion_thresh_m=0.02, motion_window_s=0.2)
    pb = mask_definition_mismatches(eval_manifest(cfg, 1, sources, ms_arch))
    assert [m.split(" ")[0] for m in pb] == ["m2", "m3"]
    assert "mask_motion" in pb[0] and "motion_thresh_m" in pb[1]


def test_mode_all_ignore_seuil_et_fenetre(faux_runs):
    cfg = _eval_cfg()
    cfg["oracle"] = {**cfg["oracle"], "mask_motion": "all"}
    sources, ms_arch = resolve_eval_sources(cfg, seed=1)
    _ecrit_config_run(sources["m1"]["run_dir"], mask_motion="all",
                      motion_thresh_m=0.5, motion_window_s=9.0)
    assert mask_definition_mismatches(eval_manifest(cfg, 1, sources, ms_arch)) == []


# ------------------------------------------------------------------- table écrite

def test_table_contient_les_cinq_lignes_et_le_delta(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_table", REPO_ROOT / "evals/eval_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = module.write_table({"vanilla": 2.75, "oracle": 1.60, "m1": 1.74}, tmp_path)
    assert [r[0] for r in rows] == [module.METHOD_LABELS[m] for m in module.METHOD_ORDER]

    par_nom = {r[0]: r for r in rows}
    assert par_nom["DEVO + oracle GT"][2] == pytest.approx(100 * (2.75 - 1.60) / 2.75)
    assert par_nom["DEVO + couplé supervisé (M2)"][1] is None, "méthode absente = ligne vide"

    csv_text = (tmp_path / "central_table.csv").read_text()
    assert "delta_pct_vs_vanilla" in csv_text
    assert (tmp_path / "central_table.md").read_text().startswith("| Configuration |")


def test_methode_inconnue_refusee(faux_runs):
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_table", REPO_ROOT / "evals/eval_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = _eval_cfg(methods=["vanilla", "m4"], sources="none")
    with pytest.raises(ValueError, match="m4"):
        module.build_factories(cfg, {}, None)


# ------------------------------------- correctif d'alignement des timestamps GT

def _providers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evimo_providers", REPO_ROOT / "evals/eval_evs/evimo_providers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offset_nul_sans_fichier_de_timestamps(tmp_path):
    providers = _providers()
    assert providers.gt_timestamp_offset(str(tmp_path), str(tmp_path / "absent.npz")) == 0.0


@pytest.mark.parametrize("fabrique", ["oracle", "learned"])
def test_les_deux_providers_appliquent_le_correctif_de_timestamps(tmp_path, monkeypatch,
                                                                  fabrique):
    """L'ancienne passe `_fixed` : sans cet offset, les masques sont associés aux
    mauvaises frames et toute la table est fausse (cf. INVENTAIRE_RUNS.md §6.5)."""
    providers = _providers()
    npz = tmp_path / "imo" / "eval" / "scene13_dyn_test_00.npz"
    npz.parent.mkdir(parents=True)
    npz.write_text("factice")

    appels = []
    monkeypatch.setattr(providers, "gt_timestamp_offset",
                        lambda datapath, npz_path: appels.append(npz_path) or 1.25)

    import ms_model.oracle as oracle_mod
    recu = {}

    class _Faux:
        @staticmethod
        def from_evimo_npz(path, **kw):
            recu.update(kw)
            return "provider"

        def __init__(self, **kw):
            recu.update(kw)

    monkeypatch.setattr(oracle_mod, "OracleDynMaskProvider", _Faux)
    monkeypatch.setattr(oracle_mod, "LearnedDynMaskProvider", _Faux)

    if fabrique == "oracle":
        factory = providers.make_oracle_factory(str(tmp_path), device="cpu")
    else:
        factory = providers.make_learned_factory(str(tmp_path), "ms.pt", "arch.yaml",
                                                 device="cpu")
    factory("imo/eval/scene13_dyn_test_00", str(tmp_path))

    assert appels, "gt_timestamp_offset doit être appelé par la fabrique"
    assert recu["ts_offset_s"] == 1.25, "l'offset doit être transmis au provider"


# ------------------------------------------- définition du masque GT dans les configs

def _mask_def(cfg, section):
    return {k: cfg[section].get(k) for k in ("mask_motion", "motion_thresh_m", "motion_window_s")}


@pytest.mark.parametrize("eval_cfg, trainings", [
    ("evimo2.yaml", ["ms/v4_evimo2.yaml", "ms/v4large_evimo2.yaml",
                     "coupled/m2_evimo2.yaml", "coupled/m3_evimo2.yaml",
                     "coupled/m2_evimo2_scratch.yaml", "coupled/m3_evimo2_scratch.yaml",
                     "coupled/m2_evimo2_large.yaml", "coupled/m3_evimo2_large.yaml"]),
    ("evimo2_all.yaml", ["ms/v4_evimo2_all.yaml",
                         "coupled/m2_evimo2_all.yaml", "coupled/m3_evimo2_all.yaml"]),
])
def test_oracle_et_entrainements_de_la_campagne_ont_le_meme_masque(eval_cfg, trainings):
    """T7 : le papier (frame) et l'ablation (all) doivent chacun être cohérents de bout en bout."""
    oracle = _mask_def(resolve_config(REPO_ROOT / "configs/eval" / eval_cfg), "oracle")
    for t in trainings:
        assert _mask_def(resolve_config(REPO_ROOT / "configs" / t), "data") == oracle, t


def test_les_ablations_all_ecrivent_ailleurs_et_partent_du_ms_all():
    ms = resolve_config(REPO_ROOT / "configs/ms/v4_evimo2_all.yaml")
    assert ms["exp"] == "v4_evimo2_all" and ms["data"]["mask_motion"] == "all"
    assert ms["data"]["train_sequences"] == resolve_config(
        REPO_ROOT / "configs/ms/v4_evimo2.yaml")["data"]["train_sequences"]
    for m in ("m2", "m3"):
        c = resolve_config(REPO_ROOT / f"configs/coupled/{m}_evimo2_all.yaml")
        assert c["exp"] == f"{m}_evimo2_all"
        assert c["ms"]["init"] == "runs/v4_evimo2_all/@latest"
    ev = resolve_config(REPO_ROOT / "configs/eval/evimo2_all.yaml")
    assert ev["sources"] == "evimo2_all" and ev["exp"] != "eval_evimo2"
