"""Métriques d'évaluation pures — aucun import DEVO, donc utilisables et testables sans GPU.

`utils/eval_utils.py` importe DEVO (et donc les extensions CUDA) dès le chargement : les
métriques qui n'en ont pas besoin vivent ici, pour que les tests CPU les couvrent vraiment.
"""
import numpy as np

AGGREGATE_KEYS = ("AUC", "AVG")


def mean_scene_ate(results_dict):
    """ATE moyen [cm] sur les scènes d'un dict produit par `compute_median_results`.

    Seules les entrées par scène (« <dataset>/<scene> ») sont moyennées : `AUC` et `AVG`
    sont des agrégats, pas des ATE. Les inclure sous-estimait l'ATE absolue de **toutes**
    les tables centrales du stage (~20 % sur EVIMO2, cf. INTEGRITY_CHECK.md, bug 2).
    """
    vals = [float(v) for k, v in results_dict.items() if k not in AGGREGATE_KEYS]
    return float(np.mean(vals)) if vals else float("nan")
