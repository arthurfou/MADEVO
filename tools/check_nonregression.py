#!/usr/bin/env python
"""Compare l'ATE de DEVO vanilla d'une éval aux valeurs de référence du stage (T10).

    python tools/check_nonregression.py evals_out/evimo2/none/<launch_id>/seed{1,2,3} --dataset evimo2
    python tools/check_nonregression.py evals_out/rpg/none/<launch_id>/seed1 --dataset rpg

Lit `central_table.csv` (moyenne) et `per_scene.csv` (par scène) écrits par
`evals/eval_table.py`, et les références de `configs/nonregression.yaml`. Code de sortie 0
si tout est dans les tolérances, 1 sinon : la campagne ne doit pas partir sur un écart.

DEVO n'est pas déterministe sur GPU (une scène EVIMO2 varie de 1,7 à 7,2 cm d'une passe à
l'autre) : passer plusieurs seeds, dont on compare la moyenne. Avec `vanilla_seed_std`, la
tolérance sur la moyenne est max(rel_tolerance, 2 σ/√n) : une seule passe est jugée large.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VANILLA = "DEVO vanilla"


def read_mean(run_dir: Path) -> float | None:
    with open(run_dir / "central_table.csv") as f:
        for row in csv.DictReader(f):
            if row["config"] == VANILLA and row["ate_mean"]:
                return float(row["ate_mean"])
    return None


def read_per_scene(run_dir: Path) -> dict[str, float]:
    path = run_dir / "per_scene.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {row["scene"].split("/")[-1]: float(row["vanilla"])
                for row in csv.DictReader(f) if row.get("vanilla")}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def check(run_dirs, ref: dict) -> list[tuple[str, float, float, float, bool]]:
    """[(nom, mesuré, référence, écart relatif, ok)] ; `run_dirs` : un dossier ou une liste
    (une seed chacun), dont on compare la moyenne."""
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    run_dirs = [Path(d) for d in run_dirs]
    out = []
    if "vanilla_mean" in ref:
        got = _mean([read_mean(d) for d in run_dirs])
        exp = ref["vanilla_mean"]
        tol = ref["rel_tolerance"]
        if "vanilla_seed_std" in ref:
            tol = max(tol, 2 * ref["vanilla_seed_std"] / len(run_dirs) ** 0.5 / exp)
        rel = float("inf") if got is None else abs(got - exp) / exp
        out.append((f"moyenne ({len(run_dirs)} seed(s), tol. {tol:.0%})", got, exp, rel, rel <= tol))
    if "vanilla_per_scene" in ref:
        per_dir = [read_per_scene(d) for d in run_dirs]
        measured = {s: _mean([m.get(s) for m in per_dir]) for s in set().union(*per_dir)}
        got_all, exp_all = [], []
        for scene, exp in ref["vanilla_per_scene"].items():
            got = measured.get(scene)
            rel = float("inf") if got is None else abs(got - exp) / exp
            out.append((scene, got, exp, rel, rel <= ref["rel_tolerance"]))
            if got is not None:
                got_all.append(got)
                exp_all.append(exp)
        if got_all:
            got, exp = sum(got_all) / len(got_all), sum(exp_all) / len(exp_all)
            rel = abs(got - exp) / exp
            out.append(("moyenne (scènes de référence)", got, exp, rel,
                        rel <= ref.get("mean_rel_tolerance", ref["rel_tolerance"])))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+",
                   help="dossiers d'éval, un par seed (contiennent central_table.csv)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--refs", type=Path, default=REPO_ROOT / "configs/nonregression.yaml")
    args = p.parse_args()

    refs = yaml.safe_load(args.refs.read_text())
    if args.dataset not in refs:
        raise SystemExit(f"pas de référence pour '{args.dataset}' dans {args.refs}")
    ref = refs[args.dataset]
    results = check(args.run_dirs, ref)

    print(f"Non-régression {args.dataset} — référence : {ref['source']}")
    for name, got, exp, rel, ok in results:
        shown = "absent" if got is None else f"{got:.4f}"
        print(f"  {'OK ' if ok else 'ÉCART'} {name:32s} mesuré {shown:>8s}  réf {exp:.4f}  ({rel:+.1%})")
    failed = [r for r in results if not r[4]]
    print("→ OK" if not failed else f"→ {len(failed)} écart(s) : ne pas lancer la campagne avant d'avoir compris")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
