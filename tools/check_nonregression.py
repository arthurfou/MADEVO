#!/usr/bin/env python
"""Compare l'ATE de DEVO vanilla d'une éval aux valeurs de référence du stage (T10).

    python tools/check_nonregression.py evals_out/evimo2/none/<launch_id>/seed1 --dataset evimo2
    python tools/check_nonregression.py evals_out/rpg/none/<launch_id>/seed1 --dataset rpg

Lit `central_table.csv` (moyenne) et `per_scene.csv` (par scène) écrits par
`evals/eval_table.py`, et les références de `configs/nonregression.yaml`. Code de sortie 0
si tout est dans les tolérances, 1 sinon : la campagne ne doit pas partir sur un écart.
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


def check(run_dir: Path, ref: dict) -> list[tuple[str, float, float, float, bool]]:
    """[(nom, mesuré, référence, écart relatif, ok)]"""
    out = []
    if "vanilla_mean" in ref:
        got = read_mean(run_dir)
        exp = ref["vanilla_mean"]
        rel = float("inf") if got is None else abs(got - exp) / exp
        out.append(("moyenne", got, exp, rel, rel <= ref["rel_tolerance"]))
    if "vanilla_per_scene" in ref:
        measured = read_per_scene(run_dir)
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
    p.add_argument("run_dir", type=Path, help="dossier d'éval d'une seed (contient central_table.csv)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--refs", type=Path, default=REPO_ROOT / "configs/nonregression.yaml")
    args = p.parse_args()

    refs = yaml.safe_load(args.refs.read_text())
    if args.dataset not in refs:
        raise SystemExit(f"pas de référence pour '{args.dataset}' dans {args.refs}")
    ref = refs[args.dataset]
    results = check(args.run_dir, ref)

    print(f"Non-régression {args.dataset} — référence : {ref['source']}")
    for name, got, exp, rel, ok in results:
        shown = "absent" if got is None else f"{got:.4f}"
        print(f"  {'OK ' if ok else 'ÉCART'} {name:32s} mesuré {shown:>8s}  réf {exp:.4f}  ({rel:+.1%})")
    failed = [r for r in results if not r[4]]
    print("→ OK" if not failed else f"→ {len(failed)} écart(s) : ne pas lancer la campagne avant d'avoir compris")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
