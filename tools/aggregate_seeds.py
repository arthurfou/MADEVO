#!/usr/bin/env python
"""Agrège les seeds d'une éval : moyenne ± écart-type par configuration.

    python tools/aggregate_seeds.py evals_out/evimo2/evimo2/2026-11-07_1100

Lit les `seed*/central_table.csv` du dossier de lancement et écrit `summary.csv` et
`summary.md` à côté. L'écart-type est celui de l'échantillon (ddof=1) ; avec une seule
seed il n'est pas défini et la colonne reste vide.

Le Δ vs vanilla est **recalculé sur les moyennes**, pas moyenné : moyenner des
pourcentages de lignes différentes n'a pas de sens.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

TABLE_NAME = "central_table.csv"
VANILLA = "DEVO vanilla"


def read_seed_tables(launch_dir: Path):
    """{configuration: {seed: ate}} pour toutes les seeds trouvées."""
    per_config: dict[str, dict[int, float]] = {}
    order: list[str] = []
    seeds: list[int] = []

    for seed_dir in sorted(launch_dir.glob("seed*"), key=lambda p: int(p.name[4:])):
        table = seed_dir / TABLE_NAME
        if not table.exists():
            print(f"[skip] {seed_dir.name} : pas de {TABLE_NAME}", file=sys.stderr)
            continue
        seed = int(seed_dir.name[4:])
        seeds.append(seed)
        with open(table, newline="") as f:
            for row in csv.DictReader(f):
                name = row["config"]
                if name not in per_config:
                    per_config[name] = {}
                    order.append(name)
                raw = (row.get("ate_mean") or "").strip()
                if raw:
                    per_config[name][seed] = float(raw)
    return per_config, order, seeds


def aggregate(per_config, order, seeds):
    base_values = per_config.get(VANILLA, {})
    base_mean = statistics.fmean(base_values.values()) if base_values else None

    rows = []
    for name in order:
        values = [per_config[name][s] for s in seeds if s in per_config[name]]
        if not values:
            rows.append({"config": name, "n_seeds": 0, "ate_mean": None,
                         "ate_std": None, "delta_pct_vs_vanilla": None})
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else None
        delta = None
        if base_mean:
            delta = 100.0 * (base_mean - mean) / base_mean
        rows.append({"config": name, "n_seeds": len(values), "ate_mean": mean,
                     "ate_std": std, "delta_pct_vs_vanilla": delta})
    return rows


def write_summary(rows, seeds, launch_dir: Path):
    csv_path = launch_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_seeds", "ate_mean", "ate_std", "delta_pct_vs_vanilla"])
        for r in rows:
            writer.writerow([
                r["config"], r["n_seeds"],
                "" if r["ate_mean"] is None else f"{r['ate_mean']:.6f}",
                "" if r["ate_std"] is None else f"{r['ate_std']:.6f}",
                "" if r["delta_pct_vs_vanilla"] is None else f"{r['delta_pct_vs_vanilla']:.2f}",
            ])

    md_path = launch_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(f"# {launch_dir.name} — {len(seeds)} seed(s) : {', '.join(map(str, seeds))}\n\n")
        f.write("| Configuration | ATE moyen | écart-type | Δ vs vanilla | seeds |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            if r["ate_mean"] is None:
                f.write(f"| {r['config']} | — | — | — | 0 |\n")
                continue
            std = "—" if r["ate_std"] is None else f"{r['ate_std']:.4f}"
            delta = "—" if r["delta_pct_vs_vanilla"] is None else f"{r['delta_pct_vs_vanilla']:+.1f}%"
            f.write(f"| {r['config']} | {r['ate_mean']:.4f} | {std} | {delta} | {r['n_seeds']} |\n")
        f.write("\nATE moyenné sur les seeds ; Δ recalculé sur les moyennes, non moyenné.\n")
    return csv_path, md_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("launch_dir", help="dossier <date> d'une éval, contenant les seed<N>/")
    args = p.parse_args()

    launch_dir = Path(args.launch_dir)
    if not launch_dir.is_dir():
        sys.exit(f"dossier introuvable : {launch_dir}")

    per_config, order, seeds = read_seed_tables(launch_dir)
    if not seeds:
        sys.exit(f"aucun seed*/{TABLE_NAME} sous {launch_dir}")

    rows = aggregate(per_config, order, seeds)
    csv_path, md_path = write_summary(rows, seeds, launch_dir)
    print(open(md_path).read())
    print(f"[summary] {csv_path}\n[summary] {md_path}")


if __name__ == "__main__":
    main()
