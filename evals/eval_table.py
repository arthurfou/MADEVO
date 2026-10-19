#!/usr/bin/env python
"""Évaluation VO unifiée : une seule éval paramétrée par `methods` et par `sources`.

Remplace les six scripts du stage (`eval_evimo_m0_oracle.py`,
`eval_evimo{,2}_m1_decoupled.py`, `eval_*_central_table.py`) : M0 et M1 ne sont que des
tables avec moins de méthodes.

    python evals/eval_table.py --config <config d'éval résolue> --run-dir <dossier> --seed 1

Tout vient de la config (cf. `configs/eval/evimo2.yaml`) ; il n'y a plus vingt options à
passer et à retrouver après coup. Chaque éval écrit un `manifest.yaml` qui dit **quel
fichier exact** a servi à chaque méthode : c'est ce qui manquait aux tables du stage, dont
on ne sait plus quels checkpoints elles ont chargés.

Invariants, à ne pas relâcher :
- `mean_scene_ate` ne moyenne **que** les scènes, jamais les agrégats `AUC` et `AVG` ;
- le checkpoint MS par défaut est `best_iou.pt` ;
- la seed d'éval est la seed d'entraînement (passée par le job, pas 1234 en dur) ;
- `gt_timestamp_offset` (l'ancienne passe `_fixed`) est appliqué par les providers.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evals" / "eval_evs"))

# `utils.eval_utils` importe DEVO, donc les extensions CUDA : importé seulement au moment
# d'évaluer, pour que `--dry-run` tourne sur une machine sans GPU.
from utils.metrics import mean_scene_ate  # noqa: E402
from utils.run_utils import eval_manifest, resolve_config, resolve_eval_sources  # noqa: E402
from utils.wandb_logger import RunLogger  # noqa: E402

# Nom affiché de chaque méthode, dans l'ordre des tables du rapport.
METHOD_LABELS = {
    "vanilla": "DEVO vanilla",
    "oracle":  "DEVO + oracle GT",
    "m1":      "DEVO + appris découplé (M1)",
    "m2":      "DEVO + couplé supervisé (M2)",
    "m3":      "DEVO + couplé auto-sup (M3)",
}
METHOD_ORDER = list(METHOD_LABELS)

# Un dataset = un runner `evaluate(...)` et un module de providers.
DATASETS = {"evimo2": "eval_evimo2_evs"}


def _get(cfg, dotted, default="__raise__"):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default == "__raise__":
                raise KeyError(f"clé absente de la config d'éval : '{dotted}'")
            return default
        node = node[part]
    return node


# ------------------------------------------------------------------ provenance

def write_manifest(path, cfg, seed, sources, ms_arch):
    manifest = eval_manifest(cfg, seed, sources, ms_arch)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    return manifest


# ------------------------------------------------------------------- évaluation

def build_factories(cfg, sources, ms_arch, device="cuda"):
    """{méthode: (poids DEVO, fabrique de provider de masque)} pour les méthodes demandées."""
    from evimo_providers import make_learned_factory, make_oracle_factory

    mask_root = str(_get(cfg, "dataset.mask_root"))
    devo_weights = str(_get(cfg, "devo.weights"))
    oracle_cfg = _get(cfg, "oracle")
    threshold = _get(cfg, "threshold", None)

    def learned(ms_weights):
        return make_learned_factory(mask_root, str(ms_weights), str(ms_arch),
                                    threshold=threshold, device=device)

    plans = {}
    for method in _get(cfg, "methods"):
        if method not in METHOD_LABELS:
            raise ValueError(f"méthode inconnue '{method}' (attendu {METHOD_ORDER})")
        if method == "vanilla":
            plans[method] = (devo_weights, None)
        elif method == "oracle":
            plans[method] = (devo_weights, make_oracle_factory(
                mask_root, thicken_radius=oracle_cfg["thicken_radius"],
                device=device, mask_motion=oracle_cfg["mask_motion"],
                motion_thresh_m=oracle_cfg["motion_thresh_m"],
                motion_window_s=oracle_cfg["motion_window_s"]))
        else:
            info = sources[method]
            plans[method] = (str(info["devo"] or devo_weights), learned(info["ms"]))
    return plans


def run_methods(cfg, args, plans, devo_cfg):
    """Lance chaque méthode et renvoie (ATE moyen par méthode, ATE par scène)."""
    runner = __import__(DATASETS[_get(cfg, "dataset.name")])
    common = dict(datapath=str(_get(cfg, "dataset.root")),
                  split_file=str(_get(cfg, "dataset.split")),
                  trials=int(_get(cfg, "trials", 1)), plot=False, save=True,
                  stride=int(_get(cfg, "stride", 1)))
    if _get(cfg, "export_map", False):
        common["map_path"] = str(Path(args.run_dir) / "map.ply")

    ate, per_scene = {}, {}
    for method in METHOD_ORDER:
        if method not in plans:
            continue
        devo_weights, factory = plans[method]
        label = METHOD_LABELS[method]
        print(f"\n=== [table] {label} (DEVO={os.path.basename(devo_weights)}) ===", flush=True)
        args.expname = f"{method}_seed{args.seed}"
        res, _ = runner.evaluate(devo_cfg, args, devo_weights,
                                 dyn_mask_provider_factory=factory, **common)
        ate[method] = mean_scene_ate(res)
        per_scene[method] = res
    return ate, per_scene


# ----------------------------------------------------------------------- sorties

def write_table(ate, out_dir):
    """`central_table.csv` et `central_table.md`, mêmes colonnes que les tables du stage."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = ate.get("vanilla")

    rows = []
    for method in METHOD_ORDER:
        value = ate.get(method)
        delta = None if (value is None or not base) else 100.0 * (base - value) / base
        rows.append((METHOD_LABELS[method], value, delta))

    with open(out_dir / "central_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "ate_mean", "delta_pct_vs_vanilla"])
        for name, value, delta in rows:
            writer.writerow([name, "" if value is None else f"{value:.6f}",
                             "" if delta is None else f"{delta:.2f}"])

    with open(out_dir / "central_table.md", "w") as f:
        f.write("| Configuration | ATE moyen | Δ vs vanilla |\n|---|---|---|\n")
        for name, value, delta in rows:
            if value is None:
                f.write(f"| {name} | — | — |\n")
            else:
                f.write(f"| {name} | {value:.4f} | "
                        f"{'' if delta is None else f'{delta:+.1f}%'} |\n")
    return rows


def log_to_wandb(cfg, args, rows, per_scene, manifest):
    logger = RunLogger("eval", cfg, exp=_get(cfg, "exp", f"eval_{_get(cfg, 'dataset.name')}"),
                       seed=args.seed, run_dir=args.run_dir,
                       launch_id=os.environ.get("LAUNCH_ID"),
                       extra_config={"manifest": manifest})
    logger.log_table("eval/table", ["configuration", "ate_mean", "delta_pct_vs_vanilla"],
                     [list(r) for r in rows])

    scenes = sorted({s for res in per_scene.values() for s in res if s not in ("AUC", "AVG")})
    if scenes:
        methods = list(per_scene)
        logger.log_table("eval/ate_par_scene", ["scene"] + methods,
                         [[s] + [per_scene[m].get(s) for m in methods] for s in scenes])

    summary = {}
    for method, label in METHOD_LABELS.items():
        value = next((v for n, v, _ in rows if n == label), None)
        if value is not None:
            summary[f"eval/ate/{method}"] = value
    for name, value, delta in rows:
        if delta is not None:
            method = next(m for m, l in METHOD_LABELS.items() if l == name)
            summary[f"eval/delta_pct/{method}"] = delta
    logger.summary(summary)
    logger.finish()


# --------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config d'éval (résolue par le job)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="CLÉ=VALEUR")
    p.add_argument("--run-dir", dest="run_dir", required=True,
                   help="dossier de sortie de cette seed")
    p.add_argument("--seed", type=int, required=True,
                   help="seed d'éval = seed d'entraînement")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true",
                   help="résout la config et la provenance, écrit le manifest, ne lance rien")
    args = p.parse_args()

    cfg = resolve_config(args.config, args.overrides)
    dataset = _get(cfg, "dataset.name")
    if dataset not in DATASETS:
        raise SystemExit(f"dataset '{dataset}' sans runner (connus : {sorted(DATASETS)})")

    sources, ms_arch = resolve_eval_sources(cfg, args.seed)
    manifest = write_manifest(Path(args.run_dir) / "manifest.yaml", cfg, args.seed,
                              sources, ms_arch)

    missing = [m for m, info in manifest["checkpoints"].items() if not info["exists"]]
    if missing:
        raise SystemExit(f"checkpoints introuvables pour {missing} — voir "
                         f"{Path(args.run_dir) / 'manifest.yaml'}")

    if args.dry_run:
        print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
        print("dry-run : rien n'a été évalué.")
        return

    from devo.config import cfg as devo_cfg
    from utils.eval_utils import assert_eval_config
    devo_cfg.merge_from_file(str(_get(cfg, "devo.config")))
    torch.manual_seed(args.seed)

    # Les runners hérités lisent ces attributs sur `args`.
    args.weights = str(_get(cfg, "devo.weights"))
    args.plot, args.save_trajectory, args.viz, args.timing, args.viz_flow = \
        False, True, False, False, False
    args.outdir = str(args.run_dir)
    args.expname = ""
    assert_eval_config(args)

    plans = build_factories(cfg, sources, ms_arch, device=args.device)
    ate, per_scene = run_methods(cfg, args, plans, devo_cfg)

    rows = write_table(ate, args.run_dir)
    print(f"\n==================== TABLE {dataset.upper()} (seed {args.seed}) ====================")
    for name, value, delta in rows:
        shown = "—" if value is None else f"{value:.4f}"
        print(f"  {name:32s}: {shown}" + ("" if delta is None else f"  ({delta:+.1f}%)"))
    log_to_wandb(cfg, args, rows, per_scene, manifest)


if __name__ == "__main__":
    main()
