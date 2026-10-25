#!/usr/bin/env python
"""Outil de configs pour les jobs : résolution, lecture, et construction des commandes.

Les jobs de `jobs/` sont du bash volontairement mince : toute la connaissance des configs
et des chemins vit ici et dans `utils/run_utils.py`, donc testable sans GPU.

    python tools/cfg.py resolve configs/ms/v4_evimo2.yaml --set training.epochs=2
    python tools/cfg.py get     configs/ms/v4_evimo2.yaml exp
    python tools/cfg.py rundir  configs/coupled/m3_evimo2.yaml --launch-id 2026-10-20_1200 --seed 3
    python tools/cfg.py argv ms configs/ms/v4_evimo2.yaml --run-dir runs/... --seed 1

`argv` écrit la ligne de commande du point d'entrée correspondant, un argument par ligne
(les jobs la relisent avec `mapfile`), ce qui évite tout problème de quoting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.run_utils import (  # noqa: E402
    EVAL_DATASETS, eval_config_errors, eval_dir, eval_manifest, latest_coupled_checkpoint, resolve_config,
    resolve_eval_sources, resolve_ref, run_dir,
)

# Une seule éval, paramétrée par la config (T6).
EVAL_SCRIPT = "evals/eval_table.py"


def _get(cfg: dict, dotted: str, default="__raise__"):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default == "__raise__":
                raise KeyError(f"clé absente de la config : '{dotted}'")
            return default
        node = node[part]
    return node


# ------------------------------------------------------------------ argv par job

def argv_ms(cfg, cfg_path, run_dir_path, seed):
    """Entraînement MS : le trainer lit la config résolue écrite dans le dossier du run."""
    return ["python", "-m", "ms_model.training.train",
            "--config", str(cfg_path),
            "--run-dir", str(run_dir_path),
            "--seed", str(seed)]


def argv_coupled(cfg, cfg_path, run_dir_path, seed):
    """Entraînement couplé M2 / M3 : traduit la config en options de train_coupled.py."""
    run_dir_path = Path(run_dir_path)
    ckpt_dir = run_dir_path / "checkpoints"

    ms_arch = _get(cfg, "ms.arch")
    init_ref = _get(cfg, "ms.init", None)
    init_dir = resolve_ref(init_ref, seed)
    ms_weights = None
    if init_dir is not None:
        ms_weights = Path(init_dir) / "checkpoints" / _get(cfg, "ms.ckpt", "best_iou.pt")

    args = ["python", "train_coupled.py",
            "--devo_weights", str(_get(cfg, "devo.weights")),
            "--ms_config", str(ms_arch),
            "--dataset", "evimo",
            "--datapath", str(_get(cfg, "data.root")),
            "--outdir", str(ckpt_dir),
            "--steps", str(_get(cfg, "train.steps")),
            "--freeze_devo_steps", str(_get(cfg, "train.freeze_devo_steps")),
            "--n_frames", str(_get(cfg, "train.n_frames")),
            "--ms_lr", str(_get(cfg, "train.ms_lr")),
            "--devo_lr", str(_get(cfg, "train.devo_lr")),
            "--clip", str(_get(cfg, "train.clip")),
            "--save_every", str(_get(cfg, "train.save_every")),
            "--log_every", str(_get(cfg, "log.every")),
            "--images_every", str(_get(cfg, "log.images_every")),
            "--dump_every", str(_get(cfg, "log.dump_every", 0) if _get(cfg, "save_viz", False) else 0),
            "--exp", str(_get(cfg, "exp")),
            "--run-dir", str(run_dir_path),
            "--mask_weight", str(_get(cfg, "loss.mask_weight")),
            "--mask_pos_weight", str(_get(cfg, "loss.mask_pos_weight", 1.0)),
            "--mask_thicken_radius", str(_get(cfg, "data.mask_thicken_radius")),
            "--mask_motion", str(_get(cfg, "data.mask_motion")),
            "--motion_thresh_m", str(_get(cfg, "data.motion_thresh_m")),
            "--motion_window_s", str(_get(cfg, "data.motion_window_s")),
            "--seed", str(seed)]

    if ms_weights is not None:
        args += ["--ms_weights", str(ms_weights)]

    mode = _get(cfg, "loss.mode")
    if mode == "supervised":
        args += ["--provide_gt_mask"]
    elif mode == "selfsup":
        args += ["--selfsup", "--selfsup_k", str(_get(cfg, "loss.selfsup_k")),
                 "--selfsup_min_res", str(_get(cfg, "loss.selfsup_min_res", 0.0))]
        if _get(cfg, "log.diag_gt", False):
            args += ["--diag_gt_mask"]  # GT pour les diagnostics, jamais dans la loss
    else:
        raise ValueError(f"loss.mode inconnu : '{mode}' (supervised | selfsup)")

    beta = float(_get(cfg, "loss.soft_prior_beta", 0.0))
    if beta:
        args += ["--soft_prior_beta", str(beta),
                 "--soft_prior_target", str(_get(cfg, "loss.soft_prior_target"))]

    # Reprise : le dernier coupled_step<K>.pt du dossier, s'il existe.
    resume = latest_coupled_checkpoint(ckpt_dir)
    if resume is not None:
        args += ["--resume", str(resume)]
    return args


def argv_eval(cfg, cfg_path, out_dir, seed):
    """Table d'évaluation : tout vient de la config résolue, rien n'est répété en options."""
    dataset = _get(cfg, "dataset.name")
    if dataset not in EVAL_DATASETS:
        raise ValueError(f"pas de runner d'éval pour le dataset '{dataset}'")
    return ["python", EVAL_SCRIPT,
            "--config", str(cfg_path),
            "--run-dir", str(out_dir),
            "--seed", str(seed)]


ARGV_BUILDERS = {"ms": argv_ms, "coupled": argv_coupled, "eval": argv_eval}


def manifest(cfg, seed):
    """Provenance des checkpoints d'une éval — même code que `evals/eval_table.py`."""
    sources, ms_arch = resolve_eval_sources(cfg, seed)
    return eval_manifest(cfg, seed, sources, ms_arch)


# ----------------------------------------------------------------------- dossiers

def out_dir_for(kind, cfg, launch_id, seed):
    if kind == "eval":
        return eval_dir(_get(cfg, "dataset.name"), str(_get(cfg, "sources")), launch_id, seed)
    return run_dir(_get(cfg, "exp"), launch_id, seed)


# --------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, with_set=True):
        sp.add_argument("config")
        if with_set:
            sp.add_argument("--set", dest="overrides", action="append", default=[],
                            metavar="CLÉ=VALEUR", help="surcharge (répétable)")

    sp = sub.add_parser("resolve", help="écrit la config résolue (YAML) sur la sortie standard")
    common(sp)
    sp.add_argument("--out", default=None, help="écrire dans ce fichier plutôt que stdout")

    sp = sub.add_parser("get", help="lit une clé pointée de la config résolue")
    common(sp)
    sp.add_argument("key")
    sp.add_argument("--default", default="__raise__")

    sp = sub.add_parser("rundir", help="affiche le dossier de sortie du run")
    common(sp)
    sp.add_argument("--kind", default=None, choices=sorted(ARGV_BUILDERS))
    sp.add_argument("--launch-id", required=True)
    sp.add_argument("--seed", type=int, required=True)

    sp = sub.add_parser("manifest", help="écrit la provenance des checkpoints d'une éval")
    common(sp)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("argv", help="affiche la commande du point d'entrée, un mot par ligne")
    sp.add_argument("kind", choices=sorted(ARGV_BUILDERS))
    common(sp)
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--resolved", default=None,
                    help="config résolue déjà écrite (passée au point d'entrée à la place "
                         "du yaml source)")

    args = p.parse_args()
    cfg = resolve_config(args.config, getattr(args, "overrides", []))

    if args.command == "resolve":
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
        else:
            sys.stdout.write(text)

    elif args.command == "get":
        value = _get(cfg, args.key, args.default)
        print("" if value is None else value)

    elif args.command == "rundir":
        kind = args.kind or ("eval" if "methods" in cfg else "train")
        print(out_dir_for("eval" if kind == "eval" else "train", cfg, args.launch_id, args.seed))

    elif args.command == "manifest":
        try:
            m = manifest(cfg, args.seed)
        except FileNotFoundError as exc:
            # Même convention que `argv` : une référence d'entraînement non encore
            # satisfaite n'est pas une erreur de configuration (code 3, cf. jobs/_common.sh).
            print(f"référence d'entraînement non résolue : {exc}", file=sys.stderr)
            sys.exit(3)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True))
        errors = eval_config_errors(cfg, m)
        if errors:
            sys.exit("config d'éval invalide :\n  " + "\n  ".join(errors))

    elif args.command == "argv":
        cfg_path = args.resolved or args.config
        for word in ARGV_BUILDERS[args.kind](cfg, cfg_path, args.run_dir, args.seed):
            print(word)


if __name__ == "__main__":
    main()
