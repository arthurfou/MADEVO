"""Configs YAML, surcharges `--set`, et arborescence des dossiers de runs.

Tout ce que les jobs (`jobs/*.sh`) et les lanceurs (`launch/*.sh`) savent des configs et
des chemins passe par ici, pour que la logique soit testable sans GPU. L'outil en ligne de
commande correspondant est `tools/cfg.py`.

Arborescence des sorties (cf. JOBS_AND_RUNS.md §4) :

    runs/<exp>/<launch_id>/seed<N>/      entraînements MS et couplés
    evals/<dataset>/<sources>/<launch_id>/seed<N>/

Racines surchargeables par `MADEVO_RUNS` et `MADEVO_EVALS` (pour mettre les sorties sur le
disque de données d'un serveur) ; par défaut `runs/` et `evals/` dans le repo, ignorés par
git.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

LAUNCH_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}")


# --------------------------------------------------------------------------- configs

def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """Fusion récursive : `overlay` gagne, les dicts sont fusionnés clé par clé."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def expand_env(value):
    """Remplace `${VAR}` et `${VAR:-défaut}` dans toutes les chaînes d'une structure."""
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def sub(match):
        name, default = match.group(1), match.group(2)
        found = os.environ.get(name)
        if found is None:
            if default is None:
                raise KeyError(f"variable d'environnement {name} non définie "
                               f"(et pas de défaut ${{{name}:-…}})")
            return default
        return found

    return _ENV_RE.sub(sub, value)


def set_in(cfg: dict, dotted: str, value, create: bool = False) -> None:
    """Pose `cfg[a][b] = value` pour `dotted = "a.b"`.

    Une clé inconnue est une erreur : c'est ce qui attrape les fautes de frappe dans les
    `--set` du catalogue. Le préfixe `+` (`--set +a.b=1`) autorise la création.
    """
    parts = dotted.split(".")
    node = cfg
    for i, part in enumerate(parts[:-1]):
        if part not in node:
            if not create:
                raise KeyError(f"clé inconnue '{'.'.join(parts[:i + 1])}' "
                               f"(utiliser --set +{dotted}=… pour la créer)")
            node[part] = {}
        node = node[part]
        if not isinstance(node, dict):
            raise KeyError(f"'{'.'.join(parts[:i + 1])}' n'est pas une section")
    leaf = parts[-1]
    if leaf not in node and not create:
        raise KeyError(f"clé inconnue '{dotted}' (utiliser --set +{dotted}=… pour la créer)")
    node[leaf] = value


def apply_overrides(cfg: dict, overrides) -> dict:
    """Applique des surcharges `clé=valeur`, la valeur étant lue comme du YAML.

    `--set train.steps=40000`, `--set methods=[vanilla,oracle]`, `--set ms.init=null`.
    """
    cfg = dict(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"surcharge mal formée '{item}' (attendu clé=valeur)")
        key, raw = item.split("=", 1)
        key = key.strip()
        create = key.startswith("+")
        set_in(cfg, key.lstrip("+"), yaml.safe_load(raw), create=create)
    return cfg


def resolve_config(path, overrides=None) -> dict:
    """Config finale : YAML + héritage `extends:` + surcharges + `${VAR}`."""
    path = Path(path)
    cfg = load_yaml(path)
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent) if not str(parent).startswith("/") else Path(parent)
        cfg = deep_merge(resolve_config(parent_path), cfg)
    cfg = apply_overrides(cfg, overrides)
    return expand_env(cfg)


# ------------------------------------------------------------------- dossiers de runs

def runs_root() -> Path:
    return Path(os.environ.get("MADEVO_RUNS", REPO_ROOT / "runs"))


def evals_root() -> Path:
    return Path(os.environ.get("MADEVO_EVALS", REPO_ROOT / "evals_out"))


def run_dir(exp: str, launch_id: str, seed: int) -> Path:
    """`runs/<exp>/<launch_id>/seed<N>/` — entraînements MS et couplés."""
    check_launch_id(launch_id)
    return runs_root() / exp / launch_id / f"seed{seed}"


def eval_dir(dataset: str, sources: str, launch_id: str, seed: int) -> Path:
    """`evals_out/<dataset>/<sources>/<launch_id>/seed<N>/`."""
    check_launch_id(launch_id)
    return evals_root() / dataset / sources / launch_id / f"seed{seed}"


def check_launch_id(launch_id: str) -> None:
    if not LAUNCH_ID_RE.match(str(launch_id)):
        raise ValueError(f"LAUNCH_ID '{launch_id}' invalide (attendu YYYY-MM-DD_HHMM)")


def is_done(path: Path) -> bool:
    return (Path(path) / "DONE").exists()


def resolve_ref(ref, seed: int, root: Path = None) -> Path | None:
    """Résout une référence d'entraînement vers un dossier de seed.

    - `null` / `""`            → None (pas de checkpoint, from scratch)
    - `<exp>/@latest`          → dernier `<launch_id>` dont `seed<N>/DONE` existe
    - `<exp>/<launch_id>`      → ce lancement précis
    - chemin existant          → tel quel

    Le préfixe `runs/` est optionnel : `runs/v4_evimo2/@latest` et `v4_evimo2/@latest`
    désignent la même chose.
    """
    if ref in (None, "", "null"):
        return None
    root = Path(root or runs_root())
    ref = str(ref)

    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    rel = ref[len("runs/"):] if ref.startswith("runs/") else ref
    parts = rel.split("/")
    exp, rest = parts[0], parts[1:]

    if rest and rest[0] == "@latest":
        base = root / exp
        if not base.is_dir():
            raise FileNotFoundError(f"aucun lancement sous {base}")
        done = sorted(d for d in base.iterdir()
                      if d.is_dir() and LAUNCH_ID_RE.match(d.name) and is_done(d / f"seed{seed}"))
        if not done:
            raise FileNotFoundError(
                f"aucun lancement terminé (DONE) de '{exp}' pour la seed {seed} sous {base}")
        return done[-1] / f"seed{seed}"

    if rest:
        return root / exp / rest[0] / f"seed{seed}"
    raise ValueError(f"référence '{ref}' incomplète (attendu <exp>/@latest ou <exp>/<launch_id>)")


def latest_coupled_checkpoint(directory) -> Path | None:
    """Dernier `coupled_step<K>.pt` d'un dossier de checkpoints (pour la reprise)."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    steps = []
    for p in directory.glob("coupled_step*.pt"):
        match = re.search(r"coupled_step(\d+)\.pt$", p.name)
        if match:
            steps.append((int(match.group(1)), p))
    return max(steps)[1] if steps else None


# ------------------------------------------------- provenance des évaluations

def resolve_eval_sources(cfg, seed, repo_root: Path = None):
    """{méthode: {ref, run_dir, devo, ms}} pour les méthodes apprises d'une éval.

    Une seule implémentation, utilisée par `tools/cfg.py manifest` (qui tourne sans GPU)
    et par `evals/eval_table.py`.
    """
    root = Path(repo_root or REPO_ROOT)
    methods = cfg.get("methods", [])
    name = str(cfg.get("sources"))
    learned = [m for m in ("m1", "m2", "m3") if m in methods]

    if name in ("none", "None", "", "null"):
        if learned:
            raise ValueError(f"`methods` demande {learned} mais `sources` vaut 'none'")
        return {}, None

    sources = resolve_config(root / "configs/eval/sources" / f"{name}.yaml")
    ms_arch = sources.get("ms_arch")
    out = {}

    if "m1" in methods:
        ref = sources["m1"]["ms"]
        run = Path(resolve_ref(ref, seed, root=root / "runs" if repo_root else None))
        out["m1"] = {"ref": str(ref), "run_dir": run, "devo": None,
                     "ms": run / "checkpoints" / sources["m1"].get("ckpt", "best_iou.pt")}
    for method in ("m2", "m3"):
        if method not in methods:
            continue
        ref = sources[method]["coupled"]
        run = Path(resolve_ref(ref, seed, root=root / "runs" if repo_root else None))
        ckpts = run / "checkpoints"
        out[method] = {"ref": str(ref), "run_dir": run,
                       "devo": ckpts / "devo_final.pth", "ms": ckpts / "ms_final.pt"}
    return out, ms_arch


# Datasets évaluables par `evals/eval_table.py`. Seul EVIMO2 a un masque GT, donc un oracle ;
# les autres servent à la validation cross-dataset (vanilla, M1, M2, M3).
EVAL_DATASETS = ("evimo2", "rpg", "mvsec", "eds", "vector", "tumvie")
ORACLE_DATASETS = ("evimo2",)

MASK_DEF_KEYS = ("mask_motion", "motion_thresh_m", "motion_window_s")


def _run_mask_definition(run_dir) -> dict | None:
    """Définition du masque GT (`data.mask_motion`…) avec laquelle un run a été entraîné."""
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        return None
    data = load_yaml(path).get("data", {})
    out = {k: data.get(k) for k in MASK_DEF_KEYS}
    out["thicken_radius"] = data.get("mask_thicken_radius", 0)
    return out


def _relevant_mask_keys(mode) -> tuple[str, ...]:
    """Seuil et fenêtre n'ont de sens que pour les modes qui les utilisent."""
    return {"all": ("mask_motion",),
            "sequence": ("mask_motion", "motion_thresh_m")}.get(mode, MASK_DEF_KEYS)


def mask_definition_mismatches(manifest) -> list[str]:
    """Méthodes apprises entraînées avec une autre définition du masque GT que l'oracle.

    Comparer un oracle `frame` à un MS entraîné en `all` (ou l'inverse) n'a pas de sens
    (cf. T7) : l'éval s'arrête sur la liste renvoyée. Un run sans `config.yaml` (fabriqué
    hors des jobs) n'est pas vérifiable et n'est pas signalé.
    """
    if "oracle" not in manifest.get("methods", []):
        return []  # rien à comparer (éval cross-dataset, ou table sans oracle)
    oracle = manifest.get("oracle", {})
    out = []
    for method, info in manifest["checkpoints"].items():
        trained = info.get("mask")
        if trained is None:
            continue
        keys = _relevant_mask_keys(oracle.get("mask_motion"))
        diff = {k: (trained.get(k), oracle.get(k)) for k in keys if trained.get(k) != oracle.get(k)}
        # Dilatation : l'oracle doit être exactement la cible que le MS a apprise.
        if "thicken_radius" in trained and trained["thicken_radius"] != oracle.get("thicken_radius", 0):
            diff["thicken_radius"] = (trained["thicken_radius"], oracle.get("thicken_radius", 0))
        if diff:
            out.append(f"{method} : " + ", ".join(f"{k} entraînement={a!r} oracle={b!r}"
                                                  for k, (a, b) in diff.items()))
    return out


def eval_config_errors(cfg, manifest) -> list[str]:
    """Erreurs de configuration d'une éval, détectables sans GPU ni checkpoint.

    Appelée par `tools/cfg.py manifest` (donc par le `--dry-run` des jobs) et par
    `evals/eval_table.py` avant d'évaluer quoi que ce soit.
    """
    errors = []
    dataset = cfg.get("dataset", {}).get("name")
    if dataset not in EVAL_DATASETS:
        errors.append(f"dataset inconnu : {dataset!r} (attendu : {', '.join(EVAL_DATASETS)})")
    if "oracle" in cfg.get("methods", []) and dataset not in ORACLE_DATASETS:
        errors.append(f"pas d'oracle sur {dataset} : seul {', '.join(ORACLE_DATASETS)} a un masque "
                      "GT de mouvement (retirer `oracle` de `methods`)")
    errors += [f"masque GT de l'oracle différent de celui de l'entraînement (T7) — {m}"
               for m in mask_definition_mismatches(manifest)]
    return errors


def eval_manifest(cfg, seed, sources, ms_arch) -> dict:
    """Provenance exacte d'une table : ce qui a été chargé, et si le fichier existe."""
    return {
        "seed": seed,
        "dataset": cfg["dataset"]["name"],
        "sources": str(cfg.get("sources")),
        "methods": list(cfg.get("methods", [])),
        "devo_weights": str(cfg["devo"]["weights"]),
        "devo_config": str(cfg["devo"].get("config", "")),
        "oracle": dict(cfg.get("oracle", {})),
        "threshold": cfg.get("threshold"),
        "ms_arch": None if ms_arch is None else str(ms_arch),
        "checkpoints": {
            method: {
                "ref": info["ref"],
                "run_dir": str(info["run_dir"]),
                "files": {k: str(info[k]) for k in ("devo", "ms") if info.get(k)},
                "exists": all(Path(info[k]).exists() for k in ("devo", "ms") if info.get(k)),
                "mask": _run_mask_definition(info["run_dir"]),
            }
            for method, info in sources.items()
        },
    }


# ------------------------------------------------------------------------ provenance

def git_description() -> str:
    """Commit, état « dirty » et diff --stat, pour `git.txt` dans le dossier du run."""
    def run(*args):
        try:
            return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "(indisponible)"

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    lines = [f"commit: {commit}", f"dirty: {'oui' if status else 'non'}"]
    if status:
        lines += ["", "git status --porcelain:", status,
                  "", "git diff --stat:", run("git", "diff", "--stat")]
    return "\n".join(lines) + "\n"
