"""Logger wandb commun à l'entraînement MS, à l'entraînement couplé et aux évals.

Un seul endroit décide du nommage des runs, de ce qui part dans la config et de la façon
de fabriquer les images, pour que les trois types de jobs soient comparables dans la même
interface wandb :

    group    = <exp>              une expérience, toutes seeds confondues
    name     = <exp>_s<seed>      un run = une seed
    job_type = train_ms | train_coupled | eval

Les métriques sont **préfixées par domaine** (`ms/`, `devo/`, `ba/`, `coupling/`) : avec
cinq seeds et plusieurs jalons, c'est ce qui rend les courbes lisibles.

Le logger ne doit jamais faire échouer un entraînement : si wandb est indisponible ou en
mode `disabled`, il devient silencieux. Et surtout, **rien ici ne pilote un chemin sur
disque** : les checkpoints vivent dans le dossier du run (`--run-dir`), pas dans un dossier
dérivé du nom wandb — c'est ce qui cassait la reprise en mode `disabled`
(cf. INTEGRITY_CHECK.md, point 5).
"""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

JOB_TYPES = ("train_ms", "train_coupled", "eval")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "(indisponible)"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
                             capture_output=True, text=True).stdout
        return bool(out.strip())
    except Exception:
        return False


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------- images

def voxel_image(voxel) -> np.ndarray:
    """Grille de voxels (C, H, W) ou (H, W) → image RGB uint8.

    Rouge = polarité positive, bleu = négative, noir = pas d'event : la même convention
    que `utils/viz_utils.render`, pour qu'un coup d'œil à wandb ressemble aux figures.
    """
    v = _to_numpy(voxel).astype(np.float32)
    if v.ndim == 3:
        v = v.sum(axis=0)
    elif v.ndim != 2:
        raise ValueError(f"voxel de forme inattendue : {v.shape}")

    img = np.zeros((*v.shape, 3), dtype=np.uint8)
    scale = np.abs(v).max() or 1.0
    pos, neg = np.clip(v, 0, None) / scale, np.clip(-v, 0, None) / scale
    img[..., 0] = (255 * pos).astype(np.uint8)
    img[..., 2] = (255 * neg).astype(np.uint8)
    return img


def mask_image(mask, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Masque / score (H, W) dans [0, 1] → image RGB uint8 (noir → jaune)."""
    m = _to_numpy(mask).astype(np.float32)
    while m.ndim > 2:
        m = m[0] if m.shape[0] == 1 else m.mean(axis=0)
    m = np.clip((m - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    img = np.zeros((*m.shape, 3), dtype=np.uint8)
    img[..., 0] = (255 * m).astype(np.uint8)
    img[..., 1] = (255 * m).astype(np.uint8)
    return img


def patches_image(voxel, patch_xy, dyn_score=None, radius: int = 2) -> np.ndarray:
    """Patches sélectionnés dessinés sur la grille de voxels.

    patch_xy : (P, 2) en pixels. `dyn_score` (P,) colore le patch du vert (statique, gardé)
    au rouge vif (dynamique, écarté par la BA) — on voit d'un coup si le masque écarte les
    patches qui tombent sur l'objet mobile.
    """
    img = voxel_image(voxel)
    H, W = img.shape[:2]
    xy = _to_numpy(patch_xy).reshape(-1, 2)
    scores = None if dyn_score is None else _to_numpy(dyn_score).reshape(-1)

    for i, (x, y) in enumerate(xy):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        s = 0.0 if scores is None or i >= len(scores) else float(scores[i])
        color = np.array([255 * s, 255 * (1 - s), 0], dtype=np.uint8)
        y0, y1 = max(yi - radius, 0), min(yi + radius + 1, H)
        x0, x1 = max(xi - radius, 0), min(xi + radius + 1, W)
        img[y0:y1, x0:x1] = color
    return img


# ---------------------------------------------------------------------------- logger

class RunLogger:
    """Enveloppe wandb partagée. Silencieuse si wandb est absent ou désactivé."""

    def __init__(self, job_type: str, config: dict, *, exp: str = None, seed: int = None,
                 run_dir=None, launch_id: str = None, project: str = None,
                 entity: str = None, resume_id: str = None, images_every: int = 0,
                 extra_config: dict = None):
        if job_type not in JOB_TYPES:
            raise ValueError(f"job_type '{job_type}' inconnu (attendu {JOB_TYPES})")

        self.job_type = job_type
        self.seed = seed
        self.exp = exp or "sans_nom"
        self.images_every = int(images_every or 0)
        self.run = None
        self.enabled = False
        self._warned = False

        full_config = dict(config or {})
        full_config.update({
            "_exp": self.exp,
            "_seed": seed,
            "_launch_id": launch_id,
            "_job_type": job_type,
            "_git_commit": _git_commit(),
            "_git_dirty": _git_dirty(),
            "_hostname": socket.gethostname(),
            "_run_dir": None if run_dir is None else str(run_dir),
        })
        full_config.update(extra_config or {})

        try:
            import wandb
        except ImportError:
            print("[wandb] paquet absent : logging désactivé", flush=True)
            return

        self._wandb = wandb
        name = self.exp if seed is None else f"{self.exp}_s{seed}"
        try:
            self.run = wandb.init(
                project=project or os.environ.get("WANDB_PROJECT", "madevo"),
                entity=entity or os.environ.get("WANDB_ENTITY") or None,
                group=self.exp,
                name=name,
                job_type=job_type,
                id=resume_id,
                resume="allow" if resume_id else None,
                config=full_config,
                tags=[job_type, self.exp],
            )
            self.enabled = True
        except Exception as exc:  # pragma: no cover — dépend de l'environnement
            print(f"[wandb] init impossible ({exc}) : logging désactivé", flush=True)
            return

        mode = os.environ.get("WANDB_MODE", "online")
        print(f"[wandb] {job_type} | group={self.exp} | name={name} | mode={mode}", flush=True)
        if run_dir is not None:
            self.write_run_file(run_dir)

    # -- identité ----------------------------------------------------------------

    @property
    def id(self):
        return getattr(self.run, "id", None)

    def write_run_file(self, run_dir) -> None:
        """`wandb.txt` dans le dossier du run, pour retrouver le run après un sync."""
        try:
            url = self.run.url or "(hors ligne)"
        except Exception:
            url = "(hors ligne)"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "wandb.txt").write_text(
            f"id: {self.id}\nname: {getattr(self.run, 'name', None)}\n"
            f"group: {self.exp}\njob_type: {self.job_type}\n"
            f"mode: {os.environ.get('WANDB_MODE', 'online')}\nurl: {url}\n")

    # -- métriques ---------------------------------------------------------------

    def log(self, metrics: dict, step: int = None, prefix: str = None) -> None:
        if not self.enabled or not metrics:
            return
        payload = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    value = float(_to_numpy(value).mean())
            payload[f"{prefix}{key}" if prefix else key] = value
        self._safe(lambda: self.run.log(payload, step=step))

    def log_histogram(self, name: str, values, step: int = None) -> None:
        if not self.enabled:
            return
        arr = _to_numpy(values).reshape(-1)
        if arr.size == 0:
            return
        self._safe(lambda: self.run.log({name: self._wandb.Histogram(arr)}, step=step))

    def log_images(self, images: dict, step: int = None, prefix: str = None) -> None:
        """`images` : {nom: image RGB uint8} (déjà fabriquée par les helpers ci-dessus)."""
        if not self.enabled or not images:
            return
        payload = {f"{prefix}{k}" if prefix else k: self._wandb.Image(v, caption=k)
                   for k, v in images.items() if v is not None}
        self._safe(lambda: self.run.log(payload, step=step))

    def should_log_images(self, step: int) -> bool:
        return bool(self.enabled and self.images_every and step % self.images_every == 0)

    def log_table(self, name: str, columns, rows) -> None:
        """Table d'éval envoyée comme `wandb.Table` (triable et comparable entre runs)."""
        if not self.enabled:
            return
        data = [list(r) for r in rows]
        self._safe(lambda: self.run.log({name: self._wandb.Table(columns=list(columns),
                                                                 data=data)}))

    def summary(self, values: dict) -> None:
        if not self.enabled or not values:
            return
        def apply():
            for key, value in values.items():
                self.run.summary[key] = value
        self._safe(apply)

    def finish(self) -> None:
        if self.enabled:
            self._safe(self.run.finish)
        self.enabled = False

    # -- interne -----------------------------------------------------------------

    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            if not self._warned:
                print(f"[wandb] erreur de logging ignorée : {exc}", flush=True)
                self._warned = True
