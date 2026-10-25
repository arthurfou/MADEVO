"""Sauvegardes de visualisation des évals : score maps et patches retenus, par frame.

Sans elles, la figure « events → masque → patches écartés » du papier demanderait de
relancer DEVO sur GPU (cf. docs/viz_app.md §6). Elles ne changent rien à l'éval :
`RecordingProvider` enveloppe le provider de masque d'une méthode (ou rien, pour vanilla)
et renvoie exactement le même score.

Un fichier par scène et par méthode, `<dump_dir>/<method>/<scene>.npz` :
- `ts_us` (F,)            : timestamp de chaque frame acceptée par DEVO ;
- `patch_xy` (F, P, 2)    : centre des P patches retenus, à la résolution de la score map
                            (H/4 × W/4), float16 ;
- `patch_dyn` (F, P)      : score dynamique sous chaque patch, uint8 (×255) ;
- `scores` (F, H/4, W/4)  : score dynamique dense, uint8 (×255) — absent pour vanilla, et
                            quand `keep_scores` est faux.
"""

from pathlib import Path

import numpy as np


def _to_uint8(x) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _score_2d(score) -> np.ndarray:
    """Score d'un provider (tenseur (…, h, w) ou tableau) → tableau float32 (h, w)."""
    if hasattr(score, "detach"):
        score = score.detach().float().cpu().numpy()
    score = np.asarray(score, dtype=np.float32)
    return score.reshape(score.shape[-2:])


class RecordingProvider:
    """Provider de masque qui enregistre ce que DEVO a vu et ce qu'il a retenu.

    `run_voxel` appelle `before_frame(slam)` et `after_frame(slam)` autour de chaque frame,
    puis `finish()` en fin de scène ; un provider sans ces méthodes n'est pas affecté.
    """

    def __init__(self, inner, out_path, keep_scores=True):
        self.inner = inner
        self.out_path = Path(out_path)
        self.keep_scores = keep_scores
        self._pending = None
        self.ts, self.patch_xy, self.patch_dyn, self.scores = [], [], [], []

    def __call__(self, ts_us):
        score = None if self.inner is None else self.inner(ts_us)
        self._pending = (ts_us, score)
        return score

    def before_frame(self, slam):
        """À appeler juste avant `slam(...)`."""
        self._n_before, self._counter_before = slam.n, slam.counter

    def after_frame(self, slam):
        """À appeler juste après `slam(...)`. Ignore les frames que DEVO a sautées.

        Les patches de la frame courante sont dans `slam.patches_[n]`, `n` étant la valeur
        de `slam.n` avant l'appel : DEVO les y écrit avant d'incrémenter `n`, et le
        retrait d'une keyframe ne recopie que les cases d'indice < n.
        """
        if slam.counter == self._counter_before or self._pending is None:
            return  # frame sautée (trop peu d'events, voxel vide)
        n_before = self._n_before
        ts_us, score = self._pending
        xy = slam.patches_[n_before, :, :2, 1, 1].detach().float().cpu().numpy()  # (P, 2)
        self.ts.append(float(ts_us))
        self.patch_xy.append(xy.astype(np.float16))
        if score is None:
            self.patch_dyn.append(np.zeros(len(xy), np.uint8))
        else:
            s = _score_2d(score)
            h, w = s.shape
            xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
            self.patch_dyn.append(_to_uint8(s[yi, xi]))
            if self.keep_scores:
                self.scores.append(_to_uint8(s))
        self._pending = None

    def finish(self):
        if not self.ts:
            return None
        out = {"ts_us": np.asarray(self.ts, dtype=np.float64),
               "patch_xy": np.stack(self.patch_xy),
               "patch_dyn": np.stack(self.patch_dyn)}
        if self.scores:
            out["scores"] = np.stack(self.scores)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_name(self.out_path.name + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        tmp.replace(self.out_path)
        return self.out_path


def recording_factory(factory, dump_dir, method, keep_scores=True):
    """Enveloppe une fabrique de providers `(scene, datapath) → provider` (ou None, vanilla)."""
    def make(scene, datapath):
        inner = None if factory is None else factory(scene, datapath)
        name = str(scene).strip("/").replace("/", "__")
        return RecordingProvider(inner, Path(dump_dir) / method / f"{name}.npz",
                                 keep_scores=keep_scores)
    return make
