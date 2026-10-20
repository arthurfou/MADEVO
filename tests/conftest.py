"""Fixtures synthétiques partagées : séquences EVIMO2 / EVIMO1 minuscules, sans dataset réel.

Une séquence EVIMO2 synthétique reproduit les pièges du vrai dataset :
- des **trous de tracking** (frames présentes dans la meta mais sans `cam` ni `gt_frame`,
  et sans masque ni depth), dont un au tout début ;
- des clés `mask_%010d` renumérotées à partir de 0 alors que les ids de frames ne le sont
  pas ;
- une caméra qui bouge (translation + rotation) et des poses objet exprimées dans le
  repère caméra, avec un objet statique dans le monde et un objet qui ne bouge que
  pendant une partie de la séquence.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVIMO2_H, EVIMO2_W = 480, 640
MASK_ID_SCALE = 1000
STATIC_ID, MOVING_ID = 3, 7     # ids d'objets VICON du fixture
RATE_HZ = 60.0


# ------------------------------------------------------------------ géométrie

def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_dict(t, q):
    return {"t": {"x": t[0], "y": t[1], "z": t[2]},
            "q": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]}}


def camera_pose(k):
    """Caméra → monde à la frame k : translation et rotation lentes (c2w)."""
    t = np.array([0.01 * k, 0.005 * k, 0.0])
    q = quat_from_axis_angle([0, 1, 0], 0.01 * k)
    return t, q


def object_world_position(oid, ts, t_start, t_stop):
    """Position monde : STATIC_ID immobile ; MOVING_ID à 0,5 m/s sur x pendant [t_start, t_stop]."""
    base = np.array([0.2, 0.0, 1.5]) if oid == STATIC_ID else np.array([-0.2, 0.1, 1.2])
    if oid == MOVING_ID:
        base = base + np.array([0.5 * (np.clip(ts, t_start, t_stop) - t_start), 0.0, 0.0])
    return base


def object_in_camera(p_world, cam_t, cam_q):
    """Inverse de ce que fait `ms_model.motion._world_tracks` : monde → repère caméra."""
    return rotmat(cam_q).T @ (p_world - cam_t)


# --------------------------------------------------------------- séquence EVIMO2

def make_evimo2_sequence(root: Path, n_frames=10, holes=(0, 4), first_id=134,
                         t_start=0.05, t_stop=0.10, events_per_frame=40, seed=0):
    """Écrit une séquence EVIMO2 synthétique dans `root` et renvoie sa description.

    Les frames dont l'indice est dans `holes` n'ont ni pose, ni masque, ni depth.
    MOVING_ID ne bouge qu'entre t_start et t_stop (s).
    """
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    frames, masks, depths, gt_ts, gt_cam = [], [], [], [], []
    for k in range(n_frames):
        ts = 0.5 + k / RATE_HZ
        f = {"id": first_id + k, "ts": ts, "classical_frame": f"{k:06d}.png"}
        if k not in holes:
            cam_t, cam_q = camera_pose(k)
            f["cam"] = {"pos": pose_dict(cam_t, cam_q)}
            f["gt_frame"] = f"depth_{k:010d}.png"
            for oid in (STATIC_ID, MOVING_ID):
                p_cam = object_in_camera(object_world_position(oid, ts - 0.5, t_start, t_stop),
                                         cam_t, cam_q)
                f[str(oid)] = {"pos": pose_dict(p_cam, [0.0, 0.0, 0.0, 1.0])}
            # masque : un rectangle par objet, valeur = id × 1000 ; position qui dérive
            # avec k, pour que deux frames voisines n'aient jamais le même masque
            m = np.zeros((EVIMO2_H, EVIMO2_W), dtype=np.uint16)
            m[100:140, 200 + 4 * k:260 + 4 * k] = STATIC_ID * MASK_ID_SCALE
            m[300:340, 400:440] = MOVING_ID * MASK_ID_SCALE
            masks.append(m)
            depths.append(np.full((EVIMO2_H, EVIMO2_W), 1500 + k, dtype=np.uint16))
            gt_ts.append(ts)
            gt_cam.append((cam_t, cam_q))
        frames.append(f)

    # events : répartis dans tout l'intervalle couvert par la meta
    t = np.sort(rng.uniform(frames[0]["ts"], frames[-1]["ts"] + 1 / RATE_HZ,
                            n_frames * events_per_frame))
    xy = np.stack([rng.integers(0, EVIMO2_W, len(t)), rng.integers(0, EVIMO2_H, len(t))], axis=1)
    p = rng.integers(0, 2, len(t))
    np.save(root / "dataset_events_t.npy", t.astype(np.float64))
    np.save(root / "dataset_events_xy.npy", xy.astype(np.uint16))
    np.save(root / "dataset_events_p.npy", p.astype(np.uint8))

    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    # `index` d'EVIMO2 : table de lookup sur une grille de 10 ms, pas un index par frame
    grid = np.arange(frames[0]["ts"], frames[-1]["ts"], 0.01)
    np.savez(root / "dataset_info.npz", K=K, D=np.zeros(4),
             index=np.searchsorted(t, grid), meta=np.array({"frames": frames}, dtype=object))
    # clés renumérotées à partir de 0 (comme la conversion officielle), ids ≠ positions
    np.savez_compressed(root / "dataset_mask.npz",
                        **{f"mask_{i:010d}": m for i, m in enumerate(masks)})
    np.savez_compressed(root / "dataset_depth.npz",
                        **{f"depth_{i:010d}": d for i, d in enumerate(depths)})
    return {"path": root, "gt_ts": np.array(gt_ts), "gt_cam": gt_cam, "masks": np.stack(masks),
            "all_ts": np.array([f["ts"] for f in frames]), "t_start": t_start, "t_stop": t_stop}


@pytest.fixture(scope="session")
def evimo2_seq(tmp_path_factory):
    """Séquence EVIMO2 synthétique en lecture seule, partagée par les tests."""
    return make_evimo2_sequence(tmp_path_factory.mktemp("evimo2") / "scene_test_000000")


@pytest.fixture(scope="session")
def evimo2_long(tmp_path_factory):
    """Séquence longue (0,8 s) : MOVING_ID immobile, bouge de 5 cm entre 0,3 et 0,4 s, puis
    s'arrête — assez de frames au repos de part et d'autre pour tester le mode `frame`."""
    return make_evimo2_sequence(tmp_path_factory.mktemp("evimo2_long") / "scene_long_000000",
                                n_frames=48, holes=(0, 20), t_start=0.30, t_stop=0.40)


@pytest.fixture
def evimo2_seq_rw(tmp_path):
    """Copie modifiable (les tests de cache écrivent à côté de la séquence)."""
    return make_evimo2_sequence(tmp_path / "scene_rw_000000")


# ------------------------------------------------------------------ npz EVIMO1

def make_evimo1_npz(path: Path, H=24, W=32, n_frames=6, with_mask=True, seed=0):
    """Petit npz au format EVIMO1 (events (N,4) [t_s, x, y, p], index par frame, mask, meta)."""
    rng = np.random.default_rng(seed)
    frame_ts = np.arange(n_frames) * 0.05
    t = np.sort(rng.uniform(0.0, frame_ts[-1], 400))
    events = np.stack([t, rng.integers(0, W, len(t)), rng.integers(0, H, len(t)),
                       rng.integers(0, 2, len(t))], axis=1).astype(np.float32)
    data = {"events": events, "index": np.searchsorted(t, frame_ts).astype(np.int64),
            "meta": np.array({"frames": [{"ts": float(s)} for s in frame_ts]}, dtype=object)}
    if with_mask:
        mask = np.zeros((n_frames, H, W), dtype=np.uint16)
        mask[:, 4:8, 4:12] = 1000
        data["mask"] = mask
    np.savez_compressed(path, **data)
    return path
