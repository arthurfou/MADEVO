import numpy as np
import torch

from ms_model.io.events import EventArray

# Repris de DEVO/utils/event_utils.py::to_voxel_grid (Klenk et al., TUM 2023),
# vendoré ici pour ne pas dépendre du chemin/installation de DEVO.


def to_voxel_grid(xs, ys, ts, ps, H=480, W=640, nb_of_time_bins=5):
    """Représentation voxel grid d'un flux d'events. (nb_of_time_bins, H, W)

    Le temps est discrétisé en `nb_of_time_bins` bins ; chaque event est
    réparti par interpolation bilinéaire (espace + temps) sur les bins
    voisins, pondéré par sa polarité.
    """
    voxel_grid = torch.zeros(nb_of_time_bins, H, W, dtype=torch.float32, device="cpu")
    voxel_grid_flat = voxel_grid.flatten()

    ps = ps.astype(np.int8)
    ps[ps == 0] = -1

    duration = ts[-1] - ts[0]
    start_timestamp = ts[0]
    features = torch.from_numpy(np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1))
    x = features[:, 0]
    y = features[:, 1]
    polarity = features[:, 3].float()
    t = (features[:, 2] - start_timestamp) * (nb_of_time_bins - 1) / duration
    t = t.to(torch.float64)

    left_t, right_t = t.floor(), t.floor() + 1
    left_x, right_x = x.floor(), x.floor() + 1
    left_y, right_y = y.floor(), y.floor() + 1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (
                    (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t)
                    & (lim_x <= W - 1) & (lim_y <= H - 1) & (lim_t <= nb_of_time_bins - 1)
                )

                lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H

                weight = polarity * (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs()) * (1 - (lim_t - t).abs())
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    return voxel_grid


def make_voxel_sequence(ea: EventArray, frame_ts: np.ndarray, nb_of_time_bins: int = 5) -> torch.Tensor:
    """Découpe ea selon frame_ts (cf. make_frames) et convertit chaque intervalle en voxel grid.

    Retourne un tenseur (len(frame_ts) - 1, nb_of_time_bins, H, W).
    """
    grids = []
    t = ea.t
    # Events triés (cas normal) : bornes par dichotomie, O(log N) par frame au lieu d'un masque
    # sur tout le flux (O(N) par frame, des heures sur une longue scène EDS / TUM-VIE).
    sorted_t = len(t) < 2 or bool(np.all(t[1:] >= t[:-1]))
    if sorted_t:
        bounds = np.searchsorted(t, np.asarray(frame_ts), side="left")
    for i in range(1, len(frame_ts)):
        if sorted_t:
            sel = slice(bounds[i - 1], bounds[i])
            n = bounds[i] - bounds[i - 1]
        else:
            sel = (t >= frame_ts[i - 1]) & (t < frame_ts[i])
            n = sel.sum()
        if n < 2:
            grids.append(torch.zeros(nb_of_time_bins, ea.H, ea.W, dtype=torch.float32))
            continue
        grids.append(to_voxel_grid(
            ea.x[sel], ea.y[sel], t[sel], ea.p[sel],
            H=ea.H, W=ea.W, nb_of_time_bins=nb_of_time_bins,
        ))
    return torch.stack(grids)
