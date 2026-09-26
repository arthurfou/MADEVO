import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.masking import thicken_mask
from ms_model.representations.voxel import make_voxel_sequence


def downsample_mask(mask: np.ndarray, patch_size: int) -> torch.Tensor:
    """(H, W) -> (H//patch_size, W//patch_size), binaire (1 = patch contient un pixel dynamique)."""
    t = torch.from_numpy((mask != 0).astype(np.float32))[None, None]
    Hp, Wp = mask.shape[0] // patch_size, mask.shape[1] // patch_size
    out = F.adaptive_max_pool2d(t, (Hp, Wp))
    return out[0, 0]


# Version du format de cache. v1 (`.voxels_bins<N>.npy` / `.pt`, avant 09/2026) a été
# construite avec des timestamps de frames désalignés sur les frames possédant une GT : ces
# caches sont inutilisables et ne doivent jamais être relus.
VOXEL_CACHE_VERSION = 2


def voxel_cache_path(path: Union[str, Path], nb_time_bins: int) -> Path:
    """Chemin du cache de voxel grids d'une séquence EVIMO (fichier ou répertoire)."""
    return Path(path).with_suffix(f".voxels_bins{nb_time_bins}.v{VOXEL_CACHE_VERSION}.npy")


def ensure_voxel_cache(
    path: Union[str, Path],
    nb_time_bins: int,
    frame_ts: np.ndarray | None = None,
    overwrite: bool = False,
) -> Path:
    """Chemin du cache disque des voxel grids d'une séquence, construit s'il manque.

    Le cache contient une voxel grid par intervalle entre deux frames GT ; il est lu en
    memmap par EvimoSegDataset et EvimoClipDataset. L'écriture passe par un fichier
    temporaire pour ne jamais laisser de cache tronqué.
    """
    path = Path(path)
    cache_path = voxel_cache_path(path, nb_time_bins)
    if cache_path.exists() and not overwrite:
        return cache_path

    print(f"[voxel cache] construction : {cache_path}", flush=True)
    if frame_ts is None:
        frame_ts = load_evimo_mask(path).ts
    voxel_seq = make_voxel_sequence(load_events(path), frame_ts, nb_of_time_bins=nb_time_bins)

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, voxel_seq.numpy())
    tmp_path.replace(cache_path)
    return cache_path


class EvimoSegDataset(Dataset):
    """Séquences (voxel grids, masques dynamiques par patch) pour la motion segmentation.

    Les voxel grids sont mis en cache sur disque (voir voxel_cache_path) et lus
    via numpy memmap : seules les slices accédées sont chargées en RAM, ce qui
    permet d'entraîner sur de grands datasets sans OOM.
    """

    def __init__(
        self,
        npz_paths: list[Union[str, Path]],
        seq_len: int = 16,
        nb_time_bins: int = 5,
        patch_size: int = 4,
        mask_thicken_radius: int = 0,
    ):
        self.seq_len = seq_len
        # (cache_path, mask_tensor) — voxels restent sur disque
        self.sequences: list[tuple[Path, torch.Tensor]] = []
        self.index: list[tuple[int, int]] = []

        for i, path in enumerate(npz_paths):
            t0 = time.time()
            print(f"[EvimoSegDataset] ({i + 1}/{len(npz_paths)}) chargement {path} ...", flush=True)

            fm = load_evimo_mask(path)
            if mask_thicken_radius > 0:
                fm = thicken_mask(fm, radius=mask_thicken_radius)

            cache_path = ensure_voxel_cache(path, nb_time_bins, frame_ts=fm.ts)

            # Memmap : l'OS ne charge en RAM que les pages réellement accédées
            voxel_mmap = np.load(cache_path, mmap_mode='r')
            n = voxel_mmap.shape[0]

            mask_seq = torch.stack([downsample_mask(m, patch_size) for m in fm.masks[:-1]])

            seq_idx = len(self.sequences)
            for start in range(0, n - seq_len + 1, seq_len):
                self.index.append((seq_idx, start))
            self.sequences.append((cache_path, mask_seq))

            print(f"[EvimoSegDataset]   -> {n} frames, {time.time() - t0:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.index[idx]
        cache_path, mask_seq = self.sequences[seq_idx]
        end = start + self.seq_len
        # Rechargement du memmap nécessaire dans les workers DataLoader (après fork)
        voxel_mmap = np.load(cache_path, mmap_mode='r')
        voxel = torch.from_numpy(np.array(voxel_mmap[start:end], dtype=np.float32))
        return voxel, mask_seq[start:end]
