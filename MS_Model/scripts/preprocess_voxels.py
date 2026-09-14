"""Prétraitement offline des voxel grids EVIMO.

Calcule et sauvegarde les voxel grids (`<séquence>.voxels_bins<N>.npy`) à côté de chaque
séquence, sans lancer d'entraînement. Facultatif : EvimoSegDataset et EvimoClipDataset
construisent le cache à la volée s'il manque ; ce script permet de le faire une seule fois
sur un nœud CPU avant de soumettre les jobs GPU.

Séquences reconnues sous --data-root :
- EVIMO2 : répertoires contenant `dataset_info.npz` ;
- EVIMO1 : fichiers `*.npz` (hors fichiers `dataset_*.npz` des répertoires EVIMO2).

Le format .npy permet un chargement lazy via numpy memmap (pas d'OOM même sur
de grands datasets). Les anciens caches .pt sont migrés automatiquement.

Usage :
    python scripts/preprocess_voxels.py --data-root ../datasets/evimo2/samsung_mono/imo/train
    python scripts/preprocess_voxels.py --data-root ../datasets/evimo_full --nb-time-bins 5
"""

import argparse
import time
from pathlib import Path

import numpy as np

from ms_model.data.evimo_dataset import ensure_voxel_cache


def find_sequences(root: Path) -> list[Path]:
    evimo2_dirs = sorted(p.parent for p in root.rglob("dataset_info.npz"))
    evimo1_npz = sorted(p for p in root.rglob("*.npz") if not p.name.startswith("dataset_"))
    return evimo2_dirs + evimo1_npz


def preprocess(data_root: str, nb_time_bins: int, force: bool) -> None:
    root = Path(data_root)
    sequences = find_sequences(root)

    if not sequences:
        print(f"Aucune séquence EVIMO trouvée dans {root}")
        return

    print(f"{len(sequences)} séquences trouvées dans {root}\n")

    for i, seq in enumerate(sequences, 1):
        cache_path = seq.with_suffix(f".voxels_bins{nb_time_bins}.npy")
        if cache_path.exists() and not force:
            print(f"[{i}/{len(sequences)}] {seq.name} — cache déjà présent, ignoré")
            continue

        t0 = time.time()
        print(f"[{i}/{len(sequences)}] {seq.name} ...", flush=True)
        cache_path = ensure_voxel_cache(seq, nb_time_bins, overwrite=force)
        n = np.load(cache_path, mmap_mode="r").shape[0]
        print(f"    {n} frames, {time.time() - t0:.1f}s")

    print("\nPrétraitement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",     required=True, help="Racine du dataset EVIMO (EVIMO1 ou EVIMO2)")
    parser.add_argument("--nb-time-bins",  type=int, default=5)
    parser.add_argument("--force",         action="store_true", help="Recalculer même si le cache existe")
    args = parser.parse_args()

    preprocess(args.data_root, args.nb_time_bins, args.force)
