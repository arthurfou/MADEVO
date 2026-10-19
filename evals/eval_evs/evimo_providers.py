"""Fournisseurs de masques dynamiques pour les évaluations EVIMO2.

Partagé par l'éval unifiée (`evals/eval_table.py`) : l'oracle lit le masque GT du dataset,
le provider « appris » fait tourner le modèle MS sur la séquence.

⚠️ `gt_timestamp_offset()` est le correctif d'alignement des timestamps GT (l'ancienne
passe `_fixed` des tables du stage) : EVIMO2 stocke `meta["frames"]["ts"]` en secondes
relatives alors que DEVO travaille en microsecondes Unix. Sans lui, les masques sont
associés aux mauvaises frames et toute la table est fausse. **Ne pas le supprimer.**
"""
import os

import numpy as np


def resolve_mask_npz(scene, mask_root):
    """scene 'imo/eval/seq_name' -> '{mask_root}/imo/eval/seq_name.npz'.

    mask_root doit pointer vers datasets/evimo2/evimo1_format
    (produit par scripts/convert_evimo2_to_evimo1.py).
    """
    scene = scene.strip("/")

    # Format principal : {mask_root}/{scene}.npz
    c1 = os.path.join(mask_root, scene + ".npz")
    if os.path.exists(c1):
        return c1

    # Fallback : seq_name seul → chercher dans imo/eval/
    parts = scene.split("/")
    seq = parts[-1]
    for subdir in ("imo/eval", "imo/train"):
        c = os.path.join(mask_root, subdir, seq + ".npz")
        if os.path.exists(c):
            return c

    return c1  # retourner pour un message d'erreur lisible


def gt_timestamp_offset(datapath_val, npz_path):
    """Offset (s) à soustraire de ts_us/1e6 pour aligner avec les ts relatifs du npz.

    Identique à EVIMO1 — ne pas supprimer : EVIMO2 stocke aussi meta["frames"]["ts"]
    en secondes relatives alors que DEVO utilise Unix epoch µs.
    """
    from ms_model.io.loaders import load_evimo_mask
    tss_file = os.path.join(datapath_val, "tss_imgs_us.txt")
    if not os.path.exists(tss_file):
        return 0.0
    tss_us = np.loadtxt(tss_file)
    if tss_us.ndim == 0:
        tss_us = np.array([float(tss_us)])
    fm = load_evimo_mask(npz_path)
    if len(tss_us) == 0 or len(fm.ts) == 0:
        return 0.0
    devo_t0_s = float(np.sort(tss_us)[0]) / 1e6
    meta_t0_s = float(fm.ts[0])
    if abs(devo_t0_s - meta_t0_s) < 100.0:
        return 0.0
    offset = devo_t0_s - meta_t0_s
    print(f"[ts_offset] {os.path.basename(datapath_val)}: offset={offset:.3f}s "
          f"(DEVO t0={devo_t0_s:.3f}s, meta t0={meta_t0_s:.6f}s)")
    return offset


def make_oracle_factory(mask_root, patch_size=4, thicken_radius=0, device="cuda",
                        mask_motion="all", motion_thresh_m=0.02, motion_window_s=0.2):
    from ms_model.oracle import OracleDynMaskProvider

    def factory(scene, datapath_val):
        npz = resolve_mask_npz(scene, mask_root)
        if not os.path.exists(npz):
            raise FileNotFoundError(f"[masque] npz GT introuvable pour '{scene}': {npz}")
        offset = gt_timestamp_offset(datapath_val, npz)
        print(f"[masque] Oracle masks <- {npz}")
        return OracleDynMaskProvider.from_evimo_npz(
            npz, patch_size=patch_size, thicken_radius=thicken_radius,
            device=device, ts_offset_s=offset, mask_motion=mask_motion,
            motion_thresh_m=motion_thresh_m, motion_window_s=motion_window_s)

    return factory


def make_learned_factory(mask_root, ms_weights, ms_config, threshold=None, device="cuda"):
    from ms_model.oracle import LearnedDynMaskProvider

    def factory(scene, datapath_val):
        npz = resolve_mask_npz(scene, mask_root)
        if not os.path.exists(npz):
            raise FileNotFoundError(f"[masque] npz introuvable pour '{scene}': {npz}")
        offset = gt_timestamp_offset(datapath_val, npz)
        print(f"[masque] Modèle MS -> masque appris sur {npz}")
        return LearnedDynMaskProvider(
            npz_path=npz, weights_path=ms_weights, config_path=ms_config,
            device=device, threshold=threshold, ts_offset_s=offset)

    return factory
