#!/usr/bin/env bash
# EVIMO2 v2 — caméra samsung_mono, catégorie IMO (motion segmentation), format NPZ (~2,4 Go).
# Page officielle : https://better-flow.github.io/evimo/download_evimo_2.html
#
# Résultat :
#   datasets/evimo2/samsung_mono/imo/{train,eval}/<seq>/  format natif (entraînements MS et couplé)
#   datasets/evimo2/evimo1_format/imo/eval/<seq>.npz      masques GT des évals (--mask_root)
#   datasets/evimo2/eval_preprocessed/<seq>/              format DEVO des évals (--datapath)
#
#   datasets/evimo2/samsung_mono/sfm/{train,eval}/<seq>/   avec --sfm : scènes statiques (caméra
#                                                         seule en mouvement), ~10,4 Go
#
# Usage : bash scripts/download/download_evimo2.sh [--sfm] [--no-preprocess] [--keep-archive]
#   --sfm            télécharge aussi la catégorie SfM (cf. docs/sfm_sequences.md)
#   --no-preprocess  télécharge et extrait seulement
#   --keep-archive   conserve le .tar.gz après extraction
#
# Le prétraitement (scripts/convert_evimo2_to_evimo1.py, scripts/preprocess_evimo2.py)
# nécessite l'env conda du projet.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

URL="https://obj.umiacs.umd.edu/evimo2v2npz/npz_samsung_mono_imo.tar.gz"
SFM_URL="https://obj.umiacs.umd.edu/evimo2v2npz/npz_samsung_mono_sfm.tar.gz"
OUT="$DATASETS_DIR/evimo2"
ARCHIVE="$OUT/npz_samsung_mono_imo.tar.gz"
EXTRACTED_MARKER="$OUT/samsung_mono/.imo_extracted"
VAL_SPLIT="$REPO_ROOT/splits/evimo2/evimo2_val.txt"

PREPROCESS=1
KEEP_ARCHIVE=0
SFM=0
for arg in "$@"; do
    case "$arg" in
        --no-preprocess) PREPROCESS=0 ;;
        --keep-archive) KEEP_ARCHIVE=1 ;;
        --sfm) SFM=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "option inconnue : $arg" ;;
    esac
done

require_cmd curl tar
(( PREPROCESS )) && require_python_modules numpy h5py
mkdir -p "$OUT"

if [[ -f "$EXTRACTED_MARKER" ]]; then
    log "déjà extrait : $OUT/samsung_mono/imo"
else
    fetch "$URL" "$ARCHIVE"
    log "extraction dans $OUT"
    tar -xzf "$ARCHIVE" -C "$OUT"
    [[ -d "$OUT/samsung_mono/imo/train" && -d "$OUT/samsung_mono/imo/eval" ]] \
        || die "arborescence inattendue après extraction (samsung_mono/imo/{train,eval} absent)"
    touch "$EXTRACTED_MARKER"
fi
if (( ! KEEP_ARCHIVE )) && [[ -f "$ARCHIVE" ]]; then
    rm -f "$ARCHIVE"
    log "archive supprimée"
fi

if (( SFM )); then
    SFM_ARCHIVE="$OUT/npz_samsung_mono_sfm.tar.gz"
    SFM_MARKER="$OUT/samsung_mono/.sfm_extracted"
    if [[ -f "$SFM_MARKER" ]]; then
        log "déjà extrait : $OUT/samsung_mono/sfm"
    else
        fetch "$SFM_URL" "$SFM_ARCHIVE"
        log "extraction de la catégorie SfM dans $OUT"
        tar -xzf "$SFM_ARCHIVE" -C "$OUT"
        [[ -d "$OUT/samsung_mono/sfm/train" ]] \
            || die "arborescence inattendue après extraction (samsung_mono/sfm/train absent)"
        touch "$SFM_MARKER"
    fi
    if (( ! KEEP_ARCHIVE )) && [[ -f "$SFM_ARCHIVE" ]]; then
        rm -f "$SFM_ARCHIVE"
    fi
fi

if (( PREPROCESS )); then
    cd "$REPO_ROOT"
    log "conversion au format EVIMO1 des séquences d'éval (masques GT)"
    python scripts/convert_evimo2_to_evimo1.py --src "$OUT/samsung_mono" --dst "$OUT/evimo1_format" --split imo/eval
    log "prétraitement DEVO des séquences d'éval"
    python scripts/preprocess_evimo2.py --src "$OUT/samsung_mono/imo/eval" --dst "$OUT/eval_preprocessed"

    # Les deux scripts Python continuent en cas d'erreur sur une séquence : on vérifie les sorties.
    while read -r seq || [[ -n "$seq" ]]; do
        [[ -z "$seq" || "$seq" == \#* ]] && continue
        [[ -s "$OUT/evimo1_format/imo/eval/$seq.npz" ]] || die "conversion incomplète : evimo1_format/imo/eval/$seq.npz manquant"
        for f in evs.npy tss_imgs_us.txt calib.txt gt_stamped.txt rectify_map.h5; do
            [[ -s "$OUT/eval_preprocessed/$seq/$f" ]] || die "prétraitement incomplet : eval_preprocessed/$seq/$f manquant"
        done
    done < "$VAL_SPLIT"
    # Marqueur : ces npz viennent du convertisseur aligné sur les frames GT (bug 6 corrigé).
    # scripts/setup_server.sh supprime tout evimo1_format/ qui ne le porte pas.
    echo "convert_evimo2_to_evimo1.py aligné sur les frames GT (INTEGRITY_CHECK.md, bug 6)" \
        > "$OUT/evimo1_format/.gt_aligned"
fi

log "terminé : $(du -sh "$OUT" | cut -f1) dans $OUT"
