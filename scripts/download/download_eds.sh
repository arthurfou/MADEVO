#!/usr/bin/env bash
# EDS — Event-aided Direct Sparse Odometry (Hidalgo-Carrió et al., CVPR 2022).
# Page officielle : https://rpg.ifi.uzh.ch/eds.html
#
# Résultat :
#   datasets/eds/<seq>/   events.h5, images/, images_timestamps_us.txt, t_offset_us.txt,
#                         stamped_groundtruth_us.txt, calib_undist_calib0.json, rectify_map_calib0.h5
#   → pour l'éval : configs/eval/eds.yaml (--datapath datasets/eds, splits/eds/eds_val.txt)
#
# Usage : bash scripts/download/download_eds.sh [--sequences a,b] [--no-preprocess] [--keep-archive]
#   --sequences      sous-ensemble du split (défaut : splits/eds/eds_val.txt, 9 séquences, ~26 Go)
#   --no-preprocess  télécharge et extrait seulement
#   --keep-archive   conserve les .tgz après extraction
#
# Le prétraitement (scripts/pp_eds.py) nécessite l'env conda du projet. Les intrinsèques
# (calib0) sont écrites en dur dans pp_eds.py et utils/load_utils.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BASE_URL="https://download.ifi.uzh.ch/rpg/eds/dataset"
OUT="$DATASETS_DIR/eds"
SPLIT="$REPO_ROOT/splits/eds/eds_val.txt"

SEQS="" PREPROCESS=1 KEEP_ARCHIVE=0
while (( $# )); do
    case "$1" in
        --sequences) SEQS="${2:?}"; shift ;;
        --no-preprocess) PREPROCESS=0 ;;
        --keep-archive) KEEP_ARCHIVE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "option inconnue : $1" ;;
    esac
    shift
done
mapfile -t SEQUENCES < <(select_sequences "$SEQS" "$SPLIT")

require_cmd curl tar
(( PREPROCESS )) && require_python_modules numpy cv2 h5py tqdm
mkdir -p "$OUT"

for seq in "${SEQUENCES[@]}"; do
    if [[ -f "$OUT/$seq/.extracted" ]]; then
        log "déjà extrait : $OUT/$seq"
        continue
    fi
    fetch "$BASE_URL/$seq/$seq.tgz" "$OUT/$seq.tgz"
    log "extraction de $seq"
    # l'archive n'a pas de dossier racine : on l'extrait dans <seq>/ (et on aplatit au cas où)
    mkdir -p "$OUT/$seq"
    tar -xzf "$OUT/$seq.tgz" -C "$OUT/$seq"
    if [[ -f "$OUT/$seq/$seq/events.h5" ]]; then
        mv "$OUT/$seq/$seq"/* "$OUT/$seq/" && rmdir "$OUT/$seq/$seq"
    fi
    [[ -f "$OUT/$seq/events.h5" ]] || die "arborescence inattendue : $OUT/$seq/events.h5 absent après extraction"
    touch "$OUT/$seq/.extracted"
    (( KEEP_ARCHIVE )) || rm -f "$OUT/$seq.tgz"
done

if (( PREPROCESS )); then
    ensure_headless_evo
    log "prétraitement DEVO (scripts/pp_eds.py)"
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python scripts/pp_eds.py --indir "$OUT")
    for seq in "${SEQUENCES[@]}"; do
        check_outputs "$OUT/$seq" events.h5 images_timestamps_us.txt t_offset_us.txt \
            stamped_groundtruth_us.txt calib_undist_calib0.json rectify_map_calib0.h5
    done
fi

log "terminé : $(du -sh "$OUT" | cut -f1) dans $OUT"
