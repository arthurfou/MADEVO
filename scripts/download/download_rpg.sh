#!/usr/bin/env bash
# RPG — Event-based Stereo Visual Odometry (Zhou et al., ECCV 2018), séquences DAVIS.
# Page officielle : https://rpg.ifi.uzh.ch/ECCV18_stereo_davis.html
#
# Résultat :
#   datasets/rpg/<seq>.bag   rosbags bruts (~4,3 Go au total)
#   datasets/rpg/<seq>/      format DEVO : evs_left.txt, calib_undist_left.txt,
#                            tss_imgs_us_left.txt, gt_stamped_left.txt, rectify_map_left.h5
#   → pour l'éval : --datapath datasets/rpg --val_split splits/rpg/rpg_val.txt
#
# Usage : bash scripts/download/download_rpg.sh [--no-preprocess] [--delete-bags]
#   --no-preprocess  télécharge seulement les rosbags
#   --delete-bags    supprime les rosbags une fois le prétraitement réussi
#
# Le prétraitement (scripts/pp_rpg.py) nécessite l'env conda du projet.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BASE_URL="https://rpg.ifi.uzh.ch/datasets/eccv18_stereo_davis/bags"
SEQUENCES=(bin boxes1 boxes2 desk1 desk2 monitor1 monitor2 reader)  # = splits/rpg/rpg_val.txt
OUT="$DATASETS_DIR/rpg"

PREPROCESS=1
DELETE_BAGS=0
for arg in "$@"; do
    case "$arg" in
        --no-preprocess) PREPROCESS=0 ;;
        --delete-bags) DELETE_BAGS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "option inconnue : $arg" ;;
    esac
done
(( PREPROCESS || ! DELETE_BAGS )) || die "--delete-bags n'a pas de sens avec --no-preprocess"

require_cmd curl
(( PREPROCESS )) && require_python_modules numpy scipy cv2 h5py tqdm rosbags
mkdir -p "$OUT"

for seq in "${SEQUENCES[@]}"; do
    fetch "$BASE_URL/$seq.bag" "$OUT/$seq.bag"
done

if (( PREPROCESS )); then
    log "prétraitement DEVO (scripts/pp_rpg.py)"
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python scripts/pp_rpg.py --indir "$OUT")
    # pp_rpg.py lance des sous-processus dont les erreurs ne remontent pas : on vérifie les sorties.
    for seq in "${SEQUENCES[@]}"; do
        for f in evs_left.txt calib_undist_left.txt tss_imgs_us_left.txt gt_stamped_left.txt rectify_map_left.h5; do
            [[ -s "$OUT/$seq/$f" ]] || die "prétraitement incomplet : $OUT/$seq/$f manquant"
        done
    done
    if (( DELETE_BAGS )); then
        for seq in "${SEQUENCES[@]}"; do rm -f "$OUT/$seq.bag"; done
        log "rosbags supprimés"
    fi
fi

log "terminé : $(du -sh "$OUT" | cut -f1) dans $OUT"
