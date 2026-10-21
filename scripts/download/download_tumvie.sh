#!/usr/bin/env bash
# TUM-VIE — TUM Stereo Visual-Inertial Event Dataset (Klenk et al., IROS 2021).
# Page officielle : https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset
#
# Résultat :
#   datasets/tumvie/<seq>/   <seq>-events_left.h5, left_images/, mocap_data.txt,
#                            left_images_undistorted/, calib_undist.json, rectify_map_left.h5
#   datasets/tumvie/camera-calibration{A,B}.json
#   → pour l'éval : configs/eval/tumvie.yaml (--datapath datasets/tumvie, splits/tumvie/tumvie_val.txt)
#
# Usage : bash scripts/download/download_tumvie.sh [--sequences a,b] [--no-preprocess] [--keep-archive]
#   --sequences      sous-ensemble (défaut : splits/tumvie/tumvie_val_public.txt, 12 séquences,
#                    ~39 Go d'events gauches + ~5 Go d'images)
#
# `mocap-1d-trans-calibB`, listée dans splits/tumvie/tumvie_val.txt (hérité de DEVO amont),
# n'existe pas sur le serveur de TUM-VIE : tumvie_val_public.txt est le même split sans elle.
#
# Seule la caméra event gauche est téléchargée (l'éval DEVO utilise camID=2, la gauche).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BASE_URL="https://tumevent-vi.vision.in.tum.de"
OUT="$DATASETS_DIR/tumvie"
SPLIT="$REPO_ROOT/splits/tumvie/tumvie_val_public.txt"

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

for c in A B; do
    fetch "$BASE_URL/camera-calibration$c.json" "$OUT/camera-calibration$c.json"
done

for seq in "${SEQUENCES[@]}"; do
    mkdir -p "$OUT/$seq"
    fetch "$BASE_URL/$seq/$seq-events_left.h5" "$OUT/$seq/$seq-events_left.h5"
    if [[ -f "$OUT/$seq/.extracted" ]]; then
        log "déjà extrait : $OUT/$seq (images, mocap)"
        continue
    fi
    fetch "$BASE_URL/$seq/$seq-vi_gt_data.tar.gz" "$OUT/$seq-vi_gt_data.tar.gz"
    log "extraction de $seq"
    tar -xzf "$OUT/$seq-vi_gt_data.tar.gz" -C "$OUT/$seq"
    # l'archive contient soit les fichiers, soit un dossier <seq>-vi_gt_data/ : on aplatit
    nested=$(find "$OUT/$seq" -mindepth 2 -maxdepth 2 -name mocap_data.txt -printf '%h\n' | head -n 1)
    if [[ -n "$nested" && "$nested" != "$OUT/$seq" ]]; then
        mv "$nested"/* "$OUT/$seq/" && rmdir "$nested"
    fi
    [[ -f "$OUT/$seq/mocap_data.txt" && -d "$OUT/$seq/left_images" ]] \
        || die "arborescence inattendue dans $OUT/$seq (mocap_data.txt ou left_images/ absent)"
    touch "$OUT/$seq/.extracted"
    (( KEEP_ARCHIVE )) || rm -f "$OUT/$seq-vi_gt_data.tar.gz"
done

if (( PREPROCESS )); then
    ensure_headless_evo
    log "prétraitement DEVO (scripts/pp_tumvie.py)"
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python scripts/pp_tumvie.py --indir "$OUT" --camId 0)
    for seq in "${SEQUENCES[@]}"; do
        check_outputs "$OUT/$seq" "*events_left.h5" mocap_data.txt calib_undist.json rectify_map_left.h5 \
            left_images_undistorted/image_timestamps_left.txt
    done
fi

log "terminé : $(du -sh "$OUT" | cut -f1) dans $OUT"
