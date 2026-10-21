#!/usr/bin/env bash
# VECtor — Versatile Event-Centric Benchmark (Gao et al., RA-L 2022).
# Page officielle : https://star-datasets.github.io/vector/
# Les séquences ne sont publiées que sur Google Drive : ids dans vector_files.tsv (extraits de
# la page de téléchargement), téléchargés avec gdown. Calibrations : site du benchmark.
#
# Résultat :
#   datasets/vector/0_calib/                  intrinsèques event gauche, extrinsèques
#   datasets/vector/<seq>/                    <seq>1.synced.left_event.hdf5,
#                                             <seq>1.synced.left_camera/, <seq>1.synced.gt.txt,
#                                             poses_evs_left.txt, tss_imgs_us_left.txt,
#                                             calib_undist_evs_left.txt, rectify_map_left.h5
#   → pour l'éval : configs/eval/vector.yaml (--datapath datasets/vector, splits/vector/vector_val.txt)
#
# Usage : bash scripts/download/download_vector.sh [--sequences a,b] [--no-preprocess] [--keep-archive]
#   --sequences      sous-ensemble (défaut : splits/vector/vector_val.txt, 17 séquences, ~15 Go)
#
# Google Drive limite le nombre de téléchargements par jour : en cas d'échec, relancer plus
# tard, les fichiers déjà complets ne sont pas retéléchargés.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CALIB_URL="https://star-datasets.github.io/vector/assets/yaml"
CALIB_FILES=(left_event_camera_intrinsic_results.yaml camera_mocap_extrinsic_results1.yaml
             small_scale_joint_camera_extrinsic_results.yaml large_scale_joint_camera_extrinsic_results.yaml)
FILES_TSV="$(dirname "${BASH_SOURCE[0]}")/vector_files.tsv"
OUT="$DATASETS_DIR/vector"
SPLIT="$REPO_ROOT/splits/vector/vector_val.txt"

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

require_cmd curl unzip
(( PREPROCESS )) && require_python_modules numpy cv2 h5py yaml
mkdir -p "$OUT/0_calib"
for f in "${CALIB_FILES[@]}"; do
    fetch "$CALIB_URL/$f" "$OUT/0_calib/$f"
done

drive_id() {  # drive_id <séquence> <fichier>
    awk -F'\t' -v s="$1" -v k="$2" '$1 == s && $2 == k { print $3 }' "$FILES_TSV"
}

for seq in "${SEQUENCES[@]}"; do
    d="$OUT/$seq"
    mkdir -p "$d"
    for kind in left_event left_camera gt; do
        [[ -n "$(drive_id "$seq" "$kind")" ]] || die "séquence VECtor absente de vector_files.tsv : $seq ($kind)"
    done
    fetch_gdrive "$(drive_id "$seq" left_event)" "$d/${seq}1.synced.left_event.hdf5"
    fetch_gdrive "$(drive_id "$seq" gt)" "$d/${seq}1.synced.gt.txt"
    if [[ ! -f "$d/.extracted" ]]; then
        fetch_gdrive "$(drive_id "$seq" left_camera)" "$d/${seq}1.synced.left_camera.zip"
        unzip -q -o "$d/${seq}1.synced.left_camera.zip" -d "$d"
        [[ -f "$d/${seq}1.synced.left_camera/timestamp.txt" ]] \
            || die "arborescence inattendue : $d/${seq}1.synced.left_camera/timestamp.txt absent"
        touch "$d/.extracted"
        (( KEEP_ARCHIVE )) || rm -f "$d/${seq}1.synced.left_camera.zip" "$d/${seq}1.synced.left_camera.zip.ok"
    fi
done

if (( PREPROCESS )); then
    ensure_headless_evo
    log "prétraitement DEVO (scripts/pp_vector.py)"
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python scripts/pp_vector.py --indir "$OUT")
    for seq in "${SEQUENCES[@]}"; do
        check_outputs "$OUT/$seq" "${seq}1.synced.left_event.hdf5" poses_evs_left.txt \
            tss_imgs_us_left.txt calib_undist_evs_left.txt rectify_map_left.h5
    done
fi

log "terminé : $(du -sh "$OUT" | cut -f1) dans $OUT"
