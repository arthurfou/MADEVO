#!/usr/bin/env bash
# MVSEC — Multi Vehicle Stereo Event Camera (Zhu et al., RA-L 2018), séquences indoor_flying.
# Page officielle : https://daniilidis-group.github.io/mvsec/download/
# Les .hdf5 ne sont publiés que sur Google Drive : téléchargés avec gdown, ids listés ici
# (extraits du dossier Drive officiel 1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV).
#
# Résultat :
#   datasets/mvsec/<seq>_data/   <seq>_data.hdf5, <seq>_gt.hdf5, tss_imgs_us_left.txt,
#                                calib_undist_left.txt, rectify_map_left.h5
#   → pour l'éval : configs/eval/mvsec.yaml (--datapath datasets/mvsec, splits/mvsec/mvsec_val.txt)
#
# Usage : bash scripts/download/download_mvsec.sh [--sequences indoor_flying1,...] [--no-preprocess] [--keep-raw]
#   --sequences      sous-ensemble (défaut : indoor_flying1 à 4, ~6 Go)
#   --keep-raw       conserve les .hdf5 d'origine (pp_mvsec.py en fait une copie par séquence)
#
# Le prétraitement (scripts/pp_mvsec.py) nécessite l'env conda du projet. Google Drive limite
# le nombre de téléchargements par jour et par fichier : en cas d'échec, relancer plus tard.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

declare -A DRIVE_IDS=(
    [indoor_flying1_data]=18lISHaEtIHxTETuuNHBRHPTtsVhq29l_
    [indoor_flying1_gt]=1U9NChDGHdBGVSZrQJeQLSG0_i_z3NquN
    [indoor_flying2_data]=1BD7cVNUdDgvqTYPdzxy9Tcyl1FrjhrjF
    [indoor_flying2_gt]=1zKJS-2Ayn2FitjRw40x_qVYLQDmh-C4u
    [indoor_flying3_data]=1Q7Mm_oZrSI_cTRktY8LcUaLSrGUmAERJ
    [indoor_flying3_gt]=1DnT5SOOuKP2xuM3clMd_L54qnRG-JOI6
    [indoor_flying4_data]=1bngs9QbVX3KtESrPLyXJbGyxuZI8j9tb
    [indoor_flying4_gt]=1UweN8PPaNlG27qp-ORc89gFRIzaid38o
    [indoor_flying_calib]=1dZNctjh7i_Myq-wACiUuBPli9FR7d96U
)
OUT="$DATASETS_DIR/mvsec"
RAW="$OUT/raw"

SEQS="" PREPROCESS=1 KEEP_RAW=0
while (( $# )); do
    case "$1" in
        --sequences) SEQS="${2:?}"; shift ;;
        --no-preprocess) PREPROCESS=0 ;;
        --keep-raw) KEEP_RAW=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "option inconnue : $1" ;;
    esac
    shift
done
# le split liste « indoor_flying1_data » ; on accepte aussi « indoor_flying1 »
mapfile -t SEQUENCES < <(select_sequences "$SEQS" "$REPO_ROOT/splits/mvsec/mvsec_val.txt" | sed 's/_data$//')

(( PREPROCESS )) && require_python_modules numpy cv2 h5py
require_cmd unzip
mkdir -p "$RAW"

# calibration (intrinsèques et distorsion lues par pp_mvsec.py dans raw/indoor_flying_calib/)
if [[ ! -f "$RAW/indoor_flying_calib/camchain-imucam-indoor_flying.yaml" ]]; then
    fetch_gdrive "${DRIVE_IDS[indoor_flying_calib]}" "$RAW/indoor_flying_calib.zip"
    unzip -q -o "$RAW/indoor_flying_calib.zip" -d "$RAW/indoor_flying_calib"
fi

for seq in "${SEQUENCES[@]}"; do
    [[ -n "${DRIVE_IDS[${seq}_data]:-}" ]] || die "séquence MVSEC inconnue : $seq"
    if [[ -f "$OUT/${seq}_data/${seq}_data.hdf5" ]]; then
        log "déjà prétraité : $OUT/${seq}_data"
        continue
    fi
    for part in data gt; do
        fetch_gdrive "${DRIVE_IDS[${seq}_${part}]}" "$RAW/${seq}_${part}.hdf5"
    done
done

if (( PREPROCESS )); then
    ensure_headless_evo
    log "prétraitement DEVO (scripts/pp_mvsec.py)"
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python scripts/pp_mvsec.py --indir "$RAW")
    for seq in "${SEQUENCES[@]}"; do
        # pp_mvsec.py écrit à côté des .hdf5 : on range chaque séquence sous datasets/mvsec/
        if [[ -d "$RAW/${seq}_data" ]]; then
            check_outputs "$RAW/${seq}_data" "${seq}_data.hdf5" "${seq}_gt.hdf5" \
                tss_imgs_us_left.txt calib_undist_left.txt rectify_map_left.h5
            rm -rf "$OUT/${seq}_data"
            mv "$RAW/${seq}_data" "$OUT/${seq}_data"
        fi
        check_outputs "$OUT/${seq}_data" "${seq}_data.hdf5" "${seq}_gt.hdf5" \
            tss_imgs_us_left.txt calib_undist_left.txt rectify_map_left.h5
        if (( ! KEEP_RAW )); then
            rm -f "$RAW/${seq}_data.hdf5"* "$RAW/${seq}_gt.hdf5"*
        fi
    done
    (( KEEP_RAW )) || rm -rf "$RAW/indoor_flying_calib.zip"*
fi

log "terminé : $(du -sh "$OUT" | cut -f1) dans $OUT"
