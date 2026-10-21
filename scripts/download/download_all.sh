#!/usr/bin/env bash
# Télécharge et prétraite tout ce que la campagne MADEVO utilise, dans l'ordre d'utilité.
#
#   1. poids DEVO officiels                       pretrained/DEVO.pth          ~40 Mo
#   2. EVIMO2 samsung_mono IMO (train + éval)     datasets/evimo2/             ~2,4 Go
#      + SfM (scènes statiques), avec --sfm                                    ~10,4 Go
#   3. cross-dataset, uniquement en éval (séquences des splits/<dataset>/<dataset>_val.txt) :
#      RPG ~4,3 Go · MVSEC ~6 Go · VECtor ~15 Go · EDS ~26 Go · TUM-VIE ~44 Go
#
# Usage : bash scripts/download/download_all.sh [--sfm] [--only a,b] [--skip a,b] [--keep-going]
#   --only        sous-ensemble parmi : devo evimo2 rpg mvsec eds vector tumvie eigen
#   --skip        tout sauf ceux-là
#   --sfm         ajoute la catégorie SfM d'EVIMO2 (cf. docs/sfm_sequences.md)
#   --keep-going  continue après l'échec d'un dataset (quota Google Drive, p. ex.) et
#                 résume les échecs à la fin
#
# Chaque script est idempotent : relancer ne retélécharge que ce qui manque. Comptez ~110 Go
# de téléchargement (~120 Go avec --sfm) et ~200 Go sur disque après prétraitement (MADEVO_DATASETS pour placer
# les données ailleurs que dans le repo).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
HERE="$(dirname "${BASH_SOURCE[0]}")"

ALL=(devo evimo2 rpg mvsec eds vector tumvie)
ONLY="" SKIP="" SFM=0 KEEP_GOING=0
while (( $# )); do
    case "$1" in
        --only) ONLY="${2:?}"; shift ;;
        --skip) SKIP="${2:?}"; shift ;;
        --sfm) SFM=1 ;;
        --keep-going) KEEP_GOING=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "option inconnue : $1" ;;
    esac
    shift
done

if [[ -n "$ONLY" ]]; then IFS=',' read -r -a TARGETS <<< "$ONLY"; else TARGETS=("${ALL[@]}"); fi
IFS=',' read -r -a SKIPS <<< "$SKIP"

failed=()
for t in "${TARGETS[@]}"; do
    [[ " ${SKIPS[*]} " == *" $t "* ]] && continue
    case "$t" in
        devo)   cmd=(bash "$HERE/download_devo_weights.sh") ;;
        eigen)  cmd=(bash "$HERE/download_eigen.sh") ;;
        evimo2) cmd=(bash "$HERE/download_evimo2.sh"); (( SFM )) && cmd+=(--sfm) ;;
        rpg|mvsec|eds|vector|tumvie) cmd=(bash "$HERE/download_$t.sh") ;;
        *) die "cible inconnue : $t (attendu : ${ALL[*]} eigen)" ;;
    esac
    log "================ $t ================"
    if ! "${cmd[@]}"; then
        (( KEEP_GOING )) || die "échec : $t"
        failed+=("$t")
    fi
done

if (( ${#failed[@]} )); then
    log "ÉCHECS : ${failed[*]} — relancer avec --only $(IFS=,; echo "${failed[*]}")"
    exit 1
fi
log "tout est en place sous $DATASETS_DIR et $PRETRAINED_DIR"
