#!/usr/bin/env bash
# Prétraitement d'un dataset (CPU, pas de GPU nécessaire).
#
# Usage : jobs/preprocess.sh <dataset> [--dry-run]
#   evimo2  conversion au format EVIMO1 + format DEVO des séquences d'éval, puis caches de voxels
#   rpg     scripts/pp_rpg.py
#
# Idempotent : relancer ne refait que ce qui manque.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET=""
DRY_RUN="${DRY_RUN:-0}"
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -*) die "option inconnue : $1" ;;
        *)  [[ -z "$DATASET" ]] || die "un seul dataset attendu"; DATASET="$1"; shift ;;
    esac
done
[[ -n "$DATASET" ]] || { usage; die "dataset manquant"; }

job_setup_env
cd "$REPO_ROOT"

case "$DATASET" in
    evimo2)
        CMDS=("bash scripts/download/download_evimo2.sh"
              "python scripts/preprocess_voxels.py --data-root datasets/evimo2/samsung_mono --nb-time-bins 5")
        ;;
    rpg)
        CMDS=("bash scripts/download/download_rpg.sh")
        ;;
    *)  die "dataset inconnu : $DATASET (evimo2 | rpg)" ;;
esac

for cmd in "${CMDS[@]}"; do
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "dry-run : $cmd"
    else
        log "$cmd"
        eval "$cmd"
    fi
done
