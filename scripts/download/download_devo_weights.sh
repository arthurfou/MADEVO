#!/usr/bin/env bash
# Poids pré-entraînés officiels de DEVO (Klenk et al., 3DV 2024), ~40 Mo.
#
# Résultat : pretrained/DEVO.pth
#
# Usage : bash scripts/download/download_devo_weights.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

URL="https://cvg.cit.tum.de/webshare/g/evs/DEVO.pth"

case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "option inconnue : $1" ;;
esac

require_cmd curl
mkdir -p "$PRETRAINED_DIR"
fetch "$URL" "$PRETRAINED_DIR/DEVO.pth"
log "terminé : $PRETRAINED_DIR/DEVO.pth"
