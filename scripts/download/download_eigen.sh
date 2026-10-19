#!/usr/bin/env bash
# En-têtes Eigen 3.4.0 pour compiler lietorch, ~2,5 Mo (repli : l'env conda
# du projet fournit déjà eigen 3.4, et setup.py le trouve tout seul).
#
# Résultat : thirdparty/eigen-3.4.0/
#
# Usage : bash scripts/download/download_eigen.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
DEST="$REPO_ROOT/thirdparty/eigen-3.4.0"

case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "option inconnue : $1" ;;
esac

require_cmd curl tar
if [[ -f "$DEST/Eigen/Dense" ]]; then
    log "déjà présent : $DEST"
    exit 0
fi

mkdir -p "$REPO_ROOT/thirdparty"
tarball="$REPO_ROOT/thirdparty/eigen-3.4.0.tar.gz"
fetch "$URL" "$tarball"
tar -xzf "$tarball" -C "$REPO_ROOT/thirdparty"
rm -f "$tarball"
[[ -f "$DEST/Eigen/Dense" ]] || die "extraction incomplète : $DEST/Eigen/Dense manquant"
log "terminé : $DEST"
