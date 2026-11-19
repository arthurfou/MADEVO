#!/usr/bin/env bash
# VECtor depuis une machine personnelle (quand Google Drive bloque gdown sur les serveurs).
#
# Télécharge, pour chaque séquence de splits/vector/vector_val.txt :
#   <seq>1.synced.left_event.hdf5, <seq>1.synced.gt.txt, et du zip d'images
#   <seq>1.synced.left_camera.zip SEULEMENT <seq>1.synced.left_camera/timestamp.txt (le zip est
#   supprimé ensuite : 16 Go d'images dont DEVO n'utilise que les timestamps).
# L'arborescence produite est celle de download_vector.sh (marqueurs .ok et .extracted compris) :
# une fois copiée sur le serveur, `download_vector.sh` n'y retélécharge rien et fait le prétraitement.
#
# Usage :
#   bash scripts/download/download_vector_local.sh [--out DIR] [--sequences a,b] [--to HOTE:CHEMIN]
#     --out        dossier de sortie (défaut : ./vector_dl) — prévoir ~8 Go + un zip de 0,4-1,3 Go à la fois
#     --sequences  sous-ensemble (défaut : les 17 séquences du split)
#     --to         copie finale par rsync, ex. royce:/Data/arthur.fournier/madevo/datasets/vector/
# Relançable : ce qui est complet n'est pas retéléchargé ; les échecs sont listés à la fin.
#
# Si gdown échoue (quota Drive), télécharger le fichier dans un navigateur (lien affiché), le déposer
# dans <out>/_navigateur/ sous son nom d'origine, et relancer : le script le range lui-même.
#
# Prérequis : bash, python3 avec gdown (`pip install gdown`), unzip, rsync (pour --to).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TSV="$HERE/vector_files.tsv"
SPLIT="$REPO/splits/vector/vector_val.txt"
OUT="./vector_dl" SEQS="" TO=""
while (( $# )); do
    case "$1" in
        --out) OUT="${2:?}"; shift ;;
        --sequences) SEQS="${2:?}"; shift ;;
        --to) TO="${2:?}"; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "option inconnue : $1" >&2; exit 2 ;;
    esac
    shift
done

log() { echo "[$(date +%H:%M:%S)] $*"; }
PY=$(command -v python3 || command -v python)
"$PY" -c "import gdown" 2>/dev/null || { echo "gdown manquant : pip install gdown" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip manquant : sudo apt install unzip" >&2; exit 1; }

mkdir -p "$OUT/_navigateur"
OUT="$(cd "$OUT" && pwd)"
if [[ -n "$SEQS" ]]; then mapfile -t SEQUENCES < <(tr ',' '\n' <<< "$SEQS")
else mapfile -t SEQUENCES < <(grep -vE '^\s*(#|$)' "$SPLIT" | tr -d '\r'); fi

drive_id() { awk -F'\t' -v s="$1" -v k="$2" '$1 == s && $2 == k { print $3 }' "$TSV"; }
FAILED=()

# get <id> <destination> : depuis _navigateur/ si le fichier y a été déposé, sinon gdown.
get() {
    local id=$1 dest=$2 name; name=$(basename "$dest")
    [[ -f "$dest.ok" && -s "$dest" ]] && { log "déjà là : $name"; return 0; }
    if [[ -s "$OUT/_navigateur/$name" ]]; then
        mv "$OUT/_navigateur/$name" "$dest" && touch "$dest.ok" && log "rangé depuis _navigateur/ : $name"
        return 0
    fi
    log "gdown : $name"
    if "$PY" -c 'import sys, gdown
sys.exit(0 if gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=False, resume=True) else 1)' "$id" "$dest" \
        && [[ -s "$dest" ]]; then
        touch "$dest.ok"; return 0
    fi
    log "ÉCHEC : $name — navigateur : https://drive.google.com/uc?id=$id&export=download → $OUT/_navigateur/$name"
    FAILED+=("$name"); return 1
}

for seq in "${SEQUENCES[@]}"; do
    d="$OUT/$seq"; mkdir -p "$d"
    for k in left_event left_camera gt; do
        [[ -n "$(drive_id "$seq" "$k")" ]] || { echo "séquence absente de vector_files.tsv : $seq ($k)" >&2; exit 1; }
    done
    get "$(drive_id "$seq" left_event)" "$d/${seq}1.synced.left_event.hdf5"
    get "$(drive_id "$seq" gt)" "$d/${seq}1.synced.gt.txt"
    if [[ ! -f "$d/.extracted" ]]; then
        z="$d/${seq}1.synced.left_camera.zip"
        if get "$(drive_id "$seq" left_camera)" "$z"; then
            unzip -q -o "$z" '*timestamp.txt' -d "$d"
            if [[ -f "$d/${seq}1.synced.left_camera/timestamp.txt" ]]; then
                touch "$d/.extracted"; rm -f "$z" "$z.ok"
                log "$seq : timestamp.txt extrait, zip supprimé"
            else
                log "ÉCHEC : $seq — timestamp.txt introuvable dans $z (arborescence inattendue, zip gardé)"
                FAILED+=("$seq/timestamp.txt")
            fi
        fi
    fi
done

if (( ${#FAILED[@]} )); then
    log "${#FAILED[@]} fichier(s) manquant(s) : ${FAILED[*]}"
    log "→ relancer plus tard, ou les déposer dans $OUT/_navigateur/ puis relancer"
    exit 1
fi
log "tout est là : $(du -sh "$OUT" | cut -f1) dans $OUT"
if [[ -n "$TO" ]]; then
    log "copie vers $TO"
    rsync -av --partial --exclude _navigateur "$OUT/" "$TO"
else
    log "copie sur le serveur : rsync -av --partial --exclude _navigateur $OUT/ royce:/Data/arthur.fournier/madevo/datasets/vector/"
fi
