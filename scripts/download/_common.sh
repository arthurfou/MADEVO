# Fonctions partagées par les scripts scripts/download/download_*.sh (à sourcer, pas à exécuter).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS_DIR="${MADEVO_DATASETS:-$REPO_ROOT/datasets}"
PRETRAINED_DIR="${MADEVO_PRETRAINED:-$REPO_ROOT/pretrained}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERREUR : $*" >&2; exit 1; }

# Affiche l'en-tête de commentaires du script appelant.
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

require_cmd() {
    local c
    for c in "$@"; do command -v "$c" > /dev/null || die "commande '$c' introuvable"; done
}

# Vérifie que les modules Python sont importables dans l'env courant.
require_python_modules() {
    require_cmd python
    local missing
    missing=$(python -c 'import importlib.util, sys; print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' "$@")
    [[ -z "$missing" ]] || die "modules Python manquants : $missing (activer l'env conda du projet)"
}

# Taille distante en octets (vide si le serveur ne la donne pas).
remote_size() {
    curl -sIL --retry 3 "$1" | tr -d '\r' | awk 'tolower($1) == "content-length:" { s = $2 } END { print s }'
}

check_space() {
    local dir=$1 need=$2 avail
    avail=$(df -PB1 "$dir" | awk 'NR == 2 { print $4 }')
    (( avail > need )) || die "espace disque insuffisant dans $dir : $(numfmt --to=iec "$avail") libres, $(numfmt --to=iec "$need") nécessaires"
}

# fetch <url> <fichier> : téléchargement reprenable, vérifie l'espace libre et la taille finale.
fetch() {
    local url=$1 dest=$2 size have=0
    size=$(remote_size "$url")
    [[ -f "$dest" ]] && have=$(stat -c %s "$dest")
    if [[ -n "$size" ]]; then
        if (( have == size )); then
            log "déjà téléchargé : $dest"
            return 0
        fi
        if (( have > size )); then
            rm -f "$dest"
            have=0
        fi
        check_space "$(dirname "$dest")" $(( size - have ))
    fi
    log "téléchargement : $url"
    curl -fL --retry 5 --retry-delay 10 -C - -o "$dest" "$url"
    if [[ -n "$size" ]]; then
        have=$(stat -c %s "$dest")
        (( have == size )) || die "taille inattendue pour $dest ($have octets au lieu de $size)"
    fi
}
