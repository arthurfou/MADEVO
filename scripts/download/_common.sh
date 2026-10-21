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

# fetch_gdrive <id> <fichier> : téléchargement Google Drive (gdown). Drive ne donne pas la
# taille à l'avance : un marqueur `<fichier>.ok` signale un téléchargement complet.
fetch_gdrive() {
    local id=$1 dest=$2
    if [[ -f "$dest.ok" && -s "$dest" ]]; then
        log "déjà téléchargé : $dest"
        return 0
    fi
    require_python_modules gdown
    mkdir -p "$(dirname "$dest")"
    log "téléchargement Google Drive : $id → $dest"
    python -c 'import sys, gdown
out = gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=False, resume=True)
sys.exit(0 if out else 1)' "$id" "$dest" \
        || die "échec gdown pour $id (quota Drive dépassé ? réessayer plus tard)"
    [[ -s "$dest" ]] || die "fichier vide : $dest"
    touch "$dest.ok"
}

# split_sequences <fichier split> : séquences du split, sans lignes vides ni commentaires.
split_sequences() {
    grep -vE '^\s*(#|$)' "$1" | tr -d '\r'
}

# select_sequences "<liste CSV ou vide>" <fichier split> : la liste demandée, ou tout le split.
select_sequences() {
    if [[ -n "$1" ]]; then tr ',' '\n' <<< "$1"; else split_sequences "$2"; fi
}

# check_outputs <dossier> <fichier>... : échoue si un fichier attendu manque ou est vide.
check_outputs() {
    local dir=$1 f; shift
    for f in "$@"; do
        compgen -G "$dir/$f" > /dev/null || die "prétraitement incomplet : $dir/$f manquant"
    done
}

# evo (évaluation de trajectoires, importé par utils/) applique à l'import son réglage
# `plot_backend`, Qt5Agg par défaut : sans Qt (serveur, WSL), tout import de utils/ échoue.
ensure_headless_evo() {
    command -v evo_config > /dev/null || return 0
    local backend
    backend=$(python -c "from evo.tools.settings import SETTINGS; print(SETTINGS.plot_backend)" 2> /dev/null || true)
    if [[ "$backend" == Qt* ]]; then
        log "evo : plot_backend passé à Agg (~/.evo/settings.json), sans Qt sur cette machine"
        evo_config set plot_backend Agg > /dev/null
    fi
}
