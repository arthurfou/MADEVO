# Sourcé par tous les jobs de jobs/ — à sourcer, pas à exécuter.
#
# Un job = un run = une seed. Les jobs ne savent pas s'ils tournent sous tmux ou Slurm :
# ils lisent SEED et LAUNCH_ID dans l'environnement et n'ont aucun en-tête #SBATCH.
#
# Variables d'entrée :
#   SEED        seed du run (défaut 1)
#   LAUNCH_ID   YYYY-MM-DD_HHMM partagé par toutes les seeds d'un lancement
#               (défaut : maintenant — à ne pas laisser au hasard quand on lance 5 seeds)
#   DRY_RUN     1 = résoudre la config, afficher le dossier et la commande, ne rien lancer
#   RESUME      1 = reprendre même si DONE est présent (posé par launch/*.sh --resume)
#   paths.env   facultatif, à la racine du repo : WANDB_*, MADEVO_RUNS, MADEVO_EVALS…

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${MADEVO_CONDA_ENV:-madevo}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '[%s] ERREUR : %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Affiche l'en-tête de commentaires du script appelant.
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

# job_parse_args "$@" : remplit CONFIG et OVERRIDES[] depuis la ligne de commande.
#   <config.yaml> [--set clé=valeur ...] [--dry-run]
job_parse_args() {
    CONFIG=""
    OVERRIDES=()
    while (( $# )); do
        case "$1" in
            -h|--help)  usage; exit 0 ;;
            --dry-run)  DRY_RUN=1; shift ;;
            --set)      [[ -n "${2:-}" ]] || die "--set attend clé=valeur"
                        OVERRIDES+=(--set "$2"); shift 2 ;;
            --set=*)    OVERRIDES+=(--set "${1#--set=}"); shift ;;
            -*)         die "option inconnue : $1" ;;
            *)          [[ -z "$CONFIG" ]] || die "une seule config attendue (reçu '$CONFIG' puis '$1')"
                        CONFIG="$1"; shift ;;
        esac
    done
    [[ -n "$CONFIG" ]] || { usage; die "config manquante"; }
    [[ -f "$CONFIG" ]] || die "config introuvable : $CONFIG"
}

# Active l'env conda du projet s'il ne l'est pas déjà, et charge paths.env.
job_setup_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
        local hook
        for hook in "$HOME/miniforge3/etc/profile.d/conda.sh" \
                    "$HOME/miniconda3/etc/profile.d/conda.sh" \
                    "$HOME/anaconda3/etc/profile.d/conda.sh" \
                    "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"; do
            [[ -f "$hook" ]] && { source "$hook"; break; }
        done
        command -v conda > /dev/null || die "conda introuvable (voir README §Setup)"
        conda activate "$CONDA_ENV" || die "env conda '$CONDA_ENV' introuvable"
    fi
    # shellcheck disable=SC1091
    [[ -f "$REPO_ROOT/paths.env" ]] && source "$REPO_ROOT/paths.env"
    # wandb `disabled` découple le nom du run du dossier de checkpoints et casse la reprise.
    if [[ "${WANDB_MODE:-}" == "disabled" ]]; then
        log "WANDB_MODE=disabled remplacé par 'offline' (cf. INTEGRITY_CHECK.md)"
        export WANDB_MODE=offline
    fi
}

# job_prepare <kind> : résout la config, calcule le dossier du run, écrit sa provenance.
# Définit RUN_DIR, RESOLVED (config résolue) et ARGV[] (commande à lancer).
# En dry-run, rien n'est écrit sous runs/ : la config résolue va dans un dossier temporaire.
job_prepare() {
    local kind=$1
    SEED="${SEED:-1}"
    LAUNCH_ID="${LAUNCH_ID:-$(date +%Y-%m-%d_%H%M)}"

    cd "$REPO_ROOT" || die "cd $REPO_ROOT"
    RUN_DIR=$(python tools/cfg.py rundir "$CONFIG" "${OVERRIDES[@]}" \
                  --kind "$kind" --launch-id "$LAUNCH_ID" --seed "$SEED") \
        || die "résolution du dossier de run"

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        WORK_DIR=$(mktemp -d)
        trap 'rm -rf "$WORK_DIR"' EXIT
    else
        if [[ -f "$RUN_DIR/DONE" && "${RESUME:-0}" != 1 ]]; then
            log "déjà terminé (DONE) : $RUN_DIR — rien à faire"
            log "(pour reprendre quand même, p. ex. avec plus d'epochs :"
            log " launch/tmux.sh --resume $LAUNCH_ID … --set training.epochs=N)"
            exit 0
        fi
        if [[ -f "$RUN_DIR/DONE" ]]; then
            log "DONE présent mais --resume demandé : reprise de $RUN_DIR"
            rm -f "$RUN_DIR/DONE"
        fi
        WORK_DIR="$RUN_DIR"
        mkdir -p "$RUN_DIR/logs"
    fi

    RESOLVED="$WORK_DIR/config.yaml"
    python tools/cfg.py resolve "$CONFIG" "${OVERRIDES[@]}" --out "$RESOLVED" \
        || die "résolution de la config"

    # La construction de la commande peut échouer légitimement : `ms.init:
    # runs/<exp>/@latest` exige que l'entraînement MS de LA MÊME SEED soit DONE. En dry-run
    # on le signale sans faire échouer la vérification du catalogue (code de sortie 3).
    local argv_err="$WORK_DIR/.argv.err"
    if ! python tools/cfg.py argv "$kind" "$CONFIG" "${OVERRIDES[@]}" \
             --run-dir "$RUN_DIR" --seed "$SEED" --resolved "$RESOLVED" \
             > "$WORK_DIR/.argv" 2> "$argv_err"; then
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            echo "--- dry-run : commande NON résolue ---"
            echo "dossier de run : $RUN_DIR"
            echo "raison         : $(tail -n 1 "$argv_err")"
            echo "(normal si l'entraînement dont ce run dépend n'a pas encore tourné"
            echo " pour cette seed — lancer les étapes dans l'ordre du §8 de JOBS_AND_RUNS.md)"
            exit 3
        fi
        cat "$argv_err" >&2
        die "construction de la commande"
    fi
    mapfile -t ARGV < "$WORK_DIR/.argv"
    rm -f "$WORK_DIR/.argv" "$argv_err"
    (( ${#ARGV[@]} )) || die "commande vide"

    local provenance
    provenance=$(
        printf 'config    : %s\n' "$CONFIG"
        printf 'surcharges: %s\n' "${OVERRIDES[*]:-(aucune)}"
        printf 'seed      : %s\n' "$SEED"
        printf 'launch_id : %s\n' "$LAUNCH_ID"
        printf 'hostname  : %s\n' "$(hostname)"
        printf 'date      : %s\n' "$(date -Is)"
        printf 'gpu       : %s\n' "${CUDA_VISIBLE_DEVICES:-(toutes)}"
        printf 'commande  :\n'
        printf '  %q' "${ARGV[@]}"; printf '\n'
    )
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        printf '%s\n' "$provenance" > "$RUN_DIR/cmd.txt"
        python -c "import sys; sys.path.insert(0, '.'); from utils.run_utils import git_description; \
                   open(sys.argv[1], 'w').write(git_description())" "$RUN_DIR/git.txt"
    fi
}

# write_manifest : provenance exacte des checkpoints d'une éval, dans le dossier du run.
write_manifest() {
    local out="${WORK_DIR:-$RUN_DIR}/manifest.yaml"
    local rc=0
    python tools/cfg.py manifest "$CONFIG" "${OVERRIDES[@]}" \
        --seed "$SEED" --out "$out" || rc=$?
    # Code 3 : les checkpoints de M1/M2/M3 n'existent pas encore pour cette seed. En
    # dry-run c'est le cas normal avant la campagne ; en vrai c'est un ordre de lancement
    # faux (cf. §8 de JOBS_AND_RUNS.md).
    if (( rc == 3 )) && [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "--- dry-run : manifest NON résolu ---"
        echo "dossier de run : $RUN_DIR"
        echo "(normal si les entraînements évalués n'ont pas encore tourné pour cette seed)"
        exit 3
    fi
    (( rc == 0 )) || die "écriture du manifest"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "--- manifest (provenance des checkpoints) ---"
        cat "$out"
    else
        log "manifest : $out"
    fi
}

# job_run : lance la commande (ou l'affiche en --dry-run), pose DONE en cas de succès.
job_run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "--- dry-run : rien n'est lancé ---"
        echo "dossier de run : $RUN_DIR"
        echo "config résolue : $RESOLVED"
        echo "commande       :"
        printf '  %q' "${ARGV[@]}"; echo
        echo "--- config résolue ---"
        cat "$RESOLVED"
        return 0
    fi

    log "run : $RUN_DIR"
    log "cmd : ${ARGV[*]}"
    set +e
    "${ARGV[@]}" 2>&1 | tee -a "$RUN_DIR/logs/stdout.log"
    local status=${PIPESTATUS[0]}
    set -e
    (( status == 0 )) || die "échec (code $status), voir $RUN_DIR/logs/stdout.log"
    date -Is > "$RUN_DIR/DONE"
    log "terminé : $RUN_DIR"
}
