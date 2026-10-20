#!/usr/bin/env bash
# Soumet un job via sbatch, une tâche de job array par seed.
#
# Usage :
#   launch/slurm.sh --profile <profil> [--seeds 1-5] [--sbatch "<options en plus>"] \
#                   [--resume <launch_id>] [--dry-run] \
#                   <jobs/xxx.sh> <config.yaml> [--set clé=valeur ...]
#
#   --profile  fichier de launch/profiles/ (gpu_train, gpu_eval, cpu)
#   --seeds    1-5, 1,3,5 ou 2 (défaut : 1) — devient --array
#   --sbatch   options sbatch supplémentaires, qui écrasent celles du profil
#   --dry-run  affiche le script sbatch généré et la commande, sans rien soumettre
#
# Le job lancé est le MÊME script bash que sous tmux : aucun en-tête #SBATCH dans jobs/.
# Les options viennent du profil, et _slurm_entry.sh traduit SLURM_ARRAY_TASK_ID en SEED.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/launch/_seeds.sh"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die()   { printf 'ERREUR : %s\n' "$*" >&2; exit 1; }

PROFILE=""
SEEDS_SPEC=1
EXTRA_SBATCH=""
LAUNCH_ID=""
RESUME=0
DRY_RUN=0
JOB=""
JOB_ARGS=()

# Tout ce qui suit le job (config, --set …) lui revient, options comprises.
while (( $# )); do
    if [[ -n "$JOB" ]]; then JOB_ARGS+=("$1"); shift; continue; fi
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --profile)  PROFILE="${2:?}"; shift 2 ;;
        --seeds)    SEEDS_SPEC="${2:?}"; shift 2 ;;
        --sbatch)   EXTRA_SBATCH="${2:?}"; shift 2 ;;
        --resume)   LAUNCH_ID="${2:?}"; RESUME=1; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -*)         die "option inconnue : $1" ;;
        *)          JOB="$1"; shift ;;
    esac
done

[[ -n "$PROFILE" ]] || { usage; die "--profile manquant"; }
PROFILE_FILE="$REPO_ROOT/launch/profiles/$PROFILE.env"
[[ -f "$PROFILE_FILE" ]] || die "profil introuvable : $PROFILE_FILE (disponibles : $(cd "$REPO_ROOT/launch/profiles" && ls *.env | sed 's/\.env//' | tr '\n' ' '))"
[[ -n "$JOB" ]] || { usage; die "job manquant"; }
(( ${#JOB_ARGS[@]} )) || { usage; die "config manquante"; }

# shellcheck disable=SC1090
source "$PROFILE_FILE"
: "${SBATCH_OPTS:?le profil doit définir SBATCH_OPTS}"

mapfile -t SEEDS < <(expand_seeds "$SEEDS_SPEC")
ARRAY=$(slurm_array "${SEEDS[@]}")
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y-%m-%d_%H%M)}"

CONFIG="${JOB_ARGS[0]}"
EXP=$(cd "$REPO_ROOT" && python tools/cfg.py get "$CONFIG" exp --default "$(basename "${CONFIG%.yaml}")")
LOG_DIR="${MADEVO_LOGS:-$REPO_ROOT/logs}/slurm"

CMD=(sbatch --job-name "$EXP" --array "$ARRAY"
     --output "$LOG_DIR/${EXP}_${LAUNCH_ID}_%a.out"
     $SBATCH_OPTS $EXTRA_SBATCH
     "$REPO_ROOT/launch/_slurm_entry.sh" "$JOB" "${JOB_ARGS[@]}")

echo "profil    : $PROFILE ($SBATCH_OPTS${EXTRA_SBATCH:+ + $EXTRA_SBATCH})"
echo "job       : $JOB ${JOB_ARGS[*]}"
echo "seeds     : ${SEEDS[*]} (--array $ARRAY)"
echo "launch_id : $LAUNCH_ID"
echo "logs      : $LOG_DIR"
echo

if (( DRY_RUN )); then
    echo "--- dry-run : rien n'est soumis ---"
    printf 'commande : '; printf '%q ' "${CMD[@]}"; echo
    echo
    echo "Les jobs eux-mêmes se résolvent ainsi (seed par seed) :"
    for seed in "${SEEDS[@]}"; do
        echo "================ seed $seed ================"
        ( cd "$REPO_ROOT" && SEED="$seed" LAUNCH_ID="$LAUNCH_ID" RESUME="$RESUME" DRY_RUN=1 \
            bash "$JOB" "${JOB_ARGS[@]}" )
    done
    exit 0
fi

command -v sbatch > /dev/null || die "sbatch introuvable (pas sur un cluster Slurm ?)"
mkdir -p "$LOG_DIR"
MADEVO_LAUNCH_ID="$LAUNCH_ID" MADEVO_RESUME="$RESUME" "${CMD[@]}"
