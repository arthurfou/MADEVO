#!/usr/bin/env bash
# Lance un job dans une session tmux, une seed à la fois, réparties sur les GPU donnés.
#
# Usage :
#   launch/tmux.sh [--seeds 1-5] [--gpus 0,1] [--dry-run] [--resume <launch_id>] \
#                  <jobs/xxx.sh> <config.yaml> [--set clé=valeur ...]
#
#   --seeds   1-5, 1,3,5 ou 2 (défaut : 1)
#   --gpus    un GPU  => les seeds s'enchaînent séquentiellement dessus
#             plusieurs => une fenêtre tmux par GPU, seeds réparties en round-robin
#             (défaut : 0 ; sans GPU, laisser tel quel, CUDA_VISIBLE_DEVICES est ignoré
#              par les jobs CPU)
#   --resume  réutilise ce LAUNCH_ID au lieu d'en créer un (ex. relancer la seed 3)
#   --dry-run n'ouvre aucune session : résout tout et affiche, seed par seed
#
# Toutes les seeds d'un lancement partagent le LAUNCH_ID, donc le même dossier de date.
# Suivi : tmux attach -t <session>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/launch/_seeds.sh"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die()   { printf 'ERREUR : %s\n' "$*" >&2; exit 1; }

SEEDS_SPEC=1
GPUS=0
DRY_RUN=0
LAUNCH_ID=""
RESUME=0
JOB=""
JOB_ARGS=()

while (( $# )); do
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --seeds)    SEEDS_SPEC="${2:?}"; shift 2 ;;
        --gpus)     GPUS="${2:?}"; shift 2 ;;
        --resume)   LAUNCH_ID="${2:?}"; RESUME=1; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -*)         die "option inconnue : $1" ;;
        *)          if [[ -z "$JOB" ]]; then JOB="$1"; else JOB_ARGS+=("$1"); fi; shift ;;
    esac
done

[[ -n "$JOB" ]] || { usage; die "job manquant"; }
[[ -f "$REPO_ROOT/$JOB" || -f "$JOB" ]] || die "job introuvable : $JOB"
(( ${#JOB_ARGS[@]} )) || { usage; die "config manquante"; }

mapfile -t SEEDS < <(expand_seeds "$SEEDS_SPEC")
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y-%m-%d_%H%M)}"

CONFIG="${JOB_ARGS[0]}"
EXP=$(cd "$REPO_ROOT" && python tools/cfg.py get "$CONFIG" exp --default "$(basename "${CONFIG%.yaml}")")
SESSION="${EXP}_${LAUNCH_ID}"

echo "job       : $JOB ${JOB_ARGS[*]}"
echo "seeds     : ${SEEDS[*]}"
echo "gpus      : ${GPU_LIST[*]}"
echo "launch_id : $LAUNCH_ID"
echo "session   : $SESSION"
echo

if (( DRY_RUN )); then
    for seed in "${SEEDS[@]}"; do
        echo "================ seed $seed ================"
        ( cd "$REPO_ROOT" && SEED="$seed" LAUNCH_ID="$LAUNCH_ID" RESUME="$RESUME" DRY_RUN=1 \
            bash "$JOB" "${JOB_ARGS[@]}" )
    done
    exit 0
fi

command -v tmux > /dev/null || die "tmux introuvable"
tmux has-session -t "$SESSION" 2> /dev/null && die "session tmux '$SESSION' déjà ouverte"

# Une fenêtre par GPU ; chaque fenêtre enchaîne ses seeds séquentiellement.
for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    cmd=""
    for j in "${!SEEDS[@]}"; do
        (( j % ${#GPU_LIST[@]} == i )) || continue
        cmd+="CUDA_VISIBLE_DEVICES=$gpu SEED=${SEEDS[$j]} LAUNCH_ID=$LAUNCH_ID RESUME=$RESUME"
        cmd+=" bash $JOB $(printf '%q ' "${JOB_ARGS[@]}"); "
    done
    [[ -n "$cmd" ]] || continue
    cmd+="echo '--- fenêtre terminée (gpu $gpu) ---'; exec bash"
    if tmux has-session -t "$SESSION" 2> /dev/null; then
        tmux new-window -t "$SESSION" -n "gpu$gpu" -c "$REPO_ROOT" "$cmd"
    else
        tmux new-session -d -s "$SESSION" -n "gpu$gpu" -c "$REPO_ROOT" "$cmd"
    fi
done

echo "lancé. Suivi : tmux attach -t $SESSION"
