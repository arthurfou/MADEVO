#!/usr/bin/env bash
# Lance une chaîne de jobs sur une autre machine (ssh, sans Slurm), détachée de la session,
# avec un arrêt forcé à une échéance (machines prêtées pour une durée limitée).
#
# Usage :
#   launch/remote.sh [--deadline "2026-11-12 08:45"] [--name <nom>] <hôte> <seed> -- \
#       "<job1> <config> [--set …]" ["<job2> …" …]
#
# Les jobs s'enchaînent dans l'ordre sur l'hôte (même seed, même LAUNCH_ID) ; un échec arrête
# la chaîne. Le repo est sur le home partagé : seul /Data est local, il doit contenir l'env
# conda et les données (launch/provision.sh). À l'échéance, `timeout` envoie SIGTERM puis
# SIGKILL 60 s plus tard : la reprise repart du dernier checkpoint (relancer avec le même LAUNCH_ID).
#
# Variables : LAUNCH_ID (défaut : maintenant), MADEVO_DEADLINE (défaut de --deadline).
# Journal : /Data/arthur.fournier/madevo/logs/remote/<nom>.log sur l'hôte, et l'hôte, la GPU
# et la commande dans <nom>.meta.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { printf 'ERREUR : %s\n' "$*" >&2; exit 1; }

DEADLINE="${MADEVO_DEADLINE:-}"
NAME=""
while [[ "${1:-}" == --* && "$1" != "--" ]]; do
    case "$1" in
        --deadline) DEADLINE="${2:?}"; shift 2 ;;
        --name)     NAME="${2:?}"; shift 2 ;;
        *)          die "option inconnue : $1" ;;
    esac
done
HOST="${1:?hôte manquant}"; SEED="${2:?seed manquante}"; shift 2
[[ "${1:-}" == "--" ]] || die "attendu -- avant les jobs"; shift
(( $# )) || die "aucun job"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y-%m-%d_%H%M)}"
NAME="${NAME:-${HOST}_s${SEED}_${LAUNCH_ID}}"

LIMIT=""
DEADLINE_TS=0
if [[ -n "$DEADLINE" ]]; then
    DEADLINE_TS=$(date -d "$DEADLINE" +%s)
    (( DEADLINE_TS - $(date +%s) > 300 )) || die "échéance $DEADLINE trop proche ou passée"
    # recalculé au début de chaque job de la chaîne, sur l'hôte
    LIMIT="timeout --kill-after=60 \$((DEADLINE_TS - \$(date +%s)))"
fi

CHAIN=""
for job in "$@"; do CHAIN+="$LIMIT bash $job && "; done
CHAIN+="echo CHAIN_OK"

LOGDIR=/Data/arthur.fournier/madevo/logs/remote
REMOTE="mkdir -p $LOGDIR && cd $REPO_ROOT && {
  echo \"host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"
  echo \"seed=$SEED launch_id=$LAUNCH_ID deadline=${DEADLINE:-aucune}\"
  printf 'job: %s\n' $(printf '%q ' "$@")
} > $LOGDIR/$NAME.meta || exit 1
nohup setsid env SEED=$SEED LAUNCH_ID=$LAUNCH_ID DEADLINE_TS=$DEADLINE_TS bash -c '
  source /usr/local/Anaconda3-2025.12/etc/profile.d/conda.sh
  $CHAIN' > $LOGDIR/$NAME.log 2>&1 < /dev/null & echo \$!"
pid=$(ssh -o BatchMode=yes "$HOST" "$REMOTE")
echo "$HOST seed $SEED → pid $pid, journal $HOST:$LOGDIR/$NAME.log"
