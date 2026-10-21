#!/usr/bin/env bash
# Campagne complète EVIMO2 (T11) en une commande : entraînements, tables, agrégation, et en
# option la validation cross-dataset.
#
#   phase 1  MS      v4_evimo2, v4_evimo2_all, v4large_evimo2            (+ couplés scratch,
#                    m{2,3}_evimo2_scratch, qui ne dépendent de rien)
#   phase 2  couplés m{2,3}_evimo2, m{2,3}_evimo2_large, m{2,3}_evimo2_all (MS de la même seed)
#   phase 3  tables  evimo2 × sources {evimo2, evimo2_scratch, evimo2_large}, evimo2_all
#                    (+ rpg, mvsec, eds, vector, tumvie avec --cross)
#   phase 4  agrégation des seeds (tools/aggregate_seeds.py), résumé dans logs/campaign/<id>/
#
# Usage :
#   launch/campaign_evimo2.sh --launcher local --gpus 0,1,2,3 [--seeds 1-5] [--cross]
#   launch/campaign_evimo2.sh --launcher slurm [--seeds 1-5] [--cross]
#   options : --phases 1,2,3,4 (défaut : toutes)  --resume <id>  --dry-run  --sbatch "<opts>"
#
#   --launcher local  une file de tâches sur les GPU listés, phase par phase (à lancer dans
#                     un tmux : `tmux new -s campagne 'launch/campaign_evimo2.sh …'`)
#   --launcher slurm  soumet tout d'un coup ; chaque seed d'une phase attend la même seed de
#                     la phase précédente (--dependency=aftercorr)
#   --resume <id>     relance une campagne : les runs DONE sont sautés, les runs interrompus
#                     reprennent depuis leur dernier checkpoint. En slurm, une seed dont un
#                     run échoue annule ses runs dépendants (--kill-on-invalid-dep=yes) :
#                     corriger, puis relancer avec --resume.
#   --dry-run         affiche le plan et vérifie que toutes les configs se résolvent
#
# Tous les runs d'une campagne partagent un même launch_id (= <id>), donc les dossiers de
# sortie sont prévisibles : runs/<exp>/<id>/seed<N>/, evals_out/<dataset>/<sources>/<id>/seed<N>/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/launch/_seeds.sh"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERREUR : $*" >&2; exit 1; }

LAUNCHER="" SEEDS_SPEC="1-5" GPUS="0" PHASES="1,2,3,4" CROSS=0 DRY_RUN=0 RESUME_ID="" EXTRA_SBATCH=""
while (( $# )); do
    case "$1" in
        --launcher) LAUNCHER="${2:?}"; shift ;;
        --seeds) SEEDS_SPEC="${2:?}"; shift ;;
        --gpus) GPUS="${2:?}"; shift ;;
        --phases) PHASES="${2:?}"; shift ;;
        --cross) CROSS=1 ;;
        --resume) RESUME_ID="${2:?}"; shift ;;
        --sbatch) EXTRA_SBATCH="${2:?}"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) die "option inconnue : $1" ;;
    esac
    shift
done
[[ "$LAUNCHER" == local || "$LAUNCHER" == slurm ]] || die "--launcher local|slurm requis"
mapfile -t SEEDS < <(expand_seeds "$SEEDS_SPEC")
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
phase_on() { [[ ",$PHASES," == *",$1,"* ]]; }

CAMPAIGN_ID="${RESUME_ID:-$(date +%Y-%m-%d_%H%M)}"
CLOG="${MADEVO_LOGS:-$REPO_ROOT/logs}/campaign/$CAMPAIGN_ID"
cd "$REPO_ROOT"

# ------------------------------------------------------------------- le plan
# Une tâche = "job|config|surcharges séparées par des espaces" ; chaque tâche tourne pour
# toutes les seeds.
PHASE1=(
    "train_ms|configs/ms/v4_evimo2.yaml|"
    "train_ms|configs/ms/v4_evimo2_all.yaml|"
    "train_ms|configs/ms/v4large_evimo2.yaml|"
    "train_coupled|configs/coupled/m2_evimo2_scratch.yaml|"
    "train_coupled|configs/coupled/m3_evimo2_scratch.yaml|"
)
PHASE2=(
    "train_coupled|configs/coupled/m2_evimo2.yaml|"
    "train_coupled|configs/coupled/m3_evimo2.yaml|"
    "train_coupled|configs/coupled/m2_evimo2_large.yaml|"
    "train_coupled|configs/coupled/m3_evimo2_large.yaml|"
    "train_coupled|configs/coupled/m2_evimo2_all.yaml|"
    "train_coupled|configs/coupled/m3_evimo2_all.yaml|"
)
PHASE3=(
    "eval_table|configs/eval/evimo2.yaml|"
    "eval_table|configs/eval/evimo2.yaml|sources=evimo2_scratch exp=eval_evimo2_scratch"
    "eval_table|configs/eval/evimo2.yaml|sources=evimo2_large exp=eval_evimo2_large"
    "eval_table|configs/eval/evimo2_all.yaml|"
)
if (( CROSS )); then
    for ds in rpg mvsec eds vector tumvie; do PHASE3+=("eval_table|configs/eval/$ds.yaml|"); done
fi
# dossiers d'éval à agréger (phase 4) : <dataset>/<sources>
AGGREGATE=(evimo2/evimo2 evimo2/evimo2_scratch evimo2/evimo2_large evimo2/evimo2_all)
if (( CROSS )); then AGGREGATE+=(rpg/evimo2 mvsec/evimo2 eds/evimo2 vector/evimo2 tumvie/evimo2); fi

task_args() {  # task_args "<tâche>" → job, config, --set …
    local job cfg sets s
    IFS='|' read -r job cfg sets <<< "$1"
    TASK_JOB="jobs/$job.sh"; TASK_CFG="$cfg"; TASK_SETS=()
    for s in $sets; do TASK_SETS+=(--set "$s"); done
}
task_name() {  # nom lisible et unique pour les journaux
    local job cfg sets
    IFS='|' read -r job cfg sets <<< "$1"
    echo "$(basename "${cfg%.yaml}")${sets:+_$(tr ' =' '_-' <<< "$sets")}"
}

print_plan() {
    local p name t
    echo "campagne   : $CAMPAIGN_ID ($LAUNCHER, seeds ${SEEDS[*]}$( (( CROSS )) && echo ", cross-dataset"))"
    echo "journaux   : $CLOG"
    for p in 1 2 3; do
        phase_on "$p" || continue
        name="PHASE$p[@]"
        echo "phase $p :"
        for t in "${!name}"; do task_args "$t"; echo "  $TASK_JOB $TASK_CFG ${TASK_SETS[*]}"; done
    done
    if phase_on 4; then echo "phase 4 : tools/aggregate_seeds.py sur ${AGGREGATE[*]}"; fi
}

check_configs() {  # chaque config de la campagne doit se résoudre (surcharges comprises)
    local p name t
    for p in 1 2 3; do
        name="PHASE$p[@]"
        for t in "${!name}"; do
            task_args "$t"
            python tools/cfg.py resolve "$TASK_CFG" "${TASK_SETS[@]}" > /dev/null \
                || die "config non résolue : $TASK_CFG ${TASK_SETS[*]}"
        done
    done
    log "les $(( ${#PHASE1[@]} + ${#PHASE2[@]} + ${#PHASE3[@]} )) configs du plan se résolvent"
}

print_plan
check_configs
if (( DRY_RUN )); then
    echo
    echo "dry-run : rien n'est lancé. Détail d'une commande : launch/tmux.sh --dry-run --seeds 1 <job> <config>"
    exit 0
fi
mkdir -p "$CLOG"

# --------------------------------------------------------------- local : file GPU
# Une tâche par GPU libre ; chaque code de sortie est récolté une seule fois (STATUS).
# Pas de RESUME=1 : avec le même launch_id, un run DONE est sauté et un run interrompu
# reprend de lui-même depuis son dernier checkpoint (RESUME=1 relancerait les runs DONE).
declare -A GPU_PID=() STATUS=() LABEL=()
FAILED=()

reap() {  # reap <pid> : récolte le code de sortie d'une tâche terminée
    local st=0
    wait "$1" || st=$?
    STATUS[$1]=$st
    (( st == 0 )) || { FAILED+=("${LABEL[$1]}"); log "ÉCHEC (code $st) : ${LABEL[$1]}"; }
}

wait_free_gpu() {  # FREE_GPU ← un GPU sans tâche en cours
    local g pid
    while true; do
        for g in "${GPU_LIST[@]}"; do
            pid="${GPU_PID[$g]:-}"
            if [[ -z "$pid" ]]; then FREE_GPU=$g; return; fi
            if ! kill -0 "$pid" 2> /dev/null; then
                reap "$pid"; unset "GPU_PID[$g]"; FREE_GPU=$g; return
            fi
        done
        sleep 10
    done
}

run_phase_local() {  # run_phase_local <n> : lance toutes les tâches de la phase puis attend
    local name="PHASE$1[@]" t seed out g
    for t in "${!name}"; do
        task_args "$t"
        for seed in "${SEEDS[@]}"; do
            wait_free_gpu
            out="$CLOG/phase$1_$(task_name "$t")_seed$seed.log"
            log "phase $1 · gpu $FREE_GPU · seed $seed · $TASK_CFG ${TASK_SETS[*]}"
            CUDA_VISIBLE_DEVICES=$FREE_GPU SEED=$seed LAUNCH_ID=$CAMPAIGN_ID \
                bash "$TASK_JOB" "$TASK_CFG" "${TASK_SETS[@]}" > "$out" 2>&1 &
            GPU_PID[$FREE_GPU]=$!
            LABEL[$!]="phase $1 · seed $seed · $TASK_CFG ${TASK_SETS[*]} → $out"
        done
    done
    for g in "${!GPU_PID[@]}"; do reap "${GPU_PID[$g]}"; done
    GPU_PID=()
    log "phase $1 terminée"
}

# ------------------------------------------------------- slurm : dépendances
declare -A PHASE_JOBS=()   # n → ids de job array de la phase, séparés par ':'

submit_phase_slurm() {  # submit_phase_slurm <n> <profil>
    local n=$1 profile=$2 name="PHASE$1[@]" t dep="" prev=$(( $1 - 1 )) out id ids=""
    [[ -n "${PHASE_JOBS[$prev]:-}" ]] && dep="--dependency=aftercorr:${PHASE_JOBS[$prev]} --kill-on-invalid-dep=yes"
    for t in "${!name}"; do
        task_args "$t"
        out=$(LAUNCH_ID=$CAMPAIGN_ID launch/slurm.sh --profile "$profile" --seeds "$SEEDS_SPEC" \
                  --sbatch "--parsable $dep $EXTRA_SBATCH" "$TASK_JOB" "$TASK_CFG" "${TASK_SETS[@]}")
        id=$(tail -n 1 <<< "$out" | cut -d';' -f1)
        [[ "$id" =~ ^[0-9]+$ ]] || die "soumission échouée pour $TASK_CFG : $out"
        log "phase $n · job $id · $TASK_CFG ${TASK_SETS[*]}"
        ids+="${ids:+:}$id"
    done
    PHASE_JOBS[$n]=$ids
}

# --------------------------------------------------------------------- lancement
if [[ "$LAUNCHER" == local ]]; then
    for p in 1 2 3; do
        if phase_on "$p"; then run_phase_local "$p"; fi
    done
else
    if phase_on 1; then submit_phase_slurm 1 gpu_train; fi
    if phase_on 2; then submit_phase_slurm 2 gpu_train; fi
    if phase_on 3; then submit_phase_slurm 3 gpu_eval; fi
    log "soumis. Suivi : squeue -u \$USER ; puis, une fois tout terminé :"
    log "  launch/campaign_evimo2.sh --launcher local --resume $CAMPAIGN_ID --phases 4"
    PHASES=""   # l'agrégation attend la fin des jobs : à relancer avec --phases 4
fi

if phase_on 4; then
    SUMMARY="$CLOG/summary.md"
    { echo "# Campagne $CAMPAIGN_ID"; echo; } > "$SUMMARY"
    for agg in "${AGGREGATE[@]}"; do
        dir="${MADEVO_EVALS:-$REPO_ROOT/evals_out}/$agg/$CAMPAIGN_ID"
        if compgen -G "$dir/seed*/central_table.csv" > /dev/null; then
            python tools/aggregate_seeds.py "$dir" && { echo "## $agg"; echo; cat "$dir/summary.md"; echo; } >> "$SUMMARY"
        else
            log "rien à agréger dans $dir"
        fi
    done
    log "résumé : $SUMMARY"
fi

if (( ${#FAILED[@]} )); then
    log "${#FAILED[@]} tâche(s) en échec :"; printf '  %s\n' "${FAILED[@]}"
    log "relancer : launch/campaign_evimo2.sh --launcher $LAUNCHER --resume $CAMPAIGN_ID …"
    exit 1
fi
log "campagne $CAMPAIGN_ID terminée"
