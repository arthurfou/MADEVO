#!/usr/bin/env bash
# Smoke test GPU de toute la chaîne, en ~15 min sur une scène EVIMO2 : à lancer une fois sur
# chaque nouveau serveur avant la campagne (T10). Aucune valeur scientifique.
#
#   1. extensions CUDA importées et exécutées, GPU visible
#   2. MS : 2 epochs (configs/ms/smoke.yaml)
#   3. couplés M2 et M3 : 200 steps chacun, MS initialisé depuis l'étape 2, dégel de DEVO à 100
#   4. table vanilla / oracle / M1 / M2 / M3 sur 2 scènes (splits/evimo2/evimo2_smoke.txt)
#
# Usage : jobs/smoke_gpu.sh [--steps N] [--only imports|ms|coupled|eval]
#
# Sorties sous logs/smoke_gpu/<date>/ (runs/ et evals_out/ du smoke, journal), à part des
# vraies sorties : rien n'est écrit dans runs/ ni evals_out/ du repo, sauf si paths.env
# impose MADEVO_RUNS / MADEVO_EVALS (les exp du smoke sont alors préfixées smoke_ / ms_smoke).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STEPS=200 ONLY=""
while (( $# )); do
    case "$1" in
        --steps) STEPS="${2:?}"; shift ;;
        --only) ONLY="${2:?}"; shift ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "option inconnue : $1" ;;
    esac
    shift
done
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

job_setup_env
cd "$REPO_ROOT"
STAMP=$(date +%Y-%m-%d_%H%M)
SMOKE_DIR="${MADEVO_LOGS:-$REPO_ROOT/logs}/smoke_gpu/$STAMP"
mkdir -p "$SMOKE_DIR"
export MADEVO_RUNS="${MADEVO_RUNS:-$SMOKE_DIR/runs}" MADEVO_EVALS="${MADEVO_EVALS:-$SMOKE_DIR/evals_out}"
export WANDB_MODE="${WANDB_MODE:-offline}" SEED=1 LAUNCH_ID="$STAMP"
exec > >(tee -a "$SMOKE_DIR/smoke.log") 2>&1
log "smoke GPU → $SMOKE_DIR"

if want imports; then
    log "[1/4] extensions CUDA"
    python - <<'PY'
import torch
assert torch.cuda.is_available(), "aucun GPU visible par torch"
import cuda_ba, cuda_corr, lietorch_backends  # noqa: F401
from devo.lietorch import SE3
x = SE3.Random(4, device="cuda")
err = (x * x.inv()).log().abs().max().item()
assert err < 1e-4, f"lietorch : x · x⁻¹ ≠ identité ({err})"
import devo.altcorr as altcorr  # noqa: F401
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)} — extensions OK")
PY
fi

if want ms; then
    log "[2/4] MS (configs/ms/smoke.yaml)"
    jobs/train_ms.sh configs/ms/smoke.yaml
fi

if want coupled; then
    for m in m2 m3; do
        log "[3/4] couplé $m, $STEPS steps"
        jobs/train_coupled.sh "configs/coupled/${m}_evimo2.yaml" \
            --set "exp=smoke_$m" --set ms.arch=configs/ms/smoke.yaml \
            --set ms.init=runs/ms_smoke/@latest \
            --set "train.steps=$STEPS" --set "train.freeze_devo_steps=$(( STEPS / 2 ))" \
            --set "train.save_every=$(( STEPS / 2 ))" --set log.images_every=0
    done
fi

if want eval; then
    log "[4/4] table vanilla / oracle / M1 / M2 / M3 sur 2 scènes"
    jobs/eval_table.sh configs/eval/evimo2.yaml --set sources=smoke --set exp=smoke_eval \
        --set dataset.split=splits/evimo2/evimo2_smoke.txt
    table=$(find "$MADEVO_EVALS" -name central_table.md -path "*smoke*" | sort | tail -n 1)
    [[ -n "$table" ]] || die "pas de central_table.md produite"
    cat "$table"
    grep -qi "nan" "$table" && die "ATE NaN dans la table du smoke : $table"
fi

log "smoke GPU terminé sans erreur — journal : $SMOKE_DIR/smoke.log"
