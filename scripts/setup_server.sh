#!/usr/bin/env bash
# Met en place MADEVO sur un serveur GPU, de zéro jusqu'au feu vert pour la campagne (T10).
#
# Étapes (chacune idempotente, relançable seule avec --steps) :
#   conda     Miniforge dans ~/miniforge3 si ni conda ni mamba ne sont disponibles
#   env       env conda `madevo` depuis environment.yml (création, ou mise à jour avec --update-env)
#   build     compilation des extensions CUDA (cuda_corr, cuda_ba, lietorch_backends) pour les GPU
#             présents, puis vérification des imports
#   paths     paths.env créé depuis paths.env.example s'il n'existe pas (wandb offline par défaut)
#   data      poids DEVO + EVIMO2 (+ --sfm, + --cross pour les 5 datasets d'éval cross-dataset)
#   clean     suppression des caches de voxels v1 et d'un evimo1_format/ antérieur au bug 6
#   tests     pytest (CPU)
#   smoke     jobs/smoke_gpu.sh : MS, M2, M3 et une table sur 2 scènes
#   nonreg    DEVO vanilla sur les 8 scènes EVIMO2 (+ RPG avec --cross) contre configs/nonregression.yaml
#
# Usage : bash scripts/setup_server.sh [--steps a,b,...] [--sfm] [--cross] [--update-env]
#                                      [--arch "8.0;8.6"] [--yes]
#   --steps       sous-ensemble des étapes ci-dessus (défaut : toutes, dans l'ordre)
#   --cross       télécharge aussi RPG, MVSEC, EDS, VECtor, TUM-VIE (~110 Go) et contrôle RPG
#   --sfm         télécharge aussi la catégorie SfM d'EVIMO2 (~10,4 Go)
#   --arch        TORCH_CUDA_ARCH_LIST imposée (défaut : déduite de nvidia-smi)
#   --yes         ne demande pas confirmation avant l'installation de Miniforge
#
# Journal complet : logs/setup_server_<date>.log. Données hors du repo : définir
# MADEVO_DATASETS / MADEVO_PRETRAINED (et MADEVO_RUNS, MADEVO_EVALS) avant de lancer, ou
# dans paths.env une fois l'étape `paths` passée.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL_STEPS=(conda env build paths data clean tests smoke nonreg)
STEPS="" CROSS=0 SFM=0 UPDATE_ENV=0 ARCH="" YES=0
ENV_NAME="${MADEVO_CONDA_ENV:-madevo}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERREUR : $*" >&2; exit 1; }

while (( $# )); do
    case "$1" in
        --steps) STEPS="${2:?}"; shift ;;
        --cross) CROSS=1 ;;
        --sfm) SFM=1 ;;
        --update-env) UPDATE_ENV=1 ;;
        --arch) ARCH="${2:?}"; shift ;;
        --yes) YES=1 ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) die "option inconnue : $1" ;;
    esac
    shift
done
if [[ -n "$STEPS" ]]; then IFS=',' read -r -a SELECTED <<< "$STEPS"; else SELECTED=("${ALL_STEPS[@]}"); fi
for s in "${SELECTED[@]}"; do
    [[ " ${ALL_STEPS[*]} " == *" $s "* ]] || die "étape inconnue : $s (attendu : ${ALL_STEPS[*]})"
done
want() { [[ " ${SELECTED[*]} " == *" $1 "* ]]; }

mkdir -p "$REPO_ROOT/logs"
LOG="$REPO_ROOT/logs/setup_server_$(date +%Y-%m-%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
log "MADEVO — mise en place serveur ($(hostname)), étapes : ${SELECTED[*]}"
log "journal : $LOG"
cd "$REPO_ROOT"

activate_conda() {
    local base
    for base in "${CONDA_PREFIX_1:-}" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "$base/etc/profile.d/conda.sh"
            [[ -f "$base/etc/profile.d/mamba.sh" ]] && source "$base/etc/profile.d/mamba.sh"
            return 0
        fi
    done
    command -v conda > /dev/null && eval "$(conda shell.bash hook)" && return 0
    return 1
}

# ------------------------------------------------------------------------- conda
if want conda; then
    if activate_conda; then
        log "[conda] déjà disponible : $(conda --version)"
    else
        url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        if (( ! YES )); then
            read -r -p "Installer Miniforge dans $HOME/miniforge3 (sans modifier ~/.bashrc) ? [o/N] " ok
            [[ "$ok" == [oOyY]* ]] || die "conda requis : installer Miniforge ou relancer avec --yes"
        fi
        log "[conda] installation de Miniforge ($url)"
        curl -fsSL -o /tmp/miniforge.sh "$url"
        bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
        rm -f /tmp/miniforge.sh
        activate_conda || die "Miniforge installé mais introuvable"
        log "[conda] pour l'avoir dans vos shells : $HOME/miniforge3/bin/conda init bash"
    fi
fi
activate_conda || die "conda introuvable (étape conda)"
SOLVER=conda; command -v mamba > /dev/null && SOLVER=mamba

# --------------------------------------------------------------------------- env
if want env; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        if (( UPDATE_ENV )); then
            log "[env] mise à jour de $ENV_NAME depuis environment.yml"
            $SOLVER env update -n "$ENV_NAME" -f environment.yml --prune
        else
            log "[env] $ENV_NAME existe déjà (--update-env pour le resynchroniser)"
        fi
    else
        log "[env] création de $ENV_NAME depuis environment.yml (torch 2.4.1 + CUDA 11.8)"
        $SOLVER env create -n "$ENV_NAME" -f environment.yml
    fi
fi
# Les scripts d'activation de conda (binutils, gcc) lisent des variables non définies.
set +u
conda activate "$ENV_NAME" || die "env $ENV_NAME introuvable (étape env)"
set -u
# evo applique son plot_backend (Qt5Agg par défaut) à l'import : sans Qt, utils/ ne s'importe pas.
if [[ "$(python -c "from evo.tools.settings import SETTINGS; print(SETTINGS.plot_backend)" 2> /dev/null || true)" == Qt* ]]; then
    log "[env] evo : plot_backend → Agg (serveur sans affichage)"
    evo_config set plot_backend Agg > /dev/null
fi

# ------------------------------------------------------------------------- build
if want build; then
    command -v nvidia-smi > /dev/null || die "[build] nvidia-smi introuvable : pas de GPU sur cette machine ?"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[build] GPU : /'
    if [[ -z "$ARCH" ]]; then
        ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | sort -u | paste -sd ';')
        [[ -n "$ARCH" ]] || die "[build] capacité de calcul introuvable : passer --arch \"8.0;8.6\""
    fi
    log "[build] TORCH_CUDA_ARCH_LIST=$ARCH"
    if ! python -c "import os, sys; sys.exit(0 if os.environ.get('EIGEN_INCLUDE_DIR') or os.path.isdir(os.path.join(sys.prefix, 'include', 'eigen3')) else 1)"; then
        log "[build] Eigen absent de l'env : téléchargement"
        bash scripts/download/download_eigen.sh
    fi
    TORCH_CUDA_ARCH_LIST="$ARCH" pip install -e . --no-build-isolation 2>&1 | tail -n 5
    python -c "import torch, cuda_ba, cuda_corr, lietorch_backends; \
assert torch.cuda.is_available(), 'torch ne voit pas le GPU'; \
print(f'[build] torch {torch.__version__} / CUDA {torch.version.cuda} / {torch.cuda.device_count()} GPU — extensions OK')"
fi

# ------------------------------------------------------------------------- paths
if want paths; then
    if [[ -f paths.env ]]; then
        log "[paths] paths.env existe déjà, laissé tel quel"
    else
        sed 's/^# export WANDB_MODE=offline/export WANDB_MODE=offline/' paths.env.example > paths.env
        log "[paths] paths.env créé (WANDB_MODE=offline) : à éditer pour placer données et sorties"
        log "        sur le disque de données du serveur (MADEVO_DATASETS, MADEVO_RUNS, …)"
    fi
fi
[[ -f paths.env ]] && source paths.env

# -------------------------------------------------------------------------- data
if want data; then
    targets="devo,evimo2"
    (( CROSS )) && targets="$targets,rpg,mvsec,eds,vector,tumvie"
    args=(--only "$targets" --keep-going)
    (( SFM )) && args+=(--sfm)
    log "[data] scripts/download/download_all.sh ${args[*]}"
    bash scripts/download/download_all.sh "${args[@]}" || die "[data] échecs de téléchargement : relancer --steps data"
fi

# ------------------------------------------------------------------------- clean
if want clean; then
    DATA="${MADEVO_DATASETS:-$REPO_ROOT/datasets}"
    mapfile -t v1 < <(find "$DATA" -name '*.voxels_bins*' ! -name '*.v2.npy' ! -name '*.v2.npy.tmp' 2> /dev/null)
    if (( ${#v1[@]} )); then
        log "[clean] ${#v1[@]} cache(s) de voxels v1 supprimé(s) (timestamps décalés, problème 5)"
        printf '        %s\n' "${v1[@]}"
        rm -f "${v1[@]}"
    else
        log "[clean] aucun cache de voxels v1"
    fi
    E1="$DATA/evimo2/evimo1_format"
    if [[ -d "$E1" && ! -f "$E1/.gt_aligned" ]]; then
        log "[clean] $E1 antérieur au correctif du bug 6 : supprimé et régénéré"
        rm -rf "$E1"
        bash scripts/download/download_evimo2.sh
    fi
    [[ -f "$E1/.gt_aligned" ]] || die "[clean] $E1/.gt_aligned absent : relancer --steps data"
fi

# ------------------------------------------------------------------------- tests
if want tests; then
    log "[tests] pytest"
    python -m pytest -q tests
fi

# ------------------------------------------------------------------------- smoke
if want smoke; then
    log "[smoke] jobs/smoke_gpu.sh"
    jobs/smoke_gpu.sh
fi

# ------------------------------------------------------------------------ nonreg
if want nonreg; then
    # DEVO n'est pas déterministe sur GPU : on compare la moyenne de plusieurs seeds.
    run_nonreg() {  # run_nonreg <dataset>
        local ds=$1 lid seed
        lid="$(date +%Y-%m-%d_%H%M)"
        log "[nonreg] DEVO vanilla sur $ds, seeds ${NONREG_SEEDS:-1 2 3}"
        for seed in ${NONREG_SEEDS:-1 2 3}; do
            SEED=$seed LAUNCH_ID=$lid jobs/eval_table.sh "configs/eval/$ds.yaml" \
                --set 'methods=[vanilla]' --set sources=none --set "exp=nonreg_$ds"
        done
        python tools/check_nonregression.py \
            "${MADEVO_EVALS:-$REPO_ROOT/evals_out}/$ds/none/$lid"/seed* --dataset "$ds"
    }
    run_nonreg evimo2
    (( CROSS )) && run_nonreg rpg
fi

log "mise en place terminée — étapes : ${SELECTED[*]}"
log "suite : bash launch/campaign_evimo2.sh --launcher local --dry-run  (cf. JOBS_AND_RUNS.md)"
