#!/usr/bin/env bash
# Entraînement couplé MS + DEVO (jalons M2 supervisé et M3 auto-supervisé). Un job = une seed.
#
# Usage : [SEED=1] [LAUNCH_ID=…] jobs/train_coupled.sh configs/coupled/<exp>.yaml [--set clé=valeur ...] [--dry-run]
#
# `ms.init` est résolu vers le checkpoint MS de LA MÊME SEED : l'entraînement MS
# correspondant doit être terminé (DONE) pour cette seed, sauf en variante scratch.
# Sortie : runs/<exp>/<launch_id>/seed<N>/
# Reprise : automatique depuis le dernier checkpoints/coupled_step<K>.pt.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

job_parse_args "$@"
job_setup_env
job_prepare coupled
job_run
