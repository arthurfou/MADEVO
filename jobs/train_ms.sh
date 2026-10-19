#!/usr/bin/env bash
# Entraînement du modèle de motion segmentation seul (jalon M1). Un job = une seed.
#
# Usage : [SEED=1] [LAUNCH_ID=…] jobs/train_ms.sh configs/ms/<exp>.yaml [--set clé=valeur ...] [--dry-run]
#
# Sortie : runs/<exp>/<launch_id>/seed<N>/ (config.yaml, cmd.txt, git.txt, checkpoints/, logs/, DONE)
# Reprise : automatique depuis checkpoints/last.pt si le dossier existe et n'est pas DONE.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

job_parse_args "$@"
job_setup_env
job_prepare ms
job_run
