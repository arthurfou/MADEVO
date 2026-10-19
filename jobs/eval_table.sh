#!/usr/bin/env bash
# Table d'évaluation VO : vanilla / oracle / M1 / M2 / M3 sur un dataset. Un job = une seed.
#
# Usage : [SEED=1] [LAUNCH_ID=…] jobs/eval_table.sh configs/eval/<dataset>.yaml [--set clé=valeur ...] [--dry-run]
#
# Les checkpoints de M1/M2/M3 viennent du préréglage `sources`, pour la seed évaluée
# (seed d'éval = seed d'entraînement). Le manifest écrit dans le dossier dit exactement
# quel fichier a été chargé pour chaque méthode.
# Sortie : evals_out/<dataset>/<sources>/<launch_id>/seed<N>/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

job_parse_args "$@"
job_setup_env
job_prepare eval
write_manifest
job_run
