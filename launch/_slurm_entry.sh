#!/usr/bin/env bash
# Point d'entrée côté nœud Slurm : traduit SLURM_ARRAY_TASK_ID en SEED, puis lance le job.
# Appelé par launch/slurm.sh, jamais à la main.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export SEED="${SLURM_ARRAY_TASK_ID:-1}"
export LAUNCH_ID="${MADEVO_LAUNCH_ID:?LAUNCH_ID non transmis par launch/slurm.sh}"
export RESUME="${MADEVO_RESUME:-0}"

JOB="$1"; shift
exec bash "$JOB" "$@"
