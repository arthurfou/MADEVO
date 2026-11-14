#!/usr/bin/env bash
# Copie l'env conda `madevo`, les données et les poids de cette machine vers le /Data local
# d'autres machines (le home, donc le repo et les extensions compilées, est déjà partagé).
#
# Usage : launch/provision.sh <hôte> [<hôte> …]     (en parallèle, relançable : rsync)
set -euo pipefail
B=/Data/arthur.fournier
for h in "$@"; do
    (
        ssh -o BatchMode=yes "$h" "mkdir -p $B/.conda/envs $B/madevo/{datasets,pretrained,runs,evals_out,logs}"
        rsync -a --delete "$B/.conda/envs/madevo/" "$h:$B/.conda/envs/madevo/"
        # caches de voxels exclus (~55 Go) : chaque machine les reconstruit en parallèle
        rsync -a --exclude "*.voxels_bins*.npy" "$B/madevo/datasets/" "$h:$B/madevo/datasets/"
        rsync -a "$B/madevo/pretrained/" "$h:$B/madevo/pretrained/"
        ssh -o BatchMode=yes "$h" "$B/.conda/envs/madevo/bin/python -c 'import torch, cuda_corr, cuda_ba, lietorch_backends; print(torch.cuda.get_device_name(0))'" \
            > /tmp/provision_$h.ok 2>&1 && echo "$h : OK ($(tail -1 /tmp/provision_$h.ok))" \
            || echo "$h : ÉCHEC ($(tail -1 /tmp/provision_$h.ok))"
    ) &
done
wait
