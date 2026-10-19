# Expansion des spécifications de seeds — sourcé par launch/tmux.sh et launch/slurm.sh.

# expand_seeds "1-5" | "1,3,5" | "2"  ->  une seed par ligne
expand_seeds() {
    local spec=$1 part lo hi
    [[ -n "$spec" ]] || { printf 'ERREUR : spec de seeds vide\n' >&2; return 1; }
    local IFS=','
    for part in $spec; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo=${BASH_REMATCH[1]}; hi=${BASH_REMATCH[2]}
            (( lo <= hi )) || { printf "ERREUR : plage '%s' inversée\n" "$part" >&2; return 1; }
            seq "$lo" "$hi"
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            echo "$part"
        else
            printf "ERREUR : seed '%s' invalide (attendu N, N-M ou N,M)\n" "$part" >&2
            return 1
        fi
    done
}

# slurm_array "1 2 3 5" -> "1,2,3,5" (pour --array)
slurm_array() { local IFS=','; echo "$*"; }
