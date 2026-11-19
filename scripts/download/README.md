# Téléchargement des données et des poids

Un script par ressource. Chaque script télécharge, extrait puis prétraite, et range le
résultat sous `datasets/<nom>/` (ou `pretrained/` pour les poids). Ces deux dossiers sont
ignorés par git.

| Script | Contenu | Taille téléchargée | Prétraitement |
|---|---|---|---|
| `download_devo_weights.sh` | `pretrained/DEVO.pth`, poids officiels DEVO | ~40 Mo | — |
| `download_evimo2.sh` | `datasets/evimo2/`, EVIMO2 v2 samsung_mono, catégorie IMO | ~2,4 Go | conversion EVIMO1 + format DEVO (séquences d'éval) |
| `download_rpg.sh` | `datasets/rpg/`, RPG stereo DAVIS (ECCV'18), 8 séquences | ~4,3 Go | `scripts/pp_rpg.py` |

```bash
bash scripts/download/download_devo_weights.sh
bash scripts/download/download_evimo2.sh      # --no-preprocess, --keep-archive
bash scripts/download/download_rpg.sh         # --no-preprocess, --delete-bags
```

- Les téléchargements reprennent là où ils se sont arrêtés, et la taille finale de chaque
  fichier est vérifiée. Relancer un script terminé ne retélécharge rien.
- Le prétraitement demande l'env conda du projet. Le téléchargement seul
  (`--no-preprocess`) ne demande que `curl` et `tar`.
- Pour stocker les données ailleurs (disque de données d'un serveur), définir
  `MADEVO_DATASETS` et/ou `MADEVO_PRETRAINED` avant de lancer les scripts.

## Arborescence obtenue

```
datasets/
  evimo2/
    samsung_mono/imo/{train,eval}/<seq>/   format natif EVIMO2 (entraînements MS et couplé)
    evimo1_format/imo/eval/<seq>.npz       masques GT pour les évals (--mask_root)
    eval_preprocessed/<seq>/               format DEVO pour les évals (--datapath)
  rpg/
    <seq>.bag                              rosbags bruts
    <seq>/                                 format DEVO (--datapath datasets/rpg)
pretrained/
  DEVO.pth
```

## VECtor quand Google Drive bloque le serveur

Les séquences VECtor ne sont que sur Google Drive, qui finit par refuser gdown depuis les
serveurs (« Too many users have viewed or downloaded this file recently »). On les télécharge
alors depuis une machine personnelle :

```bash
pip install gdown
bash scripts/download/download_vector_local.sh --out ~/vector_dl \
    --to royce:/Data/arthur.fournier/madevo/datasets/vector/
```

Le script n'extrait des zips d'images (16 Go) que `timestamp.txt`, seul fichier dont DEVO a
besoin, et supprime le zip. Il est relançable ; un fichier que gdown refuse se télécharge dans
un navigateur (le lien est affiché), se dépose dans `<out>/_navigateur/` et se range à la
relance. Sur le serveur, `download_vector.sh` ne retélécharge alors rien et fait le
prétraitement (`pp_vector.py` ne contrôle le nombre d'images que si elles sont présentes).
