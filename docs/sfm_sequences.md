# Séquences statiques SfM d'EVIMO2 — faux positifs du MS (piste T14)

**État : outillage en place, rien de mesuré.** Cette note décrit le problème, ce qui est déjà
branché dans le code, et ce qu'il reste à faire. À ne pas confondre avec une décision prise :
la campagne du papier (T11) s'entraîne sur la seule catégorie **IMO**, comme le stage.

## Le problème

Le modèle de motion segmentation n'a jamais vu de scène sans objet mobile : toutes les
séquences d'entraînement viennent de la catégorie IMO d'EVIMO2, où un ou plusieurs objets
suivis par VICON bougent devant la caméra. Rien dans l'entraînement ne lui apprend donc à
répondre « rien ne bouge ».

C'est gênant à deux endroits :

1. **Validation cross-dataset** (`configs/eval/{rpg,mvsec,eds,vector,tumvie}.yaml`) : ces
   cinq datasets sont des scènes **entièrement statiques**, caméra seule en mouvement. Le
   masque idéal y est nul partout, et DEVO devrait se comporter exactement comme vanilla.
   Tout score non nul écarte des patches utiles : un faux positif y coûte directement de
   l'ATE, sans aucun gain possible en face.
2. **Généralisation** : une scène réelle alterne des moments où quelque chose bouge et des
   moments où rien ne bouge. Un MS qui ne sait pas se taire n'est pas utilisable hors banc.

## Ce que la catégorie SfM apporte

EVIMO2 v2 publie, pour la même caméra `samsung_mono`, une catégorie **`sfm`** : scènes où
**seule la caméra bouge**, sans objet mobile. Le masque GT « dynamique » y est donc vide,
quel que soit le mode de `mask_motion` (`all`, `sequence`, `frame`) — ce sont des exemples
**entièrement négatifs**, dans le même domaine capteur que l'entraînement (même caméra, même
résolution, mêmes conditions d'éclairage, même prétraitement).

C'est exactement le signal qui manque, et il ne demande ni nouvelle annotation ni nouveau
capteur.

```bash
bash scripts/download/download_evimo2.sh --sfm      # ~10,4 Go, en plus de la catégorie IMO
# → datasets/evimo2/samsung_mono/sfm/{train,eval}/<seq>/
```

## Ce qui est déjà branché

- **`extra_train_sequences`** dans une config MS : séquences **ajoutées** à
  `train_sequences` sans le remplacer. `ms_model/training/train.py:training_sequences()` est
  la seule source de la liste, et `train_coupled.py` lit la même fonction — les
  entraînements couplés voient donc le même ensemble que le MS, sans config à synchroniser.

  ```yaml
  data:
    train_sequences: [...]              # les 21 séquences IMO du split train
    extra_train_sequences:              # scènes statiques, exemples tout négatifs
      - sfm/train/<seq>
      - ...
  ```

- **`loss.selfsup_min_res`** pour M3 (`train_coupled.py`) : plancher absolu, en pixels, du
  seuil sur les résidus de la DBA. Le seuil auto-supervisé du stage est **relatif** (médiane
  + k·MAD) : sur une scène statique, où tous les résidus sont du bruit, il marque toujours la
  queue de la distribution comme « dynamique ». Le plancher empêche cela. `0.0` (défaut, et
  valeur de toutes les configs de campagne) = comportement du stage, inchangé.

## Ce qu'il reste à faire

1. Télécharger la catégorie SfM et **compter** : combien de séquences train, quelle durée
   totale par rapport aux 21 séquences IMO.
2. Choisir la proportion ajoutée. Trop de négatifs et le MS ne prédit plus rien ; les deux
   garde-fous sont `loss_pos_weight` (7 aujourd'hui, réglé sur 2 % de cellules positives en
   mode `frame` — l'ajout de séquences tout négatives fait baisser cette part, donc le poids
   est à **remesurer**, avec `tools/mask_motion_stats.py` sur l'ensemble étendu) et l'IoU de
   validation, qui reste calculée sur les seules séquences IMO.
3. Mesurer sur **deux** axes, les deux nécessaires :
   - part de cellules positives prédites sur les scènes SfM (faux positifs : doit tomber) ;
   - IoU de validation sur IMO (doit ne pas bouger).
4. Mesurer le gain là où il compte : les tables cross-dataset
   (`launch/campaign_evimo2.sh --cross`), M1/M2/M3 contre vanilla.
5. Pour M3, régler `selfsup_min_res` sur l'ordre de grandeur du résidu observé sur une scène
   statique (les métriques `coupling/selfsup_*` du logger wandb donnent médiane, MAD, seuil et
   part positive à chaque pas).

**Ordre** : après la campagne EVIMO2 (T11) et ses chiffres. Un entraînement MS avec SfM est
un modèle **différent** : il ne remplace pas la ligne du papier, il s'ajoute comme ablation de
généralisation.

## Pièges

- La catégorie SfM a sa propre arborescence `sfm/{train,eval}/` : les chemins
  d'`extra_train_sequences` sont relatifs à `data.root`
  (`${MADEVO_DATASETS:-datasets}/evimo2/samsung_mono`), comme ceux de `train_sequences`, et
  commencent donc par `sfm/`, non par `imo/`.
- `ms_model/motion.py` filtre les masques par mouvement : sur une séquence SfM il n'y a
  aucun objet à filtrer, le masque est vide avant comme après. Rien à régler de ce côté.
- Les séquences SfM ne doivent **jamais** entrer dans un split d'éval VO : les tables du
  papier se comparent à des chiffres existants sur `splits/evimo2/evimo2_val.txt`.
