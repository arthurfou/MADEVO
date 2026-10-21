# Séquences statiques SfM d'EVIMO2 — faux positifs du MS (piste T14)

**État : outillage en place, catégorie mesurée (21/10), aucun entraînement lancé.** Cette note décrit le problème, ce qui est déjà
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
seule la caméra est censée bouger. Même domaine capteur que l'entraînement (même caméra,
même résolution, même prétraitement), et aucune annotation nouvelle à produire.

⚠️ **Ces séquences ne sont des exemples négatifs qu'en mode `frame`.** Mesuré le 21/10 sur
les 35 séquences de `sfm/train` (`tools/mask_motion_stats.py`, tables complètes dans
[`mask_motion_stats_sfm_train.md`](mask_motion_stats_sfm_train.md)) :

| mode | cellules positives | frames sans positif | séquences non vides |
|---|---|---|---|
| `all` | **67,4 %** | 0,6 % | 35 / 35 |
| `sequence`, 1 cm | **8,0 %** | 69,6 % | 11 / 35 |
| **`frame`, 1 cm, 0,2 s** | **0,11 %** | 99,6 % | 8 / 35 |

En mode `all`, la catégorie SfM est masquée aux deux tiers — c'est du décor annoté par
VICON, exactement le problème du point 4 d'`INTEGRITY_CHECK.md`, et elle n'apporterait que
des faux positifs. En mode `sequence`, 11 séquences gardent des objets marqués mobiles
(jusqu'à 36 % des pixels d'une séquence). Seul le mode **`frame` retenu en T7** en fait de
vrais négatifs : 0,11 % de cellules positives, 99,6 % des frames entièrement vides.

Le résidu de 0,11 % n'est pas du bruit de mesure : 8 séquences gardent une trace (au plus
1 % des pixels d'une séquence). La catégorie SfM n'est donc pas *parfaitement* statique, et
il faut la traiter comme « presque tout négatif », pas comme un zéro garanti.

**Conséquence pour l'ablation `all`** : les configs `*_all.yaml` ne peuvent pas recevoir de
séquences SfM. L'ablation reproduit le stage, où la question ne se posait pas ; y ajouter
des scènes masquées aux deux tiers n'aurait aucun sens.

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

1. ~~Télécharger la catégorie SfM et compter.~~ **Fait le 21/10** : `sfm/train` compte
   **35 séquences et 28 326 frames GT (8,2 min)**, contre 21 séquences et 9 478 frames
   (2,7 min) pour `imo/train` — la catégorie SfM pèse **3 fois** l'ensemble d'entraînement
   actuel. L'ajouter en entier déséquilibrerait donc l'entraînement vers le négatif.
2. Choisir la proportion ajoutée, et **remesurer `loss_pos_weight` en conséquence** : il
   vaut 7 aujourd'hui, pour 2,00 % de cellules positives sur `imo/train` seul (cf.
   [`mask_motion.md`](mask_motion.md)). En supposant l'heuristique de T7 inchangée
   (`pos_weight ≈ √(neg/pos)`), et SfM à 0,11 % de positifs :

   | ensemble d'entraînement | frames | cellules positives | `loss_pos_weight` |
   |---|---|---|---|
   | IMO seul (campagne T11) | 9 478 | 2,00 % | **7** (actuel) |
   | IMO + SfM à parts égales (~12 séq.) | ~19 000 | ~1,06 % | ~10 |
   | IMO + tout SfM | 37 804 | ~0,58 % | ~13 |

   Ces valeurs sont des **prévisions arithmétiques**, pas des mesures : à confirmer en
   relançant `tools/mask_motion_stats.py` sur l'ensemble effectivement retenu. Le second
   garde-fou est l'IoU de validation, qui reste calculée sur les seules séquences IMO.
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
- `ms_model/motion.py` filtre les masques par mouvement, et c'est **lui** qui rend la
  catégorie SfM exploitable : sans filtre (mode `all`), ces séquences sont masquées à 67 %.
  Le mode, le seuil et la fenêtre doivent donc être les mêmes que pour les séquences IMO —
  ce que `tests/test_eval.py` vérifie déjà sur les configs.
- Les séquences SfM ne doivent **jamais** entrer dans un split d'éval VO : les tables du
  papier se comparent à des chiffres existants sur `splits/evimo2/evimo2_val.txt`.
