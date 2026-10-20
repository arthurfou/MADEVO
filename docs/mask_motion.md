# Définition du masque GT « dynamique » sur EVIMO2 (T7)

**Décision** : `mask_motion: frame`, `motion_thresh_m: 0.01`, `motion_window_s: 0.2`,
`loss: bce_dice` avec `loss_pos_weight: 7.0` (MS seul) et `loss.mask_pos_weight: 7.0`
(supervision M2). Ablation : `mask_motion: all`, la définition du stage, avec sa loss
(`loss_pos_weight: 10.0`, BCE non pondérée pour M2).

Un objet est donc **dynamique à l'instant t** s'il s'écarte de plus de 1 cm de sa position
médiane (repère monde) dans la fenêtre [t − 0,1 s, t + 0,1 s]. Cela revient à un seuil de
vitesse d'environ 10 cm/s : à vitesse constante, l'écart maximal à la médiane vaut la moitié
du trajet parcouru dans la fenêtre.

## Le problème

`dataset_mask.npz` segmente tous les objets suivis par VICON, décor immobile compris. Prendre
`mask != 0` comme cible (`all`) revient à apprendre à reconnaître les objets annotés, et
l'oracle masque surtout du décor statique. `ms_model/motion.py` reconstruit la trajectoire
monde de chaque objet à partir des poses de la meta et filtre les masques au chargement.

## Mesures

Commande (CPU, quelques minutes) :

```bash
python tools/mask_motion_stats.py --data-root datasets/evimo2/samsung_mono/imo/train \
    --out docs/mask_motion_stats_train.md
```

Tables complètes : [`mask_motion_stats_train.md`](mask_motion_stats_train.md) (choix) et
[`mask_motion_stats_eval.md`](mask_motion_stats_eval.md) (pour information seulement, le
choix n'en dépend pas). Cellules à la résolution du MS (H/4 × W/4), masques dilatés de 3 px
comme à l'entraînement, poolées sur toutes les frames (1 sur 2).

**Split train** (21 séquences, 4745 frames) :

| mode | positifs | frames sans positif | neg/pos |
|---|---|---|---|
| `all` | 61,4 % | 1,8 % | 0,6 |
| `sequence`, 1 cm | 28,1 % | 3,8 % | 2,6 |
| `sequence`, 2 cm | 27,9 % | 3,8 % | 2,6 |
| `frame`, 0,5 cm, 0,2 s | 3,05 % | 63,7 % | 31,8 |
| **`frame`, 1 cm, 0,2 s** | **2,00 %** | **65,6 %** | **49,0** |
| `frame`, 2 cm, 0,2 s | 1,42 % | 69,1 % | 69,4 |
| `frame`, 1 cm, 0,1 s | 1,18 % | 70,3 % | 83,6 |
| `frame`, 1 cm, 0,5 s | 3,64 % | 59,2 % | 26,4 |

Pour comparaison, sur l'éval (8 séquences) : `all` 74,9 %, `sequence` (2 cm) 11,8 %,
`frame` (1 cm, 0,2 s) 3,77 % de positifs.

## Pourquoi `frame` plutôt que `sequence`

- Les objets mobiles sont **au repos la plupart du temps**. Sur les 92 objets du train qui
  bougent de plus de 2 cm sur la séquence, 59 ont un écart médian sous 1 cm dans une fenêtre
  de 0,2 s : ils sont immobiles plus de la moitié du temps. `sequence` les marque dynamiques
  pendant tout ce temps : 93 % des positifs de `sequence` (28,1 % de cellules) ne bougent pas
  à cet instant (`frame` : 2,0 %).
- Pour la VO, un objet au repos est un amer statique valide : l'écarter ne fait que perdre
  des patches. Et pour un ConvLSTM causal sur des events, un objet au repos ne se distingue
  du décor que par sa mémoire ou son apparence : `sequence` redonne en partie la tâche de
  reconnaissance qu'on voulait éviter avec `all`.

## Pourquoi 1 cm

- **Bruit des poses** : sur les 83 objets du train dont l'écart sur toute la séquence reste
  sous 1 cm (décor), l'écart maximal dans une fenêtre glissante atteint **0,66 cm**
  (0,52 cm sur l'éval). Il vient surtout des poses caméra (l'objet est exprimé dans le
  repère caméra) et de quelques sauts VICON. Un seuil de 0,5 cm est donc dans le bruit.
- 1 cm laisse une marge de 1,5× au-dessus de ce bruit, et 2 cm de 3×. Mais 2 cm perd les
  mouvements lents (objets poussés, qui restent sous 1,5 cm dans 0,2 s), soit 29 % des
  positifs de 1 cm. On garde 1 cm.
- Le passage 1 cm → 2 cm ne change presque rien en mode `sequence` (28,1 % → 27,9 %) : les
  objets du train sont soit franchement statiques, soit franchement mobiles. Le seuil ne
  devient sensible qu'en mode `frame`, où il trie les phases lentes.

## Pourquoi 0,2 s

- 0,1 s : l'écart dû à une vitesse donnée est deux fois plus petit, alors que le bruit, lui,
  ne diminue pas. Le seuil effectif monte à ~20 cm/s et les étiquettes clignotent dès qu'un
  objet ralentit (70 % de frames vides).
- 0,5 s : l'objet est marqué dynamique 0,25 s **avant** de commencer à bouger. Un modèle
  causal ne peut pas le prédire, et la cible devient en partie inatteignable.
- 0,2 s (±0,1 s) laisse une douzaine d'échantillons de pose pour la médiane, et le retard
  de l'étiquette reste court.

## Loss

- Avec 2 % de positifs, le `pos_weight` qui équilibre les classes vaudrait 49.
  `loss_pos_weight: 10.0` avait été réglé pour `all`, où il **sur-pondérait** une classe déjà
  majoritaire (61 %). On prend **7 ≈ √49**, une heuristique courante. Plein équilibre (49) :
  le masque déborderait largement des objets. Sans pondération (1) : la BCE pousserait le
  masque vers 0.
  Le biais vers le rappel est voulu : un faux positif ne fait que déplacer un patch sur le
  décor, alors qu'un faux négatif laisse un patch dynamique dans la BA.
- On garde **`bce_dice`** (poids 0,5). Le Dice est calculé sur tout le batch (8 clips × 16
  frames), donc il reste défini malgré 66 % de frames sans positif. Il compense le
  déséquilibre sans réglage. Un Dice seul serait instable sur les batchs presque vides.
- **M2** utilisait une BCE non pondérée sur le masque. Elle reçoit le même `pos_weight`
  (`loss.mask_pos_weight`, option `--mask_pos_weight` de `train_coupled.py`). Défaut 1.0 :
  le comportement du stage, conservé pour l'ablation `all`.
- **M3** n'est pas touché : sa cible (résidus > médiane + 3·MAD) ne dépend pas du GT.

⚠️ **7 est une heuristique, pas une valeur mesurée.** Le vérifier demande un entraînement
GPU. Si on le règle, il faut le faire sur des séquences **du train** tenues à l'écart : le
split `val_sequences` actuel est le split d'éval des tables (voir « Points ouverts »).

## Cohérence de bout en bout

La définition du masque doit être la même pour l'entraînement MS, la supervision M2 et
l'oracle des tables :

- configs du papier : `configs/ms/v4_evimo2.yaml`, `v4large_evimo2.yaml`,
  `configs/coupled/_base_evimo2.yaml`, `configs/eval/evimo2.yaml` ;
- ablation : `configs/ms/v4_evimo2_all.yaml`, `configs/coupled/m{2,3}_evimo2_all.yaml`,
  `configs/eval/evimo2_all.yaml` (des `exp` distincts, donc des `runs/` distincts) ;
- `tests/test_eval.py` vérifie que chaque famille est cohérente ;
- à l'exécution, `evals/eval_table.py` lit le `config.yaml` de chaque run évalué, trace sa
  définition du masque dans `manifest.yaml` et **refuse** d'évaluer si elle diffère de celle
  de l'oracle.

## Points ouverts

- `val_sequences` des configs MS = les 8 séquences d'éval. `best_iou.pt` est donc choisi
  sur le split de test des tables. Pour le papier, il faudrait une validation tirée du
  train.
- L'oracle des tables dilate de 2 px, l'entraînement de 3 px : c'est l'héritage du stage,
  non tranché ici.
- Variantes scratch : `soft_prior_target: 0.1` (moyenne visée du masque) a été fixé pour
  `all`. En `frame`, le taux réel est de 2 à 4 %.
- Avec `frame`, 66 % des frames n'ont aucun objet mobile, et l'oracle y vaut DEVO vanilla.
  Le gain de l'oracle mesurera donc l'effet des seuls vrais objets mobiles. Il sera
  probablement bien plus petit que les +41 % du stage (obtenus en masquant du décor), et
  c'est la grandeur honnête.
