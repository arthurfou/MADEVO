# Interface de visualisation MADEVO (T14) — note de conception

**État : rien n'est construit, rien n'est décidé.** Cette note dit quoi construire, dans quel
ordre, et surtout **ce que la campagne doit écrire en plus dès maintenant** pour que l'app
soit possible plus tard. Ce dernier point est urgent : une fois les 5 seeds lancées, ajouter
un artefact veut dire tout relancer sur GPU.

**Recommandation en une phrase** : deux interfaces distinctes — un **inspecteur local
Streamlit** qui rejoue les dossiers de runs pour le débogage scientifique, et une **page
statique exportée** pour le papier et les candidatures — construites sur un **contrat
d'artefacts** que les jobs doivent produire à partir de maintenant.

---

## 1. Le problème et les publics

Aujourd'hui, pour comprendre pourquoi une scène a une ATE aberrante, il faut ouvrir un CSV,
retrouver le dossier d'éval, lire un `manifest.yaml`, relancer un script de plot, et croiser
tout ça à la main. Rien ne permet de **voir** ce que le modèle a fait sur une frame donnée :
la score map, qui est le cœur de la méthode, n'est écrite nulle part (cf. §6).

Trois publics, trois besoins incompatibles :

| Public | Ce qu'il veut | Contrainte dominante |
|---|---|---|
| **Arthur qui travaille** | vérifier qu'un run n'a pas déraillé, comparer deux seeds, comprendre une ATE aberrante, inspecter le masque frame par frame | accès aux données lourdes (5,5 à 44 Go), interactivité profonde, local |
| **Le papier / les relecteurs** | produire la figure teaser, remonter d'un chiffre au dossier qui l'a produit | reproductibilité, export figé (PDF/PNG vectoriel), pas d'interactivité |
| **Candidatures / entretiens** | comprendre le sujet en 2 minutes sans connaître DEVO ni les events | rien à installer, rien à télécharger, lien cliquable, public |

### Une interface ou deux ?

**Deux.** Les contraintes sont réellement incompatibles, sur trois axes :

1. **Les données.** L'inspecteur a besoin des datasets (EVIMO2 5,5 Go, TUM-VIE 44 Go) et des
   checkpoints. La démo publique ne peut rien télécharger et ne doit rien exiger.
2. **Le temps d'attention.** L'inspecteur assume qu'on sait ce qu'on cherche et qu'on reste
   dix minutes. La démo a deux minutes et un spectateur qui ignore ce qu'est un event.
3. **La confidentialité et la durée de vie.** L'inspecteur manipule des runs en cours, des
   chiffres non publiés, des chemins serveur. La démo est publique et doit survivre au
   projet sans maintenance.

Vouloir une seule app qui fait les deux, c'est exactement l'erreur d'EVOwithMS (§3) : une
interface qui sert à tout finit par ne servir à personne. En revanche les deux **partagent
leur socle** : le même paquet `ms_model/viz/` et les mêmes artefacts. La démo est un **export
figé de l'inspecteur**, pas un second développement.

---

## 2. Ce qui existe déjà, et la frontière avec wandb

### Briques réutilisables (ne rien réécrire)

| Fonction | Fichier | Ce qu'elle rend |
|---|---|---|
| `render_event_frame`, `make_frames` | `ms_model/viz/render.py` | events → image (H, W, 3) |
| `render_mask_frame`, `render_mask_overlay`, `make_overlay_frames` | `ms_model/viz/mask.py` | masque coloré par id d'objet, overlay sur une frame |
| `frames_to_video` | `ms_model/viz/video.py` | liste de frames → mp4 |
| `voxel_image`, `mask_image`, **`patches_image`** | `utils/wandb_logger.py` | voxel colorisé, score map colorisée, **patches sélectionnés dessinés** |
| `plot_trajectory`, `fig_trajectory` | `utils/viz_utils.py` | trajectoire estimée vs GT, alignée Sim(3) |
| `mean_scene_ate` | `utils/metrics.py` | moyenne ATE sans les agrégats (pur, sans CUDA) |

⚠️ `patches_image()` vit dans `utils/wandb_logger.py` alors qu'elle n'a rien de wandb. Elle
est à **déplacer dans `ms_model/viz/`** quand l'app la réutilisera (le logger l'importera de
là), sinon l'app dépendra du logger pour dessiner une image.

### Ce qui est mort et qu'il faut supprimer

`ms_model/viz/gradio_demo.py` pointe vers `../datasets/EV-IMO/eval` — chemins EVIMO1, dataset
abandonné et supprimé de la machine. Ce fichier **ne tourne plus**. C'est le second cadavre
Gradio du projet (§3) ; il doit partir avec la première étape de T14, pas rester comme
« point de départ » : sa structure est précisément celle qu'il ne faut pas reprendre.

### La frontière avec wandb

**wandb garde tout ce qui est série temporelle d'entraînement. L'app ne la duplique jamais.**

| Sujet | Qui | Pourquoi |
|---|---|---|
| courbes de loss, IoU, LR, gradient norm, par pas | **wandb** | il est fait pour ça, il compare les seeds nativement, il est déjà branché (`RunLogger`) |
| images d'entraînement périodiques (voxel/masque prédit/GT tous les N pas) | **wandb** | `log_images`, déjà en place |
| tables d'éval, ATE par méthode | **wandb** (`log_table`) *et* app | wandb pour l'historique, l'app pour la navigation vers la frame fautive |
| **replay frame par frame d'une scène évaluée** | **app** | wandb ne sait pas rejouer une séquence à la demande |
| **comparaison M0–M3 sur la même frame** | **app** | demande de charger 4 artefacts et de les aligner |
| **trajectoire + erreur le long du temps** | **app** | wandb stocke des scalaires, pas des trajectoires navigables |
| **figures du papier** | **app** (export) | il faut un PDF vectoriel reproductible, pas une capture |
| **provenance (quel checkpoint → quel chiffre)** | **app** | le `manifest.yaml` est sur disque, wandb n'en a qu'une copie dans la config |

Règle : **si la question est « comment ça a évolué pendant l'entraînement », c'est wandb. Si
la question est « qu'est-ce que ce modèle a fait sur cette image », c'est l'app.**

Cas limite assumé : wandb est utilisé en `offline` sur les serveurs sans internet. Tant que
les runs ne sont pas synchronisés, les courbes ne sont pas consultables. L'app doit donc
savoir afficher, en dépannage, les métriques parsées depuis `logs/stdout.log` (format
`epoch N: train_loss=… train_iou=… val_loss=… val_iou=…`, déjà écrit par
`ms_model/training/train.py`). C'est un filet de sécurité, pas un remplacement.

---

## 3. Pourquoi EVOwithMS est mort — la contrainte de conception numéro un

Reconstitution à partir de `reports/june_05/main.tex` (slide « Gradio Interface & Motion
Segmentation ») et de l'historique :

> « While waiting for long computations to finish, started building a Gradio web interface
> to: visualise event frames from EV-IMO sequences, overlay GT masks […], **interactively
> debug the mask alignment issue** »

L'app a été construite en juin **pour déboguer une question précise** : le désalignement des
masques — qui s'est révélé être le problème 5 d'`INTEGRITY_CHECK.md` (le i-ème masque associé
au i-ème timestamp toutes frames confondues). Quatre causes de mort, toutes structurelles :

1. **Construite pour une question transitoire.** Le bug compris et corrigé, l'app n'avait
   plus de raison d'être. Un outil qui répond à une seule question meurt avec elle.
2. **Couplée à un dataset qui a été abandonné.** Tout était indexé sur l'arborescence EVIMO1
   (`EVOwithMS/datasets/EV-IMO`). EVIMO1 abandonné → app morte, sans qu'une ligne de son code
   soit fausse.
3. **Elle recalculait au lieu de rejouer.** Elle chargeait un checkpoint et lançait
   l'inférence en direct. Elle dépendait donc de l'environnement, des poids, des chemins, du
   GPU — c'est-à-dire de tout ce qui change. `gradio_demo.py` a exactement la même maladie
   (`Predictor` + inférence live) et il est mort de la même façon.
4. **Elle vivait hors du repo**, avec sa propre copie des données. Rien ne la testait, rien
   ne la cassait visiblement quand l'amont bougeait : elle a pourri en silence.

**Les quatre règles qui en découlent, non négociables :**

| Règle | Conséquence concrète |
|---|---|
| **Rejouer, jamais recalculer** | l'app ne charge aucun checkpoint, n'importe jamais `devo`, ne touche pas au GPU. Elle lit des fichiers. |
| **Indexer sur le contrat de run, pas sur un dataset** | l'entrée de l'app est `runs/<exp>/<id>/seed<N>/` et `evals_out/…`, structures stables (T4) et déjà testées. Un dataset qui disparaît dégrade une vue, il ne tue pas l'app. |
| **Dans le repo, couverte par les tests** | `tests/test_viz_app.py` vérifie au moins le chargement d'un dossier de run synthétique. Si l'amont casse le format, le test rougit. |
| **Dégradation gracieuse** | artefact absent → section grisée avec « non disponible, relancer avec `--dump-scores` », jamais une exception. |

---

## 4. Ce qu'on pourrait montrer — inventaire trié

Notation : **valeur** = utilité réelle pour l'un des trois publics ; **effort** une fois les
artefacts disponibles. Les éléments marqués 🔒 sont **bloqués** par un artefact manquant (§6).

### Rang A — à faire, le cœur de l'app

| # | Vue | Public | Valeur / effort | Note |
|---|---|---|---|---|
| A1 | **Index des runs** : tous les `runs/` et `evals_out/`, état (DONE/en cours/échec), exp, seed, date, commit git | Arthur | ⭐⭐⭐ / faible | répond à « où en est la campagne », marche **aujourd'hui**, sans aucun artefact nouveau |
| A2 | **Table centrale + provenance** : `central_table.csv` + `per_scene.csv` + `manifest.yaml` côte à côte, chaque chiffre cliquable vers le checkpoint qui l'a produit | papier, Arthur | ⭐⭐⭐ / faible | c'est la réponse à « d'où sort ce chiffre », exigée par T12 |
| A3 | **Replay d'une scène** : events + masque GT + **masque prédit** + score map, curseur de frame | Arthur | ⭐⭐⭐ / moyen | 🔒 score maps. Le cœur du débogage scientifique |
| A4 | **Comparaison M0–M3 sur la même frame** : 4 vignettes synchronisées | papier, Arthur | ⭐⭐⭐ / moyen | 🔒 score maps. C'est **la figure teaser** du papier |
| A5 | **Trajectoire estimée vs GT** + **erreur le long du temps**, avec curseur lié au replay | Arthur | ⭐⭐⭐ / moyen | les trajectoires existent déjà (TUM format) ; le lien erreur→frame est ce qui explique une ATE aberrante |
| A6 | **Agrégat 5 seeds** : moyenne ± écart-type, et **quelle scène est instable entre seeds** | papier | ⭐⭐⭐ / faible | `summary.csv` existe ; la dispersion par scène est ce qui dit si un gain est réel |

### Rang B — utile, après le rang A

| # | Vue | Valeur / effort | Note |
|---|---|---|---|
| B1 | **Score map avant/après `(1 − s)`** et **patches sélectionnés** superposés | ⭐⭐ / moyen | 🔒 patches. Montre le mécanisme réel : ce n'est pas le masque qui compte, c'est quels patches disparaissent |
| B2 | **Résidus DBA de M3** (le signal d'auto-supervision) : carte des résidus + seuil médiane+k·MAD | ⭐⭐ / moyen | 🔒 résidus. C'est l'argument central du papier (« le résidu est un meilleur signal que l'annotation ») — invisible aujourd'hui |
| B3 | **Faux positifs sur scènes statiques** (SfM, cross-dataset) : part de cellules positives prédites là où la vérité est « rien ne bouge » | ⭐⭐ / faible | 🔒 score maps. Mesure directement la piste de `docs/sfm_sequences.md` |
| B4 | **Définition du masque GT** : les 3 modes `all`/`sequence`/`frame` sur la même frame | ⭐⭐ / faible | `scripts/plot_mask_definition.py` le fait déjà en statique — à rendre interactif (seuil, fenêtre) |
| B5 | **Diff de configs entre deux runs** : ce qui change entre deux `config.yaml` | ⭐⭐ / faible | répond à « pourquoi ces deux runs diffèrent », purement textuel |
| B6 | **Frise de couverture GT** : trous de tracking VICON d'EVIMO2 sur un axe temporel | ⭐⭐ / faible | rend visible le piège qui a causé les bugs 5 et 6 |

### Rang C — si le temps le permet

| # | Vue | Note |
|---|---|---|
| C1 | **Nuage de points `map.ply`** en 3D | `export_map` existe mais est à `false` partout ; joli en démo, faible valeur scientifique. Un viewer 3D web est coûteux — préférer un rendu offline |
| C2 | **Courbes d'entraînement hors-ligne** depuis `logs/stdout.log` | dépannage quand wandb n'est pas synchronisé |
| C3 | **Vue « avant/après correctif »** : table buggée vs corrigée | pédagogique pour le rapport d'intégrité, périmé une fois T12 fait |
| C4 | **Drift** : erreur cumulée vs longueur de trajectoire (MPE est déjà calculé) | complète A5 |

### Ce à quoi la demande initiale ne pensait pas, et qui vaut le détour

- **A6-bis — la dispersion inter-seeds par scène.** Avec 5 seeds, la vraie question du papier
  n'est pas « quelle est l'ATE » mais « ce gain dépasse-t-il le bruit ». Une scène dont
  l'écart-type dépasse l'écart entre deux méthodes invalide la comparaison. **Aucun outil ne
  le montre aujourd'hui**, et `summary.csv` agrège trop tôt pour le voir.
- **A5-bis — la navigation chiffre → instant fautif.** L'intérêt de lier l'erreur temporelle
  au replay, c'est de cliquer sur le pic d'erreur et d'arriver sur la frame correspondante.
  C'est ce qui transforme « cette scène est mauvaise » en « cette scène casse à t = 4,2 s
  quand l'objet entre dans le champ ».
- **A4-bis — l'écart oracle ↔ appris sur la même frame.** M0 est le plafond. Afficher
  l'oracle et le masque appris sur la même image montre *ce qui reste à gagner*, et où.
- **Le contrôle d'alignement cross-dataset en image.** `tools/check_cross_alignment.py` sort
  des nombres (corrélation, décalage). Les deux densités superposées rendraient le bug 7
  évident en un coup d'œil, et rejouables pour n'importe quelle scène.
- **La reproductibilité affichée.** `git.txt` et `cmd.txt` sont déjà écrits dans chaque run :
  les afficher répond à « comment refaire exactement ce chiffre » sans fouiller.

---

## 5. Le choix technique

### Inspecteur : **Streamlit**, dans le repo, `madevo-viz`

```bash
streamlit run tools/viz/app.py -- --runs runs/ --evals evals_out/
```

**Pourquoi Streamlit plutôt que Gradio.** Gradio est conçu pour *« une fonction, une UI »* :
on lui donne un modèle, il fabrique un formulaire autour. C'est exactement le modèle
model-in-the-loop qui a tué EVOwithMS et `gradio_demo.py`. Streamlit est conçu pour
*parcourir des données* : état de session, widgets de navigation, et surtout
**`@st.cache_data` / `@st.cache_resource`**, qui est précisément la primitive dont on a besoin
avec des datasets de 5 à 44 Go (charger une séquence une fois, la garder entre deux
mouvements de curseur). Avec Gradio il faudrait gérer ce cache à la main.

**Pourquoi pas une pile JS.** Une personne maintiendra ça, après la fin du stage, dans un env
conda unique. Une app React/Three.js serait plus belle et serait abandonnée en six mois. Le
critère est la survie, pas l'esthétique.

**Pourquoi pas un notebook.** Un notebook est parfait pour explorer une fois, mauvais pour
revenir trois mois plus tard : l'état est implicite, le diff git est illisible, et rien ne
garantit qu'il tourne encore. Pour les analyses ponctuelles, un notebook reste le bon outil —
mais il ne doit pas être *l'* interface.

**Coût accepté** : Streamlit ajoute une dépendance à `environment-cpu.yml`. Elle est pure
Python, sans compilation, et n'est requise que pour l'app (import paresseux, et les tests de
l'app sont sautés si elle est absente — comme les tests EVIMO1 l'étaient).

### Démo publique : **page HTML statique auto-portante**

Pas un service, pas un serveur : `tools/viz/export_demo.py` produit un dossier
`demo/` (HTML + GIF/MP4 + PNG + un petit JSON de chiffres) déployable sur GitHub Pages. Rien
à installer pour le spectateur, rien à maintenir, pas de coût d'hébergement, et ça survit au
projet. Les figures viennent des **mêmes artefacts** que l'inspecteur : la démo est un export,
pas un second code.

C'est aussi ce qui alimente le README (qui réclame aujourd'hui « une figure ou un GIF »,
cf. T9) et la section Results, vide jusqu'à T12.

### Où ça vit

```
tools/viz/
  app.py              # entrée Streamlit (navigation, onglets)
  loaders.py          # lecture des dossiers de runs → objets Python, avec cache
  panels/             # une vue par fichier (index, table, replay, trajectoire, seeds)
  export_demo.py      # export de la page statique + figures du papier
ms_model/viz/         # rendus réutilisables (existant, + patches_image déplacée ici)
tests/test_viz_app.py # chargement d'un dossier de run synthétique, sans Streamlit
```

`tools/viz/loaders.py` ne doit **jamais** importer `devo` ni `torch` : c'est ce qui garantit
que l'app tourne sur ce portable sans GPU. Les tests le vérifient explicitement.

---

## 6. Les artefacts à faire écrire par les jobs — **le point urgent**

**C'est la partie actionnable de cette note.** La campagne T11 n'a pas encore tourné. Chaque
artefact ajouté *après* coup coûte un re-run GPU complet. Chaque artefact ci-dessous est
produit par du code **à nous** (`utils/eval_utils.py`, `evals/eval_table.py`,
`train_coupled.py`), jamais dans `devo/`.

### 6.1 Score maps par frame — 🔴 bloquant, priorité absolue

**Rien ne les écrit aujourd'hui.** Dans `utils/eval_utils.py::run_voxel`, la ligne

```python
dyn_score = None if dyn_mask_provider is None else dyn_mask_provider(t)
```

tient le score en main, le passe à DEVO, puis le jette. Sans dump, **la figure teaser du
papier est impossible** sans relancer DEVO sur GPU — alors que le papier la décrit comme
« the figure that carries the paper; produce it first ».

- **Quoi** : le score `(h/4, w/4)` de chaque frame, par scène et par méthode.
- **Où** : `evals_out/<dataset>/<sources>/<id>/seed<N>/scores/<method>/<scene>.npz`
- **Format** : `uint8` (score ×255) + le vecteur des `ts_us`, en `np.savez_compressed`.
  256 niveaux suffisent largement pour une visualisation, et c'est 2 fois plus petit que
  float16.
- **Taille** : EVIMO2 ≈ 120×160 = 19 200 cellules/frame ; ~250 frames/scène ; 8 scènes ;
  4 méthodes (oracle, M1, M2, M3) → **~150 Mo par seed avant compression**, nettement moins
  après (les scores sont lisses). Acceptable pour **la seed 1 seulement**, piloté par une clé
  de config `dump_scores: true`, par défaut **false** pour les seeds 2 à 5.
- **Comment** : un paramètre `score_dump_dir` sur `run_voxel`, qui accumule et écrit en fin de
  scène. ~15 lignes, dans notre code.

### 6.2 Patches sélectionnés — 🟠 important

- **Quoi** : par frame, les coordonnées `(x, y)` des P patches retenus, et leur poids après
  `(1 − s)`.
- **Pourquoi** : c'est le mécanisme réel de la méthode. Le masque est un intermédiaire ; ce
  qui change la pose, c'est **quels patches ont disparu**. `patches_image()` sait déjà les
  dessiner.
- **Où** : dans le même `.npz` que les scores (clés `patch_xy`, `patch_w`).
- **Taille** : négligeable (P ≈ 80 à 96 par frame).

### 6.3 Résidus DBA de M3 — 🟠 important

- **Quoi** : par pas d'entraînement, les statistiques déjà calculées (`median`, `mad`,
  `threshold`, `pos_frac` — elles existent dans `selfsup_stats`, `train_coupled.py`) **et**,
  périodiquement, le vecteur complet des résidus par patch.
- **Pourquoi** : l'argument central du papier est que le résidu est un meilleur signal
  d'apprentissage que l'annotation. Aujourd'hui on ne peut ni le montrer ni le vérifier.
- **Où** : `runs/<exp>/<id>/seed<N>/selfsup/step_<K>.npz`, tous les `log.images_every` pas.
- **Taille** : négligeable.

### 6.4 Index d'éval — 🟡 confort, mais évite un piège réel

⚠️ **Piège mesuré** : `utils/eval_utils.py::make_outfolder` construit son chemin avec
`datetime.today()`, **pas** avec le `LAUNCH_ID` du job. Le dossier des trajectoires est donc

```
<run_dir>/<dataset_name>/<date_du_jour>_<expname>/<scene>_trial_0_step_<step>/
```

où `<date_du_jour>` est la date **d'exécution de l'éval**, qui peut différer du `launch_id`
(éval lancée après minuit, job repris le lendemain). Une app qui devine ce chemin se trompera.

- **Quoi** : un `artifacts.yaml` à la racine de chaque dossier de seed, listant les chemins
  réellement écrits (trajectoires, scores, map, stats RPG), en relatif.
- **Pourquoi** : rend l'app indépendante de cette convention fragile, et documente ce que le
  run contient. C'est aussi le mécanisme de dégradation gracieuse (§3).
- **Où** : `evals_out/<dataset>/<sources>/<id>/seed<N>/artifacts.yaml`

### 6.5 Erreur par frame — 🟡 confort

- **Quoi** : l'erreur de translation après alignement Sim(3), par instant.
- **Pourquoi** : c'est ce qui permet de cliquer sur un pic et d'arriver sur la frame (A5-bis).
- **Note** : recalculable à froid depuis `stamped_groundtruth.txt` et
  `stamped_traj_estimate.txt`, qui sont **déjà écrits** par `run_rpg_eval`. **Donc non
  bloquant** — à faire dans l'app, pas dans le job.

### 6.6 Vignette de scène — 🟢 optionnel

Un PNG léger (une frame d'events rendue) par scène, pour que l'index soit lisible sans
charger un dataset. Quelques dizaines de Ko.

### Récapitulatif — ce qu'il faut décider **avant** de lancer T11

| Artefact | Bloquant pour | À écrire par | Priorité |
|---|---|---|---|
| **Score maps** (6.1) | figure teaser du papier, A3, A4, B1, B3 | `utils/eval_utils.py::run_voxel` | 🔴 **avant la campagne** |
| **Patches** (6.2) | B1 | idem, même fichier | 🟠 avant la campagne |
| **Résidus M3** (6.3) | B2, argument du papier | `train_coupled.py` | 🟠 avant la campagne |
| **`artifacts.yaml`** (6.4) | robustesse de l'app | `evals/eval_table.py` | 🟡 avant la campagne (peu cher) |
| Erreur par frame (6.5) | A5 | — (recalculable) | 🟢 après |
| Vignettes (6.6) | confort de l'index | — (recalculable) | 🟢 après |

**Les quatre premiers coûtent ~1 journée de dev CPU et se testent sur le smoke GPU.** Les
omettre coûte une campagne entière à relancer.

---

## 7. Plan par étapes — chaque étape livre quelque chose d'utilisable

### Étape 0 — le contrat d'artefacts *(avant la campagne, incompressible)*
Ajouter les dumps 6.1 à 6.4, les piloter par config (`dump_scores`, `dump_patches`),
les vérifier dans `jobs/smoke_gpu.sh` (qui tourne déjà la chaîne complète sur 2 scènes), et
supprimer `ms_model/viz/gradio_demo.py`.
**Livrable** : la campagne produit de quoi tout construire ensuite. *Rien de visuel, et c'est
la seule étape vraiment urgente.*

### Étape 1 — l'index et la provenance *(utilisable immédiatement, sans artefact nouveau)*
`tools/viz/loaders.py` + un onglet « Runs » : tous les runs, état, seed, commit, config ; et
un onglet « Tables » : `central_table.csv`, `per_scene.csv`, `summary.csv`, `manifest.yaml`.
**Livrable** : « où en est la campagne » et « d'où sort ce chiffre ». Marche **aujourd'hui**
sur `runs/ms_smoke/`, et répond déjà à un besoin de T12.

### Étape 2 — le replay *(le cœur)*
Onglet « Scène » : curseur de frame, events + masque GT + masque prédit + score map, les
4 méthodes synchronisées. Cache par scène.
**Livrable** : le débogage scientifique devient possible. A3, A4, B4.

### Étape 3 — trajectoires et erreur
Onglet « Trajectoire » : estimée vs GT alignée Sim(3), erreur le long du temps, curseur lié au
replay de l'étape 2, dispersion inter-seeds par scène.
**Livrable** : on explique une ATE aberrante au lieu de la constater. A5, A6.

### Étape 4 — les figures du papier
`tools/viz/export_demo.py --figure teaser` : PDF vectoriel reproductible, commande archivée à
côté de la figure.
**Livrable** : la figure teaser, et le chemin de chaque chiffre du papier vers son dossier.

### Étape 5 — la démo publique
Export statique : un GIF (events → masque → trajectoire corrigée), la table de résultats, 3
paragraphes d'explication. Déployable sur GitHub Pages, lié depuis le README.
**Livrable** : le lien qu'on met dans une candidature.

**Ordre imposé** : l'étape 0 avant T11. Les étapes 1 à 3 peuvent se faire pendant que la
campagne tourne. Les étapes 4 et 5 après T12, quand les chiffres sont définitifs — une démo
construite sur des chiffres faux est pire que pas de démo.

---

## 8. Pièges

- **Le chemin daté des trajectoires** (§6.4) : `make_outfolder` utilise la date du jour, pas
  le `launch_id`. Ne jamais reconstruire ce chemin par convention — passer par
  `artifacts.yaml`, ou à défaut par un `glob` qui tolère plusieurs dates.
- **Les ATE `NaN`** : quand l'alignement échoue, `log_results` écrit `nan` et continue. Les
  vues doivent distinguer « scène ratée » de « scène absente » — sinon une méthode qui plante
  partout paraîtra excellente, ses `NaN` disparaissant d'une moyenne.
- **Les trous de tracking EVIMO2** : les frames avec GT ne sont pas toutes les frames
  (bugs 5 et 6). Toujours indexer par **timestamp**, jamais par numéro de frame — c'est
  exactement ce qui a produit les deux bugs.
- **La rectification** (bug 7) : les coordonnées des events cross-dataset sont rectifiées
  côté DEVO. Une superposition masque/events faite sur des coordonnées brutes sera décalée et
  donnera l'impression d'un bug du modèle.
- **Le mode de masque** : afficher un masque GT sans dire son `mask_motion` n'a pas de sens
  (`all` montre 61 % de l'image, `frame` 2 %). L'afficher toujours, lu dans le `config.yaml`
  du run.
- **La mémoire** : EDS dépasse déjà 16 Go en chargeant tous ses events (constaté). L'app doit
  charger **par fenêtre temporelle**, jamais une séquence entière — réutiliser l'approche de
  `tools/check_cross_alignment.py --max-events`.
- **L'invalidation du cache** : `@st.cache_data` doit être clé sur `(chemin, mtime)`, sinon un
  run relancé affichera les anciens chiffres — le genre d'erreur qui fait croire à un
  résultat.
- **Les chemins absolus** : `args.yaml` contient les chemins de la machine qui a évalué. Ne
  jamais les suivre tels quels ; les réinterpréter relativement à `MADEVO_EVALS`.
- **Taille du repo** : les dumps de scores ne doivent **jamais** être commités.
  `evals_out/` est déjà dans `.gitignore` ; vérifier que `scores/` l'est aussi par héritage.

---

## 9. Ce qu'on ne fera pas, et pourquoi

| Écarté | Raison |
|---|---|
| **Lancer l'inférence depuis l'app** | c'est ce qui a tué EVOwithMS et `gradio_demo.py`. L'app rejoue, elle ne calcule pas. Sans exception. |
| **Un viewer 3D web du nuage de points** | coût élevé (Three.js ou équivalent), valeur scientifique faible, `export_map` est à `false` partout. Un rendu offline suffit. |
| **Refaire les courbes d'entraînement** | wandb le fait mieux et c'est déjà branché. Seul le repli hors-ligne depuis `stdout.log` est admis. |
| **Un backend, une base, des comptes** | les artefacts sur disque *sont* la base de données. Tout ajout de service est un coût de maintenance pour une personne seule. |
| **Supporter les 7 datasets dès le départ** | EVIMO2 d'abord (c'est le seul avec masque GT, donc le seul où le replay complet a du sens). Les autres n'ont que trajectoire + score map. |
| **Une app qui sert les trois publics** | §1. Deux interfaces, un socle commun. |
| **Éditer les configs ou relancer des jobs depuis l'app** | les lanceurs existent et sont testés (`launch/`). Une UI qui lance des runs est une UI qui lance de mauvais runs. |

---

## 10. Décisions attendues

1. **Les dumps entrent-ils dans la campagne T11 ?** C'est la seule question urgente. Coût :
   ~1 journée CPU + ~150 Mo par seed évaluée (seed 1 par défaut). Ne pas le faire rend la
   figure teaser du papier impossible sans une campagne supplémentaire.
2. **Sur quelles seeds dumper ?** Proposition : seed 1 seulement pour les scores (les 5 seeds
   pour les tables, qui sont légères). Dumper les 5 coûterait ~750 Mo par table.
3. **Streamlit accepté comme dépendance** de `environment-cpu.yml` ? Sinon le repli est une
   page statique générée, sans interactivité — nettement moins utile pour le débogage.
4. **La démo publique est-elle dans le périmètre avant les candidatures**, ou après T12 ?
   Elle dépend de chiffres définitifs ; la faire trop tôt veut dire la refaire.
5. **`ms_model/viz/gradio_demo.py` : supprimé maintenant** (il ne tourne plus) ou gardé
   jusqu'à ce que l'app le remplace ? Recommandation : supprimé à l'étape 0, pour qu'il ne
   serve pas de modèle.
