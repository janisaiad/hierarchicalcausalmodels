# Résultats préliminaires — découverte causale STAR (`star_causal_discovery`)

Synthèse après exécution complète de `star_causal_discovery.py` et agrégation des graphes disponibles.

## Ce qui a réellement tourné

- **causal-learn** : PC, FCI, DirectLiNGAM, ExactBIC — OK sur $n = 4000$.
- **CDT** : les 11 méthodes testées échouent ici (`Rscript` absent) → aucun graphe CDT à mélanger.
- **DoWhy** : ATE avec le DAG ExactBIC seulement (pas un quatrième graphe indépendant).
- **SAM (PyTorch)** : lancé à part pour un quatrième avis ; le `DiGraph` renvoyé a **42 arcs** sur 7 nœuds (= toutes les paires ordonnées), donc **pas un DAG utilisable** pour un consensus — **exclu** du « graphe moyen ».

En pratique, le **graphe moyen** se calcule sur **trois** méthodes : **PC** (arcs orientés du CPDAG), **LiNGAM**, **ExactBIC**.

## Vote majoritaire sur les arcs dirigés

| Votes | Arcs |
|--------|------|
| **3/3** | `ethnicity → lunchk`, `gender → readk`, `schoolk → lunchk` |
| **2/3** | `ethnicity → mathk`, `ethnicity → schoolk`, `lunchk → readk`, `mathk → lunchk`, `mathk → readk`, `schoolk → readk` |
| **1/3** | dont `stark → readk` (ExactBIC seul), `stark → mathk` (LiNGAM seul), `schoolk → stark` (ExactBIC seul), `readk → mathk` (ExactBIC seul, sens contraire aux deux autres), etc. |

## Conflits d’orientation (important pour un « DAG moyen »)

- **`mathk` ↔ `readk`** : 2 méthodes en `mathk → readk` (PC, LiNGAM), 1 en `readk → mathk` (ExactBIC). Une règle de **majorité** donnerait **`mathk → readk`**.
- **`mathk` ↔ `lunchk`** : 2 en `mathk → lunchk`, 1 en `lunchk → mathk` (LiNGAM) → majorité **`mathk → lunchk`**.

## Graphe causal « moyen » proposé (interprétation)

### Noyau très robuste (3/3)

Cohérent avec une lecture **socio-école → déjeuner / scores** :

- origine ethnique → programme déjeuner  
- genre → lecture  
- école (`schoolk`) → programme déjeuner  

### Couche majoritaire (≥ 2/3), en résolvant les deux conflits au vote

- `ethnicity → mathk`, `ethnicity → schoolk`  
- `mathk → lunchk`, `mathk → readk`  
- `lunchk → readk`, `schoolk → readk`  

### Traitement `stark`

Aucun arc **2/3** ou **3/3** ; seulement des **hypothèses isolées** (surtout ExactBIC / LiNGAM). Pour une **lecture causale du RCT**, il ne faut **pas** s’appuyer sur ce graphe appris : il faut le **design** (randomisation).

### Synthèse en une phrase

Le graphe moyen des algorithmes sur ce tableau encodé ressemble à un **bloc socio-démographique + école → `lunchk` → chemins vers scores**, avec **`mathk` et `readk` fortement couplés** (orientation **`mathk → readk`** si l’on impose une majorité), et **peu de consensus sur `stark`**.

### Piste d’implémentation

Pour écrire automatiquement ce « graphe moyen » dans `causal_discovery_star.json` (votes 2/3, 3/3, exclusion SAM), on peut l’ajouter dans `star_causal_discovery.py` dans une prochaine passe.

---

## Variable `schoolk` (documentation STAR)

D’après la documentation du jeu STAR dans le dépôt (`data/star/STAR_doc.html`, reprise du package R **AER**) :

**`schoolk`** est une variable **catégorielle** qui indique le **type de zone / d’implantation de l’école en maternelle** (*kindergarten*), avec les modalités :

- `inner-city` (centre-ville)  
- `suburban` (banlieue)  
- `rural`  
- `urban` (urbain)  

Ce n’est **pas** l’identifiant d’une école précise : pour l’ID école en maternelle, le jeu prévoit une autre variable, **`schoolidk`**.

Dans le notebook de découverte causale, `schoolk` est encodée en entiers avec `LabelEncoder` : les codes **0, 1, 2, 3** correspondent à un **ordre arbitraire** des quatre catégories, **pas** à une échelle métrique.



## feedback

Le document est utile comme **compte rendu d'exécution** et comme **synthèse exploratoire** des sorties de `star_causal_discovery.py`. Il a plusieurs qualités importantes:

- il distingue clairement ce qui a tourné de ce qui n'a pas tourné
- il explicite que `DoWhy` n'ajoute pas un graphe indépendant mais exploite le DAG `ExactBIC`
- il exclut à juste titre `SAM` du consensus lorsque la sortie n'est pas un DAG exploitable
- il note correctement que, pour le traitement `stark`, une lecture causale sérieuse doit s'appuyer d'abord sur le **design randomisé** de `STAR`, et non sur un graphe appris

Sur le fond, le texte me paraît donc bon comme **note de travail interne**. En revanche, il est actuellement **trop affirmatif** si on veut lui donner un statut de résultat causal stabilisé.

### Point méthodologique principal

Le principal problème est l'expression **« graphe causal moyen »**. En l'état, ce n'est pas vraiment un objet causal standard. C'est plutôt un **résumé heuristique** construit à partir de plusieurs sorties de méthodes de découverte causale ayant:

- des hypothèses différentes
- des objets de sortie différents
- des statuts orientationnels différents

Par exemple:

- `PC` produit un **CPDAG**, donc certaines orientations sont partiellement identifiées seulement
- `FCI` produit un **PAG**, qui admet des variables latentes et ne joue pas le même rôle qu'un DAG
- `DirectLiNGAM` impose un modèle linéaire à bruit non gaussien
- `ExactBIC` cherche un DAG score-based sous hypothèse i.i.d.

Autrement dit, faire un **vote majoritaire sur des arcs orientés** est acceptable comme **résumé exploratoire**, mais il ne faut pas le présenter comme si l'on obtenait un DAG consensuel rigoureux au sens causal.

### Deuxième limite importante: l'encodage des variables catégorielles

Le notebook encode plusieurs variables catégorielles avec `LabelEncoder`, notamment:

- `stark`
- `gender`
- `ethnicity`
- `lunchk`
- `schoolk`

Cela signifie que des catégories nominales sont transformées en entiers `0, 1, 2, ...`, ce qui induit artificiellement une structure d'ordre et de distance. Pour des méthodes comme `PC` avec test gaussien, `FCI` avec test gaussien, ou `LiNGAM`, cela rend l'interprétation des arcs encore plus délicate.

En particulier, il faut être très prudent avec des lectures du type:

- `ethnicity → schoolk`
- `mathk → lunchk`
- `schoolk → readk`

Ces flèches peuvent refléter:

- une vraie structure de dépendance
- un artefact d'encodage
- un compromis imposé par les hypothèses des algorithmes

Elles sont donc intéressantes comme **signal exploratoire**, mais pas comme conclusion causale robuste.

### Point fort du document

Le passage sur `stark` est, à mon sens, le plus juste. Le fait qu'aucun arc robuste ne relie fortement `stark` aux scores dans le consensus appris n'est **pas** un argument contre l'effet causal du traitement. Cela dit surtout que:

- les algorithmes de découverte causale utilisés ici ne retrouvent pas bien le traitement expérimental sur ce tableau encodé
- l'information causale principale sur `stark` provient du **protocole expérimental STAR**

C'est une distinction très importante. Pour `STAR`, le bon point de départ causal n'est pas la structure apprise par discovery, mais la randomisation connue du traitement.

### Comment je reformulerais l'interprétation

Je remplacerais mentalement:

- **« graphe causal moyen »**

par:

- **« résumé de consensus exploratoire des graphes appris »**

Et je remplacerais:

- **« noyau très robuste »**

par quelque chose comme:

- **« arêtes orientées retrouvées par toutes les méthodes orientées retenues, sous les hypothèses et approximations du prétraitement »**

Cela paraît plus lourd, mais c'est beaucoup plus correct.

### Ce que le document permet malgré tout d'affirmer

À mon avis, on peut garder les messages suivants:

- les algorithmes convergent vers un bloc de dépendances entre variables socio-démographiques, environnement scolaire, déjeuner et scores
- `mathk` et `readk` sont structurellement très liés dans toutes les sorties
- il y a peu de consensus algorithmique direct sur `stark`
- les résultats sont sensibles aux choix de représentation et ne doivent pas remplacer l'interprétation causale fondée sur le design expérimental

### Conclusion

Mon avis global est donc le suivant: c'est un **bon document exploratoire**, clair et déjà utile, mais il doit être lu comme une **note d'analyse algorithmique** et non comme une validation causale du graphe sous-jacent à `STAR`. Sa meilleure valeur est de montrer:

- ce que les méthodes de causal discovery voient ou ne voient pas
- à quel point les résultats dépendent des hypothèses algorithmiques
- pourquoi, dans `STAR`, le design expérimental reste plus crédible que le graphe appris pour raisonner causalement sur `stark`

Si ce document doit servir dans un texte plus académique, je recommanderais de renforcer explicitement trois avertissements:

- le consensus n'est pas un DAG causal identifié
- l'encodage des variables catégorielles limite fortement l'interprétation des arcs
- les conclusions sur le traitement `stark` doivent rester ancrées dans le RCT, pas dans la discovery

---

## Comparaison avec `sota.md`

Comparé à `sota.md`, ce document a une force différente. `sota.md` situe `STAR` dans la littérature causale existante, rappelle les résultats déjà établis sur les petites classes, les effets de long terme, les effets de pairs et les limites du protocole. `preliminar_results.md`, lui, ne fait pas de revue de littérature: il documente ce que produisent concrètement plusieurs algorithmes de découverte causale sur un sous-tableau encodé de `STAR`.

Les deux documents sont donc complémentaires:

- `sota.md` répond à la question **« que sait-on déjà causalement sur STAR ? »**
- `preliminar_results.md` répond à la question **« que retrouvent ici nos algorithmes de causal discovery sur cette représentation des données ? »**

### Ce qui est bien dans `preliminar_results.md` par rapport à `sota.md`

Par rapport à `sota.md`, ce document a plusieurs qualités spécifiques.

#### 1. Il apporte des résultats empiriques concrets

Là où `sota.md` donne le contexte scientifique général, `preliminar_results.md` montre ce qui se passe effectivement quand on exécute les méthodes sur les données disponibles. Il donne:

- les méthodes réellement exécutées
- les méthodes qui échouent
- les arcs retrouvés
- les conflits d'orientation
- les limites pratiques du pipeline

De ce point de vue, il transforme le discours général de l'état de l'art en observation expérimentale locale.

#### 2. Il est transparent sur le pipeline

Le document est honnête sur le fait que:

- `CDT` n'a pas tourné faute de `Rscript`
- `SAM` n'est pas retenu
- `DoWhy` n'est pas une quatrième source indépendante

Cette transparence est précieuse. `sota.md` explique pourquoi `STAR` est important; `preliminar_results.md` montre ce qui est réellement reproductible dans cet environnement précis.

#### 3. Il fait apparaître une dissociation utile entre discovery et identification

Un point très fort est que le document montre implicitement quelque chose d'intéressant: les algorithmes de découverte causale ne retrouvent pas clairement `stark`, alors même que la littérature causale sur `STAR` attribue une importance centrale au traitement expérimental. Cette tension est intellectuellement utile, car elle rappelle qu'un bon design causal et un bon graphe appris ne sont pas la même chose.

#### 4. Il aide à préparer un futur travail méthodologique

Comme note de travail, ce document est plus directement actionnable que `sota.md`. Il permet de voir:

- quelles méthodes comparer ensuite
- quelles sorties conserver
- quels objets sérialiser
- où renforcer le pipeline

Autrement dit, `sota.md` aide à poser la motivation scientifique; `preliminar_results.md` aide à organiser la suite technique.

### Ce qui est à revoir dans `preliminar_results.md` à la lumière de `sota.md`

En revanche, la comparaison avec `sota.md` fait apparaître plusieurs faiblesses.

#### 1. Le document n'est pas encore assez bien repositionné par rapport à la littérature

`sota.md` rappelle que `STAR` est un dataset où l'effet causal principal de la taille de classe est déjà très documenté. Du coup, `preliminar_results.md` devrait expliciter davantage que son objectif n'est **pas** de redécouvrir à lui seul la causalité de `STAR`, mais d'évaluer ce que des méthodes de discovery récupèrent ou ratent sur une représentation simplifiée du dataset.

Sans cette mise au point, un lecteur peut croire que le document cherche à infirmer ou confirmer directement la littérature causale existante, ce qui serait trop ambitieux compte tenu du pipeline utilisé.

#### 2. Le statut du « graphe moyen » reste trop fort

Vu ce que `sota.md` rappelle sur la solidité du design expérimental et la complexité réelle de `STAR`, la formule **« graphe causal moyen »** paraît encore plus problématique. Elle suggère un objet causal relativement stabilisé, alors qu'il s'agit plutôt d'un résumé empirique de plusieurs méthodes appliquées à un tableau prétraité et encodé.

Le document gagnerait donc à adopter systématiquement une formulation du type:

- **consensus exploratoire**
- **résumé des graphes appris**
- **signal algorithmique partagé**

plutôt que de parler d'un DAG moyen quasi interprétable causalement.

#### 3. L'analyse ne relie pas encore assez ses résultats aux faits connus sur `STAR`

Par exemple, `sota.md` rappelle que:

- l'effet causal de la petite classe est bien documenté
- les effets de pairs sont plausibles et étudiés
- l'attrition et la non-conformité existent
- la structure hiérarchique est importante

`preliminar_results.md` devrait donc commenter davantage ses propres résultats à cette lumière. En particulier:

- le faible consensus sur `stark` n'est pas une surprise décisive
- il peut refléter les limites de la discovery sur des données encodées, plutôt qu'une absence de signal causal
- les liens trouvés autour de `schoolk`, `lunchk`, `ethnicity`, `readk`, `mathk` sont plausibles descriptivement, mais doivent être rapprochés de la littérature avant toute interprétation forte

#### 4. La dimension hiérarchique est presque absente

C'est probablement le point le plus important si l'on lit `preliminar_results.md` à la lumière de `sota.md`. L'état de l'art insiste sur le fait que `STAR` est naturellement structuré par niveaux:

- élèves
- classes / enseignants
- écoles
- éventuellement temps

Or ici, l'analyse travaille sur un tableau réduit, au niveau maternelle, avec peu de variables, sans traitement explicite de la hiérarchie. Ce n'est pas un défaut pour une première exploration, mais cela devrait être dit plus explicitement. Sinon, on risque de sur-vendre la portée structurelle des graphes appris.

#### 5. Il manque une conclusion comparative explicite

Le document gagnerait à dire noir sur blanc quelque chose comme:

> Cette analyse ne contredit pas l'état de l'art causal sur `STAR`; elle montre plutôt ce que des méthodes de découverte causale retrouvent partiellement, manquent, ou orientent de façon instable lorsqu'elles sont appliquées à une représentation simplifiée, encodée et non hiérarchique du problème.

Cette phrase ferait un pont clair entre `sota.md` et `preliminar_results.md`.

### Conclusion comparative

En résumé:

- `sota.md` est meilleur pour **situer**, **justifier** et **problématiser**
- `preliminar_results.md` est meilleur pour **documenter**, **tester** et **montrer les limites concrètes du pipeline**

Le premier document dit pourquoi `STAR` est intéressant causalement. Le second montre ce qui se passe quand on applique des outils de causal discovery sur une version simplifiée des données. La bonne lecture n'est donc pas de les opposer, mais de les articuler:

- `sota.md` fournit le cadre scientifique
- `preliminar_results.md` fournit le retour expérimental

Si `preliminar_results.md` est révisé avec plus de prudence sur l'interprétation causale, plus de liens explicites avec la littérature, et une mention plus claire de la perte de structure hiérarchique, alors l'ensemble des deux documents devient beaucoup plus solide.