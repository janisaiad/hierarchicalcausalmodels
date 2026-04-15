# Synthèse (discussion) — pipeline HCM, ATE, warnings, papier

Ce document regroupe les points techniques discutés sur l’estimation causale hiérarchique (STAR / librairie), les warnings `UserWarning: No data for outcome …; skipping term`, le calcul de l’ATE, et le lien avec le formalisme du papier HCM.

---

## 1. Warning « No data for outcome […]; skipping term »

### Signification

À l’étape de **fit des densités** dans `ast_to_estimator` / `estimate_causal_effect`, pour chaque terme de la formule d’identification de type \(P(\text{outcome} \mid \text{parents})\), le code résout le nom de l’outcome dans le dictionnaire `enriched` via `resolve(out_vars[0])`. Si **aucune** clé ne correspond (notation papier ni version sanitizée), un `UserWarning` est émis et **aucun estimateur** n’est associé à ce terme.

### Cause fréquente : \(Q\) conditionnels « multi-parents » (historique)

- La formule (LaTeX / pyAgrum) peut référencer des symboles du type \(Q^{y \mid g,l,m}\) (sanitized : `Qy_g_l_m`, via `_sanitize` dans `causal_estimators.py`).
- **Avant correctif** : `_precompute_conditional_q_vars` ne gérait que le motif **binaire** `Q` + outcome + `_` + **un** parent (`Qy_a` ↔ \(Q^{y \mid a}\)). Les symboles multi-segments (`Qy_g_l_m`, `Qm_a_e_g_l`, …) n’étaient pas précalculés.
- Conséquence : pas de colonne dans `enriched` → `resolve` échoue → warning → facteur **1** à l’évaluation → autre fonctionnelle \(\tilde{\mathfrak{F}}\) que \(\mathfrak{F}\) (voir ci-dessous).

Autres causes possibles (toujours valides) : variable absente des données d’entrée ; décalage de noms entre formule et clés.

### Correction code (multi-parent) — où on en est

**Fichier** : `src/hierarchicalcausalmodels/estimation/causal_estimators.py` — `_precompute_conditional_q_vars`, helpers `_find_subunit_matrix_for_letter`, `_fit_conditional_q_multiparent_row_task`.

**Comportement ajouté** : pour tout symbole `Q[outcome]_[p1]_[p2]_…` (ex. `Qy_g_l_m`), le code

1. retrouve la matrice sous-unitaire de l’outcome (clé dont le nom commence par la lettre d’outcome, ex. `Y`) ;
2. retrouve une matrice par parent (`G`, `L`, `M`, …) ;
3. empile les parents en \(X\) de forme \((m_i, p)\) par unité \(i\) ;
4. ajuste un `ConditionalDensityEstimator` sur les sous-unités de l’unité, puis stocke **un scalaire par unité** = moyenne sur \(j\) des \(\mathbb{E}[Y \mid X_{ij}]\) ;
5. enregistre la colonne sous le **sanitized** (`Qy_g_l_m`) et la notation papier (`Q^{y|g,l,m}` avec virgules dans les indices du `|`).

**Ce que ça règle** : plus de colonne manquante pour ces motifs → disparition des warnings associés **dans ce cas** → les facteurs \(P(\cdot\mid\cdot)\) correspondants peuvent être fittés au lieu de valoir **1** ; l’ATE se rapproche du **plug-in de la formule complète** renvoyée par pyAgrum.

**Ce que ça ne remplace pas (papier HCM au sens strict)** : la construction théorique de \(Q^{v \mid \mathrm{pa}}\) (intégration sur le bruit sous-unitaire \(\epsilon\), loi sur l’espace des distributions, etc., voir `paper.tex`) n’est pas recodée intégralement. Le scalaire stocké est un **résumé régressionnel / espérance conditionnelle** par classe, pas l’objet distribution-valued complet du manuscrit. Pour la publication théorique stricte, il faudrait soit aligner la définition sur le papier, soit justifier ce proxy dans le texte méthodo.

**Limites techniques actuelles** :

- Si le symbole multi-parent est aussi une **variable de sommation** (`\Sigma_{Q^{…}}`), il n’y a pas encore de profil 2D « une colonne par valeur de parent » : le code **repli** sur le **même résumé scalaire** multi-parent que hors somme (approximation explicite ; évite l’absence totale de colonne).
- Le chemin **Torch** batchisé utilisé pour `Qy_a` (un parent) n’est pas réutilisé pour le multi-parent ; l’exécution passe par le **chemin NumPy** (`parallel_map` + `ConditionalDensityEstimator` par unité).

**Vérification** : relancer `estimate_causal_effect` / `star_hcm_v2_teacher_student.py` et contrôler l’absence de `UserWarning: No data for outcome …` sur les termes concernés ; comparer ATE avant/après pour la sensibilité numérique.

### Rerun après correctif (`star_hcm_v2_teacher_student.py`, outcome **Y**)

Commande :  
`uv run python examples/STAR/star_hcm_v2_teacher_student.py --graphs DirectLiNGAM,ExactBIC --outcome Y`  
(merge dans `examples/STAR/results/star_hcm_v2_teacher_student.json`).

| Graphe       | ATE (`E_do_1 - E_do_0`) | `E_do_1` | `E_do_0` | Warnings « No data for outcome » |
|-------------|-------------------------:|---------:|---------:|----------------------------------|
| DirectLiNGAM | **4308.53** (anormalement élevé vs échelle score ~10²–10³) | ≈ 2.89×10⁵ | ≈ 2.85×10⁵ | **Aucun** sur ce run |
| ExactBIC   | **−18.46** | ≈ 1040.0 | ≈ 1058.5 | **Aucun** sur ce run |

**Lecture** : l’absence de warnings indique que les colonnes multi-parents sont bien présentes et que les facteurs ne sont plus remplacés par **1** de ce côté. Les ordres de grandeur **DirectLiNGAM** (potentiels ~10⁵) restent **suspects** (produit de densités / GMM sur \(Q\) dérivés, stabilité numérique, repli Σ) — à diagnostiquer avant de tirer conclusion sur l’effet. **ExactBIC** est dans une plage plus cohérente avec des scores de lecture agrégés. Les ATE « historiques » (~17 et ~21) avec warnings correspondaient à une **autre** fonctionnelle \(\tilde{\mathfrak{F}}\) ; les nouveaux chiffres visent \(\mathfrak{F}\) plug-in **sans** facteurs manquants, mais avec les approximations ci-dessus.
 et 
### Effet sur l’évaluation numérique

Si aucun estimateur n’est enregistré pour un nœud `_ASTConditional`, `_eval_formula` retourne le facteur **constant 1.0** pour ce nœud (produit global inchangé **multiplicativement** par ce facteur « neutre » au lieu du vrai noyau).

Ce n’est **pas** un bug aléatoire : c’est une **limitation d’implémentation** (précalcul \(Q\) incomplet par rapport aux symboles que l’identification peut produire), avec impact sur la **fidélité** du plug-in par rapport à la formule théorique complète.

### Nature fondamentale (papier) : quel **type** d’erreur ?

Le point important n’est **pas** seulement « un warning Python » : c’est une question de **quelle quantité** est réellement calculée.

**Ce que dit le papier (identification).**  
Sous les hypothèses du modèle (graphe HCM, collapse, augment, marginalisation, puis do-calcul), il existe une **fonctionnelle** \(\mathfrak{F}\) de la loi observée \(\mathbb{P}\) telle que l’estimande causal vaut

$$
\tau \;=\; \mathfrak{F}(\mathbb{P}) \,,
$$

typiquement un **produit** de noyaux conditionnels \(\mathbb{P}(V \mid \mathrm{pa}(V))\) (ou espérances conditionnelles quand l’outcome n’est pas fixé), **sommé ou intégré** sur les variables d’ajustement — la formule pyAgrum / LaTeX est une représentation finie de cette \(\mathfrak{F}\).

**Ce que fait le code quand un terme saute.**  
Un facteur entier de \(\mathfrak{F}\) est **retiré** du produit et **remplacé par la constante \(1\)**. On n’obtient plus \(\mathfrak{F}(\mathbb{P})\) mais une **autre** fonctionnelle

$$
\tilde{\tau} \;=\; \tilde{\mathfrak{F}}(\mathbb{P}) \,,
$$

où \(\tilde{\mathfrak{F}}\) est la formule **tronquée** (même arbre syntaxique, mais certains facteurs forcés à \(1\)).

**Classification.**

| Ce que ce n’est **pas** | Ce que c’est |
|-------------------------|--------------|
| Erreur de **Monte Carlo** (variance \(\sim 1/\sqrt{K}\)) | Non : la cible \(\tilde{\tau}\) est déjà fausse **avant** le tirage. |
| **Biais** d’un estimateur **consistant** de \(\tau\) (converge vers \(\tau\) quand \(n\to\infty\)) | Non : avec des données infinies, un plug-in parfait de \(\tilde{\mathfrak{F}}\) converge vers \(\tilde{\tau}\), pas vers \(\tau\). |
| Erreur de **spécification** d’**un** modèle (ex. mauvaise famille Gaussienne pour un facteur présent) | Partiellement différent : là, au moins la **structure** \(\mathfrak{F}\) est celle du papier ; ici la structure est **cassée** (facteur absent). |

**Nom correct au sens théorique** : **non-identification numérique de la fonctionnelle identifiée** — ou **estimation d’une autre fonctionnelle** que celle prouvée égale à l’effet causal. Ce n’est pas « l’effet causal avec du bruit » : c’est en général **un autre paramètre** (souvent sans interprétation causal claire sous le même théorème).

**Le papier « foire-t-il » ?**  
Non : les **énoncés** d’identification et les **formules** du manuscrit restent valides **si** l’on implémente exactement \(\mathfrak{F}\). Ce qui peut faillir, c’est l’**alignement implémentation ↔ formule symbolique** : symboles \(Q^{v\mid\mathrm{pa}}\) présents dans la formule mais **jamais construits** dans les données enrichies → le logiciel **ne calcule pas** \(\mathfrak{F}\), il calcule \(\tilde{\mathfrak{F}}\). La **responsabilité** est côté pipeline logiciel (précalcul / noms / colonnes), pas côté contradiction interne du papier.

**Conséquence pour l’ATE.**  
Si \(\tau(q^{a}_{(1)})\) et \(\tau(q^{a}_{(0)})\) sont tous deux remplacés par \(\tilde{\tau}(\cdot)\) avec les **mêmes** facteurs manquants, la **différence** peut parfois rester « proche » de l’ATE vrai (erreurs partiellement communes), mais **sans garantie théorique** ; si l’intervention change quels chemins ou quels symboles sont actifs, le biais sur la différence peut être **arbitraire**.

---

## 2. ATE : sous-estimation ou sur-estimation ?

**Pas de réponse universelle.**

- Remplacer un facteur manquant par **1** dans un **produit** change la valeur par rapport à \(\prod_j S_j\) si les vrais facteurs \(S_j\) n’étaient pas 1.
- Si les facteurs sont des **probabilités** dans \((0,1)\), mettre 1 à la place **gonfle** typiquement ce morceau du produit.
- En **continu**, les facteurs sont souvent des **densités** (peuvent être &gt; 1 ou &lt; 1) : le signe de l’erreur sur ce facteur n’est pas fixe.
- L’**ATE** est en général une **différence** (ou parfois un rapport) entre deux scénarios \(\mathrm{do}(\cdot)\). Le biais sur l’ATE dépend de la **symétrie** (ou non) des termes absents entre les deux bras et de la structure de la formule ; il faut analyser **quel** terme saute et **pour quels** niveaux d’intervention.

---

## 3. Pourquoi « aucune donnée trouvée » ?

Ici « pas de données » signifie : **pas de colonne dans `enriched`** dont le nom résolu correspond à l’**outcome** du terme conditionnel — pas que le fichier de données est vide.

---

## 4. Calcul de l’ATE : code vs formalisme du papier (`refs/hcmpaper/paper.tex`)

### Notation du papier (macros utiles)

- Espérances : \(\mathbb{E}[\cdot]\), \(\mathbb{E}_{\mathbb{P}}[\cdot]\), \(\mathbb{E}_{Q}[\cdot]\) / \(\mathbb{E}_{Q^{y}}[Y]\).
- Intervention : \(\mathrm{do}(q^{a} = q^{a}_{\star})\) (ou masses \(\delta_0\), \(\delta_1\) pour un traitement binaire).

### Quantité cible pour **une** intervention

Exemple **modèle augmenté confounder** : l’estimande s’écrit (double espérance, intervention sur \(q^{a}\)) :

\[
\mathbb{E}_{\mathbb{P}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{\star})\right].
\]

L’identification passe par la loi de \(q^{y}\) après \(\mathrm{do}\) (backdoor sur le graphe effondré/augmenté), avec le lien déterministe \(q^{y} = \mathfrak{m}(q^{a}, q^{y \mid a})\) et une intégrale sur \(q^{y \mid a}\) (équations type `eq:collapse_id`, `eqn:deterministic_qy_mech`, `eq:augmented_estimand` dans le `.tex`).

### ATE binaire dans le papier

\[
\mathbb{E}_{\mathbb{P}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = \delta_{1})\right]
-
\mathbb{E}_{\mathbb{P}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = \delta_{0})\right]
\]

(`eq:confounder_effect`).

### Ce que fait la librairie

- **`estimate_causal_effect` / `ast_to_estimator`** : un appel retourne l’estimation d’**un seul** potentiel pour l’**intervention donnée** (un \(q^{a}_{\star}\) ou équivalent). L’**ATE** se construit en pratique par **deux appels** (deux niveaux d’intervention) puis **différence** (éventuellement rapport selon le contexte), pas par un seul float « ATE » interne.
- **Formule** : l’AST pyAgrum (ou parse LaTeX) donne une fonctionnelle \(\mathcal{F}\) (produits, sommes, facteurs \(P(\cdot\mid\cdot)\)). Chaque facteur est estimé (**plug-in** : \(\mathbb{P} \mapsto \hat{\mathbb{P}}\)) puis la formule est évaluée récursivement.
- **Sommes** : discrètes énumérées ; continues marginalisées par **Monte Carlo** (`n_mc_samples`).
- **Conditionnels** : si l’outcome n’est **pas** dans le contexte unitaire, l’évaluateur utilise **\(\mathbb{E}[Y \mid X=x]\)** ; sinon densité/probabilité ponctuelle — cohérent avec des espérances imbriquées type \(\mathbb{E}_{Q^{y}}[Y]\).
- **Intervention** : valeurs injectées dans `context` ; un scalaire sert aussi au **précalcul** des \(Q^{v\mid u}\) compatibles (`_iv_scalar`). Pour une intervention scalaire anonyme, seul ce précalcul peut porter la valeur si les clés nommées ne sont pas passées en dict.
- **Moyenne sur unités** : avec \(n\) unités,
  \[
  \hat{\tau}(q^{a}_{\star}) = \frac{1}{n}\sum_{i=1}^{n} \widehat{\mathcal{F}}_{i}(q^{a}_{\star}),
  \]
  où \(\widehat{\mathcal{F}}_{i}\) est l’évaluation au **contexte de l’unité \(i\)** (données enrichies + intervention). Esprit proche de la moyenne empirique des \(\hat{\mu}^{y}_{i}\) du papier (`eq:confounder_estimator`), avec nuance d’implémentation : le papier décrit aussi des modèles **par unité** pour \(\hat{q}^{y\mid a}_{i}\) ; le code ajuste souvent des estimateurs **piscinés** sur tout l’échantillon et les évalue au covariable de \(i\).

### Termes manquants et formule

Tout facteur non fitté compte comme **1** dans le produit : \(\widehat{\mathcal{F}}\) n’est plus le plug-in de la formule **complète** identifiée théoriquement.

---

## 5. Autres éléments de contexte (session / dépôt)

- **Device Torch / JAX** : défauts possibles via variables d’environnement et helpers (`default_torch_device_str`, etc.) ; tests et benchmarks GPU vs CPU / Numba.
- **Artefacts** : `return_artifacts=True` renvoie `enriched`, `fitted`, formule, etc. ; le précalcul \(Q\) dépend de l’intervention → **un fit / export par niveau** \(\mathrm{do}\) si l’on veut des artefacts complets comparables.
- **Fichiers STAR / JSON** : régénérer les exports si colonnes ou familles (ex. GMM) manquent par rapport au code actuel.
- **Permissions** : certains fichiers peuvent être en `root:root` et bloquer les sauvegardes IDE sans `chown`.

---

## 6. Références rapides dans le code

- Warning + skip fit : `causal_estimators.py` (boucle sur `unique_terms`, `resolve(out_vars[0])`, `warnings.warn`).
- Facteur 1 si pas d’estimateur : `_eval_formula`, branche `_ASTConditional`, `if est is None: return 1.0`.
- Précalcul \(Q\) restreint : `_precompute_conditional_q_vars`, regex `^Q([a-zA-Z]+)_([a-zA-Z]+)$`.
- Moyenne sur unités : `ast_to_estimator`, fin de fonction (`np.mean(unit_vals)`).
- Papier : `refs/hcmpaper/paper.tex` — estimandes `eq:confounder_estimand`, `eq:augmented_estimand`, identification `eq:collapse_id`, estimateur moyenne `eq:confounder_estimator`, ATE `eq:confounder_effect`.

---

## 7. Formalisme papier : formules « complètes » avec toutes les intégrales intérieures

Références : `refs/hcmpaper/paper.tex` (sections identification, graphe **confounder**, graphe **confounder \& interference**, appendice augmentation). On note $\mathrm{pr}$ comme dans le papier pour les lois observées / conditionnelles ; $\mathcal{A}$, $\mathcal{Y}$ les domaines de $a$ et $y$ ; les intégrales sur $q^{a}$, $q^{y|a}$, etc. sont prises sur l’espace des distributions (mesure image induite par le modèle effondré), comme dans les écritures intégrales du texte.

### 7.1 HCGM général (une unité, sous-unités)

Pour tout $v \in \mathcal{S}$ (variables sous-unitaires), le papier écrit le mécanisme hiérarchique (éq. `eqn:hcgm`) :

$$
Q^{v \mid \mathrm{pa}_{\mathcal{S}}(v)}_{i} \sim \mathrm{pr}\!\left(q^{v \mid \mathrm{pa}_{\mathcal{S}}(v)} \,\middle|\, x^{\mathrm{pa}_{\mathcal{U}}(v)}_{i}\right), \qquad
X^{v}_{ij} \sim q^{v \mid \mathrm{pa}_{\mathcal{S}}(v)}_{i}\!\left(x^{v} \,\middle|\, x^{\mathrm{pa}_{\mathcal{S}}(v)}_{ij}\right),
$$

et pour $v \in \mathcal{U}$ (variables unitaires) :

$$
X^{v}_{i} \sim \mathrm{pr}\!\left(x^{v} \,\middle|\, x^{\mathrm{pa}_{\mathcal{U}}(v)}_{i}, \left\{x^{\mathrm{pa}_{\mathcal{S}}(v)}_{ij}\right\}_{j=1}^{m}\right).
$$

### 7.2 Estimande de base : espérance interne sur les sous-unités

Pour une sous-unité $Y$, l’espérance **à l’intérieur** d’une unité, sous la loi jointe $Q(a,y)$ des sous-unités (papier, autour de `eq:confounder_estimand`) :

$$
\mathbb{E}_{Q}[Y] \;=\; \iint_{\mathcal{Y}\times\mathcal{A}} y \, Q(a, y) \, \mathrm{d}a \, \mathrm{d}y \;=\; \int_{\mathcal{Y}} y \, Q^{y}(y) \, \mathrm{d}y,
$$

avec la marginale $Q^{y}(y) = \int_{\mathcal{A}} Q(a,y)\,\mathrm{d}a$.

### 7.3 Lien déterministe $q^{y} = \mathfrak{m}(q^{a}, q^{y \mid a})$ (augmentation)

Le papier définit (éq. `eqn:deterministic_qy_mech`) :

$$
q^{y}(y) \;=\; \int_{\mathcal{A}} q^{a}(a) \, q^{y \mid a}(y \mid a) \, \mathrm{d}a \;\triangleq\; \mathfrak{m}\!\left(q^{a}, q^{y \mid a}\right)(y).
$$

Donc, pour une **intervention** qui fixe $q^{a} = q^{a}_{\star}$ :

$$
\mathbb{E}_{Q^{y}}[Y] \;=\; \int_{\mathcal{Y}} y \, \mathfrak{m}\!\left(q^{a}_{\star}, q^{y \mid a}\right)(y) \, \mathrm{d}y \;=\; \int_{\mathcal{Y}} \int_{\mathcal{A}} y \, q^{a}_{\star}(a) \, q^{y \mid a}(y \mid a) \, \mathrm{d}a \, \mathrm{d}y,
$$

expression valable pour un $q^{y \mid a}$ **donné** (réalisation unitaire).

### 7.4 Graphe **confounder** (backdoor) : loi de $q^{y}$ après $\mathrm{do}(q^{a} = q^{a}_{\star})$ puis espérence externe

**Étape identification** (`eq:collapse_id`) :

$$
\mathrm{pr}\!\left(q^{y} \,\middle|\, \mathrm{do}(q^{a} = q^{a}_{\star})\right) \;=\; \int \mathrm{pr}(q^{y \mid a}) \, \mathrm{pr}\!\left(q^{y} \,\middle|\, q^{a}_{\star}, q^{y \mid a}\right) \, \mathrm{d}q^{y \mid a} \;=\; \int \mathrm{pr}(q^{y \mid a}) \, \mathfrak{m}\!\left(q^{a}_{\star}, q^{y \mid a}\right) \, \mathrm{d}q^{y \mid a},
$$

la dernière égalité venant du fait que $q^{y}$ est une fonction **déterministe** de $(q^{a}_{\star}, q^{y \mid a})$.

L’**estimande augmenté** (`eq:augmented_estimand`) est $\mathbb{E}_{\mathrm{pr}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{\star})\right]$. Comme $\mathbb{E}_{Q^{y}}[Y] = \phi(q^{y})$ avec $\phi(q^{y}) = \int_{\mathcal{Y}} y\, q^{y}(y)\,\mathrm{d}y$, et sous la loi identifiée la masse sur $q^{y}$ est portée par $\mathfrak{m}(q^{a}_{\star}, q^{y \mid a})$ quand $q^{y \mid a}$ varie, on obtient la **chaîne d’intégrales complète** (une seule « couche » d’incertitude sur $q^{y \mid a}$) :

$$
\mathbb{E}_{\mathrm{pr}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{\star})\right] \;=\; \int \mathrm{pr}(q^{y \mid a}) \left[ \int_{\mathcal{Y}} \int_{\mathcal{A}} y \, q^{a}_{\star}(a) \, q^{y \mid a}(y \mid a) \, \mathrm{d}a \, \mathrm{d}y \right] \mathrm{d}q^{y \mid a}.
$$

C’est la version « tout intérieur explicite » du couple **backdoor sur $(Q^{a}, Q^{y \mid a}, Q^{y})$** + **espérance de $Y$ dans la sous-population**.

### 7.5 Graphe **confounder \& interference** (front-door sur $Z$) : intégrales imbriquées maximales (dans le papier pour cet estimande)

**Loi intermédiaire** (`eq:interfere_front_door`) :

$$
\mathrm{pr}\!\left(q^{y \mid a} \,\middle|\, \mathrm{do}(q^{a} = q^{a}_{\star})\right) \;=\; \int \mathrm{pr}(z \mid q^{a}_{\star}) \left[ \int \mathrm{pr}(q^{a}) \, \mathrm{pr}(q^{y \mid a} \mid q^{a}, z) \, \mathrm{d}q^{a} \right] \mathrm{d}z.
$$

**Loi de $q^{y}$** (`eq:interfere_intervention`) :

$$
\mathrm{pr}\!\left(q^{y} \,\middle|\, \mathrm{do}(q^{a} = q^{a}_{\star})\right) \;=\; \int \mathrm{pr}\!\left(q^{y} \,\middle|\, q^{a}_{\star}, q^{y \mid a}\right) \, \mathrm{pr}\!\left(q^{y \mid a} \,\middle|\, \mathrm{do}(q^{a} = q^{a}_{\star})\right) \, \mathrm{d}q^{y \mid a} \;=\; \int \mathfrak{m}\!\left(q^{a}_{\star}, q^{y \mid a}\right) \, \mathrm{pr}\!\left(q^{y \mid a} \,\middle|\, \mathrm{do}(q^{a} = q^{a}_{\star})\right) \, \mathrm{d}q^{y \mid a}.
$$

En injectant la première dans l’espérance de $\phi(q^{y}) = \int y\, q^{y}(y)\,\mathrm{d}y$, on obtient l’**identification complète en une ligne** (toutes les intégrales apparentes) :

$$
\mathbb{E}_{\mathrm{pr}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{\star})\right] \;=\; \int \left\{ \int \mathrm{pr}(z \mid q^{a}_{\star}) \left[ \int \mathrm{pr}(q^{a}) \, \mathrm{pr}(q^{y \mid a} \mid q^{a}, z) \, \mathrm{d}q^{a} \right] \mathrm{d}z \right\} \left[ \int_{\mathcal{Y}} \int_{\mathcal{A}} y \, q^{a}_{\star}(a) \, q^{y \mid a}(y \mid a) \, \mathrm{d}a \, \mathrm{d}y \right] \mathrm{d}q^{y \mid a}.
$$

Ici l’accolade intérieure est exactement $\mathrm{pr}(q^{y \mid a} \mid \mathrm{do}(q^{a} = q^{a}_{\star}))$ ; le dernier crochet est $\mathbb{E}_{Q^{y}}[Y]$ pour ce $q^{y \mid a}$ et ce $q^{a}_{\star}$ fixés.

### 7.6 ATE (différence de deux potentiels)

Comme dans `eq:confounder_effect` / l’exemple interférence (deux lois $q^{a}_{\star}$) :

$$
\mathrm{ATE} \;=\; \mathbb{E}_{\mathrm{pr}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{(1)})\right] \;-\; \mathbb{E}_{\mathrm{pr}}\!\left[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{(0)})\right],
$$

chacun des deux termes étant donné par la chaîne d’intégrales du **§7.4** ou du **§7.5** selon le graphe identifié.

### 7.7 Augmentation générale (appendice, sous-graphe sous-unitaire relatif à $\mathcal{L}$, $\mathcal{R}$)

Pour $\mathcal{L} \subseteq \mathcal{S}$ et $\mathcal{R} \subseteq \mathrm{da}_{\mathcal{S}}(\mathcal{L}) = \big(\bigcup_{v \in \mathcal{L}} \mathrm{da}_{\mathcal{S}}(v)\big)\setminus \mathcal{L}$ (ancêtres directs sous-unitaires de $\mathcal{L}$, au sens du papier), l’équation **`eqn:augment_mech`** (`paper.tex`, appendice `apx:aug_var_form`) donne la loi **au sein de l’unité** après intervention sur $x^{\mathcal{R}}$ :

$$
q_{i}^{\mathcal{L} \mid \mathcal{R}}\!\left(x^{\mathcal{L}} \,\middle|\, \mathrm{do}(x^{\mathcal{R}})\right) \;=\; \int \cdots \int \; \prod_{v \,\in\, \mathcal{L} \,\cup\, \mathrm{da}_{\mathcal{S}}(\mathcal{L}) \setminus \mathcal{R}} q_{i}^{v \mid \mathrm{pa}_{\mathcal{S}}(v)}\!\left(x^{v} \,\middle|\, x^{\mathrm{pa}_{\mathcal{S}}(v)}\right) \; \prod_{w \,\in\, \mathrm{da}_{\mathcal{S}}(\mathcal{L}) \setminus \mathcal{R}} \mathrm{d}x^{w}.
$$

(Les intégrales portent sur toutes les variables $x^{w}$ apparaissant dans le second produit ; le premier produit regroupe les facteurs $q_{i}^{v \mid \mathrm{pa}_{\mathcal{S}}(v)}$ du mécanisme sous-unitaire marginalisé.)

**Remarque :** un HCM « complet » en application (plusieurs $Q$, confondeurs, interférence, etc.) se traduit en pratique par une **formule symbolique** (produits / sommes / conditionnelles) renvoyée par le do-calcul sur le modèle effondré augmenté ; ses facteurs se spécialisent aux blocs **§7.3–7.5** et aux marginales du type **`eqn:augment_mech`** selon le motif graphique.

---

## 8. STAR `teacher_student` — graphes **DirectLiNGAM** et **ExactBIC** (pas le motif « confounder » seul)

Contexte code : `examples/STAR/star_hcm_v2_teacher_student.py` — arêtes issues de `causal_discovery_star.json` (DirectLiNGAM / ExactBIC), traduites en $(A,Y,M,G,E,L)$, plus **toujours** $(U \to A)$ et $(U \to Y)$ ; collapse + `suggest_augment_for_outcome` sur la sous-unité cible (ici **lecture** $Y$) ; `identify_effect(..., Y=\{Q^y\}, X=\{Q^a\}, \text{unobserved}=\{U\})$ ; formule LaTeX stockée dans `examples/STAR/results/star_hcm_v2_teacher_student.json` (`formula_latex`).

L’estimande affiché dans ce JSON est $\mathbb{E}\big[ Q^{y} \,\big|\, \mathrm{do}(Q^{a} = q^{a}_{\star}) \big]$ au niveau **unitaire** (variable $Q^{y}$ du modèle effondré augmenté). Dans le formalisme double-plaque du papier, $Q^{y}$ agrège la variabilité sous-unitaire de la lecture ; lorsque $Q^{y}$ est résumé par son **espérance sous-unitaire** (ex. moyenne $\mu$ d’une Gaussienne sur les scores), cette quantité coïncide avec $\mathbb{E}_{Q^{y}}[Y]$ à l’intérieur de l’unité. L’**espérance externe** $\mathbb{E}_{\mathrm{pr}}[\,\cdot\,]$ sur les unités est alors la même structure que $\mathbb{E}_{\mathrm{pr}}[\mathbb{E}_{Q^{y}}[Y] \,;\, \mathrm{do}(q^{a} = q^{a}_{\star})]$ du papier, mais avec un **ensemble d’ajustement** dicté par le **DAG observé** LiNGAM ou BIC (pas seulement le triplet $(Q^{a}, Q^{y|a}, U)$ du exemple confounder).

Ci-dessous, on note $\mathrm{pr}$ les densités / masses **observées** du modèle effondré augmenté ; $q^{a}_{\star}$ la valeur d’intervention sur $Q^{a}$ ; les symboles $Q^{l|e}$, $Q^{m|a,e,g,l}$, $Q^{y|g,l,m}$, $Q^{y|a,g}$ sont les $Q$-variables conditionnelles / jointes produites par l’augmentation et le graphe (notation papier $Q^{v|\mathrm{pa}_{\mathcal{S}}(v)}}$).

### 8.1 Continu vs discret : sommes du code $\leftrightarrow$ intégrales population

La sortie pyAgrum est écrite avec $\sum_{\cdots}$ sur les variables d’ajustement. Pour des $Q$ à composantes **continues** (Gaussienne, mélange, etc.), la formule population équivalente remplace ces sommes par des **intégrales** sur la loi jointe observée :

$$
\sum_{w \in \mathcal{W}} f(w) \;\longleftrightarrow\; \int_{\mathcal{W}} f(w)\,\mathrm{d}\mathrm{pr}(w).
$$

Dans `estimate_causal_effect`, les variables d’ajustement continues sont en pratique marginalisées par **Monte Carlo** (`n_mc_samples`), ce qui approche ces intégrales.

### 8.2 Graphe **DirectLiNGAM** (outcome lecture $Y$, même pipeline que le JSON)

**Formule identifiée** (traduction de `formula_latex` du run `outcome_subunit: "Y"`) :

$$
\begin{aligned}
\tau_{\text{DL}}(q^{a}_{\star})
&= \sum_{q^{e},\,q^{g},\,q^{l|e},\,q^{m|a,e,g,l},\,q^{y|g,l,m},\,s}
\mathrm{pr}(s \mid q^{e})
\,\mathrm{pr}(q^{y|g,l,m})
\,\mathrm{pr}(q^{m|a,e,g,l})
\,\mathrm{pr}(q^{l|e} \mid s) \\
&\qquad \cdot
\underbrace{\mathbb{E}\big[ Q^{y} \,\big|\, Q^{a}{=}q^{a}_{\star},\, Q^{e}{=}q^{e},\, Q^{g}{=}q^{g},\, Q^{l|e}{=}q^{l|e},\, Q^{m|a,e,g,l}{=}q^{m|a,e,g,l},\, Q^{y|g,l,m}{=}q^{y|g,l,m} \big]}_{\text{facteur } P(Q^{y}\mid\cdots)\text{ évalué en espérance conditionnelle (code)}}
\,\mathrm{pr}(q^{g})
\,\mathrm{pr}(q^{e}) \, .
\end{aligned}
$$

**Lien avec l’espérance interne du papier** : pour une unité $i$, si $Q^{y}_{i}$ désigne la loi marginale sur les scores de lecture des élèves, alors $\mathbb{E}_{Q^{y}_{i}}[Y] = \int_{\mathcal{Y}} y\, q^{y}_{i}(y)\,\mathrm{d}y$ (comme après `eq:confounder_estimand`). Dans le pipeline STAR, $Q^{y}$ est souvent **paramétré** (Gaussienne / mélange) : l’espérance $\mathbb{E}[Q^{y}\mid\cdots]$ utilisée dans le plug-in est alors typiquement ce **premier moment** (ex. $\mu$) plutôt que la notation densité complète ci-dessus. La forme **opérationnelle identifiée** pour le STAR LiNGAM reste le **produit de noyaux** ci-dessus (ajustement sur $\{Q^{e},Q^{g},Q^{l|e},Q^{m|\cdot},Q^{y|\cdot},S\}$), chaque $\mathrm{pr}(\cdot\mid\cdot)$ étant la loi conditionnelle induite par le DAG effondré augmenté ; les « sommes » $\sum$ deviennent des **intégrales** sur l’espace des paramètres $Q$ lorsque ces composantes sont continues.

*(Les symboles $Q^{y|g,l,m}$, $Q^{m|a,e,g,l}$ viennent des parents sous-unitaires du graphe discovery traduit ; certains blocs peuvent être absents de `enriched` → warnings §1.)*

### 8.3 Graphe **ExactBIC** (même outcome $Y$, même pipeline)

**Formule identifiée** (`formula_latex` du même JSON) :

$$
\begin{aligned}
\tau_{\text{BIC}}(q^{a}_{\star})
&= \sum_{q^{e},\,q^{g},\,q^{y|a,g},\,s}
\mathrm{pr}(s \mid q^{e})
\,\mathrm{pr}(q^{y|a,g} \mid s)
\,\mathbb{E}\big[ Q^{y} \,\big|\, Q^{a}{=}q^{a}_{\star},\, Q^{g}{=}q^{g},\, Q^{y|a,g}{=}q^{y|a,g} \big]
\,\mathrm{pr}(q^{g})
\,\mathrm{pr}(q^{e}) \, .
\end{aligned}
$$

**Version intégrale population** (même remarque que §8.1) :

$$
\tau_{\text{BIC}}(q^{a}_{\star})
= \int \mathbb{E}\big[ Q^{y} \,\big|\, Q^{a}{=}q^{a}_{\star},\, Q^{g},\, Q^{y|a,g} \big]
\,\mathrm{pr}(q^{y|a,g} \mid s)\,\mathrm{pr}(s \mid q^{e})\,\mathrm{d}\mathrm{pr}(q^{e}, q^{g}, s) \, .
$$

### 8.4 ATE numérique STAR

Comme pour `eq:confounder_effect`, on forme

$$
\mathrm{ATE} = \tau(q^{a}_{(1)}) - \tau(q^{a}_{(0)}),
$$

ex. $q^{a}_{(1)} \leftrightarrow \text{petite classe}$, $q^{a}_{(0)} \leftrightarrow \text{grande classe}$ dans le script (`intervention \{"Q^a": 1.0\}` vs `0.0`).

### 8.5 Liens utiles

- JSON : `examples/STAR/results/star_hcm_v2_teacher_student.json` (`formula_latex` par graphe).
- Graphes discovery : `examples/STAR/results/causal_discovery_star.json` (`DirectLiNGAM.directed_edges`, `ExactBIC.directed_edges`).
