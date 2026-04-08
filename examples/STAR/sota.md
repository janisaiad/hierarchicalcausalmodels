# Mini état de l'art sur Project STAR

## Contexte

`Project STAR` (`Student/Teacher Achievement Ratio`) est l'un des jeux de données les plus connus en économie de l'éducation et en inférence causale appliquée. Sa notoriété vient du fait qu'il repose sur un protocole expérimental randomisé mené dans le Tennessee, où des élèves et des enseignants ont été assignés à différents types de classes: petites classes, classes régulières, et classes régulières avec aide.

Ce jeu de données a donc servi de base à plusieurs générations d'analyses causales. Les premières études ont surtout cherché à estimer l'effet causal moyen de la taille de classe sur les scores scolaires. Par la suite, la littérature s'est élargie à des questions de persistance à long terme, d'hétérogénéité des effets, d'effets de pairs, ainsi qu'aux difficultés méthodologiques liées à l'attrition, à la non-conformité au protocole et à la nature multi-période de l'expérience.

Dans ce sens, `STAR` n'est pas simplement un dataset éducatif célèbre: c'est un cas d'école pour discuter ce qu'une identification causale solide permet, mais aussi ce qu'elle ne résout pas automatiquement lorsque les données sont hiérarchiques, longitudinales et imparfaitement observées.

## Grandes lignes de la littérature causale sur STAR

### 1. Effet causal de la taille de classe

La première vague de travaux utilise directement la randomisation du protocole `STAR` pour estimer l'effet causal des petites classes sur les performances scolaires. Le résultat dominant de cette littérature est qu'une réduction importante de la taille de classe améliore les scores des élèves, en particulier dans les premières années de scolarité.

### 2. Persistance et effets de long terme

Une seconde vague de travaux étudie si ces gains initiaux persistent au-delà de la période expérimentale. Certains papiers suivent les élèves plus tard dans leur parcours scolaire, voire dans leur vie adulte, pour relier la qualité de la classe de maternelle à des outcomes comme l'accès à l'université ou les revenus.

### 3. Réanalyses structurelles

D'autres études ne se contentent pas d'estimer un effet moyen du traitement. Elles cherchent à comprendre les mécanismes possibles: effets de pairs, qualité de l'enseignant, dynamique d'exposition sur plusieurs années, ou rôle de l'environnement de classe.

### 4. Limites et débats

Même si `STAR` est expérimental, la littérature souligne plusieurs difficultés:

- l'attrition n'est pas nécessairement ignorable
- la conformité au protocole n'est pas parfaite
- le traitement agit dans un cadre hiérarchique `école -> classe -> élève`
- des effets d'interférence ou de composition peuvent compliquer l'interprétation simple en termes de SUTVA

Autrement dit, `STAR` est à la fois un benchmark de causalité et un rappel qu'un essai randomisé réel ne supprime pas tous les problèmes d'identification ou de modélisation.

## Bibliographie commentée

### 1. Krueger (1999)

**Krueger, Alan B. (1999). _Experimental Estimates of Education Production Functions_. _The Quarterly Journal of Economics_, 114(2), 497-532.**

Cette étude est la référence fondatrice sur `Project STAR`. Krueger exploite l'assignation aléatoire aux différents types de classes pour estimer l'effet causal de la taille de classe sur les scores scolaires. Le papier montre qu'une petite classe améliore les performances, avec des effets particulièrement visibles dans les premières années et pour certains sous-groupes défavorisés. C'est la référence centrale pour justifier que `STAR` est un dataset canonique en inférence causale.

### 2. Krueger et Whitmore (2001)

**Krueger, Alan B., and Diane M. Whitmore (2001). _The Effect of Attending a Small Class in the Early Grades on College-Test Taking and Middle School Test Results: Evidence from Project STAR_. _The Economic Journal_, 111(468), 1-28.**

Ce papier prolonge l'analyse au-delà des scores immédiats. Les auteurs étudient les résultats au collège et la probabilité de passer des examens d'entrée à l'université. L'intérêt de cette référence est de montrer que l'effet du traitement ne se limite pas aux tests contemporains de l'expérience: il peut avoir des conséquences plus tardives sur la trajectoire scolaire. C'est une référence utile pour motiver des estimands dynamiques ou de long terme.

### 3. Hanushek (1999)

**Hanushek, Eric A. (1999). _Some Findings from an Independent Investigation of the Tennessee STAR Experiment and from Other Investigations of Class Size Effects_. _Educational Evaluation and Policy Analysis_, 21(2), 143-163.**

Cette référence joue un rôle critique important dans la littérature. Hanushek réexamine les résultats du Tennessee `STAR` et discute leur robustesse ainsi que leur interprétation politique. Le papier est utile non pas parce qu'il invaliderait simplement les analyses précédentes, mais parce qu'il rappelle que même un dispositif expérimental très influent fait l'objet de débats sur l'ampleur exacte des effets, leur généralisation et leur coût. Dans un état de l'art sérieux, cette référence permet d'éviter une présentation trop univoque.

### 4. Chetty et al. (2011)

**Chetty, Raj, John N. Friedman, Nathaniel Hilger, Emmanuel Saez, Diane Whitmore Schanzenbach, and Danny Yagan (2011). _How Does Your Kindergarten Classroom Affect Your Earnings? Evidence from Project STAR_. _The Quarterly Journal of Economics_, 126(4), 1593-1660.**

Ce travail est devenu la grande référence sur les effets de long terme de `STAR`. En reliant les données scolaires à des outcomes adultes, les auteurs montrent que la qualité de la classe en maternelle est associée à des différences ultérieures en études supérieures, comportement économique et revenus. Pour un projet de modélisation causale, ce papier montre que `STAR` peut être vu comme une expérience initiale dont les effets dépassent largement les scores à court terme.

### 5. Boozer et Cacciola (2001)

**Boozer, Michael A., and Stephen E. Cacciola (2001). _Inside the "Black Box" of Project STAR: Estimation of Peer Effects Using Experimental Data_. Yale University Economic Growth Center Discussion Paper No. 832.**

Cette étude est particulièrement intéressante pour une lecture hiérarchique ou mécaniste de `STAR`. Les auteurs utilisent la variation expérimentale pour identifier des effets de pairs, c'est-à-dire des effets qui passent par la composition et les interactions au sein de la classe. Cette référence est utile pour souligner que le traitement "taille de classe" ne doit pas nécessairement être interprété comme un levier purement individuel: une partie de l'effet peut passer par des mécanismes collectifs au niveau de la classe.

### 6. Ding et Lehrer (2010)

**Ding, Weili, and Steven F. Lehrer (2010). _Estimating Treatment Effects from Contaminated Multiperiod Education Experiments: The Dynamic Impacts of Class Size Reductions_. _The Review of Economics and Statistics_, 92(1), 31-42.**

Ce papier est important du point de vue méthodologique. Il traite `STAR` comme une expérience multi-période avec attrition, transitions sélectives et non-conformité au protocole. L'intérêt principal est de montrer qu'un essai randomisé réel doit souvent être analysé avec des outils plus riches qu'une simple comparaison entre groupes assignés. Cette référence est donc très utile si l'on veut motiver une approche causale attentive à la dynamique du traitement et à la qualité effective des données.

## Synthèse

Pris ensemble, ces travaux montrent que `STAR` est déjà un terrain causal largement étudié. La littérature a solidement établi son intérêt pour l'estimation d'effets causaux de la taille de classe, puis a progressivement déplacé l'attention vers:

- la persistance des effets
- l'hétérogénéité entre sous-populations
- les effets de pairs
- la dynamique temporelle du traitement
- les problèmes d'attrition et de mise en oeuvre

## Lien avec une approche HCM

L'intérêt d'une approche par `Hierarchical Causal Models` ne serait donc pas de "rendre enfin causale" l'analyse de `STAR`, puisque cela existe déjà. La valeur ajoutée serait plutôt de reformuler proprement l'analyse quand on prend au sérieux la structure imbriquée des données:

- élèves dans classes
- classes dans écoles
- parfois répétitions sur plusieurs années

Un cadre HCM pourrait ainsi aider à distinguer clairement les variables de niveau école, de niveau classe et de niveau élève, à expliciter les confondeurs partagés au sein d'une école ou d'une classe, et à poser plus proprement les questions d'intervention lorsque le traitement et les outcomes vivent à des niveaux hiérarchiques différents.
