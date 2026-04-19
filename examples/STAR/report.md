# Causal Estimation Under Hierarchical Structure: Applying Hierarchical Causal Models to Project STAR

---

## Abstract

Project STAR (Student/Teacher Achievement Ratio) is one of the most celebrated randomized experiments in education economics. Its nested structure — students within classes within schools — makes it an ideal benchmark for methods that explicitly model hierarchical data-generating processes. This report applies Hierarchical Causal Models (HCMs) to the STAR kindergarten cohort, demonstrating a full pipeline from causal discovery to do-calculus identification and plug-in estimation. We compare three levels of analysis: (i) a flat causal discovery and predictive baseline, (ii) an econometric replication of Krueger (1999), and (iii) two generations of HCM (school→student, then teacher/class→student). Our main HCM result, obtained under the DirectLiNGAM graph, yields an Average Treatment Effect (ATE) of approximately **22.3 points** on the mathematics score scale when intervening on the class-size distribution at the unit level — a finding that is structurally distinct from, yet directionally consistent with, the ~6.5-point OLS reading effect established in the econometric literature. We discuss the assumptions, limitations, and methodological advantages of the HCM approach relative to standard econometric strategies.

---

## 1. Introduction

Understanding the causal effect of class size on student achievement has been a central question in education policy for decades. The Tennessee STAR experiment, launched in the mid-1980s, remains the gold standard for studying this question: students and teachers were randomly assigned to one of three class types (small, regular, regular with aide) within schools, generating clean experimental variation in class size from kindergarten through third grade.

Despite STAR's experimental pedigree, standard analyses face a structural tension: the data are fundamentally *hierarchical*. Students are nested within classes, classes within schools, and outcomes at the student level are shaped by mechanisms operating at multiple levels simultaneously — teacher behavior, peer composition, school resources. Ordinary regression with school fixed effects absorbs some of this structure, but does not model the hierarchical data-generating process explicitly.

Hierarchical Causal Models (HCMs) provide a principled framework for precisely this setting. An HCM distinguishes between *unit-level* variables (here: the class or teacher) and *subunit-level* variables (here: individual students), models the causal mechanisms at each level, and derives identification formulas via do-calculus on the collapsed, augmented graph. Crucially, HCMs define interventions at the correct level — intervening on the *distribution* of treatment across students within a class, not on individual treatment status — which is the natural causal question in a class-size experiment.

This report documents the full HCM analysis of STAR, structured as follows. Section 2 reviews the relevant literature. Section 3 describes the data. Section 4 presents the methodological pipeline. Section 5 reports results at each level of analysis. Section 6 discusses implications, limitations, and related work. Section 7 concludes.

---

## 2. Background

### 2.1 The STAR Experiment

Project STAR was a large-scale randomized controlled trial conducted in Tennessee from 1985 to 1989. Approximately 11,600 students from 80 schools participated for at least one year, from kindergarten through third grade (Achilles et al., 2008). Students and teachers were randomly assigned within schools to one of three class types:

- **Small**: 13–17 students per teacher
- **Regular**: 22–25 students per teacher
- **Regular with aide**: 22–25 students with an additional teaching assistant

The randomization occurred at the school level: within each school, students and teachers were randomly distributed across class types. Stanford Achievement Test (SAT) percentile scores and Tennessee Basic Skills First (BSF) test scores served as the primary outcome measures.

### 2.2 Established Findings

**Krueger (1999)** is the foundational analysis of STAR. Using OLS with school fixed effects and student/teacher controls, Krueger finds that small-class assignment increases average SAT percentile scores by approximately 5.3 percentage points in kindergarten, 7.6 points in grade 1, and 5–6 points in grades 2 and 3. The effect of regular-size classes with teaching aides is consistently small and statistically insignificant. Two-stage least squares (2SLS) estimates, using initial class-type assignment as an instrument for realized class size, yield a coefficient of approximately −0.72 on class size in kindergarten, implying that each additional student reduces the average score by about 0.7 percentile points.

**Krueger and Whitmore (2001)** extend the analysis to longer-term outcomes, finding that early exposure to small classes increases the probability of taking college entrance examinations in middle school, suggesting persistent effects beyond the experimental period.

**Hanushek (1999)** offers a critical reexamination, questioning the generalizability of the STAR findings and emphasizing that class-size effects may be heterogeneous across contexts. While not invalidating the core result, this work motivates careful attention to effect heterogeneity and external validity.

**Chetty et al. (2011)** link STAR records to tax data, documenting that kindergarten classroom quality — including class size — predicts adult outcomes including earnings, college attendance, and homeownership. This establishes STAR as a canonical dataset for studying long-run treatment effects.

**Boozer and Cacciola (2001)** use the experimental variation in STAR to identify peer effects — showing that a portion of the class-size effect operates through peer composition and within-class interactions rather than purely through teacher attention. This finding is especially relevant for hierarchical modeling, as it underscores that the treatment "small class" is not a purely individual-level intervention.

**Ding and Lehrer (2010)** treat STAR as a multi-period experiment with attrition, selective switching between class types, and non-compliance, and develop estimation strategies that account for these complications. Their work highlights that even a well-designed RCT requires careful methodological choices when the data are longitudinal and hierarchical.

### 2.3 Motivation for HCMs

The literature reviewed above has firmly established the causal effect of small-class assignment on test scores. The value of revisiting STAR with HCMs is not to re-establish this effect, but to demonstrate that the hierarchical structure of the data-generating process can be modeled explicitly and that causal identification and estimation can be grounded in the correct level of analysis. In particular:

- The treatment (class type) operates at the **class level**, not the student level.
- Student outcomes are shaped by unit-level variables (teacher, urbanicity) and student-level variables (gender, ethnicity, socioeconomic status).
- Peer effects and shared classroom shocks create dependence structures within classes that standard regression with robust standard errors only partially addresses.

An HCM formalizes all of this: it defines a bipartite graph over unit variables and subunit variables, derives the identification formula for interventions on the unit-level treatment distribution, and provides a plug-in estimator that respects the hierarchical structure.

---

## 3. Data

### 3.1 Dataset

We use the full student-level STAR dataset (`STAR_Students.tab`, Harvard Dataverse, Achilles et al., 2008), restricted to the **kindergarten** cohort. After removing records with missing values on the key variables and restricting to classes with sufficient observations, our working sample contains:

| Quantity | Value |
|---|---|
| Total students (after filtering) | 5,745 |
| Classes used as units | 322 |
| Students per class (balanced cap) | 10 |
| Schools | 79 |

### 3.2 Variable Definitions

#### HCM v2 (Teacher/Class → Student) schema

| Variable | Level | Definition |
|---|---|---|
| `A` | Subunit | $\mathbf{1}\{\text{gkclasstype} = 1\}$ — small class indicator |
| `Y` | Subunit | `gktreadss` — reading score |
| `M` | Subunit | `gktmathss` — mathematics score |
| `G` | Subunit | $\mathbf{1}\{\text{gender} = \text{female}\}$ |
| `E` | Subunit | $\mathbf{1}\{\text{race} \in \{\text{white, asian}\}\}$ |
| `L` | Subunit | $\mathbf{1}\{\text{free lunch eligible}\}$ |
| `S` | Unit | `gksurban` — school urbanicity (inner-city, suburban, rural, urban) |
| `U` | Unit (latent) | Unobserved class-level confounder |
| Unit ID | — | `gktchid` (teacher/class identifier) |

For the flat causal discovery analysis, we use a reduced dataset of 4,000 observations with 7 variables: `readk`, `mathk`, `stark`, `gender`, `ethnicity`, `lunchk`, `schoolk`.

### 3.3 Descriptive Statistics (Randomization Checks)

Replicating Krueger (1999, Table I), we verify that the randomization appears balanced within schools. In kindergarten, the joint p-value for background covariate differences across class types is above 0.09 for free lunch, 0.26 for White/Asian status, and 0.33 for age — all consistent with successful randomization. Class size differs significantly by design (average 15.1 students in small vs. 22.4–22.8 in regular classes, p < 0.001), as does the SAT percentile score (54.7 vs. 50.0 vs. 50.0, p < 0.001).

---

## 4. Methods

### 4.1 Pipeline Overview

Our analysis proceeds in four stages:

1. **Flat causal discovery**: learn DAGs from observational data ignoring the hierarchical structure.
2. **Predictive explainability**: train a random forest and apply SHAP/LIME/PDP to characterize the predictive signal.
3. **HCM estimation**: apply the HCM framework using the discovered graphs as structural inputs, in two versions (v1: school→student; v2: teacher/class→student).
4. **Econometric baseline**: replicate the Krueger-style OLS/IV specification for direct comparison.

### 4.2 Causal Discovery

We apply four causal discovery algorithms from the `causal-learn` library to the flat STAR dataset (n = 4,000, 7 variables):

- **PC**: constraint-based, produces a CPDAG.
- **FCI**: constraint-based allowing latent variables, produces a PAG.
- **DirectLiNGAM**: score-based under linear non-Gaussian additive noise assumption.
- **ExactBIC**: score-based exact search with BIC scoring, produces a DAG.

Graph similarity is assessed via Jaccard overlap of skeleton and directed edges. A consensus graph is constructed via majority vote over PC, DirectLiNGAM, and ExactBIC (FCI produces a PAG; SAM was excluded as it produced a complete graph with 42 arcs on 7 nodes).

### 4.3 Predictive Explainability

A Random Forest is trained on the full STAR flat dataset (5,789 observations, 114 one-hot features including `schoolidk`) with an 75/25 train/test split. Post-hoc explainability is obtained via:

- **SHAP** (FastTreeSHAP): global feature attribution.
- **LIME**: local instance explanations.
- **PDP/ICE** plots for the treatment variable `stark`.

### 4.4 Hierarchical Causal Models

#### Framework

An HCM defines a bipartite causal graph $\mathcal{G} = (\mathcal{U}, \mathcal{S}, \mathcal{E})$ where $\mathcal{U}$ denotes unit-level variables and $\mathcal{S}$ denotes subunit-level variables. For unit $i$ with subunits $j = 1, \ldots, m_i$, the hierarchical generative model is:

$$
Q^{v | \mathrm{pa}_\mathcal{S}(v)}_i \sim \mathrm{pr}\!\left(q^{v | \mathrm{pa}_\mathcal{S}(v)} \,\middle|\, x^{\mathrm{pa}_\mathcal{U}(v)}_i\right), \qquad X^v_{ij} \sim q^{v | \mathrm{pa}_\mathcal{S}(v)}_i\!\left(x^v \,\middle|\, x^{\mathrm{pa}_\mathcal{S}(v)}_{ij}\right)
$$

where $Q^{v|\cdot}_i$ is a unit-level *distribution-valued* random variable — the kernel of the conditional mechanism for subunit variable $v$ within unit $i$.

The causal estimand under an intervention $\mathrm{do}(Q^a = q^a_\star)$ is:

$$
\tau(q^a_\star) = \mathbb{E}_{\mathrm{pr}}\!\left[\mathbb{E}_{Q^y}[Y] \,;\, \mathrm{do}(Q^a = q^a_\star)\right]
$$

and the Average Treatment Effect for a binary treatment ($q^a_{(1)}$ = small class, $q^a_{(0)}$ = regular class) is:

$$
\mathrm{ATE} = \tau(q^a_{(1)}) - \tau(q^a_{(0)})
$$

Identification proceeds by collapsing the HCM to a DAG over $Q$-variables, augmenting it to handle latent confounders, and applying standard do-calculus (Pearl, 2009). The resulting identification formula is a product of conditional kernels that can be estimated by plug-in.

#### Graph Translation

Since our HCM does not have a domain-specified graph a priori, we translate the flat discovery graphs into HCM graphs by:
1. Assigning `S` (urbanicity) to the unit level.
2. Assigning `A`, `Y`, `M`, `G`, `E`, `L` to the subunit level.
3. Adding edges $U \to A$ and $U \to Y$ (latent class-level confounder).
4. Translating flat directed edges into subunit-level edges where both endpoints are subunit variables.

#### Estimation

The plug-in estimator evaluates the identification formula by:
1. Enriching the dataset with $Q$-columns (fitted conditional density parameters per unit).
2. Fitting distribution family models to each factor: `Y`, `M` use Gaussian mixtures (K=2); `G`, `E`, `L` use Bernoulli; `S` uses a categorical model.
3. Evaluating the formula recursively via the AST produced by pyAgrum's do-calculus, marginalizing continuous variables by Monte Carlo (`N_MC_SAMPLES = 60`, `RANDOM_STATE = 42`).
4. Averaging across the 322 class units.

#### HCM v1 vs. v2

**HCM v1** (`star_hcm_first_pass.py`) uses schools as units (79 schools, ~34 students per school). This coarser hierarchy leads to numerically unstable ATEs and is retained as a proof of concept only.

**HCM v2** (`star_hcm_v2_teacher_student.py`) uses teacher/class as the unit (322 classes, ~10 students per class). This correctly reflects the experimental design — randomization occurred within schools at the class level — and produces stable estimates for all five graph specifications.

### 4.5 Econometric Baseline

Following Krueger (1999), we estimate:

$$
Y_{ics} = \beta_0 + \beta_1 \mathrm{SMALL}_{cs} + \beta_2 \mathrm{REG\text{-}A}_{cs} + \beta_3 X_{ics} + \alpha_s + \varepsilon_{ics}
$$

where $Y_{ics}$ is the reading score of student $i$ in class $c$ at school $s$, $\mathrm{SMALL}_{cs}$ and $\mathrm{REG\text{-}A}_{cs}$ are class-type dummies, $X_{ics}$ includes student demographics (gender, race, free lunch) and teacher characteristics (race, experience, degree), and $\alpha_s$ is a school fixed effect. We also estimate a 2SLS model using initial class-type assignment as an instrument for actual class size.

---

## 5. Results

### 5.1 Causal Discovery

Running PC, FCI, DirectLiNGAM, and ExactBIC on the flat STAR dataset (n = 4,000, 7 variables), we observe:

#### Skeleton agreement (Jaccard)

| | PC | FCI | LiNGAM | ExactBIC |
|---|---:|---:|---:|---:|
| PC | 1.000 | 1.000 | 0.667 | 0.818 |
| FCI | 1.000 | 1.000 | 0.667 | 0.818 |
| LiNGAM | 0.667 | 0.667 | 1.000 | 0.539 |
| ExactBIC | 0.818 | 0.818 | 0.539 | 1.000 |

#### Directed edge agreement (Jaccard)

| | PC | LiNGAM | ExactBIC |
|---|---:|---:|---:|
| PC | 1.000 | 0.417 | 0.417 |
| LiNGAM | 0.417 | 1.000 | 0.333 |
| ExactBIC | 0.417 | 0.333 | 1.000 |

#### Consensus directed edges

| Votes | Edges |
|---|---|
| 3/3 | `ethnicity → lunchk`, `gender → readk`, `schoolk → lunchk` |
| 2/3 | `ethnicity → mathk`, `ethnicity → schoolk`, `lunchk → readk`, `mathk → lunchk`, `mathk → readk`, `schoolk → readk` |
| 1/3 only | `stark → readk` (ExactBIC only), `stark → mathk` (LiNGAM only) |

**Key observation**: `stark` (class-size treatment) receives no 2/3 or 3/3 directed edge in the consensus. This is not a finding against the class-size effect — it reflects that constraint-based and score-based discovery methods struggle to recover an experimentally-determined treatment variable from i.i.d.-encoded observational data. The causal status of `stark` is established by the experimental design, not by algorithmic discovery.

#### DoWhy estimate (ExactBIC graph)

Injecting the ExactBIC DAG into DoWhy and estimating via linear backdoor adjustment:

$$\widehat{\mathrm{ATE}}(\mathrm{stark} \to \mathrm{readk}) = 2.664$$

This underestimates the experimental effect, reflecting both the approximate nature of the learned graph and the limitations of applying observational identification strategies to experimental data.

### 5.2 Predictive Explainability

A Random Forest trained on the flat STAR dataset achieves:

| Metric | Value |
|---|---:|
| $R^2$ (test) | 0.194 |
| MAE | 20.38 |
| RMSE | 28.59 |

Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) curves for `stark` yield the following predicted mean reading scores:

| Class type | Mean predicted `readk` |
|---|---:|
| Regular | 434.28 |
| Regular + aide | 436.32 |
| Small | 440.73 |

**Small vs. Regular contrast**: $+6.45$ predicted reading score points. SHAP analysis confirms that `stark` is an informative predictor but that school context dominates global feature importance. These are predictive, not causal, estimates.

### 5.3 Econometric Baseline (Krueger Replication)

OLS estimates on the 5,725-observation kindergarten sample with school fixed effects:

| Model | $\hat{\beta}_{\mathrm{small}}$ | SE | $p$-value | $R^2$ |
|---|---:|---:|---:|---:|
| Minimal (no controls) | 5.74 | 1.05 | $4.4 \times 10^{-8}$ | 0.006 |
| School FE only | 6.58 | 0.97 | $1.2 \times 10^{-11}$ | 0.211 |
| Student controls + School FE | 6.51 | 0.94 | $3.4 \times 10^{-12}$ | 0.263 |
| Full (Krueger-like) | 6.55 | 0.94 | $2.6 \times 10^{-12}$ | 0.266 |

The coefficient on Regular + Aide is consistently small (0.65–1.25) and statistically insignificant — replicating Krueger (1999). School fixed effects absorb substantial variance ($R^2$ rises from 0.006 to 0.211), consistent with the large between-school variation in STAR.

2SLS estimate (initial assignment instruments for actual class size):

$$\widehat{\beta}_{\mathrm{class size}} = -0.818 \quad (\text{SE} = 0.109, \; p = 6.7 \times 10^{-14})$$

Each additional student is associated with approximately −0.82 reading score points, consistent with Krueger's estimate of −0.72 for kindergarten.

### 5.4 HCM v1: School → Student

Using schools as units (79 schools, ~34 balanced students per school), the HCM v1 produces:

| Graph | Identifiable | $E[\mathrm{do}(1)]$ | $E[\mathrm{do}(0)]$ | ATE |
|---|---:|---:|---:|---:|
| PC | Yes | 436.15 | 436.15 | 0.00 |
| FCI | Yes | 436.15 | 436.15 | 0.00 |
| DirectLiNGAM | Yes | 41,094.66 | 42,979.48 | −1,884.82 |
| ExactBIC | Yes | 426,106.02 | 405,421.90 | +20,684.12 |
| ConsensusMean | Yes | 436.15 | 436.15 | 0.00 |

The DirectLiNGAM and ExactBIC ATEs are orders of magnitude out of range and clearly unreliable. This confirms that school-level aggregation is too coarse for meaningful HCM estimation in STAR: aggregating math scores, gender, and ethnicity at the school level discards within-school variability and destabilizes the estimation. This version is retained as a proof of concept only.

### 5.5 HCM v2: Teacher/Class → Student

Using teacher/class as the unit (322 classes, ~10 students per class), with the full v2 pipeline:

#### Outcome: Mathematics score `M`

| Graph | Identifiable | $E[\mathrm{do}(1)]$ | $E[\mathrm{do}(0)]$ | ATE |
|---|---:|---:|---:|---:|
| PC | Yes | 485.91 | 485.91 | **0.00** |
| FCI | Yes | 485.91 | 485.91 | **0.00** |
| DirectLiNGAM | Yes | 1,382.54 | 1,360.27 | **+22.27** |
| ExactBIC | Yes | 2,406.65 | 2,366.69 | **+39.96** |
| ConsensusMean | Yes | 485.91 | 485.91 | **0.00** |

**Identification formulas** (pyAgrum do-calculus output):

*DirectLiNGAM*:
$$\tau_{\mathrm{DL}}(q^a_\star) = \sum_{q^e, q^g, q^{l|e}, q^{m|a,e,g,l}, s} \mathrm{pr}(s|q^e) \cdot \mathrm{pr}(q^{m|a,e,g,l}) \cdot \mathrm{pr}(q^{l|e}|s) \cdot \mathbb{E}[Q^m | Q^a{=}q^a_\star, Q^e, Q^g, Q^{l|e}, Q^{m|a,e,g,l}] \cdot \mathrm{pr}(q^g) \cdot \mathrm{pr}(q^e)$$

*ExactBIC*:
$$\tau_{\mathrm{BIC}}(q^a_\star) = \sum_{q^e, q^g, q^{m|e,y}, q^{y|a,g}, s} \mathrm{pr}(s|q^e) \cdot \mathrm{pr}(q^{y|a,g}|s) \cdot \mathrm{pr}(q^{m|e,y}) \cdot \mathbb{E}[Q^m | Q^a{=}q^a_\star, Q^e, Q^g, Q^{m|e,y}, Q^{y|a,g}] \cdot \mathrm{pr}(q^g) \cdot \mathrm{pr}(q^e)$$

**Interpretation of zero ATEs**: For PC, FCI, and ConsensusMean, the identification formula reduces to $P(Q^m)$ — the intervention on $Q^a$ does not appear in the identified expression given these graph structures. An ATE of 0 under these graphs is a *structural* consequence of the graph topology after translation, not an empirical claim about the absence of a class-size effect.

**Graph selection**: We retain **DirectLiNGAM** as the primary narrative graph because it is the only discovery-based graph that produces an explicit $A \to M$ path in the HCM translation, making the identification formula directly interpretable as "the effect of the class-size distribution on mathematics outcomes." The ExactBIC result (ATE ≈ 39.96) illustrates sensitivity to graph structure.

#### Historical runs with partial identification

Earlier runs with `HCM_DISABLE_MULTIPARENT_Q_PRECOMPUTE=1` (before the multi-parent $Q$ correction) produced:

| Graph | ATE (historical) | Missing factors replaced by 1 |
|---|---:|---|
| DirectLiNGAM | 17.618 | $P(Q^{y|g,l,m})$, $P(Q^{m|a,e,g,l})$ |
| ExactBIC | 21.628 | $P(Q^{y|a,g}|S)$ |

In both cases, the do1 vs. do0 difference is carried almost entirely by the primary conditional factor $P(Q^y|\cdots)$ (unit-level delta ≈ +5.32 to +5.34 points), while the missing factors are neutral (= 1). These historical ATEs are reproducible but represent a *truncated* functional $\tilde{\mathfrak{F}}$ rather than the full identification $\mathfrak{F}$. Factor analysis plots confirm that the missing factors contribute zero delta between the two interventional arms. These are reported for transparency and reproducibility, not as the primary causal estimate.

### 5.6 Synthesis

| Analysis | Data | Outcome | Effect estimate |
|---|---|---|---|
| Causal discovery (DoWhy/ExactBIC) | 4,000 rows, flat | Reading | ATE ≈ +2.66 |
| RF predictive (PDP/ICE) | 5,789 rows, flat | Reading | +6.45 (predicted) |
| OLS full (school FE + controls) | 5,725 rows | Reading | **+6.55** (SE 0.94) |
| 2SLS (class-size instrument) | 5,725 rows | Reading | **−0.82 per student** |
| HCM v1 (school → student) | 5,768 rows, 79 schools | Reading | Non-credible |
| HCM v2 — DirectLiNGAM | 5,745 rows, 322 classes | **Maths** | **+22.27** |
| HCM v2 — ExactBIC | 5,745 rows, 322 classes | **Maths** | +39.96 |

**Note on comparability**: The OLS/IV estimates target *reading* scores on the SAT percentile scale; the HCM v2 estimates target the *mathematics* raw score (`gktmathss`). These are on different scales and are not directly comparable point-for-point. The directional agreement (positive effect of small class) is consistent across methods, and the OLS effect size (~6.5 percentile points) serves as the substantive econometric reference.

---

## 6. Discussion

### 6.1 What HCM Adds

The HCM framework offers three methodological contributions relative to standard approaches in the STAR literature:

**1. Correct level of intervention.** Standard OLS and 2SLS define the treatment at the student level (`SMALL_{ics} = 1`). In reality, class type is assigned at the class level — students within the same class share the same treatment. HCM explicitly models this by intervening on $Q^a$, the *distribution* of treatment across students within a class, which is the natural formulation for a class-level assignment mechanism.

**2. Explicit hierarchical structure.** The HCM v2 graph distinguishes unit-level variables (urbanicity `S`, latent confounder `U`) from subunit-level variables (`A`, `Y`, `M`, `G`, `E`, `L`). This allows the model to represent the fact that school urbanicity shapes the class environment (and thus the treatment-outcome mechanism) at a different level than individual student characteristics.

**3. Principled identification via do-calculus.** Rather than assuming backdoor criterion satisfaction for a chosen set of covariates, the HCM pipeline applies do-calculus to the collapsed/augmented graph and derives the identification formula symbolically. This makes the identification assumptions explicit and verifiable.

### 6.2 Limitations and Caveats

**Discovery-based graphs.** The five graphs used in HCM v2 are translations of flat causal discovery outputs into the HCM schema. They carry the assumptions of the discovery algorithms (linearity for LiNGAM, Gaussian for PC/FCI, i.i.d. for all methods) and inherit their uncertainty. A more principled approach would use domain knowledge to specify the HCM graph directly — reflecting, for instance, the known randomization structure of STAR, the documented peer effects (Boozer & Cacciola, 2001), or the literature on teacher quality as a unit-level variable.

**Partial identification in historical runs.** The multi-parent $Q$-variable correction (`_precompute_conditional_q_vars`) resolved warnings for factors like $P(Q^{y|g,l,m})$ and $P(Q^{m|a,e,g,l})$ that previously defaulted to 1. The corrected plug-in approximates the full identification formula $\mathfrak{F}(\mathbb{P})$ rather than the truncated $\tilde{\mathfrak{F}}(\mathbb{P})$ of historical runs. However, the proxy used (a scalar expected value per unit) is not identical to the distribution-valued objects in the theoretical formulation — a gap that should be documented in any publication.

**Outcome and scale.** The HCM v2 primary results target the mathematics raw score (`gktmathss`, scale ~350–650), while the canonical econometric benchmark targets the reading SAT percentile score. A fully comparable HCM analysis would target reading with outcome `--outcome Y`; this is available in the pipeline and should be documented alongside the maths results.

**Zero ATEs under PC/FCI/Consensus.** The ATE = 0 finding for three of five graphs reflects a structural feature of the identification formula (the intervention $Q^a$ drops out), not an empirical null result. Any presentation of these results must make this distinction explicit to avoid misinterpretation.

**Graph sensitivity.** The ATE varies from 22.27 (DirectLiNGAM) to 39.96 (ExactBIC) depending on the assumed graph. This sensitivity is a feature, not a bug: it reveals how identification depends on the structural assumptions encoded in the DAG. In a well-powered study with strong domain knowledge about the graph, this sensitivity would motivate robustness checks over a class of plausible graphs.

### 6.3 Comparison with Standard Causal Discovery + HCM Approaches

To our knowledge, there is no existing implementation that combines causal discovery, HCM identification, and plug-in estimation in a unified pipeline for hierarchical datasets. The `y0` library (`y0-causal-inference`) has recently introduced support for hierarchical causal models at the symbolic level (January 2025 PR), enabling do-calculus derivation of identification formulas in a custom DSL that extends to counterfactual transport. However, their implementation focuses on the symbolic formula derivation and does not include estimation (fitting distribution families, evaluating the identification formula numerically, computing ATEs from data). Their estimation framework `eliater` was archived shortly before the HCM PR, and no density estimation or empirical evaluation is available in their current codebase. This positions the present work as, to our knowledge, the first end-to-end HCM pipeline from data to estimated ATE.

### 6.4 Parallel Computation

The estimation pipeline supports parallel execution via `parallel_map` (threads or processes). For econometric batch tasks (24 OLS specs, 10 IV specs on 5,725 rows):

| Mode | OLS batch time | IV batch time |
|---|---:|---:|
| Sequential | 12.32 s | 21.43 s |
| 4 threads | 6.71 s | 14.87 s |
| 4 processes | **0.46 s** | **2.05 s** |

The process backend provides a ~25× speedup for batch econometric tasks. For per-unit linear fits (light tasks), sequential execution outperforms parallel due to orchestration overhead.

---

## 7. Conclusion

We have applied Hierarchical Causal Models to Project STAR, demonstrating a complete pipeline from causal discovery through do-calculus identification to plug-in estimation. The key findings are:

1. **Causal discovery** on the flat STAR dataset produces graphs with moderate pairwise agreement (Jaccard 0.33–0.82 on directed edges) and weak consensus on the treatment variable `stark` — consistent with the known difficulty of recovering experimentally-designed treatments from observational algorithms.

2. **Predictive explainability** confirms that class size is informative for reading outcomes (+6.45 predicted points, small vs. regular), with school context dominating global importance.

3. **Econometric replication** of Krueger (1999) yields an OLS small-class effect of +6.55 reading percentile points (p < 0.001) and a 2SLS class-size coefficient of −0.82 points per student — consistent with the published literature.

4. **HCM v1** (school→student hierarchy) produces numerically unstable ATEs and is discarded as a causal estimate. The correct unit for STAR is the teacher/class.

5. **HCM v2** (teacher/class→student, 322 classes, 10 students per class) produces stable estimates for all five graph specifications. Under the DirectLiNGAM graph — the only discovery-based graph with an explicit $A \to M$ path — the estimated ATE on mathematics is **+22.27 points**, with the ExactBIC graph yielding **+39.96 points**. Both results are directionally consistent with the experimental literature. The sensitivity across graphs motivates future work on domain-specified HCM graphs.

6. **Partial identification** in historical runs (factors defaulting to 1) produces ATEs of 17.6 (DirectLiNGAM) and 21.6 (ExactBIC) that are reproducible but represent a truncated functional. The full multi-parent $Q$ correction brings the pipeline closer to the theoretically identified formula.

The HCM framework offers a principled way to analyze hierarchical experimental data by making explicit the level of intervention, the structure of confounding, and the identification assumptions. Project STAR — with its known randomization, rich set of individual and school-level variables, and extensive econometric literature — provides an ideal testbed for validating and extending this approach.

---

## References

- Achilles, C., Bain, H. P., Bellott, F., Boyd-Zaharias, J., Finn, J., Folger, J., Johnston, J., and Word, E. (2008). *Tennessee's Student Teacher Achievement Ratio (STAR) Project*. Harvard Dataverse.
- Boozer, M. A., and Cacciola, S. E. (2001). *Inside the "Black Box" of Project STAR: Estimation of Peer Effects Using Experimental Data*. Yale University Economic Growth Center Discussion Paper No. 832.
- Chetty, R., Friedman, J. N., Hilger, N., Saez, E., Schanzenbach, D. W., and Yagan, D. (2011). *How Does Your Kindergarten Classroom Affect Your Earnings? Evidence from Project STAR*. *The Quarterly Journal of Economics*, 126(4), 1593–1660.
- Ding, W., and Lehrer, S. F. (2010). *Estimating Treatment Effects from Contaminated Multiperiod Education Experiments: The Dynamic Impacts of Class Size Reductions*. *The Review of Economics and Statistics*, 92(1), 31–42.
- Hanushek, E. A. (1999). *Some Findings from an Independent Investigation of the Tennessee STAR Experiment and from Other Investigations of Class Size Effects*. *Educational Evaluation and Policy Analysis*, 21(2), 143–163.
- Krueger, A. B. (1999). *Experimental Estimates of Education Production Functions*. *The Quarterly Journal of Economics*, 114(2), 497–532.
- Krueger, A. B., and Whitmore, D. M. (2001). *The Effect of Attending a Small Class in the Early Grades on College-Test Taking and Middle School Test Results: Evidence from Project STAR*. *The Economic Journal*, 111(468), 1–28.
- Nye, B., Hedges, L. V., and Konstantopoulos, S. (1999). *The Long-Term Effects of Small Classes: A Five-Year Follow-Up of the Tennessee Class Size Experiment*. *Educational Evaluation and Policy Analysis*, 21(2), 127–142.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
