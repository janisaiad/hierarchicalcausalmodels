# Tennessee Student/Teacher Achievement (STAR) Project

## Replication Study for ECON 582

**Author:** Harshvardhan  
**ID:** 609162  
**Institution:** University of Tennessee, Knoxville

## Abstract

This replication paper attempts to recreate the results obtained by Krueger (1999). Through this exercise, I aim to rediscover the author’s results and thereby strengthen my understanding of the study.

The paper considers 11,600 students in Tennessee public schools from kindergarten through grade 3. Project STAR was a large-scale randomized trial costing $12$ million over four years. Although the study had some limitations, it was one of the closest attempts to study the impact of class size on teaching and learning outcomes.

Krueger (1999) found that students in smaller classes perform better than students in larger classes. Further, students with teaching aides did not perform significantly better than those without teaching aides. In this replication, I reproduce the main tables and figures and discuss the corresponding inferences.

**Keywords:** Tennessee STAR Project, replication, ordinary least squares, two-stage least squares

## 1. Introduction

The Tennessee Student/Teacher Achievement Ratio (STAR) Project was a large-scale randomized experiment on class size conducted in Tennessee schools. Students and teachers were randomly assigned to one of three class types:

- small class: 13 to 17 students per teacher
- regular-size class: 22 to 25 students
- regular-size class with teacher’s aide: 22 to 25 students

Over four years, about 11,600 students from 80 schools participated in the study. The randomization, meaning the assignment of teachers and students to one of these classroom types, occurred at the school level.

Both Project STAR and this replication study are concerned with the importance and impact of class size on learning outcomes. Informally, the hypothesis is that smaller classes, that is, classes with a lower student-teacher ratio, produce better learning outcomes. Krueger (1999) quantified these learning outcomes using the STAR dataset. Stanford Achievement Test (SAT) and Tennessee Basic Skills First (BSF) tests were used as proxy measures of student achievement.

Krueger (1999) considers class-size and regular-size-with-aide dummy variables as the main explanatory variables for student achievement, while controlling for student and teacher characteristics and for the school in which the student is enrolled. The average SAT percentile score is the outcome measure. Class-size dummies and aide status are taken directly from the project database. Additional control variables include student demographics such as gender and age, and school effects are included as a separate control.

This leads to the regression model

$$
Y_{ics} = \beta_0 + \beta_1 SMALL_{cs} + \beta_2 REG\text{-}A_{cs} + \beta_3 X_{ics} + \alpha_s + \epsilon_{ics}.
$$

where:

- $Y_{ics}$ is the average percentile score on the SAT test of student $i$ in class $c$ at school $s$
- $SMALL_{cs}$ is a dummy variable indicating whether the student was assigned to a small class that year
- $REG\text{-}A_{cs}$ is a dummy variable indicating whether the student was assigned to a regular-size class with an aide that year
- $X_{ics}$ is a vector of student and teacher covariates
- $\alpha_s$ is the school effect
- $\epsilon_{ics}$ is the error term

In my notation, $Y_{ics}$ is $Y$, and all the other variables are part of $X_1$.

Thankfully, this was a randomized experiment. If this had not been a randomized experiment, several assumptions would have been necessary to support causal interpretation. One such assumption would concern conditional independence across variables in the model, for example $Cov(X_i, X_j) = 0$ for all $i \neq j$. We would also need to assume that the error term is independent of included regressors.

There could still be omitted variables. For example, the study does not directly measure students’ inherent ability. Some students may be more able than others, and such heterogeneity is not captured explicitly in the regression. A proxy like IQ could have been included. Inherent ability may affect both student outcomes and school sorting, since higher-performing students may cluster in some schools more than others. Moreover, several studies have hypothesized that gender, age, and race may correlate with such latent traits.

Randomization does not solve every issue automatically. School effects are partly captured, but individual traits such as gender, age, and race were not themselves randomized, or at least the study does not describe them that way. Therefore, some estimates may still be biased.

The rest of this paper is organized as follows. Section 2 discusses the data and code organization. Section 3 presents replicated tables and figures from Krueger (1999). Section 4 offers exploratory discussion of the replicated results. Section 5 discusses the regression design and results. Section 6 lists limitations. Section 7 concludes.

## 2. Data and Codebase Organisation

The complete dataset was available from Harvard Dataverse (Achilles et al., 2008). It contains raw student-level and school-level data from the longitudinal experiment. Student-level data are available for 11,601 students who participated for at least one year. Demographic variables, school and class identifiers, school and teacher information, experimental conditions, achievement test scores, motivation scores, and self-concept scores are all included.

For this study, I organized the working directory into the following folders:

- `DO files`: coding files such as Stata `.do` scripts
- `DTA files`: input dataset files
- `Figures`: generated figures
- `TEX`: LaTeX files
- `others`: files not belonging to the above categories

I wrote modular code so that each `.do` file performs one analysis only, such as generating a single table or figure. For version control and backup, I used GitHub. For additional protection, I also stored files locally in a private Dropbox folder.

The GitHub repository for the project is:  
[https://github.com/harshvardhaniimi/krueger1991-replication](https://github.com/harshvardhaniimi/krueger1991-replication)

## 3. Replicated Tables and Figures from Krueger (1999)

This section presents tables and figures replicated from Krueger (1999). Later sections refer back to these results.

### 3.1. Table I

#### Table 1. Comparison of mean characteristics of treatment and control groups for students who entered STAR in kindergarten

| Variable | Small | Regular | Regular + Aide | Joint p-value |
|---|---:|---:|---:|---:|
| Free Lunch | 0.47 | 0.48 | 0.50 | 0.09 |
| White/Asian | 0.68 | 0.67 | 0.66 | 0.26 |
| Age in 1985 | 5.44 | 5.43 | 5.43 | 0.33 |
| Attrition Rate | 0.49 | 0.52 | 0.53 | 0.02 |
| Class Size | 15.12 | 22.38 | 22.77 | 0.00 |
| SAT Percentile Score | 54.73 | 49.95 | 49.99 | 0.00 |

#### Table 2. Comparison of mean characteristics of treatment and control groups for students who entered STAR in grade 1

| Variable | Small | Regular | Regular + Aide | Joint p-value |
|---|---:|---:|---:|---:|
| Free Lunch | 0.59 | 0.62 | 0.61 | 0.52 |
| White/Asian | 0.62 | 0.56 | 0.64 | 0.00 |
| Age in 1985 | 5.78 | 5.86 | 5.88 | 0.03 |
| Attrition Rate | 0.53 | 0.51 | 0.47 | 0.07 |
| Class Size | 15.87 | 22.71 | 23.46 | 0.00 |
| SAT Percentile Score | 49.52 | 42.90 | 48.02 | 0.00 |

#### Table 3. Comparison of mean characteristics of treatment and control groups for students who entered STAR in grade 2

| Variable | Small | Regular | Regular + Aide | Joint p-value |
|---|---:|---:|---:|---:|
| Free Lunch | 0.66 | 0.63 | 0.66 | 0.60 |
| White/Asian | 0.53 | 0.54 | 0.44 | 0.00 |
| Age in 1985 | 5.88 | 5.91 | 5.94 | 0.41 |
| Attrition Rate | 0.37 | 0.34 | 0.35 | 0.58 |
| Class Size | 15.50 | 23.71 | 23.59 | 0.00 |
| SAT Percentile Score | 46.56 | 45.45 | 41.84 | 0.01 |

Note: the joint p-value for age in 1985 does not match exactly, likely due to replication issues.

#### Table 4. Comparison of mean characteristics of treatment and control groups for students who entered STAR in grade 3

| Variable | Small | Regular | Regular + Aide | Joint p-value |
|---|---:|---:|---:|---:|
| Free Lunch | 0.60 | 0.64 | 0.69 | 0.04 |
| White/Asian | 0.66 | 0.57 | 0.55 | 0.00 |
| Age in 1985 | 5.95 | 5.93 | 5.99 | 0.50 |
| Class Size | 15.97 | 24.05 | 24.43 | 0.01 |
| SAT Percentile Score | 47.86 | 44.51 | 41.54 | 0.01 |

Note: the joint p-value for age in 1985 does not match exactly, likely due to replication issues.

### 3.2. Table II

#### Table 5. P-values for tests of within-school differences between small, regular, and regular-with-aide classes by program entry grade

| Variable | K | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| Free Lunch | 0.46 | 0.29 | 0.58 | 0.18 |
| White/Asian | 0.66 | 0.28 | 0.18 | 0.27 |
| Age in 1985 | 0.44 | 0.12 | 0.43 | 0.48 |
| Attrition Rate | 0.01 | 0.37 | 0.85 | NA |
| Actual Class Size | 0.00 | 0.00 | 0.00 | 0.00 |
| SAT Percentile Score | 0.00 | 0.00 | 0.47 | 0.00 |

Note: some values do not match exactly, especially age, likely due to replication errors.

### 3.3. Table III

#### Table 6. Distribution of children across actual class sizes in grade 1, by random assignment group

| Actual class size in first grade | Small | Regular | Regular with Aide |
|---|---:|---:|---:|
| 12 | 24 | 0 | 0 |
| 13 | 182 | 0 | 0 |
| 14 | 252 | 0 | 0 |
| 15 | 465 | 0 | 0 |
| 16 | 256 | 16 | 0 |
| 17 | 561 | 17 | 0 |
| 18 | 108 | 36 | 0 |
| 19 | 57 | 76 | 57 |
| 20 | 20 | 200 | 120 |
| 21 | 0 | 378 | 378 |
| 22 | 0 | 594 | 330 |
| 23 | 0 | 437 | 460 |
| 24 | 0 | 384 | 264 |
| 25 | 0 | 175 | 225 |
| 26 | 0 | 130 | 234 |
| 27 | 0 | 54 | 108 |
| 28 | 0 | 28 | 56 |
| 29 | 0 | 29 | 58 |
| 30 | 0 | 30 | 30 |
| Average | 15.7 | 22.7 | 23.4 |

Replicated from Table III.

### 3.4. Figure I

**Figure 1.** Density plot of SAT percentile distributions for each class type and grade:

- kindergarten
- grade 1
- grade 2
- grade 3

Regular classes with aides are grouped together with regular classes in these plots.

### 3.5. Table V regenerated

#### Table 7. OLS and reduced-form estimates of the effect of class-size assignment on average SAT percentile score for kindergarten students

| Explanatory variable | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small class | 4.72* | 5.34*** | 5.32*** | 5.34*** | 4.72* | 5.34*** | 5.32*** | 5.34*** |
|  | (2.20) | (1.26) | (1.22) | (1.19) | (2.20) | (1.26) | (1.22) | (1.19) |
| Regular/aide class | -0.03 | 0.22 | 0.44 | 0.26 | -0.03 | 0.22 | 0.44 | 0.26 |
|  | (2.24) | (1.12) | (1.09) | (1.06) | (2.24) | (1.12) | (1.09) | (1.06) |
| White/Asian |  |  | 8.31*** | 8.41*** |  |  | 8.31*** | 8.41*** |
|  |  |  | (1.35) | (1.36) |  |  | (1.35) | (1.36) |
| Gender (girl) |  |  | 4.49*** | 4.41*** |  |  | 4.49*** | 4.41*** |
|  |  |  | (0.63) | (0.63) |  |  | (0.63) | (0.63) |
| Free lunch |  |  | -13.16*** | -13.08*** |  |  | -13.16*** | -13.08*** |
|  |  |  | (0.78) | (0.77) |  |  | (0.78) | (0.77) |
| White teacher |  |  |  | -1.22 |  |  |  | -1.22 |
|  |  |  |  | (2.15) |  |  |  | (2.15) |
| Teacher’s experience |  |  |  | 0.26* |  |  |  | 0.26* |
|  |  |  |  | (0.10) |  |  |  | (0.10) |
| Master’s degree |  |  |  | -0.49 |  |  |  | -0.49 |
|  |  |  |  | (1.08) |  |  |  | (1.08) |
| School fixed effects | No | Yes | Yes | Yes | No | Yes | Yes | Yes |
| $R^2$ | 0.01 | 0.25 | 0.31 | 0.31 | 0.01 | 0.25 | 0.31 | 0.31 |

Standard errors are in parentheses.  
* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$

#### Table 8. OLS and reduced-form estimates of the effect of class-size assignment on average SAT percentile score for grade 1 students

| Explanatory variable | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small class | 8.60*** | 8.45*** | 7.93*** | 7.61*** | 7.59*** | 7.21*** | 6.83*** | 6.60*** |
|  | (1.98) | (1.21) | (1.17) | (1.17) | (1.77) | (1.14) | (1.10) | (1.10) |
| Regular/aide class | 3.42 | 2.20* | 2.21* | 1.77 | 1.94 | 1.71* | 1.66* | 1.53* |
|  | (2.05) | (0.99) | (0.97) | (0.97) | (1.12) | (0.80) | (0.76) | (0.76) |
| White/Asian |  |  | 6.99*** | 6.98*** |  |  | 6.87*** | 6.86*** |
|  |  |  | (1.18) | (1.19) |  |  | (1.18) | (1.19) |
| Gender (girl) |  |  | 3.79*** | 3.83*** |  |  | 3.76*** | 3.80*** |
|  |  |  | (0.56) | (0.56) |  |  | (0.56) | (0.56) |
| Free lunch |  |  | -13.43*** | -13.53*** |  |  | -13.59*** | -13.70*** |
|  |  |  | (0.87) | (0.87) |  |  | (0.88) | (0.88) |
| White teacher |  |  |  | -4.05* |  |  |  | -4.14* |
|  |  |  |  | (1.95) |  |  |  | (1.97) |
| Teacher experience |  |  |  | 0.06 |  |  |  | 0.07 |
|  |  |  |  | (0.06) |  |  |  | (0.06) |
| Master’s degree |  |  |  | 0.34 |  |  |  | 0.48 |
|  |  |  |  | (1.07) |  |  |  | (1.10) |
| School fixed effects | No | Yes | Yes | Yes | No | Yes | Yes | Yes |
| $R^2$ | 0.02 | 0.24 | 0.30 | 0.30 | 0.01 | 0.23 | 0.29 | 0.30 |

Standard errors are in parentheses.  
* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$

#### Table 9. OLS and reduced-form estimates of the effect of class-size assignment on average SAT percentile score for grade 2 students

| Explanatory variable | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small class | 5.96** | 6.28*** | 5.83*** | 5.75*** | 5.30** | 5.46*** | 5.27*** | 5.24*** |
|  | (1.98) | (1.29) | (1.23) | (1.22) | (1.70) | (1.16) | (1.10) | (1.09) |
| Regular/aide class | 2.07 | 1.97 | 1.74 | 1.67 | 0.59 | 1.56 | 1.29 | 1.30 |
|  | (2.07) | (1.10) | (1.07) | (1.06) | (1.25) | (0.87) | (0.82) | (0.81) |
| White/Asian |  |  | 7.05*** | 7.06*** |  |  | 6.98*** | 7.00*** |
|  |  |  | (1.18) | (1.18) |  |  | (1.19) | (1.19) |
| Gender (girl) |  |  | 3.30*** | 3.27*** |  |  | 3.30*** | 3.27*** |
|  |  |  | (0.60) | (0.60) |  |  | (0.60) | (0.60) |
| Free lunch |  |  | -13.55*** | -13.55*** |  |  | -13.68*** | -13.67*** |
|  |  |  | (0.72) | (0.72) |  |  | (0.73) | (0.73) |
| White teacher |  |  |  | 0.43 |  |  |  | 0.46 |
|  |  |  |  | (1.75) |  |  |  | (1.77) |
| Teaching experience |  |  |  | 0.10 |  |  |  | 0.10 |
|  |  |  |  | (0.06) |  |  |  | (0.07) |
| Master’s degree |  |  |  | -1.00 |  |  |  | -1.10 |
|  |  |  |  | (1.06) |  |  |  | (1.05) |
| School fixed effects | No | Yes | Yes | Yes | No | Yes | Yes | Yes |
| $R^2$ | 0.01 | 0.22 | 0.28 | 0.28 | 0.01 | 0.21 | 0.28 | 0.28 |

Standard errors are in parentheses.  
* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$

#### Table 10. OLS and reduced-form estimates of the effect of class-size assignment on average SAT percentile score for grade 3 students

| Explanatory variable | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small class | 5.32** | 5.57*** | 5.02*** | 5.12*** | 5.50*** | 5.42*** | 5.31*** | 5.39*** |
|  | (1.93) | (1.22) | (1.20) | (1.22) | (1.47) | (1.08) | (1.03) | (1.06) |
| Regular/aide class | -0.28 | -0.19 | -0.36 | -0.48 | -0.38 | 0.09 | 0.09 | 0.06 |
|  | (1.96) | (1.13) | (1.12) | (1.10) | (1.18) | (0.86) | (0.81) | (0.80) |
| White/Asian |  |  | 6.10*** | 6.09*** |  |  | 5.95*** | 5.95*** |
|  |  |  | (1.45) | (1.44) |  |  | (1.44) | (1.43) |
| Gender (girl) |  |  | 4.14*** | 4.14*** |  |  | 4.15*** | 4.15*** |
|  |  |  | (0.66) | (0.66) |  |  | (0.66) | (0.66) |
| Free lunch |  |  | -13.03*** | -13.00*** |  |  | -13.21*** | -13.20*** |
|  |  |  | (0.81) | (0.81) |  |  | (0.82) | (0.82) |
| White teacher |  |  |  | 0.37 |  |  |  | -0.05 |
|  |  |  |  | (1.80) |  |  |  | (1.80) |
| Teacher’s experience |  |  |  | 0.06 |  |  |  | 0.05 |
|  |  |  |  | (0.06) |  |  |  | (0.06) |
| Master’s degree |  |  |  | 0.74 |  |  |  | 0.56 |
|  |  |  |  | (1.18) |  |  |  | (1.18) |
| School fixed effects | No | Yes | Yes | Yes | No | Yes | Yes | Yes |
| $R^2$ | 0.01 | 0.17 | 0.22 | 0.22 | 0.01 | 0.16 | 0.22 | 0.22 |

Standard errors are in parentheses.  
* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$

### 3.6. Table VII regenerated

#### Table 11. OLS and 2SLS estimates of the effect of class size on achievement

| Grade | OLS | 2SLS | Sample size |
|---|---:|---:|---:|
| K | -0.62 (0.14) | -0.72 (0.14) | 5840 |
| 1 | -0.85 (0.13) | -0.67 (0.15) | 6455 |
| 2 | -0.60 (0.12) | -0.53 (0.13) | 6011 |
| 3 | -0.61 (0.13) | -0.67 (0.13) | 6124 |

The dependent variable is the average SAT percentile score. The models control for school fixed effects, student race, gender, and free lunch status, as well as teachers’ race, experience, and education. Standard errors in parentheses are robust to correlated errors among students.

## 4. Exploratory Investigations

### 4.1. How random was initial assignment?

As discussed above, this experiment does not have a control group in the usual sense. In a perfect setting, we would observe the same student under different treatment assignments. Since that is impossible, we instead assess whether the randomization appears plausible by examining whether students in different treatment groups differ substantially in observed characteristics. Tables 1 to 4 address this question.

Students were assigned to groups when they entered the program. By comparing SAT percentile scores and background covariates across groups, we can assess whether treatment groups differed systematically at baseline. Tables 1 to 4 show that some differences are statistically significant in some grades, which suggests the value of including controls in regression models.

Schools also strongly influence outcomes and group composition. For example, a school in a poorer neighborhood may have more students eligible for free lunch. Therefore, it is useful to inspect between-group differences conditional on school fixed effects. Table 5 reports joint F-tests comparing small, regular, and regular-with-aide classes within schools across entry grades.

None of the main background variables, namely free lunch, White/Asian status, and age, are significantly different across treatment assignment at the 10 percent level. This evidence suggests that assignment was fairly random. Krueger (1999) also carried out the same analysis pooling all four grades and similarly found no strong association in background variables.

### 4.2. How distributed were assignment groups?

Not all classes within the same assignment group had the same number of students. As shown in Table 6, small classes had fewer students than regular classes, which is consistent with treatment design. The minimum and maximum observed class sizes in the small treatment group were 12 and 20, respectively. Some regular classes had fewer than 20 students, but most had 20 or more.

This provides evidence that the assigned treatments were implemented broadly in line with the definitions used by the researchers. This is another useful check for causal interpretation.

### 4.3. Are smaller classes better?

Figure 1 shows the kernel density of average test scores by class type and grade. Each density plot compares SAT percentile scores between small and regular classes. Regular classes with aides are grouped with regular classes in these plots.

From the figure, the density for small classes shifts to the right of the density for regular classes around a SAT percentile score of 50. Students in smaller classes appear to perform better than students in regular classes. In fact, many students above the median appear to come from small classes, and many below the median appear to come from regular classes.

Krueger (1999) tests the robustness of this conclusion using more formal regression tools, which are discussed below.

## 5. Regression: Design and Results

Krueger (1999) uses the following regression model to estimate the effect of school resources on student achievement:

$$
Y_{ics} = \beta_0 + \beta_1 SMALL_{cs} + \beta_2 REG/A_{cs} + \beta_3 X_{ics} + \alpha_s + \epsilon_{ics}.
$$

where $Y_{ics}$ is the average SAT percentile score of student $i$ in class $c$ at school $s$, $SMALL_{cs}$ and $REG/A_{cs}$ are class assignment indicators, $X_{ics}$ is the set of student background and school characteristics, and $\alpha_s$ captures school-level factors.

### 5.1. Design

#### Ordinary Least Squares (OLS)

The regression equation above can be estimated by ordinary least squares. Because some assignments changed over time, Krueger (1999) also estimates models using the student’s initial assignment. These are labeled reduced-form models because initial assignment is used as an excluded variable correlated with actual class size.

To interpret coefficients causally, we still need assumptions about the experimental design. Even though the study is close to a randomized experiment, omitted factors may remain. For example, the stochastic error term $\epsilon_{ics}$ may include teacher quality, motivation, or other classroom-specific influences. Thus, causal interpretation requires assuming that assignment to a small class is independent of such omitted factors.

#### Two-stage Least Squares (2SLS)

As Table 6 shows, actual class sizes overlap across assignment groups. A 2SLS strategy can account for this variation using initial assignment as an instrument for realized class size:

$$
CS_{ics} = \pi_0 + \pi_1 S_{ios} + \pi_2 R_{ios} + \pi_3 X_{ics} + \tau_{ics}
$$

$$
Y_{ics} = \beta_0 + \beta_1 CS_{ics} + \beta_2 X_{ics} + \alpha_s + \epsilon_{ics}
$$

where:

- $CS_{ics}$ is actual class size
- $S_{ios}$ is an indicator for initial assignment to a small class
- $R_{ios}$ is an indicator for initial assignment to a regular class

Under this setup, initial assignment generates exogenous variation in actual class size. Because assignment was randomized, the excluded instrument should not be correlated with $\epsilon_{ics}$. However, because there were non-random transfers between groups over time, this assumption is controversial beyond kindergarten, when no such switching had yet occurred.

Therefore, OLS and 2SLS estimates coincide most closely in kindergarten.

### 5.2. Results

Regression results from OLS are presented in Tables 7 to 10. The main conclusion is that students in smaller classrooms perform better than students in larger classrooms.

Students in small classrooms score:

- about 5 percentile points higher in kindergarten
- about 8.6 percentile points higher in grade 1
- between 5 and 6 percentile points higher in grade 2
- around 5.1 to 5.4 percentile points higher in grade 3

For kindergarten, there is no difference between reduced-form and OLS estimates because students could not change assignments before the academic year started. In column 4, which controls for background variables, being in a small class increases the SAT percentile score by 5.34 points, and the estimate is highly significant. Having a teaching aide does not increase achievement.

For grade 1, the increase is 7.61 percentile points under current assignment and 6.60 under initial assignment. Having a teaching aide raises scores by roughly 1.5 to 1.7 points, but these effects are generally not statistically significant.

For grade 2, students in smaller classes score about 5.24 to 5.75 points higher, and these effects are statistically significant. Again, teaching aides do not appear to improve outcomes substantially.

For grade 3, the same qualitative pattern remains: students in smaller classes score about 5.12 to 5.39 points higher, while teacher aides do not yield major gains.

The 2SLS estimates in Table 11 are slightly larger in magnitude. This matches the general conclusion in Krueger (1999), although in my replication the coefficients are slightly smaller for grades 1 and 2. Overall, the results suggest that students in smaller classes tend to score higher after entering the program.

## 6. Limitations

A major limitation, in my view, is the lack of a direct measure of teacher quality. Teachers may differ in motivation, pedagogy, or ability to explain concepts. The study does not measure teaching quality directly. If teacher race, experience, and education are orthogonal to true teaching quality, as is plausible, then that aspect remains unobserved. For more discussion of design limitations, see Hanushek (1999).

If the difference were this strong, one might also consider a regression discontinuity design, provided the assignment mechanism could support it. Under sufficiently strong assumptions, such an approach might help estimate the difference between small and regular classes more directly.

Furthermore, Nye et al. (1999) question the validity of the results because of high attrition. More than half of the students who joined in kindergarten had left the experiment by grade 3. However, they also argue that these inconsistencies were not large enough to invalidate the experiment’s core conclusions.

## 7. Concluding Remarks

In this replication paper, I attempted to rediscover the results of Krueger (1999). I found that most tables and figures could be reproduced using the methods described in the original paper.

In one sentence, the main conclusion is that students in small classes perform better than students in regular classes, while having a teacher’s aide does not appear to have a statistically significant effect on student achievement.

Finally, I discussed the study’s limitations and possible ways it could be extended or improved.

## Acknowledgement

Sincere thanks to Dr. Carruthers and Cathy Wu for their support and help throughout the project. Most of the codebase was originally written by Dr. Carruthers, Adrienne Sudbury, Ge Wu, and others. I reused parts of it and added additional material for this replication.

The complete codebase is available at:  
[https://github.com/harshvardhaniimi/krueger1991-replication](https://github.com/harshvardhaniimi/krueger1991-replication)

## References

- Achilles, C., Bain, H. P., Bellott, F., Boyd-Zaharias, J., Finn, J., Folger, J., Johnston, J., and Word, E. (2008). *Tennessee’s Student Teacher Achievement Ratio (STAR) Project*.
- Hanushek, E. A. (1999). *Some findings from an independent investigation of the Tennessee STAR experiment and from other investigations of class size effects*. *Educational Evaluation and Policy Analysis*, 21(2), 143-163.
- Krueger, A. B. (1999). *Experimental estimates of education production functions*. *The Quarterly Journal of Economics*, 114(2), 497-532.
- Nye, B., Hedges, L. V., and Konstantopoulos, S. (1999). *The long-term effects of small classes: A five-year follow-up of the Tennessee class size experiment*. *Educational Evaluation and Policy Analysis*, 21(2), 127-142.
