# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Previous STA-207 Analysis in Python
#
# This notebook reproduces the main empirical analysis from `final.Rmd` in Python so we can inspect what the former baseline actually reported.

# %%
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
warnings.filterwarnings("ignore", message="As of SciPy 1.17", category=FutureWarning)

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
TAB_PATH = THIS_DIR / "STAR_Students.tab"
AER_PATH = ROOT_DIR / "data" / "star" / "STAR.csv"
OUTPUT_DIR = THIS_DIR / "previous_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
HTML_DIR = OUTPUT_DIR / "html"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)

DROP_SCHOOLS = [244728, 244796, 244736, 244839]
CLASS_TYPE_MAP = {1: "Small", 2: "Regular", 3: "Regular+Aide"}
SURBAN_MAP = {1: "Inner City", 2: "Suburban", 3: "Rural", 4: "Urban"}
GENDER_MAP = {1: "Male", 2: "Female"}
TRACE_MAP = {1: "White", 2: "Black", 3: "Other"}


def savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def make_class_level_data(star_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keep = ["g1classtype", "g1tchid", "g1schid", "g1tmathss"]
    data = star_raw.loc[:, keep].copy()
    summary = {
        "n_raw_rows": int(len(data)),
        "n_missing_teacher_id": int(data["g1tchid"].isna().sum()),
    }
    data = data.loc[data["g1tchid"].notna()].copy()
    summary["n_after_teacher_filter"] = int(len(data))
    summary["n_drop_school_rows"] = int(data["g1schid"].isin(DROP_SCHOOLS).sum())
    data = data.loc[~data["g1schid"].isin(DROP_SCHOOLS)].copy()
    summary["n_after_school_filter"] = int(len(data))
    data = data.loc[data["g1tmathss"].notna()].copy()
    summary["n_after_math_filter"] = int(len(data))
    class_level = (
        data.groupby("g1tchid", as_index=False)
        .mean(numeric_only=True)
        .sort_values("g1tchid")
        .reset_index(drop=True)
    )
    class_level["class_type_label"] = class_level["g1classtype"].round().astype("Int64").map(CLASS_TYPE_MAP)
    summary["n_class_level_rows"] = int(len(class_level))
    summary["n_unique_schools"] = int(class_level["g1schid"].nunique())
    return class_level, summary


def make_teacher_regression_data(star_raw: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "g1classtype",
        "g1schid",
        "g1surban",
        "g1tchid",
        "g1tgen",
        "g1trace",
        "g1thighdegree",
        "g1tcareer",
        "g1tyears",
        "g1tmathss",
    ]
    data = star_raw.loc[:, keep].copy()
    data = data.loc[data["g1tchid"].notna()].copy()
    data = data.loc[~data["g1schid"].isin(DROP_SCHOOLS)].copy()
    data = data.loc[data["g1tmathss"].notna()].copy()
    data = (
        data.groupby("g1tchid", as_index=False)
        .mean(numeric_only=True)
        .sort_values("g1tchid")
        .reset_index(drop=True)
    )
    data["class_type_label"] = data["g1classtype"].round().astype("Int64").map(CLASS_TYPE_MAP)
    data["school_area_label"] = data["g1surban"].round().astype("Int64").map(SURBAN_MAP)
    data["teacher_gender_label"] = data["g1tgen"].round().astype("Int64").map(GENDER_MAP)
    data["teacher_race_label"] = data["g1trace"].round().astype("Int64").map(TRACE_MAP).fillna("Other")
    return data


def tukey_additivity_test(y: pd.Series, a: pd.Series, b: pd.Series) -> dict[str, float]:
    df = pd.DataFrame({"y": y.to_numpy(), "a": a.to_numpy(), "b": b.to_numpy()}).dropna()
    df["a"] = pd.Categorical(df["a"])
    df["b"] = pd.Categorical(df["b"])
    ybar = float(df["y"].mean())
    ybari = df.groupby("a", observed=False)["y"].mean()
    ybarj = df.groupby("b", observed=False)["y"].mean()
    cell_means = df.pivot_table(index="a", columns="b", values="y", aggfunc="mean")
    aligned_cells = cell_means.loc[ybari.index, ybarj.index]
    term_matrix = np.outer(ybari.to_numpy() - ybar, ybarj.to_numpy() - ybar) * aligned_cells.to_numpy()
    numerator = float(np.nansum(term_matrix) ** 2)
    denominator = float(np.sum((ybari.to_numpy() - ybar) ** 2) * np.sum((ybarj.to_numpy() - ybar) ** 2))
    ssab = numerator / denominator
    additive_model = smf.ols("y ~ C(a) + C(b)", data=df).fit()
    additive_anova = sm.stats.anova_lm(additive_model, typ=1)
    ssrem = float(additive_anova.loc["Residual", "sum_sq"] - ssab)
    dfdenom = int(additive_anova.loc["Residual", "df"] - 1)
    statistic = float((ssab / ssrem) * dfdenom)
    p_value = float(1.0 - stats.f.cdf(statistic, 1, dfdenom))
    estimate = float(np.sqrt(ssab / denominator))
    return {
        "F": statistic,
        "p_value": p_value,
        "D_estimate": estimate,
        "num_df": 1.0,
        "denom_df": float(dfdenom),
    }


star_raw = pd.read_csv(TAB_PATH, sep="\t", low_memory=False)
class_level, class_level_summary = make_class_level_data(star_raw)
teacher_regression = make_teacher_regression_data(star_raw)

# %% [markdown]
# ## 1. Exploratory data analysis

# %%
eda_columns = [
    "gender",
    "g1classtype",
    "g1schid",
    "g1surban",
    "g1tchid",
    "g1tgen",
    "g1trace",
    "g1thighdegree",
    "g1tcareer",
    "g1tyears",
    "g1tmathss",
]
eda_data = star_raw.loc[:, eda_columns].copy()
missing_pct = (eda_data.isna().mean().sort_values(ascending=False) * 100.0).rename("missing_pct")
fig, ax = plt.subplots(figsize=(9, 4.5))
missing_pct.sort_values().plot(kind="barh", ax=ax, color="#6e6e6e")
ax.set_title("Missing values proportion")
ax.set_xlabel("Missing proportion (%)")
ax.set_ylabel("Feature")
savefig("figure_1_missing_proportion.png")

eda_box = class_level.copy()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=eda_box, x="class_type_label", y="g1tmathss", color="#b0b0b0", ax=axes[0])
axes[0].set_title("Math scaled scores by class type")
axes[0].set_xlabel("Class type")
axes[0].set_ylabel("Math scaled scores")
sns.histplot(eda_box["g1tmathss"], bins=30, color="#7f7f7f", ax=axes[1])
axes[1].set_title("Distribution of averaged math scores")
axes[1].set_xlabel("Averaged math scores")
axes[1].set_ylabel("Count")
savefig("figure_2_boxplot_histogram.png")

class_type_summary = (
    eda_box.groupby("class_type_label", observed=False)["g1tmathss"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)
class_type_summary.to_csv(TABLE_DIR / "class_type_summary.csv", index=False)

# %% [markdown]
# ## 2. Two-way ANOVA

# %%
anova_data = class_level.loc[:, ["g1tchid", "g1classtype", "g1schid", "g1tmathss", "class_type_label"]].copy()
anova_data = anova_data.rename(
    columns={
        "g1tchid": "teacher_id",
        "g1classtype": "class_type",
        "g1schid": "school_id",
        "g1tmathss": "math_scores",
    }
)
anova_data["class_type"] = anova_data["class_type"].round().astype("Int64").astype("category")
anova_data["school_id"] = anova_data["school_id"].round().astype("Int64").astype("category")

additive_model = smf.ols("math_scores ~ C(class_type) + C(school_id)", data=anova_data).fit()
transformed_model = smf.ols("I(math_scores ** -2) ~ C(class_type) + C(school_id)", data=anova_data).fit()
full_model = smf.ols("math_scores ~ C(class_type) * C(school_id)", data=anova_data).fit()

anova_table = sm.stats.anova_lm(additive_model, typ=1).reset_index().rename(columns={"index": "source"})
anova_table.to_csv(TABLE_DIR / "anova_table_type1.csv", index=False)
interaction_compare = sm.stats.anova_lm(additive_model, full_model).reset_index().rename(columns={"index": "model"})
interaction_compare.to_csv(TABLE_DIR / "anova_additive_vs_full.csv", index=False)
tukey_additivity = tukey_additivity_test(anova_data["math_scores"], anova_data["class_type"], anova_data["school_id"])

print("ANOVA table")
print(anova_table)
print()
print("Additive vs full model comparison")
print(interaction_compare)
print()
print("Tukey additivity test")
print(tukey_additivity)

# %% [markdown]
# ## 3. Diagnostics and hypothesis testing

# %%
residuals = additive_model.resid
fitted = additive_model.fittedvalues
standardized_residuals = additive_model.get_influence().resid_studentized_internal

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(fitted, residuals, color="dimgray", alpha=0.8, s=18)
axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
axes[0].set_title("Residual plot")
axes[0].set_xlabel("Fitted scores")
axes[0].set_ylabel("Residuals")
stats.probplot(standardized_residuals, dist="norm", plot=axes[1])
axes[1].set_title("QQ plot")
axes[1].set_xlabel("Theoretical quantile")
axes[1].set_ylabel("Standardized residual quantile")
savefig("figure_3_diagnostics.png")

diag_plot = anova_data.copy()
diag_plot["residual"] = residuals
diag_plot["school_id_str"] = diag_plot["school_id"].astype(str)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.stripplot(data=diag_plot, x="class_type_label", y="residual", color="dimgray", alpha=0.7, ax=axes[0])
axes[0].set_title("Residual vs class type")
axes[0].set_xlabel("Class type")
axes[0].set_ylabel("Residuals")
sns.stripplot(data=diag_plot, x="school_id_str", y="residual", color="dimgray", alpha=0.6, size=2.5, ax=axes[1])
axes[1].set_title("Residual vs school id")
axes[1].set_xlabel("School id")
axes[1].set_ylabel("Residuals")
axes[1].tick_params(axis="x", labelbottom=False)
savefig("figure_4_more_residuals.png")

grouped_residuals = [
    group["math_scores"].to_numpy()
    for _, group in anova_data.groupby(["class_type", "school_id"], observed=False)
    if len(group) > 1
]
levene_stat, levene_p = stats.levene(*grouped_residuals, center="median")
shapiro_additive = stats.shapiro(additive_model.resid)
shapiro_transformed = stats.shapiro(transformed_model.resid)
ad_additive = stats.anderson(additive_model.resid, dist="norm")
ad_transformed = stats.anderson(transformed_model.resid, dist="norm")
tukey_pairs = pairwise_tukeyhsd(endog=anova_data["math_scores"], groups=anova_data["class_type_label"], alpha=0.05)

fig = tukey_pairs.plot_simultaneous(figsize=(8, 4.5))
fig.suptitle("Tukey HSD for class type", y=1.02)
savefig("figure_5_tukey_hsd.png")

print("Levene test:", {"statistic": float(levene_stat), "p_value": float(levene_p)})
print("Shapiro additive:", {"statistic": float(shapiro_additive.statistic), "p_value": float(shapiro_additive.pvalue)})
print("Shapiro transformed:", {"statistic": float(shapiro_transformed.statistic), "p_value": float(shapiro_transformed.pvalue)})

# %% [markdown]
# ## 4. Extensions from the former report

# %%
extension_available = AER_PATH.exists()
extension_notes: list[str] = []

if extension_available:
    aer_star = pd.read_csv(AER_PATH)
    if "rownames" in aer_star.columns:
        aer_star = aer_star.drop(columns=["rownames"])

    long_columns = [
        "gender",
        "birth",
        "star1",
        "star2",
        "star3",
        "math1",
        "math2",
        "math3",
        "school1",
        "school2",
        "school3",
        "degree1",
        "degree2",
        "degree3",
        "ladder1",
        "ladder2",
        "ladder3",
        "experience1",
        "experience2",
        "experience3",
        "schoolid1",
        "schoolid2",
        "schoolid3",
        "system1",
        "system2",
        "system3",
    ]
    long_data = aer_star.loc[:, long_columns].copy()
    nona_long = long_data.dropna().copy()

    sorted_school = nona_long.copy()
    sorted_school["school_mean"] = sorted_school.groupby("schoolid1")["math1"].transform("mean")
    school_order = (
        sorted_school.loc[:, ["schoolid1", "school_mean"]]
        .drop_duplicates()
        .sort_values("school_mean")["schoolid1"]
        .tolist()
    )
    sorted_school["schoolid1_ordered"] = pd.Categorical(sorted_school["schoolid1"], categories=school_order, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [4, 3]})
    sns.scatterplot(
        data=sorted_school,
        x="schoolid1_ordered",
        y="math1",
        hue="school1",
        alpha=0.3,
        linewidth=0.0,
        ax=axes[0],
    )
    axes[0].set_title("Math scores by ordered school id")
    axes[0].set_xlabel("School id")
    axes[0].set_ylabel("Math score")
    axes[0].tick_params(axis="x", labelbottom=False)
    box_area = teacher_regression.loc[teacher_regression["school_area_label"].notna()].copy()
    sns.boxplot(
        data=box_area,
        x="school_area_label",
        y="g1tmathss",
        hue="class_type_label",
        ax=axes[1],
    )
    axes[1].set_title("Class-level math scores by area and class type")
    axes[1].set_xlabel("Area")
    axes[1].set_ylabel("Math test scores")
    axes[1].legend(title="Class type", fontsize=8, title_fontsize=9)
    savefig("figure_6_school_location.png")

    lim_data_long = aer_star.loc[:, ["gender", "birth", "star1", "star2", "star3", "math1", "math2", "math3", "school1", "school2", "school3"]].copy()
    sort_keys = pd.DataFrame(
        {
            "star1": lim_data_long["star1"].isna().astype(int),
            "star2": lim_data_long["star2"].isna().astype(int),
            "star3": lim_data_long["star3"].isna().astype(int),
            "math1": lim_data_long["math1"].isna().astype(int),
            "math2": lim_data_long["math2"].isna().astype(int),
            "math3": lim_data_long["math3"].isna().astype(int),
        }
    )
    sorted_idx = sort_keys.sort_values(
        by=["star1", "star2", "star3", "math1", "math2", "math3"],
        ascending=False,
    ).index
    sorted_missing = lim_data_long.loc[sorted_idx]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.imshow(sorted_missing.isna().to_numpy(), aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_title("Missing data pattern")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Rows")
    ax.set_xticks(range(len(sorted_missing.columns)))
    ax.set_xticklabels(sorted_missing.columns, rotation=45, ha="right")
    savefig("figure_7_missing_data_pattern.png")

    alluvial_counts = (
        nona_long.groupby(["star1", "star2", "star3"], observed=False)
        .size()
        .reset_index(name="Freq")
        .sort_values(["star1", "star2", "star3"])
        .reset_index(drop=True)
    )
    alluvial_counts["star1_node"] = alluvial_counts["star1"] + "_1"
    alluvial_counts["star2_node"] = alluvial_counts["star2"] + "_2"
    alluvial_counts["star3_node"] = alluvial_counts["star3"] + "_3"
    links_stage_1 = alluvial_counts.groupby(["star1_node", "star2_node"], observed=False)["Freq"].sum().reset_index()
    links_stage_2 = alluvial_counts.groupby(["star2_node", "star3_node"], observed=False)["Freq"].sum().reset_index()
    links_stage_1["source"] = links_stage_1["star1_node"]
    links_stage_1["target"] = links_stage_1["star2_node"]
    links_stage_2["source"] = links_stage_2["star2_node"]
    links_stage_2["target"] = links_stage_2["star3_node"]
    links = pd.concat(
        [
            links_stage_1.loc[:, ["source", "target", "Freq"]],
            links_stage_2.loc[:, ["source", "target", "Freq"]],
        ],
        ignore_index=True,
    )
    node_names = pd.Index(pd.unique(pd.concat([links["source"], links["target"]], ignore_index=True)))
    node_lookup = {name: idx for idx, name in enumerate(node_names)}
    node_labels = [name.replace("_1", "").replace("_2", "").replace("_3", "") for name in node_names]
    node_colors = []
    for name in node_names:
        if "small" in name:
            node_colors.append("tomato")
        elif "regular+aide" in name:
            node_colors.append("khaki")
        else:
            node_colors.append("skyblue")
    link_colors = []
    for source_name in links["source"]:
        if "small" in source_name:
            link_colors.append("rgba(255,99,71,0.55)")
        elif "regular+aide" in source_name:
            link_colors.append("rgba(240,230,140,0.55)")
        else:
            link_colors.append("rgba(135,206,235,0.55)")
    sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=18,
                    line=dict(color="black", width=0.4),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=[node_lookup[value] for value in links["source"]],
                    target=[node_lookup[value] for value in links["target"]],
                    value=links["Freq"].tolist(),
                    color=link_colors,
                ),
            )
        ]
    )
    sankey.update_layout(title_text="Continuity of program", font_size=10)
    sankey.write_html(HTML_DIR / "figure_8_alluvial.html")

    final_long = nona_long.loc[(nona_long["star1"] == nona_long["star2"]) & (nona_long["star2"] == nona_long["star3"])].copy()
    final_long["id"] = final_long.index.astype(str)
    part_1 = final_long.loc[:, ["gender", "birth", "id", "star1", "math1", "school1", "degree1", "ladder1", "experience1", "schoolid1", "system1"]].rename(
        columns={
            "star1": "star",
            "math1": "math",
            "school1": "school",
            "degree1": "degree",
            "ladder1": "ladder",
            "experience1": "experience",
            "schoolid1": "schoolid",
            "system1": "system",
        }
    )
    part_1["year"] = "1"
    part_2 = final_long.loc[:, ["gender", "birth", "id", "star2", "math2", "school2", "degree2", "ladder2", "experience2", "schoolid2", "system2"]].rename(
        columns={
            "star2": "star",
            "math2": "math",
            "school2": "school",
            "degree2": "degree",
            "ladder2": "ladder",
            "experience2": "experience",
            "schoolid2": "schoolid",
            "system2": "system",
        }
    )
    part_2["year"] = "2"
    part_3 = final_long.loc[:, ["gender", "birth", "id", "star3", "math3", "school3", "degree3", "ladder3", "experience3", "schoolid3", "system3"]].rename(
        columns={
            "star3": "star",
            "math3": "math",
            "school3": "school",
            "degree3": "degree",
            "ladder3": "ladder",
            "experience3": "experience",
            "schoolid3": "schoolid",
            "system3": "system",
        }
    )
    part_3["year"] = "3"
    chrono = pd.concat([part_1, part_2, part_3], ignore_index=True)
    loc_summary = chrono.groupby(["school", "star", "year"], observed=False)["math"].mean().reset_index(name="locmean")
    star_summary = chrono.groupby(["star", "year"], observed=False)["math"].mean().reset_index(name="starmean")

    schools = [value for value in ["inner-city", "suburban", "rural", "urban"] if value in set(loc_summary["school"])]
    fig, axes = plt.subplots(1, len(schools) + 1, figsize=(3.8 * (len(schools) + 1), 4.5), sharey=True)
    if len(schools) == 0:
        axes = [axes]
    for idx, school_value in enumerate(schools):
        school_df = loc_summary.loc[loc_summary["school"] == school_value]
        sns.lineplot(data=school_df, x="star", y="locmean", hue="year", marker="o", ax=axes[idx])
        axes[idx].set_title(str(school_value))
        axes[idx].set_xlabel("Class type")
        axes[idx].set_ylabel("Average math score")
        axes[idx].legend_.remove()
    sns.lineplot(data=star_summary, x="star", y="starmean", hue="year", marker="o", ax=axes[-1])
    axes[-1].set_title("average")
    axes[-1].set_xlabel("Class type")
    axes[-1].set_ylabel("")
    savefig("figure_9_longitudinal_scores.png")

else:
    extension_notes.append(f"Skipped AER extension because {AER_PATH} was not found.")

# %% [markdown]
# ## 5. Regression with school fixed effects

# %%
regression_data = teacher_regression.copy()
regression_model = smf.ols(
    "g1tmathss ~ C(class_type_label) + g1tyears + C(teacher_gender_label) + C(g1tcareer) + C(teacher_race_label) + C(g1thighdegree) + C(g1schid)",
    data=regression_data,
).fit()
regression_anova = sm.stats.anova_lm(regression_model, typ=1).reset_index().rename(columns={"index": "source"})
regression_anova.to_csv(TABLE_DIR / "regression_anova.csv", index=False)
print("Regression ANOVA")
print(regression_anova)

reported_from_rmd = {
    "counts": {
        "n_raw_rows": 11601,
        "n_missing_teacher_id": 4772,
        "n_after_math_filter": 6334,
        "n_class_level_rows": 325,
        "n_unique_schools": 72,
    },
    "anova_table": {
        "class_type_sum_sq": 6559.0,
        "class_type_F": 22.6,
        "class_type_p_value": 3.35e-06,
        "school_sum_sq": 128366.0,
        "school_F": 6.2,
        "school_p_value_upper_bound": 2e-16,
        "residual_sum_sq": 73131.0,
    },
    "tukey_additivity_p_value": 0.4235,
    "regression_table": {
        "class_type_F": 17.76,
        "class_type_p_value": 6.45e-08,
        "teaching_experience_F": 0.79,
        "teaching_experience_p_value": 0.38,
        "teacher_gender_F": 0.39,
        "teacher_gender_p_value": 0.53,
        "teacher_race_F": 0.37,
        "teacher_race_p_value": 0.54,
        "career_ladder_F": 1.51,
        "career_ladder_p_value": 0.21,
        "highest_degree_F": 0.11,
        "highest_degree_p_value": 0.95,
        "residual_df": 239.0,
    },
}

python_vs_rmd = {
    "counts_match": {
        "n_raw_rows": class_level_summary["n_raw_rows"] == reported_from_rmd["counts"]["n_raw_rows"],
        "n_missing_teacher_id": class_level_summary["n_missing_teacher_id"] == reported_from_rmd["counts"]["n_missing_teacher_id"],
        "n_after_math_filter": class_level_summary["n_after_math_filter"] == reported_from_rmd["counts"]["n_after_math_filter"],
        "n_class_level_rows": class_level_summary["n_class_level_rows"] == reported_from_rmd["counts"]["n_class_level_rows"],
        "n_unique_schools": class_level_summary["n_unique_schools"] == reported_from_rmd["counts"]["n_unique_schools"],
    },
    "anova": {
        "class_type_sum_sq_python": float(anova_table.loc[anova_table["source"] == "C(class_type)", "sum_sq"].iloc[0]),
        "class_type_sum_sq_rmd": reported_from_rmd["anova_table"]["class_type_sum_sq"],
        "class_type_F_python": float(anova_table.loc[anova_table["source"] == "C(class_type)", "F"].iloc[0]),
        "class_type_F_rmd": reported_from_rmd["anova_table"]["class_type_F"],
        "class_type_p_python": float(anova_table.loc[anova_table["source"] == "C(class_type)", "PR(>F)"].iloc[0]),
        "class_type_p_rmd": reported_from_rmd["anova_table"]["class_type_p_value"],
        "school_sum_sq_python": float(anova_table.loc[anova_table["source"] == "C(school_id)", "sum_sq"].iloc[0]),
        "school_sum_sq_rmd": reported_from_rmd["anova_table"]["school_sum_sq"],
        "school_F_python": float(anova_table.loc[anova_table["source"] == "C(school_id)", "F"].iloc[0]),
        "school_F_rmd": reported_from_rmd["anova_table"]["school_F"],
        "residual_sum_sq_python": float(anova_table.loc[anova_table["source"] == "Residual", "sum_sq"].iloc[0]),
        "residual_sum_sq_rmd": reported_from_rmd["anova_table"]["residual_sum_sq"],
    },
    "tukey_additivity": {
        "p_value_python": tukey_additivity["p_value"],
        "p_value_rmd": reported_from_rmd["tukey_additivity_p_value"],
    },
    "regression": {
        "class_type_F_python": float(regression_anova.loc[regression_anova["source"] == "C(class_type_label)", "F"].iloc[0]),
        "class_type_F_rmd": reported_from_rmd["regression_table"]["class_type_F"],
        "class_type_p_python": float(regression_anova.loc[regression_anova["source"] == "C(class_type_label)", "PR(>F)"].iloc[0]),
        "class_type_p_rmd": reported_from_rmd["regression_table"]["class_type_p_value"],
        "teaching_experience_F_python": float(regression_anova.loc[regression_anova["source"] == "g1tyears", "F"].iloc[0]),
        "teaching_experience_F_rmd": reported_from_rmd["regression_table"]["teaching_experience_F"],
        "teaching_experience_p_python": float(regression_anova.loc[regression_anova["source"] == "g1tyears", "PR(>F)"].iloc[0]),
        "teaching_experience_p_rmd": reported_from_rmd["regression_table"]["teaching_experience_p_value"],
        "teacher_gender_F_python": float(regression_anova.loc[regression_anova["source"] == "C(teacher_gender_label)", "F"].iloc[0]),
        "teacher_gender_F_rmd": reported_from_rmd["regression_table"]["teacher_gender_F"],
        "teacher_gender_p_python": float(regression_anova.loc[regression_anova["source"] == "C(teacher_gender_label)", "PR(>F)"].iloc[0]),
        "teacher_gender_p_rmd": reported_from_rmd["regression_table"]["teacher_gender_p_value"],
        "highest_degree_F_python": float(regression_anova.loc[regression_anova["source"] == "C(g1thighdegree)", "F"].iloc[0]),
        "highest_degree_F_rmd": reported_from_rmd["regression_table"]["highest_degree_F"],
        "highest_degree_p_python": float(regression_anova.loc[regression_anova["source"] == "C(g1thighdegree)", "PR(>F)"].iloc[0]),
        "highest_degree_p_rmd": reported_from_rmd["regression_table"]["highest_degree_p_value"],
        "residual_df_python": float(regression_anova.loc[regression_anova["source"] == "Residual", "df"].iloc[0]),
        "residual_df_rmd": reported_from_rmd["regression_table"]["residual_df"],
    },
}

# %% [markdown]
# ## 6. Export the numbers we need

# %%
results: dict[str, Any] = {
    "paths": {
        "tab_path": TAB_PATH,
        "aer_path": AER_PATH,
        "output_dir": OUTPUT_DIR,
    },
    "class_level_summary": class_level_summary,
    "class_type_summary": class_type_summary.to_dict(orient="records"),
    "anova_table_type1": anova_table.to_dict(orient="records"),
    "anova_additive_vs_full": interaction_compare.to_dict(orient="records"),
    "tukey_additivity": tukey_additivity,
    "diagnostics": {
        "levene": {"statistic": float(levene_stat), "p_value": float(levene_p)},
        "shapiro_additive": {
            "statistic": float(shapiro_additive.statistic),
            "p_value": float(shapiro_additive.pvalue),
        },
        "shapiro_transformed": {
            "statistic": float(shapiro_transformed.statistic),
            "p_value": float(shapiro_transformed.pvalue),
        },
        "anderson_additive": {
            "statistic": float(ad_additive.statistic),
            "critical_values": [float(value) for value in ad_additive.critical_values],
            "significance_levels": [float(value) for value in ad_additive.significance_level],
        },
        "anderson_transformed": {
            "statistic": float(ad_transformed.statistic),
            "critical_values": [float(value) for value in ad_transformed.critical_values],
            "significance_levels": [float(value) for value in ad_transformed.significance_level],
        },
    },
    "tukey_hsd_table": pd.DataFrame(data=tukey_pairs._results_table.data[1:], columns=tukey_pairs._results_table.data[0]).to_dict(orient="records"),
    "regression_anova": regression_anova.to_dict(orient="records"),
    "reported_from_rmd": reported_from_rmd,
    "python_vs_rmd": python_vs_rmd,
    "extension_available": extension_available,
    "extension_notes": extension_notes,
    "generated_artifacts": {
        "figures": sorted(str(path.relative_to(ROOT_DIR)) for path in FIG_DIR.glob("*")),
        "tables": sorted(str(path.relative_to(ROOT_DIR)) for path in TABLE_DIR.glob("*")),
        "html": sorted(str(path.relative_to(ROOT_DIR)) for path in HTML_DIR.glob("*")),
    },
}

results_path = OUTPUT_DIR / "previous_results.json"
results_path.write_text(json.dumps(to_builtin(results), indent=2), encoding="utf-8")

summary_lines = [
    "Python reproduction of final.Rmd",
    f"Raw rows: {class_level_summary['n_raw_rows']}",
    f"Missing teacher id rows: {class_level_summary['n_missing_teacher_id']}",
    f"Rows after math filter: {class_level_summary['n_after_math_filter']}",
    f"Teacher-level rows: {class_level_summary['n_class_level_rows']}",
    f"Unique schools: {class_level_summary['n_unique_schools']}",
    f"ANOVA class type p-value: {float(anova_table.loc[anova_table['source'] == 'C(class_type)', 'PR(>F)'].iloc[0]):.6g}",
    f"ANOVA school p-value: {float(anova_table.loc[anova_table['source'] == 'C(school_id)', 'PR(>F)'].iloc[0]):.6g}",
    f"Tukey additivity p-value: {tukey_additivity['p_value']:.6g}",
    f"Levene p-value: {levene_p:.6g}",
    f"Regression class type p-value: {float(regression_anova.loc[regression_anova['source'] == 'C(class_type_label)', 'PR(>F)'].iloc[0]):.6g}",
    f"Counts match Rmd: {all(python_vs_rmd['counts_match'].values())}",
    f"ANOVA class sumsq delta vs Rmd: {python_vs_rmd['anova']['class_type_sum_sq_python'] - python_vs_rmd['anova']['class_type_sum_sq_rmd']:.6f}",
    f"Tukey additivity delta vs Rmd: {python_vs_rmd['tukey_additivity']['p_value_python'] - python_vs_rmd['tukey_additivity']['p_value_rmd']:.6f}",
]
summary_path = OUTPUT_DIR / "previous_summary.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print()
print("Saved results to", results_path)
print("Saved summary to", summary_path)
print("Generated figures:", len(list(FIG_DIR.glob("*"))))
