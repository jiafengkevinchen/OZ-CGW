# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %load_ext rpy2.ipython
# %matplotlib inline

# %%
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from IPython.display import clear_output
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.formula import api as smf

from util.data import (
    generate_zillow_data,
    get_census_shapefiles,
    get_census_tract_attributes,
    get_oz_data,
    get_pairs,
    get_file_suffix,
)
from util.plot import plot_with_error_bars

try:
    import pandas_tools.latex as tex
except ImportError:
    print("Run pip install git+https://github.com/jiafengkevinchen/pandas_tools")

try:
    from janitor.utils import skiperror, skipna
except ImportError:
    try:
        from pandas_tools.latex import skiperror, skipna
    except ImportError:
        print("Run pip install git+https://github.com/jiafengkevinchen/pandas_tools")

sns.set_color_codes()
sns.set_style("white")
plt.rcParams["figure.dpi"] = 100

# %%
with open("settings.json", "r") as f:
    settings = json.load(f)

LIC_ONLY = settings["LIC_ONLY"]
OVERWRITE = settings["OVERWRITE"]
START_YEAR = settings["START_YEAR"]
file_suffix = get_file_suffix(LIC_ONLY)


# %%
# Load data
df, annual_change, oz_irs, oz_ui = get_oz_data(overwrite=OVERWRITE, lic_only=LIC_ONLY)
tracts_df, var_dict = get_census_tract_attributes()


# %% [markdown]
# # Zillow design

# %% [markdown]
# ## No covariates

# %%
# Plot raw trend plot
ax = annual_change.plot_with_error_bars(start_year=START_YEAR)
plt.xticks(list(range(START_YEAR, 2020)))
sns.despine()
plt.ylabel("Annual Change (%) in HPI [FHFA Tract-level index]")
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    np.array(handles)[[-1, -3, -2]], np.array(labels)[[-1, -3, -2]], frameon=False
)
plt.savefig(f"exhibits/zillow_raw_trend{file_suffix}.pdf")

# %%
zillow_data = (
    df.generate_zillow_data(tracts_df, yr=START_YEAR)
    .assign(post=lambda x: (x.year >= 2018).astype(int))
    .assign(treatment_and_post=lambda x: x.treatment * x.post)
)
assert zillow_data.notnull().all().all()

# %%
# %Rpush zillow_data

# %% {"language": "R"}
# zillow_data$tract <- factor(zillow_data$tract)
# zillow_data$year <- relevel(factor(zillow_data$year), ref = "2017")

# %% {"language": "R"}
# source("util/twfe_did.r")  # Contains the utilities for R code
# zillow_model <- fit_did_lfe(
#     fmla = annual_change ~ 1 + treatment * post | year + tract | 0 | state_abbr,
#     pretest_fmla = annual_change ~ 1 + year * treatment | year + tract | 0 | state_abbr,
#     pretest_cols = c(
#       "year2014:treatmentTRUE",
#       "year2015:treatmentTRUE",
#       "year2016:treatmentTRUE"
#     ),
#     data = zillow_data
# )

# %%
# Collect pretest coefficients into a DataFrame
pretest_coefs = %R coeftest(zillow_model$model_pretest)
pretest_coefs = pd.DataFrame(
    pretest_coefs[-5:, :2],
    index=[2014, 2015, 2016, 2018, 2019],  # TODO: Refactor code when changing START_YEAR
    columns=["coef", "se"],
)
pretest_coefs.loc[2017, :] = [0, 0]
pretest_coefs = pretest_coefs.sort_index()

# Important stats that go in the table; TODO: naming
model_results = %R coeftest(zillow_model$model)
tau, se, _, pval = model_results[-1, :]
pretest = %R zillow_model$lh_pretest
pval_pretest, chi2, degfree = pretest[:3]
control, treat = zillow_data.groupby("treatment")["tract"].nunique()

# %% [markdown]
# ## Covariates

# %% {"language": "R"}
# zillow_model_cov <- fit_did_lfe(
#     fmla = annual_change ~ 1 + treatment * post + year * (log_median_household_income + total_housing + pct_white + pct_higher_ed + pct_rent + pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed) | year + tract | 0 | state_abbr, 
#     pretest_fmla = annual_change ~ 1 + year * treatment + year * (log_median_household_income + total_housing + pct_white + pct_higher_ed + pct_rent + pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed) | year + tract | 0 | state_abbr, 
#     pretest_cols = c("year2014:treatmentTRUE", "year2015:treatmentTRUE", "year2016:treatmentTRUE"), 
#     data = zillow_data
# )

# %%
# Collect stats for the table
# zillow_model_cov = %Rget zillow_model_cov
coefs = %R coeftest(zillow_model_cov$model)
tau_cov, se_cov, _, pval_cov = coefs[~np.isnan(coefs)[:, 0]][0, :]
pretest = %R zillow_model_cov$lh_pretest
pretest_pval_cov = pretest[0]


# Add pretest coefs from with-covariate design to the DataFrame
# TODO: refactor code when changing START_YEAR
pretest_coefs_cov = %R coeftest(zillow_model_cov$model_pretest)
pretest_coefs_cov = pretest_coefs_cov[~np.isnan(pretest_coefs_cov)[:, 0]][:5]
pretest_coefs.loc[[2014, 2015, 2016, 2018, 2019], "coef_cov"] = pretest_coefs_cov[:, 0]
pretest_coefs.loc[[2014, 2015, 2016, 2018, 2019], "se_cov"] = pretest_coefs_cov[:, 1]
pretest_coefs = pretest_coefs.fillna(0)

# %%
table = pd.DataFrame(
    {
        "TWFE": [tau, f"({se})", pval, pval_pretest, f"({treat}, {control})", "No"],
        "TWFE ": [
            tau_cov,
            f"({se_cov})",
            pval_cov,
            pretest_pval_cov,
            f"({treat}, {control})",
            "Yes",
        ],
    },
    index=[
        "$\hat\tau$",
        "",
        "$p$-value",
        "Pre-trend test $p$-value",
        "$(N_1, N_0)$",
        "Covariates",
    ],
)
table.loc["Sample", :] = f"Balanced ({START_YEAR}--2019)"

# %%
table

# %% [markdown]
# ## Sparse covariates

# %% {"language": "R"}
# zillow_model_cov_sp <- fit_did_lfe(
#     fmla = annual_change ~ 1 + treatment * post + year * (log_median_household_income + pct_white) | year + tract | 0 | state_abbr, 
#     pretest_fmla = annual_change ~ 1 + year * treatment + year * (log_median_household_income + pct_white) | year + tract | 0 | state_abbr, 
#     pretest_cols = c("year2014:treatmentTRUE", "year2015:treatmentTRUE", "year2016:treatmentTRUE"), 
#     data = zillow_data
# )

# %%
coef_cov_sp = %R coeftest(zillow_model_cov_sp$model)
coef_cov_sp = coef_cov_sp[~np.isnan(coef_cov_sp[:, 0])]

# %%
tau_sp, se_sp, _, _ = coef_cov_sp[0, :]

# %% [markdown]
# ## Weighting estimators

# %% [markdown]
# ## Callaway--Sant'Anna

# %%
tracts_data = zillow_data.assign(
    first_treat=lambda x: np.where(x["treatment"].values, 2018, 0)
)

# %%
# %Rpush tracts_data

# %% {"language": "R"}
# library(did)
#
# out <- mp.spatt(
#     annual_change ~ treatment,
#     xformla=~log_median_household_income + total_housing +
#     pct_white + pct_higher_ed + pct_rent +
#     pct_native_hc_covered + pct_poverty +
#     pct_supplemental_income + pct_employed,
#     data=tracts_data,
#     panel=TRUE,
#     first.treat.name="first_treat",
#     idname="tract",
#     tname="year",
#     bstrap=FALSE,
#     se=TRUE,
#     cband=FALSE
# )
#
# summary(out$aggte, type="dynamic")

# %%
cs_estimates = %R (out$aggte[1:2])
cs_pretest_stat = %R out$Wpval
coef_cs, se_cs = [c[0] for c in cs_estimates]

# %%
table["Weighting CS"] = [
    coef_cs,
    f"({se_cs})",
    (2 * (1 - scipy.stats.norm.cdf(abs(coef_cs / se_cs)))),
    cs_pretest_stat[0][0],
    f"({treat}, {control})",
    "Yes",
    f"Balanced ({START_YEAR}--2019)",
]

# %% [markdown]
# ## Doubly-robust

# %%
zillow_data_two_period = (
    df.generate_zillow_data(tracts_df, yr=2017)
    .set_index("tract")
    .assign(treated_and_post=lambda x: x.treatment & (x.year >= 2018))
)
covs = [
    "log_median_household_income",
    "total_housing",
    "pct_white",
    "pct_higher_ed",
    "pct_rent",
    "pct_native_hc_covered",
    "pct_poverty",
    "pct_supplemental_income",
    "pct_employed",
]

pre = zillow_data_two_period.query("year == 2017").sort_values("tract").reset_index()
post = (
    zillow_data_two_period.query("year == 2018 | year == 2019")
    .sort_values("tract")
    .groupby(["state_abbr", "tract"])
    .mean()
    .reset_index()
)
pre["year"] = 0
post["year"] = 1

two_periods = pd.concat([pre, post], axis=0).reset_index(drop=True).drop("geoid", axis=1)

# %%
# %Rpush two_periods

# %% {"language": "R"}
# library(DRDID)
# drdid_out <- drdid(yname = "annual_change", tname = "year", idname = "tract", dname = "treatment",
#              xformla= ~log_median_household_income + total_housing +
#     pct_white + pct_higher_ed + pct_rent +
#     pct_native_hc_covered + pct_poverty +
#     pct_supplemental_income + pct_employed,
#              data = two_periods, panel = TRUE)

# %%
drdid_att, drdid_se = %R c(drdid_out$ATT, drdid_out$se)

# %%
c, t = post.groupby("treatment").size()
table["Weighting DR"] = [
    drdid_att,
    f"({drdid_se})",
    (2 * (1 - scipy.stats.norm.cdf(abs(drdid_att / drdid_se)))),
    None,
    f"({t}, {c})",
    "Yes",
    f"Balanced (2017--2019)",
]

# %%
table

# %% [markdown]
# # Matched pair design

# %%
geos = get_census_shapefiles(
    oz_ui[["geoid", "designated"]]
    .rename_column("geoid", "tract")
    .rename_column("designated", "status")
).assign(status=lambda x: x.status.fillna("Ineligible"))

# %%
if not LIC_ONLY:
    geos.plot(
        column="status",
        linewidth=0.01,
        categorical=True,
        cmap="coolwarm",
        figsize=(7, 7),
        legend=True,
        legend_kwds=dict(frameon=False, loc=(1.05, 0)),
    )
    plt.ylim((24, 50))
    plt.xlim((-130, -65))
    plt.axis("off")
    plt.savefig("exhibits/map.png", dpi=200, bbox_inches="tight")

# %%
pair_df = geos.get_pairs(df)

# %%
# Check balance of covariates
tracts_df, var_dict = get_census_tract_attributes()

balance = (
    pair_df[["pair_id", "treatment", "tract"]]
    .drop_duplicates()
    .assign(geoid=lambda x: x.tract.astype(str).str.zfill(11))
    .drop("tract", axis=1)
    .merge(tracts_df, how="left", on=["geoid"])
    .drop("geoid", axis=1)
    .drop("master", axis=1)
)


def get_summary(balance):
    return tex.mathify_table(
        (
            balance.query("treatment == 'treated'")
            .set_index("pair_id")
            .drop("treatment", axis=1)
            - balance.query("treatment == 'untreated'")
            .set_index("pair_id")
            .drop("treatment", axis=1)
        )
        .describe()
        .T.assign(std=lambda x: x["std"] / (x["count"] ** 0.5))
        .assign(tstat=lambda x: x["mean"] / x["std"])
        .rename_column("std", "se")
    )


# %%
bal_table = (
    get_summary(balance)[["count", "mean", "se", "tstat"]]
    .T.drop(
        [
            "supplemental_income",
            "log_median_earnings",
            "log_median_household_income",
            "log_median_gross_rent",
        ],
        axis=1,
    )
    .T
)

# %%
bal_table.index = [
    "Population",
    "Employed pop.",
    "Avg. commute (min)",
    "Median household income",
    "Median earnings",
    "Total housing",
    "Median gross rent",
    "\% White",
    "\% Higher ed.",
    "\% Rent",
    "\% Healthcare",
    "\% Poverty",
    "\% Supplemental income",
    "\% Employed",
]

bal_table.columns = ["$N$", "Mean", "Standard Err.", "$t$-statistic"]

# %%
_ = bal_table.to_latex_table(
    caption="Covariate balance between geographical pairs (treated minus untreated)",
    label="paired_balance",
    filename=f"exhibits/paired_balance{file_suffix}.tex",
)

# %%
pair_diff_panel = pair_df.paired_data_for_did(yr=START_YEAR)

distance_summary = (
    pair_diff_panel[["pair_id", "dist"]].drop_duplicates()["dist"] * 6371
).describe()

pair_diff_panel = (
    pair_diff_panel.drop("dist", axis=1)
    .assign(treatment=lambda x: x.treatment == "treated")
    .set_index(["pair_id", "treatment", "year"])["annual_change"]
    .unstack(-2)
    .diff(axis=1)[True]
    .reset_index()
    .rename_column(True, "treated_minus_untreated")
    .assign(linear_year=lambda x: x.year.astype(float))
)

# %%
with open(f"exhibits/dist_between_pairs{file_suffix}.tex", "w") as f:
    f.write(
        f"The average centroid distance between pairs \
        is {tex.mathify(distance_summary['mean'])} \
        ({tex.mathify(distance_summary['std'])}) kilometers."
    )

# %%
# %Rpush pair_diff_panel

# %% {"language": "R"}
# library(lmtest)
# library(multiwayvcov)
# library(lfe)
#
# pair_diff_panel$post <- pair_diff_panel$year == "2018"
# pair_diff_panel$year <- relevel(factor(pair_diff_panel$year), ref = "2017")
#
# model_diff <- plm(
#   formula = treated_minus_untreated ~ 1 + year,
#   data = pair_diff_panel, model = "within", index = c("pair_id", "year")
# )
#
#
# print(
#   coeftest(model_diff,
#     vcov = vcovHC(model_diff, type = "sss", cluster = "group")
#   )
# )
#
# pair_pretest <- (
#   linearHypothesis(
#     model_diff,
#     c(
#       "year2014=0",
#       "year2015=0",
#       "year2016=0"
#     ),
#     vcov. = vcovHC(model_diff, type = "sss", cluster = "group")
#   )
# )
#
# model_diff_ <- plm(
#   formula = treated_minus_untreated ~ 1 + post,
#   data = pair_diff_panel, model = "within", index = c("pair_id", "year")
# )
#
# pair_coef <- coeftest(model_diff_,
#   vcov = vcovHC(model_diff_, type = "sss", cluster = "group")
# )
# pair_coef_names <- colnames(pair_coef)
#
#
# model_diff_trend <- plm(
#   formula = treated_minus_untreated ~ 1 + linear_year + post,
#   data = pair_diff_panel, model = "within", index = c("pair_id", "year")
# )
#
# pair_coef_trend <- (
#   coeftest(model_diff_trend,
#     vcov = vcovHC(model_diff_trend, type = "sss", cluster = "group")
#   )
# )
# pair_coef_trend_names <- colnames(pair_coef_trend)

# %%
pair_pretest = %Rget pair_pretest
pair_coef = %Rget pair_coef
pair_coef_names = %Rget pair_coef_names
pair_coef_trend = %Rget pair_coef_trend
pair_coef_trend_names = %Rget pair_coef_trend_names

# %%
table["Paired"] = [
    pair_coef[0][0],
    f"({pair_coef[0][1]})",
    pair_coef[0][-1],
    pair_pretest.iloc[-1, -1],
    (pair_diff_panel["pair_id"].nunique(), pair_diff_panel["pair_id"].nunique()),
    None,
    f"Paired ({START_YEAR}--2019)",
]

# %%
table["Paired"] = [
    pair_coef[0][0],
    f"({pair_coef[0][1]})",
    pair_coef[0][-1],
    pair_pretest.iloc[-1, -1],
    (pair_diff_panel["pair_id"].nunique(), pair_diff_panel["pair_id"].nunique()),
    None,
    f"Paired ({START_YEAR}--2019)",
]

table["Paired "] = [
    pair_coef_trend[1][0],
    f"({pair_coef_trend[1][1]})",
    pair_coef_trend[1][-1],
    None,
    (pair_diff_panel["pair_id"].nunique(), pair_diff_panel["pair_id"].nunique()),
    None,
    f"Paired ({START_YEAR}--2019)",
]
table.loc["Trend", :] = [None, None, None, None, "None", "Linear"]
table.loc["Model", :] = [
    "Within",
    "Within",
    "Weighting",
    "Weighting",
    "Within",
    "Within",
]

# %%
table.to_pickle(f"exhibits/table_script1{file_suffix}.pickle")
with open(f"exhibits/script1_data{file_suffix}.txt", "w") as f:
    print(f"{tau_sp} {se_sp}", file=f)

# %%
diff_stat = (
    pair_diff_panel.groupby("year")["treated_minus_untreated"]
    .agg(["mean", "std", "count"])
    .assign(se=lambda x: x["std"] / x["count"] ** 0.5)
)

# %%
plt.errorbar(
    x=pretest_coefs.index - 0.15,
    y=pretest_coefs["coef"],
    yerr=pretest_coefs["se"] * 1.96,
    ls="",
    marker="o",
    capsize=5,
    capthick=2,
    elinewidth=2,
    label="Without covariates",
)

plt.errorbar(
    x=pretest_coefs.index,
    y=pretest_coefs["coef_cov"],
    yerr=pretest_coefs["se_cov"] * 1.96,
    ls="",
    marker="o",
    capsize=5,
    capthick=2,
    elinewidth=2,
    label="With covariates",
)

plt.errorbar(
    y=diff_stat["mean"] - diff_stat["mean"].loc[2017],
    x=diff_stat.index + 0.15,
    yerr=1.96 * diff_stat["se"],
    ls="",
    marker="o",
    capsize=5,
    capthick=2,
    elinewidth=2,
    label="Paired",
)

plt.xticks(list(range(START_YEAR, 2020)))
plt.axhline(0, ls="--", color="k")
plt.axvline(2017.5, ls="-", color="r")
plt.legend(frameon=False)

plt.ylabel("Pre-trend test coefficients (base year 2017)")
plt.xlabel("Year")
sns.despine()
plt.savefig(f"exhibits/zillow_pretest_event_study_plot{file_suffix}.pdf")
