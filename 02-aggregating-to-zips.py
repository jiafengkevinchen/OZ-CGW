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
import scipy
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
    get_zips,
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

# %%
with open("settings.json", "r") as f:
    settings = json.load(f)

LIC_ONLY = settings["LIC_ONLY"]
OVERWRITE = settings["OVERWRITE"]
START_YEAR = settings["START_YEAR"]
file_suffix = get_file_suffix(LIC_ONLY)

# %%
table = pd.read_pickle(f"exhibits/table_script1{file_suffix}.pickle")
df, annual_change, oz_irs, oz_ui = get_oz_data()
tracts_df, var_dict = get_census_tract_attributes()

zip_panel = get_zips(start_year=START_YEAR, overwrite=OVERWRITE, lic_only=LIC_ONLY)

# %%
with open(f"data/zip_missing{file_suffix}.txt", "r") as f:
    n_selected_tracts_covered, non_missing_zips, total_zips = [
        float(s) for s in f.read().split()
    ]

# %%
with open(f"exhibits/zip_coverage{file_suffix}.tex", "w") as f:
    print(
        f"""\
        Although only {tex.mathify(non_missing_zips)} of
        the total {tex.mathify(total_zips)} ZIP codes with crosswalk data
        do not have missing data in 2018,
        these ZIP codes intersect with
        {tex.mathify(n_selected_tracts_covered)} selected Opportunity Zones.% """,
        file=f,
    )

# %%
zip_panel = zip_panel.query("status_eligible_not_selected + status_selected > 0").assign(post=lambda x : x.year >= 2018)

# %%
coverage = (
    zip_panel.query("year == 2018")[["zip_code", "status_selected"]]
    .drop_duplicates()
    .describe()
)

# %%
mean_coverage = coverage.loc["mean"].iloc[0]
median_coverage = coverage.loc["50%"].iloc[0]
coverage75 = coverage.loc["75%"].iloc[0]

with open(f"exhibits/zip_coverage_distro{file_suffix}.tex", "w") as f:
    print(
        f"""\
        The average ZIP code has {mean_coverage*100:.1f}\% of
        its addresses in a selected Opportunity Zone;
        the median ZIP code has {median_coverage*100:.1f}\%;
        and the 75th percentile has {coverage75*100:.1f}\%.%""",
        file=f,
    )

# %% [markdown]
# # Zillow design

# %%
# %Rpush zip_panel

# %% [markdown]
# ## TWFE

# %% {"language": "R"}
# source("util/twfe_did.r")
# zip_panel$zip_code <- factor(zip_panel$zip_code)
# zip_panel$year <- relevel(factor(zip_panel$year), ref = "2017")
# zip_panel$treatment <- zip_panel$status_selected
#
# model_pretest_zip <- fit_did_lfe(
#   fmla = annual_change ~ 1 + treatment * post | year + zip_code | 0 | zip_code,
#   pretest_fmla = annual_change ~ 1 + year * treatment | year + zip_code | 0 | zip_code,
#   data = zip_panel,
#   pretest_cols = c(
#     "year2014:treatment",
#     "year2015:treatment",
#     "year2016:treatment"
#   )
# )
#
# model_pretest_zip_covs <- fit_did_lfe(
#     fmla = annual_change ~ 1 + treatment * post + year * (log_median_household_income + total_housing + pct_white + pct_higher_ed + pct_rent + pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed) | year + zip_code | 0 | zip_code, 
#     pretest_fmla = annual_change ~ 1 + year * treatment + year * (log_median_household_income + total_housing + pct_white + pct_higher_ed + pct_rent + pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed) | year + zip_code | 0 | zip_code, 
#     pretest_cols = c("year2014:treatment", "year2015:treatment", "year2016:treatment"), 
#     data = zip_panel
# )

# %%
coefs = %R coeftest(model_pretest_zip$model)
tau, se, _, pval = coefs[~np.isnan(coefs)[:, 0]][0, :]

coefs = %R coeftest(model_pretest_zip_covs$model)
tau_cov, se_cov, _, pval_cov = coefs[~np.isnan(coefs)[:, 0]][0, :]

pretest_zip_pval = %R model_pretest_zip$lh_pretest
pretest_zip_pval = pretest_zip_pval[0]

pretest_zip_cov_pval = %R model_pretest_zip_covs$lh_pretest
pretest_zip_cov_pval = pretest_zip_cov_pval[0]
n = zip_panel["zip_code"].nunique()

# %%
table_zip = pd.DataFrame(
    {
        "TWFE": [
            tau,
            f"({se})",
            pval,
            pretest_zip_pval,
            n,
            "No",
            f"Unbalanced ({START_YEAR}--2019)",
        ],
        "TWFE ": [
            tau_cov,
            f"({se_cov})",
            pval_cov,
            pretest_zip_cov_pval,
            n,
            "Yes",
            f"Unbalanced ({START_YEAR}--2019)",
        ],
    },
    index=[
        r"$\hat \tau$",
        "",
        r"$p$-value",
        "Pre-trend test $p$-value",
        "$N$",
        "Covariates",
        "Sample",
    ],
)

# %% [markdown]
# ## TWFE variable selection

# %% {"language": "R"}
# model_pretest_zip_cov_test <- fit_did_lfe(
#     fmla = annual_change ~ 1 + treatment * post + year * (log_median_household_income + pct_white) | year + zip_code | 0 | zip_code, 
#     pretest_fmla = annual_change ~ 1 + year * treatment + year * (log_median_household_income + pct_white) | year + zip_code | 0 | zip_code, 
#     pretest_cols = c("year2014:treatment", "year2015:treatment", "year2016:treatment"), 
#   data = zip_panel
# )

# %%
coefs = %R coeftest(model_pretest_zip_cov_test$model)
t_spz, se_spz, _, _ = coefs[~np.isnan(coefs)[:, 0]][0, :]

with open(f"exhibits/script1_data{file_suffix}.txt", "r") as f:
    tau_sp, se_sp = map(float, f.read().split())

covariate_choice = f"""\
For Column (2), only including log median household income and
percent white as covariates gives {tex.mathify(tau_sp)} ({tex.mathify(se_sp)}) for the top panel and
{tex.mathify(t_spz)} ({tex.mathify(se_spz)}) for the bottom panel.
"""

# %% [markdown]
# ## Weighting

# %%
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

pct_treated = oz_ui.query("designated == 'Selected'").count().iloc[0] / len(tracts_df)

two_period_zip = (
    zip_panel.query("year == 2017 or year == 2018 or year == 2019")
    .set_index("zip_code")
    .assign(
        treatment=lambda x: (
            x.status_selected >= x.status_selected.quantile(1 - pct_treated)
        ).astype(int)
    )
    .groupby("zip_code")
    .filter(lambda x: not x[covs + ["annual_change"]].isnull().any().any())
)
pre = two_period_zip.query("year == 2017").sort_values("zip_code").reset_index()
post = (
    two_period_zip.query("year == 2018 or year == 2019")
    .groupby("zip_code")
    .mean()
    .sort_values("zip_code")
    .reset_index()
)
pre["year"] = 0
post["year"] = 1
two_periods = pd.concat([pre, post]).reset_index(drop=True)

# %%
# %Rpush two_periods

# %% {"language": "R"}
# library(DRDID)
# drdid_out <- drdid(
#     yname="annual_change",
#     tname="year",
#     idname="zip_code",
#     dname="treatment",
#     xformla=~log_median_household_income
#     + total_housing
#     + pct_white
#     + pct_higher_ed
#     + pct_rent
#     + pct_native_hc_covered
#     + pct_poverty
#     + pct_supplemental_income
#     + pct_employed,
#     data=two_periods,
#     panel=TRUE,
# )

# %%
drdid_att, drdid_se = %R c(drdid_out$ATT, drdid_out$se)

# %%
table_zip["Weighting DR"] = [
    drdid_att,
    f"({drdid_se})",
    (2 * (1 - scipy.stats.norm.cdf(abs(drdid_att / drdid_se)))),
    None,
    (int(post["treatment"].sum()), int((1 - post["treatment"]).sum())),
    "Yes",
    "Balanced (2017--2019)",
]
table_zip.loc["Model", :] = ["Within", "Within", "Weighting"]


# %%
def calculate_ci(table):
    upper = table.iloc[0,] + 1.96 * table.iloc[1,].str.slice(1, -1).astype(float)
    lower = table.iloc[0,] - 1.96 * table.iloc[1,].str.slice(1, -1).astype(float)
    return (
        tex.mathify_column(table.iloc[0,])
        + " ["
        + tex.mathify_column(lower)
        + ", "
        + tex.mathify_column(upper)
        + "]"
    )


# %%
if "TWFE" not in table.columns:
    table.columns = [
        "TWFE",
        "TWFE ",
        "Weighting CS",
        "Weighting DR",
        "Paired",
        "Paired ",
    ]

table.iloc[0, :] = calculate_ci(table)
table_zip.iloc[0, :] = calculate_ci(table_zip)

final_table = pd.concat(
    [
        pd.DataFrame(
            {c: "" for c in table.columns}, index=["\\textbf{Tract-level data}"]
        ),
        table.fillna("---").rename(index={c: f"\\quad {c}" for c in table.index}),
        pd.DataFrame(
            {c: "" for c in table.columns}, index=["\\midrule \\textbf{ZIP-level data}"]
        ),
        table_zip.fillna("---").rename(
            index={c: f"\\quad {c}" for c in table_zip.index}
        ),
    ],
    sort=False,
).fillna("")

_ = final_table.to_latex_table(
    caption="Estimation of ATT using FHFA Tract and ZIP-level data",
    label=f"tract_and_zip{file_suffix}",
    additional_text="\\scriptsize",
    notes=f"""\
    \\begin{{enumerate}}
    
    \\item Standard errors are in parenthesis and 95\% confidence intervals are in square brackets. 
    Standard errors are clustered at the state level for the tract-level analysis (top panel)
    and clustered at the ZIP level for the ZIP-level analysis (bottom panel). 
    Clustering the top panel at the tract level does not qualitatively change results.
    
    \\item Covariates include log median household income, total housing units, percent white,
    percent with post-secondary education,
    percent rental units, percent covered by health insurance among native-born individuals,
    percent below poverty line, percent receiving supplemental income, and percent employed.
    {covariate_choice}
    
    \\item Pretest for Column (2) interacts covariates with time dummies.
    
    \\item Years 2018 and 2019 are mean-aggregated in Column (4) since the doubly-robust estimation 
    only handles two periods.
    
    \\item Discrete treatment in Column (4) is defined as
    the highest {(1 - pct_treated) * 100:.1f}\% of treated
    tract coverage, so as to keep the percentage of treated ZIPs the same as treated tracts.
    
    \\end{{enumerate}}
    """,
    filename=f"exhibits/join_tab{file_suffix}.tex",
)


# %% [markdown]
# # Redoing ZIP code TWFE by splitting on covariate

# %%
def get_emp_pct():
    population = pd.read_csv("data/ACS_16_5YR_DP05/ACS_16_5YR_DP05_with_ann.csv").iloc[
        1:, [1, 3]
    ]
    population.columns = ["zip_code", "population"]
    population["population"] = population["population"].astype(float)
    population.loc[population["population"] < 1, "population"] = np.nan

    emp_pct = (
        pd.read_csv("data/zbp16totals.txt")[["zip", "emp"]]
        .assign(zip_code=lambda x: x.zip.astype(str).str.zfill(5))
        .merge(
            population.assign(zip_code=lambda x: x.zip_code.astype(str).str.zfill(5)),
            how="outer",
        )
        .drop("zip", axis=1)
        .assign(emp_pct=lambda x: x.emp / x.population)
    )[["zip_code", "emp_pct"]]
    return emp_pct


# %%
try:
    emp_pct = get_emp_pct()
except FileNotFoundError:
    !wget -q -O data/cbp.zip https://www2.census.gov/programs-surveys/cbp/datasets/2016/zbp16totals.zip
    !unzip -q data/cbp.zip -d data/

# %%
median_residential = (
    zip_panel.merge(emp_pct)[["zip_code", "emp_pct"]]
    .drop_duplicates()["emp_pct"]
    .median()
)
heterogeneous_effect = zip_panel.merge(emp_pct).assign(
    residential=lambda x: x.emp_pct < median_residential
)

# %%
# %Rpush heterogeneous_effect

# %% {"language": "R"}
# source("util/twfe_did.r")
# heterogeneous_effect$zip_code <- factor(heterogeneous_effect$zip_code)
# heterogeneous_effect$year <- relevel(factor(heterogeneous_effect$year), ref = "2017")
# heterogeneous_effect$treatment <- heterogeneous_effect$status_selected
#
# model_hetero <- model_hetero_cov <- fit_did_lfe(
#   fmla = annual_change ~ 1 + treatment * post * residential  | year + zip_code | 0 | zip_code,
#   pretest_fmla = annual_change ~ 1 + year * treatment * residential | year + zip_code | 0 | zip_code,
#   data = heterogeneous_effect,
#   pretest_cols = c(
#       "year2014:treatment:residentialTRUE",
#       "year2015:treatment:residentialTRUE",
#       "year2016:treatment:residentialTRUE",
#       "year2014:treatment",
#       "year2015:treatment",
#       "year2016:treatment"
#   )
# )
#
# model_hetero_few_cov <- fit_did_lfe(
#   fmla = annual_change ~ 1 + treatment * post * residential + year * (log_median_household_income
#     + total_housing
#    ) | year + zip_code | 0 | zip_code,
#   pretest_fmla = annual_change ~ 1 + year * treatment * residential + year * (log_median_household_income
#     + total_housing
#     ) | year + zip_code | 0 | zip_code,
#   data = heterogeneous_effect,
#   pretest_cols = c(
#       "year2014:treatment:residentialTRUE",
#       "year2015:treatment:residentialTRUE",
#       "year2016:treatment:residentialTRUE",
#       "year2014:treatment",
#       "year2015:treatment",
#       "year2016:treatment"
#   )
# )
#
# model_hetero_cov <- fit_did_lfe(
#   fmla = annual_change ~ 1 + treatment * post * residential + year * (log_median_household_income
#     + total_housing
#     + pct_white
#     + pct_higher_ed
#     + pct_rent
#     + pct_native_hc_covered
#     + pct_poverty
#     + pct_supplemental_income
#     + pct_employed
#    ) | year + zip_code | 0 | zip_code,
#   pretest_fmla = annual_change ~ 1 + year * treatment * residential + year * (log_median_household_income
#     + total_housing
#     + pct_white
#     + pct_higher_ed
#     + pct_rent
#     + pct_native_hc_covered
#     + pct_poverty
#     + pct_supplemental_income
#     + pct_employed
#     ) | year + zip_code | 0 | zip_code,
#   data = heterogeneous_effect,
#   pretest_cols = c(
#       "year2014:treatment:residentialTRUE",
#       "year2015:treatment:residentialTRUE",
#       "year2016:treatment:residentialTRUE",
#       "year2014:treatment",
#       "year2015:treatment",
#       "year2016:treatment"
#   )
# )

# %%
coefs = %R coeftest(model_hetero$model)
coef_mat = coefs[~np.isnan(coefs)[:, 0]][[0, -1], :]

coefs = %R coeftest(model_hetero_few_cov$model)
coef_mat_few_cov = coefs[~np.isnan(coefs)[:, 0]][[0, -1], :]

coefs = %R coeftest(model_hetero_cov$model)
coef_mat_cov = coefs[~np.isnan(coefs)[:, 0]][[0, -1], :]


pretest_zip_pval = %R model_hetero$lh_pretest
pretest_zip_pval = pretest_zip_pval[0]

pretest_zip_pval_few_cov = %R model_hetero_few_cov$lh_pretest
pretest_zip_pval_few_cov = pretest_zip_pval_few_cov[0]

pretest_zip_pval_cov = %R model_hetero_cov$lh_pretest
pretest_zip_pval_cov = pretest_zip_pval_cov[0]

# %%
hetero_table = pd.DataFrame(
    np.c_[coef_mat[:, :2], coef_mat_few_cov[:, :2], coef_mat_cov[:, :2]],
    columns=[
        "No Covariates",
        "SE",
        "Few Covariates",
        "SE_few_cov",
        "All Covariates",
        "SE_cov",
    ],
    index=[
        "Treatment $\\times$ Post",
        "Treatment $\\times$ Post $\\times$ Residential",
    ],
)

hetero_table = tex.consolidate_se(
    hetero_table,
    ["No Covariates", "Few Covariates", "All Covariates"],
    ["SE", "SE_few_cov", "SE_cov"],
)
hetero_table.loc["Pretest $p$-value"] = [pretest_zip_pval, pretest_zip_pval_few_cov, pretest_zip_pval_cov]
hetero_table.iloc[0, :] = calculate_ci(hetero_table)
hetero_table.iloc[2, :] = calculate_ci(hetero_table.iloc[[2, 3], :])

# %%
_ = hetero_table.to_latex_table(
    caption="Heterogeneous treatment effect by residential population",
    label=f"hetero_effects_{file_suffix}",
    additional_text="\\scriptsize",
    notes=f"""\
    \\begin{{enumerate}}
    
     \\item The table reports the regression \[
    Y_{{it}}^{{\obs}} = \\mu_i +   \\alpha_{{it}} + \\tau_0 \\one(t\\ge t_0, D_i=1) +
    \\tau_1
    \\one
    (t\\ge t_0, D_i=1, R_i = 1) + \\gamma  \\one(t\ge t_0, R_i = 1)
    \]
    and Treatment $\\times$ Post reports $\\tau_0$, while Treatment $\\times$ Post
    $\\times$ Residential reports $\\tau_1$. Here $\\alpha_{{it}} = \\alpha_t$ in the
    no-covariate specification and $\\alpha_{{it}} = \\alpha_{{t}}'X_i$ in the covariate
    specification. $R_i$ is an indicator for whether the employment to
    residential population ratio is lower than median.
    
    
    \\item Standard errors are in parenthesis and 95\% confidence intervals are in square brackets. 
    Standard errors are clustered at the ZIP level. 
    
    \\item ``All covariates'' consists of log median household income, total housing units, percent white,
    percent with post-secondary education,
    percent rental units, percent covered by health insurance among native-born individuals,
    percent below poverty line, percent receiving supplemental income, and percent employed. ``Few covariates''
    consists of only log median household income and total housing units.
    \\end{{enumerate}}
    """,
    filename=f"exhibits/hetero_effects_{file_suffix}.tex",
)

# %%
