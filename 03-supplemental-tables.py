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
from util.data import get_oz_data, get_census_shapefiles, get_census_tract_attributes

import cenpy
from cenpy import products

import pandas as pd
import numpy as np
from statsmodels.formula import api as smf

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

# %% [markdown]
# # Table for covariates in census data

# %%
with open("data/tracts_covs_var_dict.json", "r") as f:
    covars = json.load(f)

# %%
covars = {
    **covars,
    **{
        "pct_white": "white_population / population",
        "minutes_commute": "minutes_commute / employed_population",
        "pct_higher_ed": "(associate + bachelor + professional_school + doctoral) / population",
        "pct_rent": "renter_occupied / total_housing",
        "pct_native_hc_covered": "native_born_hc_covered / native_born",
        "pct_poverty": "poverty / population",
        "log_median_earnings": "log(median_earnings)",
        "log_median_household_income": "log(median_household_income)",
        "log_median_gross_rent": "log(median_gross_rent)",
        "pct_supplemental_income": "supplemental_income / population",
        "pct_employed": "employed_population / population",
    },
}
rep = "\\_"
covars = {k: f"\\texttt{{{v.replace('_', rep)}}}" for k, v in covars.items()}

# %%
_ = (
    pd.Series(covars)
    .to_frame("description")
    .to_latex_table(
        filename="exhibits/defs.tex",
        caption="Variable definitions, ACS codes, descriptions, and transformations",
        label="defs",
        mathify_args=dict(texttt_index=True, texttt_column=True),
    )
)

# %% [markdown]
# # Summary statistics

# %%
tracts_df, var_dict = get_census_tract_attributes()

# %%
df, annual_change, oz_irs, oz_ui = get_oz_data()

# %%
tracts_df = tracts_df.merge(
    oz_ui.assign(geoid=lambda x: x.geoid.astype(str).str.zfill(11))[
        ["geoid", "designated"]
    ],
    how="left",
).fill_empty("designated", "Ineligible")

# %%
summary_stats = (
    tracts_df.groupby("designated")
    .describe()
    .stack(-1)
    .stack()
    .unstack(-1)
    .unstack(0)
    .T
)
tex.mathify_table(
    summary_stats.loc[
        ~summary_stats.reset_index()["level_0"].str.startswith("log_").values
    ]
)

# %%
two_sample_summary = (
    summary_stats.astype(float)
    .reset_index()
    .query("designated in ['Selected', 'Eligible, not selected']")
    .assign(std=lambda x: x["std"] / (x["count"] ** 0.5))
    .set_index(["level_0", "designated"])
    .unstack(-1)
)

# %%
two_sample_summary["t_stat"] = two_sample_summary["mean"].diff(axis=1)[
    "Selected"
] / np.sqrt((two_sample_summary["std"] ** 2).sum(axis=1))

# %%
two_sample_summary["Difference"] = (
    two_sample_summary["mean"]["Selected"]
    - two_sample_summary["mean"]["Eligible, not selected"]
)

# %%
tb = two_sample_summary[["mean", "Difference", "std", "t_stat"]]

# %%
tb.columns = pd.MultiIndex.from_arrays(
    np.array(
        [
            ("Mean", "Not Selected"),
            ("Mean", "Selected"),
            ("Diff.", ""),
            ("SE", "Not Selected"),
            ("SE", "Selected"),
            ("$t$", ""),
        ]
    ).T
)

# %%
tb = tb.loc[
    [
        "population",
        "employed_population",
        "minutes_commute",
        "median_earnings",
        "median_household_income",
        "total_housing",
        "median_gross_rent",
        "pct_white",
        "pct_higher_ed",
        "pct_rent",
        "pct_native_hc_covered",
        "pct_poverty",
        "pct_supplemental_income",
        "pct_employed",
    ],
    :,
]

# %%
tb.index = [
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

# %%
tbstr = tb.to_latex_table(
    caption="Balance of selected opportunity zones and eligible census tracts",
    notes="""\
    ``Not Selected\'\' refers to eligible but not selected opportunity zones.
    Difference is selected minus not selected. Two-sample $t$-statistic reported.""",
    additional_text="\\scriptsize",
    to_latex_args={"column_format" : "lccccccc"}
)

with open("exhibits/zillow_balance.tex", "w") as f:
    f.write(
        tbstr.replace("\multicolumn{2}{l}{Mean}", "\multicolumn{2}{c}{Mean}").replace(
            "\multicolumn{2}{l}{SE}", "\multicolumn{2}{c}{SE}"
        )
    )

# %%
