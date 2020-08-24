import json
import os

import cenpy
import geopandas as gpd
import janitor
import numpy as np
import pandas as pd
import pandas_flavor as pf
from sklearn.metrics.pairwise import haversine_distances
from tqdm.auto import tqdm

tqdm.pandas()


def get_file_suffix(lic_only):
    return "" if lic_only else "_lic_only_false"


def get_oz_data(overwrite=False, lic_only=True):
    """
    Get data:
    - Opportunity Zones from the Urban Institute
    - Opportunity Zones from IRS (unused)
    - FHFA tract level census data

    Parameters
    ----------
    overwrite : bool, optional
        Whether to overwrite cached data in PROJ/data, by default False
    lic_only : bool, optional
        Whether to include only low-income tracts in the eligible group, by default True
        Ignored if loading from cached data.

    Returns
    -------
    df: pd.DataFrame
        Merged dataframe of OZ data with housing prices
    annual_change: pd.DataFrame
        Aggregate statistics by selection status of the zones
    oz_irs: pd.DataFrame
        Data from IRS on OZs
    oz_ui: pd.DataFrame
        Data from the Urban Institute on OZs
    """

    file_suffix = get_file_suffix(lic_only)

    if not overwrite and os.path.isfile(f"data/tracts{file_suffix}.feather"):
        # If using cached data
        df = pd.read_feather(f"data/tracts{file_suffix}.feather")
        annual_change = (
            df[["tract", "state_abbr", "year", "annual_change", "status"]]
            .groupby(["status", "year"])["annual_change"]
            .agg(["mean", "std", "count"])
            .unstack(0)
        )
        oz_irs = pd.read_feather(f"data/oz_irs{file_suffix}.feather")
        oz_ui = pd.read_feather(f"data/oz_ui{file_suffix}.feather")
        return df, annual_change, oz_irs, oz_ui

    url_fhfa_price = (
        "https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_BDL_tract.csv"
    )
    url_irs_oz = "https://www.cdfifund.gov/Documents/Designated%20QOZs.12.14.18.xlsx"
    url_ui_oz = (
        "https://edit.urban.org/sites/default/files/urbaninstitute_"
        "tractlevelozanalysis_update1242018.xlsx"
    )

    # Loading data
    df = pd.read_csv(url_fhfa_price, na_values=".")
    print(f"There are {df.tract.nunique()} tracts in the FHFA data")
    oz_irs = pd.read_excel(url_irs_oz, header=4).clean_names().reset_index(drop=True)
    oz_ui = (
        pd.read_excel(url_ui_oz)
        .clean_names()
        .assign(
            designated=lambda x: np.where(
                x.designated.isnull(), "Eligible, not selected", "Selected"
            )
        )
    ).reset_index(drop=True)
    if lic_only:
        oz_ui = oz_ui.query("type == 'Low-Income Community'").reset_index(drop=True)
    print("The following tracts are in IRS data but not in Urban Institute's data:")
    print(
        oz_irs["census_tract_number"][
            ~oz_irs["census_tract_number"].isin(oz_ui["geoid"])
        ]
    )

    # Merging OZ data to the housing price data
    df = (
        df.merge(
            oz_ui[["geoid", "designated"]],
            left_on="tract",
            right_on="geoid",
            how="left",
        )
        .drop("geoid", axis=1)
        .fill_empty("designated", "Ineligible")
        .rename_column("designated", "status")
    ).reset_index(drop=True)

    # Compute aggregate statistics in annual_change
    annual_change = (
        df[["tract", "state_abbr", "year", "annual_change", "status"]]
        .groupby(["status", "year"])["annual_change"]
        .agg(["mean", "std", "count"])
        .unstack(0)
    )

    # Cache data to disk
    df.to_feather(f"data/tracts{file_suffix}.feather")
    oz_irs.to_feather(f"data/oz_irs{file_suffix}.feather")
    oz_ui.to_feather(f"data/oz_ui{file_suffix}.feather")

    return df, annual_change, oz_irs, oz_ui


def get_census_shapefiles(df):
    """
    Download TIGER shapefiles and return

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with tract and status columns. Only used when not loading from disk.

    Returns
    -------
    geos: gpd.GeoDataFrame
        GeoDataFrame of Census tracts
    """
    # Read Census shapefiles
    if not os.path.isfile("data/tracts/all_tracts_with_status.shp"):
        if not os.path.isfile("data/tracts/tl_2018_01_tract.shp"):
            from IPython import get_ipython, display

            ipython = get_ipython()
            ipython.system("mkdir data/tracts")
            for fips in tqdm(range(1, 79)):
                url = (
                    f"http://www2.census.gov/geo/tiger/TIGER2018/TRACT/tl_"
                    f"2018_{str(fips).zfill(2)}_tract.zip"
                )
                fn = f"tl_2018_{str(fips).zfill(2)}_tract.zip"
                ipython.system("wget -q $url")
                ipython.system("mv $fn data/tracts/$fn")
                ipython.system("unzip data/tracts/$fn -d data/tracts/")
            display.clear_output()

        geos = []
        for fips in tqdm(range(1, 79)):
            fn = f"data/tracts/tl_2018_{str(fips).zfill(2)}_tract.shp"
            if os.path.isfile(fn):
                geos.append(gpd.read_file(fn))
        geos = (
            gpd.GeoDataFrame(pd.concat(geos).reset_index(drop=True))
            .clean_names()
            .assign(geoid=lambda x: x["geoid"].astype(int))
        )
        print(f"There are {len(geos)} tracts in the 2018 TIGER Census files")

        geos = geos.merge(
            df[["tract", "status"]].drop_duplicates(),
            how="left",
            left_on="geoid",
            right_on="tract",
        )
        geos.to_file("data/tracts/all_tracts_with_status.shp")
    else:
        geos = gpd.read_file("data/tracts/all_tracts_with_status.shp")

    return geos


def get_census_tract_attributes():
    """
    Get tract-level demographic attributes from the ACS 2017 5-year estimates.
    Census tracts with < 100 in population are replaced with missing values.

    Returns
    -------
    tract_df, var_dict
        The data and definitions for each variable
    """
    if os.path.isfile("data/tracts_covs.feather") and os.path.isfile(
        "data/tracts_covs_var_dict.json"
    ):
        with open("data/tracts_covs_var_dict.json", "r") as f:
            var_dict = json.load(f)
        return (pd.read_feather("data/tracts_covs.feather"), var_dict)
    var_dict = {
        "B01003_001E": "population",
        "B02001_002E": "white_population",
        "C24020_001E": "employed_population",
        "B08131_001E": "minutes_commute",
        "B09010_002E": "supplemental_income",
        "B15003_021E": "associate",
        "B15003_022E": "bachelor",
        "B15003_023E": "master",
        "B15003_024E": "professional_school",
        "B15003_025E": "doctoral",
        "B16009_002E": "poverty",
        "B18140_001E": "median_earnings",
        "B19019_001E": "median_household_income",
        "B25011_001E": "total_housing",
        "B25011_026E": "renter_occupied",
        "B25031_001E": "median_gross_rent",
        "B27020_002E": "native_born",
        "B27020_003E": "native_born_hc_covered",
    }

    state_codes = "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 \
        27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 \
        55 56".split()

    conn = cenpy.remote.APIConnection("ACSDT5Y2017")

    tracts_df = (
        pd.concat(
            [
                conn.query(
                    cols=list(var_dict.keys()),
                    geo_unit="tract:*",
                    geo_filter={"state": st},
                )
                for st in tqdm(state_codes)
            ],
            axis=0,
        )
        .astype(float)
        .rename_columns(var_dict)
        .assign(
            **{
                "pct_white": lambda x: x.white_population / x.population,
                "minutes_commute": lambda x: x.minutes_commute / x.employed_population,
                "pct_higher_ed": lambda x: (
                    x.associate + x.bachelor + x.professional_school + x.doctoral
                )
                / x.population,
                "pct_rent": lambda x: x.renter_occupied / x.total_housing,
                "pct_native_hc_covered": lambda x: x.native_born_hc_covered
                / x.native_born,
                "pct_poverty": lambda x: x.poverty / x.population,
                "log_median_earnings": lambda x: np.log(x.median_earnings),
                "log_median_household_income": lambda x: np.log(
                    x.median_household_income
                ),
                "log_median_gross_rent": lambda x: np.log(x.median_gross_rent),
                "pct_supplemental_income": lambda x: x.supplemental_income
                / x.population,
                "pct_employed": lambda x: x.employed_population / x.population,
                "geoid": lambda x: x.state.astype(int).astype(str).str.zfill(2)
                + x.county.astype(int).astype(str).str.zfill(3)
                + x.tract.astype(int).astype(str).str.zfill(6),
            }
        )
        .drop(
            "white_population associate bachelor professional_school doctoral \
            renter_occupied native_born_hc_covered native_born \
            poverty state county tract".split(),
            axis=1,
        )
        .reset_index(drop=True)
    )

    tracts_df = tracts_df.assign(
        **{
            col: (
                tracts_df[col]
                .where(tracts_df[col].ge(0))
                .replace(to_replace=[np.inf, -np.inf], value=np.nan)
            )
            for col in tracts_df.columns
            if tracts_df[col].dtype == "float"
        }
    )

    gid = tracts_df.loc[tracts_df["population"] == 0, "geoid"].copy()
    nan_idx = (tracts_df["population"] < 100).copy()
    tracts_df.loc[nan_idx, :] = np.nan
    tracts_df.loc[nan_idx, "geoid"] = gid

    tracts_df.to_feather("data/tracts_covs.feather")
    with open("data/tracts_covs_var_dict.json", "w") as f:
        json.dump(var_dict, f)

    return tracts_df, var_dict


@pf.register_dataframe_method
def generate_zillow_data(df, tracts_df, balanced=True, yr=2014):
    """
    Clean data for comparison between Selected and Eligible tracts:
    - Drops unnecessary columns
    - Restrict to Selected or Eligible
    - Restrict to year >= yr
    - Balance by default
    - Merge in census data and drop columns with NA
    """
    zillow_data = (
        df.assign(treatment=lambda x: x.status == "Selected")
        .query("status == 'Selected' or status == 'Eligible, not selected'")
        .drop("status", axis=1)
        .reset_index(drop=True)
        .drop(["hpi1990", "hpi2000"], axis=1)
        .assign(tract=lambda x: x.tract.astype(str))
        .query(f"year >= {yr}")
    )

    if balanced:
        balanced_panel = set(
            zillow_data.set_index(["tract", "year"])["annual_change"]
            .unstack(-1)
            .dropna()
            .reset_index()["tract"]
        )
        zillow_data = zillow_data.query(f"tract in @balanced_panel").reset_index(
            drop=True
        )

    zillow_data = zillow_data.merge(
        tracts_df.dropna(subset=["geoid"]).assign(
            tract=lambda x: x.geoid.astype(int).astype(str)
        ),
        how="left",
    ).dropna(
        axis=1  # Remove _columns_ with NA
    )

    return zillow_data


@pf.register_dataframe_method
def get_pairs(geos, df):
    """Get pairwise comparisons"""

    # Clean centroid data
    geos_for_pairwise_comp = (
        geos.set_index("geoid")
        .assign(treated=lambda x: x["status"] == "Selected")[
            ["statefp", "intptlat", "intptlon", "treated"]
        ]
        .transform_column("intptlat", float)
        .transform_column("intptlon", float)
    )

    # Tracts with nonmissing housing price data
    with_data = set(
        df.query("year == 2018").dropna(subset=["annual_change"])["tract"].unique()
    )

    pair_dfs = []
    for state in geos_for_pairwise_comp.statefp.unique():
        state_data = geos_for_pairwise_comp.query(f"statefp == @state").copy()
        rad_per_degree = 1 / 360 * 2 * np.pi
        x = state_data.query("treated")[["intptlon", "intptlat"]] * rad_per_degree
        x_index = x.index
        y = state_data.query("not treated")[["intptlon", "intptlat"]] * rad_per_degree
        y_index = y.index

        y_index_data = y_index.isin(with_data)
        dist_mat = haversine_distances(X=x, Y=y)

        # Distance is infinity to places with missing data in order to exclude them
        dist_mat[:, ~y_index_data] = np.inf
        min_dist_control = y_index[dist_mat.argmin(axis=1)]

        pair_dfs.append(
            pd.DataFrame(
                {
                    "treated": x_index,
                    "untreated": min_dist_control,
                    "dist": dist_mat.min(axis=1),
                }
            ).assign(statefp=state)
        )
    pair_df = pd.concat(pair_dfs)

    pair_df = (
        pair_df.reset_index(drop=True)
        .reset_index()
        .melt(["statefp", "index", "dist"])
        .sort_values("index")
        .rename_column("variable", "treatment")
        .rename_column("value", "tract")
        .reset_index(drop=True)
        .merge(df[["tract", "annual_change", "year"]], on="tract", how="left")
        .sort_values(["statefp", "year", "index", "treatment"])
        .rename_column("index", "pair_id")
        .assign(post_treatment=lambda x: x.year >= 2018)
    )
    return pair_df


@pf.register_dataframe_method
def paired_data_for_did(pair_df, yr=2014, balanced=True):
    """
    Clean data for comparison between Selected and Eligible tracts:
    - Keeps only pair-years with nonmissing data
    - Keeps only year >= yr
    - Balances panel by default
    """
    pairs = (
        pair_df[
            ["pair_id", "annual_change", "treatment", "post_treatment", "year", "dist"]
        ]
        .merge(
            (
                pair_df.assign(
                    annual_change_notnull=lambda x: x.annual_change.notnull()
                )
                .groupby(["pair_id", "year"], group_keys=False)["annual_change_notnull"]
                .sum()
                == 2
            ).reset_index()
        )
        .query("annual_change_notnull")
        .drop("annual_change_notnull", axis=1)
        .sort_values(["pair_id", "year", "treatment"])
        .query(f"year >= {yr}")
        .reset_index(drop=True)
    )

    if balanced:
        balanced_panel = set(
            pairs.query(f"year >= {yr}")
            .set_index(["pair_id", "treatment", "year"])["annual_change"]
            .unstack(-1)
            .dropna()
            .reset_index()["pair_id"]
        )
        pairs = pairs.query(f"pair_id in @balanced_panel").reset_index(drop=True)
    return pairs


def agg_with_res_ratio(
    x,
    num_columns=[
        "population",
        "employed_population",
        "minutes_commute",
        "supplemental_income",
        "master",
        "median_earnings",
        "median_household_income",
        "total_housing",
        "median_gross_rent",
        "pct_white",
        "pct_higher_ed",
        "pct_rent",
        "pct_native_hc_covered",
        "pct_poverty",
        "log_median_earnings",
        "log_median_household_income",
        "log_median_gross_rent",
        "pct_supplemental_income",
        "pct_employed",
    ],
    cat_columns=["status"],
    cat_column_values={"status": ["Selected", "Ineligible", "Eligible, not selected"]},
):
    """
    Aggregate data to ZIP code level. Called by groupby(...).apply

    Parameters
    ----------
    x : pd.DataFrame
        Cross walk data frame
    num_columns : list, optional
        Numerical columns, by default [
            "population",
            "employed_population",
            "minutes_commute",
            "supplemental_income",
            "master",
            "median_earnings",
            "median_household_income",
            "total_housing",
            "median_gross_rent",
            "pct_white",
            "pct_higher_ed",
            "pct_rent",
            "pct_native_hc_covered",
            "pct_poverty",
            "log_median_earnings",
            "log_median_household_income",
            "log_median_gross_rent",
            "pct_supplemental_income",
            "pct_employed",
        ]
    cat_columns : list, optional
        Categorical columns, by default ["status"]
    cat_column_values : dict, optional
        Values used by categorical columns, by default
        {"status": ["Selected", "Ineligible", "Eligible, not selected"]}

    Returns
    -------
    return_dict: pd.Series
        ZIP-aggregated values
    """
    weights = x["ratio"]
    return_dict = {}

    # \sum ratio * V_ij * 1(Vij != null)
    weight_times_variable = (
        x[num_columns].fillna(0).values.reshape((-1, len(num_columns)))
        * weights.values.reshape((-1, 1))
    ).sum(axis=0)

    # \sum ratio * 1(Vij != null)
    weight_sum = (
        weights.values.reshape((-1, 1))
        * x[num_columns].notnull().values.reshape((-1, len(num_columns)))
    ).sum()

    num_mat = weight_times_variable / weight_sum
    num_mat_notnull = np.where(x[num_columns].isnull().any().values, np.nan, num_mat)

    return_dict.update(pd.Series(num_mat, index=num_columns).to_dict())
    return_dict.update(
        pd.Series(
            num_mat_notnull, index=[f"{c}_notnull" for c in num_columns]
        ).to_dict()
    )

    # Deal with categorical columns
    for c in cat_columns:
        for val in cat_column_values[c]:
            return_dict[f"{c}_{val}"] = ((x[c] == val).fillna(False) * weights).sum()
    return_dict["ratio_sum"] = weights.sum()
    return pd.Series(return_dict)


def get_zips(overwrite=False, lic_only=True, start_year=2014):
    file_suffix = get_file_suffix(lic_only)

    if not overwrite and os.path.isfile(f"data/zip_panel{file_suffix}.feather"):
        return pd.read_feather(f"data/zip_panel{file_suffix}.feather")

    # Get data
    zips = (
        pd.read_excel(
            "https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_BDL_ZIP5.xlsx",
            header=6,
            na_values=["."],
        )
        .clean_names()
        .rename_column("annual_change_%_", "annual_change")
        .assign(zip_code=lambda x: x["five_digit_zip_code"].astype(str).str.zfill(5))
        .drop("five_digit_zip_code", axis=1)
    )

    # Cross-walk file
    # https://www.huduser.gov/portal/datasets/usps_crosswalk.html
    # Get for Q1 2017 and put in data/
    cross = (
        pd.read_excel("data/ZIP_TRACT_032017.xlsx")
        .clean_names()
        .assign(zip_code=lambda x: x["zip"].astype(str).str.zfill(5))
        .drop("zip", axis=1)[["zip_code", "tract", "tot_ratio"]]
        .rename_column("tot_ratio", "ratio")
        .copy()
    )

    df, annual_change, oz_irs, oz_ui = get_oz_data(
        overwrite=overwrite, lic_only=lic_only
    )
    tracts_df, var_dict = get_census_tract_attributes()

    # Cross-walk file merged with characteristics columns
    zips_characteristics = (
        cross.merge(
            oz_ui[["geoid", "designated"]].rename_column("designated", "status"),
            left_on="tract",
            right_on="geoid",
            how="left",
        )
        .fill_empty("status", "Ineligible")
        .drop("geoid", axis=1)
        .merge(
            tracts_df.dropna(subset=["geoid"])
            .assign(tract=lambda x: x.geoid.astype(int))
            .drop("geoid", axis=1),
            how="left",
        )
        .sort_values("zip_code", ascending=False)
    )

    # Number of selected census tracts with nonempty
    # intersection with ZIPs with no missing data
    n_selected_tracts_covered = (
        zips_characteristics.merge(zips, on="zip_code", how="left")
        .query("status == 'Selected' and year == 2018")
        .dropna(subset=["annual_change"])["tract"]
        .nunique()
    )

    # Aggregate with agg_with_res_ratio to ZIP level data
    agged_zips = (
        zips_characteristics.groupby("zip_code")
        .progress_apply(agg_with_res_ratio)
        .unstack(-1)
        .unstack(0)
        .clean_names()
        .reset_index()
    )

    # Merge aggregated characteristics onto ZIP housing prices data
    zip_panel = (
        zips.merge(agged_zips, on="zip_code", how="left")
        .query(f"year >= {start_year}")
        .reset_index(drop=True)
    )

    # Number of nonmissing ZIPs
    non_missing_zips = (
        zip_panel.query("year == 2018")
        .dropna(subset=["annual_change"])["zip_code"]
        .nunique()
    )

    # Number of total ZIP codes in the crosswalk file
    total_zips = cross["zip_code"].nunique()

    # Cache data
    zip_panel.to_feather(f"data/zip_panel{file_suffix}.feather")

    with open(f"data/zip_missing{file_suffix}.txt", "w") as f:
        print(
            " ".join(
                map(str, [n_selected_tracts_covered, non_missing_zips, total_zips])
            ),
            file=f,
        )

    return zip_panel
