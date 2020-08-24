import pandas_flavor as pf


@pf.register_dataframe_method
def plot_with_error_bars(df, start_year=2012, color=["b", "orange", "green"]):
    se = df["std"].loc[start_year:] / df["count"].loc[start_year:] ** 0.5
    ax = (df["mean"].loc[start_year:] + 1.96 * se).plot(
        ls="--", color=color, lw=0.5, legend=False
    )

    (df["mean"].loc[start_year:] - 1.96 * se).plot(
        ax=ax, ls="--", color=color, lw=0.5, legend=False
    )

    df["mean"].loc[start_year:].plot(ax=ax, color=color, legend=True, marker=".")

    return ax
