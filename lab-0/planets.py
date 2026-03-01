import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Лабораторная работа №0. EDA. Ближайшие к Земле объекты
    """)
    return


@app.cell
def _():
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    warnings.filterwarnings("ignore")
    return mo, pd, plt, sns, stats


@app.cell
def _(pd):
    df_raw = pd.read_csv("../data/raw/neo_task.csv")
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    def clean_planets_data(df):
        df = df.copy()

        cols_to_drop = ["id", "name"]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        if df["hazardous"].dtype == "object":
            df["hazardous"] = df["hazardous"].map(
                {"true": 1, "false": 0, "True": 1, "False": 0}
            )
        else:
            df["hazardous"] = df["hazardous"].astype(int)

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        return df

    df_clean = clean_planets_data(df_raw)
    df_clean
    return (df_clean,)


@app.cell
def _(df_clean):
    def create_filtered_view(df):
        df = df.copy()
        numeric_cols = [
            "est_diameter_min",
            "relative_velocity",
            "miss_distance",
            "absolute_magnitude",
        ]

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower) & (df[col] <= upper)]

        return df

    df_filtered_view = create_filtered_view(df_clean)

    lost_hazardous = df_clean["hazardous"].sum() - df_filtered_view["hazardous"].sum()
    print(f"Hazardous planets LOST by filtering: {lost_hazardous}")
    print("(This is why we will train on df_clean, not df_filtered!)")
    return (df_filtered_view,)


@app.cell
def _(df_clean, mo, pd, stats):
    def get_correlations(df):
        numeric_cols = [
            "est_diameter_min",
            "est_diameter_max",
            "relative_velocity",
            "miss_distance",
            "absolute_magnitude",
        ]

        pb_correlations = {}
        p_values = {}

        for col in numeric_cols:
            corr, p_val = stats.pointbiserialr(df["hazardous"], df[col])
            pb_correlations[col] = corr
            p_values[col] = p_val

        return pd.DataFrame(
            {"Correlation": pb_correlations, "P-Value": p_values}
        ).sort_values(by="Correlation", key=abs, ascending=False)

    mo.ui.table(get_correlations(df_clean))
    return


@app.cell
def _(df_clean, df_filtered_view, plt, sns):
    def plot_planet_hists(df_raw, df_filt):
        cols = ["absolute_magnitude", "relative_velocity", "est_diameter_min"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, col in enumerate(cols):
            sns.histplot(
                df_raw[col],
                ax=axes[i],
                color="red",
                alpha=0.3,
                label="Raw (With Outliers)",
            )
            sns.histplot(
                df_filt[col],
                ax=axes[i],
                color="green",
                alpha=0.6,
                label="Filtered (Normal Range)",
            )

            axes[i].set_title(col)
            axes[i].set_yscale("log")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    plot_planet_hists(df_clean, df_filtered_view)
    return


@app.cell
def _(df_clean, plt, sns):
    def plot_separation(df):
        cols = ["absolute_magnitude", "relative_velocity", "miss_distance"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        for i, col in enumerate(cols):
            sns.boxplot(x="hazardous", y=col, data=df, ax=axes[i], palette="Set2")
            axes[i].set_title(f"{col} by Class")
            if col != "absolute_magnitude":
                axes[i].set_yscale("log")

        plt.tight_layout()
        plt.show()

    plot_separation(df_clean)
    return


@app.cell
def _(df_clean):
    def save_planets(df):
        save_path = "../data/processed/planets.csv"

        df.to_csv(save_path, index=False)

    save_planets(df_clean)
    return


if __name__ == "__main__":
    app.run()
