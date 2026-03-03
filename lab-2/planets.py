import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Классификация. Планеты
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    return SMOTE, pd, train_test_split


@app.cell
def _(pd):
    df_raw = pd.read_csv("./data/processed/planets.csv")
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    df_raw.hazardous.value_counts()
    return


@app.cell
def _(SMOTE, df_raw):
    oversample = SMOTE()

    X = df_raw.drop('hazardous', axis=1)
    y = df_raw['hazardous']

    transformed_X, transformed_y = oversample.fit_resample(X, y)

    transformed_y.value_counts()
    return transformed_X, transformed_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Разделение данных на обучающую и тестовую выборки
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hold out
    """)
    return


@app.cell
def _(train_test_split, transformed_X, transformed_y):
    X_holdout_train, X_holdout_test, y_holdout_train, y_holdout_test = train_test_split(transformed_X, transformed_y, test_size=0.2, random_state=42, stratify=transformed_y)
    return X_holdout_test, X_holdout_train, y_holdout_test, y_holdout_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KFold
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import KFold

    kFold = KFold(n_splits=5, shuffle=True, random_state=42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Алгоритмы классификации
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Логистическая регрессия
    """)
    return


@app.cell
def _(X_holdout_test, X_holdout_train, y_holdout_test, y_holdout_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    lr = LogisticRegression()
    lr.fit(X_holdout_train, y_holdout_train)
    predictions = lr.predict(X_holdout_test)

    classification_report(y_holdout_test, predictions)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
