import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    import joblib

    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, StackingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    from catboost import CatBoostClassifier

    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN

    import warnings

    return (
        BaggingClassifier,
        CatBoostClassifier,
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        RandomForestClassifier,
        RandomUnderSampler,
        RidgeClassifier,
        SMOTEENN,
        StackingClassifier,
        StandardScaler,
        accuracy_score,
        confusion_matrix,
        f1_score,
        joblib,
        mo,
        np,
        pd,
        precision_score,
        recall_score,
        roc_auc_score,
        train_test_split,
        warnings,
    )


@app.cell(hide_code=True)
def _(
    accuracy_score,
    confusion_matrix,
    f1_score,
    mo,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
):
    def calculate_classification_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        return {
            "Confusion Matrix": cm,
            "metrics": {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc
            }
        }

    def confusion_matrix_df(cm):
        tn, fp, fn, tp = cm.ravel()

        cm_df = pd.DataFrame([[tn, fp], [fn, tp]], columns=['Actually Positive', 'Actually Negative'], index=['Predicted Positive', 'Predicted Negative'])

        return cm_df.round(5)
    def get_classification_report(y_train, y_pred_train, y_test, y_pred_test):
        metrics_train = calculate_classification_metrics(y_train, y_pred_train)
        metrics_test = calculate_classification_metrics(y_test, y_pred_test)

        cm_train = confusion_matrix_df(metrics_train["Confusion Matrix"].flatten())
        cm_test = confusion_matrix_df(metrics_test["Confusion Matrix"].flatten())

        metrics_df_train = pd.DataFrame([metrics_train['metrics']]).round(5)

        metrics_df_test = pd.DataFrame([metrics_test['metrics']]).round(5)

        return mo.vstack([
            mo.md("Метрики модели"),
            mo.md("**Train**"),
            metrics_df_train,
            mo.md("Confusion matrix"),
            cm_train,
            mo.md("**Test**"),
            metrics_df_test,
            mo.md("Confusion matrix"),
            cm_test
        ])

    return (get_classification_report,)


@app.cell
def _(np, warnings):
    warnings.filterwarnings('ignore')

    SEED = 42
    np.random.seed(42)
    return (SEED,)


@app.cell
def _(pd):
    df = pd.read_csv('./data/processed/planets.csv')
    X = df.drop(columns=['hazardous'])
    y = df['hazardous']
    return X, df, y


@app.cell
def _(StandardScaler, X):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    return (X_scaled,)


@app.cell
def _(SEED, X_scaled, df, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=SEED, stratify=df['hazardous'])
    X_train.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(RandomUnderSampler, SEED, X_train, y_train):
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=SEED)

    X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

    X_train_undersampled.shape
    return X_train_undersampled, y_train_undersampled


@app.cell
def _(SEED, SMOTEENN, X_train, y_train):
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=SEED)

    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    X_train_resampled.shape
    return


@app.cell
def _(get_classification_report):
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        return get_classification_report(y_train, y_train_pred, y_test, y_pred)

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classic
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    SEED,
    X_train_undersampled,
    y_train_undersampled,
):
    best_classic = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=10,
        random_state=SEED
    )

    best_classic.fit(X_train_undersampled, y_train_undersampled)
    return (best_classic,)


@app.cell
def _(X_test, X_train, best_classic, evaluate_model, y_test, y_train):
    evaluate_model(best_classic, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(best_classic, joblib):
    joblib.dump(best_classic, './models/best_classic_model.joblib')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boosting
    """)
    return


@app.cell
def _(GradientBoostingClassifier, SEED, X_train, y_train):
    best_boosting = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=7,
        min_samples_leaf=20,
        random_state=SEED
    )

    best_boosting.fit(X_train, y_train)
    return (best_boosting,)


@app.cell
def _(X_test, X_train, best_boosting, evaluate_model, y_test, y_train):
    evaluate_model(best_boosting, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(best_boosting, joblib):
    joblib.dump(best_boosting, './models/best_boosting_model.joblib')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CatBoost
    """)
    return


@app.cell
def _(CatBoostClassifier, SEED, X_train, y_train):
    best_catboost = CatBoostClassifier(
        depth=10,
        learning_rate=0.001,
        verbose=0,
        random_seed=SEED,
        class_weights=[1, 10]
    )

    best_catboost.fit(X_train, y_train)
    return (best_catboost,)


@app.cell
def _(best_catboost):
    best_catboost.get_feature_importance(prettified=True)
    return


@app.cell
def _(X_test, X_train, best_catboost, evaluate_model, y_test, y_train):
    evaluate_model(best_catboost, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(best_catboost, joblib):
    joblib.dump(best_catboost, './models/best_catboost_model.joblib')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bagging
    """)
    return


@app.cell
def _(BaggingClassifier, DecisionTreeClassifier, SEED, X_train, y_train):
    tree = DecisionTreeClassifier(max_depth=10)
    best_bagging = BaggingClassifier(
        estimator=tree,
        n_estimators=250,
        random_state=SEED
    )

    best_bagging.fit(X_train, y_train)
    return (best_bagging,)


@app.cell
def _(X_test, X_train, best_bagging, evaluate_model, y_test, y_train):
    evaluate_model(best_bagging, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(best_bagging, joblib):
    joblib.dump(best_bagging, './models/best_bagging_model.joblib')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stacking
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    KNeighborsClassifier,
    RandomForestClassifier,
    RidgeClassifier,
    SEED,
    StackingClassifier,
    X_train,
    y_train,
):
    base_stacking_models = [
        ('knn1', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=SEED)),
        ('rf', RandomForestClassifier(class_weight='balanced', max_depth=10, n_estimators=100, random_state=SEED, n_jobs=-1)),
    ]

    final_stacking_estimator = RidgeClassifier(class_weight='balanced', random_state=SEED)

    best_stacking = StackingClassifier(
        estimators=base_stacking_models,
        cv=5,
        final_estimator=final_stacking_estimator,
        n_jobs=-1,
    )

    best_stacking.fit(X_train, y_train)
    return (best_stacking,)


@app.cell
def _(X_test, X_train, best_stacking, evaluate_model, y_test, y_train):
    evaluate_model(best_stacking, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(best_stacking, joblib):
    joblib.dump(best_stacking, './models/best_stacking_model.joblib')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
