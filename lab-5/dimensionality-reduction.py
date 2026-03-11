import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    from sklearn.preprocessing import StandardScaler

    from sklearn.manifold import TSNE, Isomap

    from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE
    from sklearn.decomposition import PCA, KernelPCA
    import umap
    import warnings

    return (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        HistGradientBoostingRegressor,
        Isomap,
        KernelPCA,
        PCA,
        RFE,
        RandomForestClassifier,
        SelectKBest,
        StandardScaler,
        TSNE,
        VarianceThreshold,
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        root_mean_squared_error,
        train_test_split,
        umap,
        warnings,
    )


@app.cell
def _(np, warnings):
    warnings.filterwarnings('ignore')

    SEED = 42
    np.random.seed(SEED)
    return (SEED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dimensionality reduction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Загрузка датасетов
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Такси. Регрессия
    """)
    return


@app.cell(hide_code=True)
def _(
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mo,
    pd,
    r2_score,
    root_mean_squared_error,
):
    def calculate_regression_metrics(y_true, y_pred):
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "R^2": r2_score(y_true, y_pred)
        }


    def get_regression_report(y_train, y_pred_train, y_test, y_pred_test):
        train_metrics = calculate_regression_metrics(y_train, y_pred_train)
        test_metrics = calculate_regression_metrics(y_test, y_pred_test)

        metrics_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["Train", "Test"],
            columns=["MSE", "RMSE", "MAE", "MAPE", "R^2"]
        )

        return mo.vstack([
            mo.md("Метрики модели"),
            metrics_df.round(5),
        ])

    return (get_regression_report,)


@app.cell
def _(
    HistGradientBoostingRegressor,
    SEED,
    get_regression_report,
    train_test_split,
):
    def run_regression(X, y, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, **kwargs)
        model = HistGradientBoostingRegressor(max_depth=10, random_state=SEED).fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        return get_regression_report(y_train, y_pred_train, y_test, y_pred)

    return (run_regression,)


@app.cell
def _(pd):
    df_taxi = pd.read_csv('./data/processed/taxi.csv')
    df_taxi
    return (df_taxi,)


@app.cell
def _(df_taxi):
    X_taxi = df_taxi.drop(columns=['trip_duration'])
    y_taxi = df_taxi['trip_duration']
    return X_taxi, y_taxi


@app.cell
def _(X_taxi, run_regression, y_taxi):
    run_regression(X_taxi, y_taxi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### NEO. Классификация
    """)
    return


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
def _(
    RandomForestClassifier,
    SEED,
    get_classification_report,
    train_test_split,
):
    def run_classification(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=SEED
                ).fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        return get_classification_report(y_train, y_pred_train, y_test, y_pred)

    return (run_classification,)


@app.cell
def _(pd):
    df_neo = pd.read_csv('./data/processed/planets.csv')
    df_neo
    return (df_neo,)


@app.cell
def _(df_neo):
    X_neo = df_neo.drop(columns=['hazardous'])
    y_neo = df_neo['hazardous']
    return X_neo, y_neo


@app.cell
def _(X_neo, run_classification, y_neo):
    run_classification(X_neo, y_neo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Понижение размерности
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Отбор признаков с высокой дисперсией
    """)
    return


@app.cell
def _(df_taxi):
    df_taxi.describe()
    return


@app.cell
def _(df_neo):
    df_neo.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance Threshold
    """)
    return


@app.cell
def _(VarianceThreshold, X_taxi):
    vt_taxi = VarianceThreshold(1.4)

    vt_taxi.fit(X_taxi)

    X_taxi_vt = vt_taxi.transform(X_taxi)

    X_taxi_vt.shape
    return (X_taxi_vt,)


@app.cell
def _(X_taxi_vt, run_regression, y_taxi):
    run_regression(X_taxi_vt, y_taxi)
    return


@app.cell
def _(VarianceThreshold, X_neo):
    vt_neo = VarianceThreshold(2)

    X_neo_vt = vt_neo.fit_transform(X_neo)
    X_neo_vt.shape
    return (X_neo_vt,)


@app.cell
def _(X_neo_vt, run_classification, y_neo):
    run_classification(X_neo_vt, y_neo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SelectKBest
    """)
    return


@app.cell
def _(SelectKBest, X_taxi, run_regression, y_taxi):
    skb_taxi = SelectKBest(k=6)

    skb_taxi.fit(X_taxi, y_taxi)

    X_taxi_skb = skb_taxi.transform(X_taxi)

    run_regression(X_taxi_skb, y_taxi)
    return


@app.cell
def _(SelectKBest, X_neo, run_classification, y_neo):
    skb_neo = SelectKBest(k=3)

    skb_neo.fit(X_neo, y_neo)

    X_neo_skb = skb_neo.transform(X_neo)

    run_classification(X_neo_skb, y_neo)
    return


@app.cell
def _(DecisionTreeRegressor, RFE, X_taxi, pd, y_taxi):
    tree_taxi = DecisionTreeRegressor().fit(X_taxi, y_taxi)

    rfe_taxi = RFE(estimator=tree_taxi, n_features_to_select=6, step=1).fit(X_taxi, y_taxi)
    X_taxi_rfe = pd.DataFrame(rfe_taxi.transform(X_taxi), columns=rfe_taxi.get_feature_names_out())
    X_taxi_rfe
    return X_taxi_rfe, tree_taxi


@app.cell
def _(X_taxi_rfe, run_regression, y_taxi):
    run_regression(X_taxi_rfe, y_taxi)
    return


@app.cell
def _(DecisionTreeClassifier, RFE, X_neo, pd, y_neo):
    tree_neo = DecisionTreeClassifier().fit(X_neo, y_neo)

    rfe_neo = RFE(estimator=tree_neo, n_features_to_select=4, step=1).fit(X_neo, y_neo)
    X_neo_rfe = pd.DataFrame(rfe_neo.transform(X_neo), columns=rfe_neo.get_feature_names_out())
    X_neo_rfe
    return X_neo_rfe, tree_neo


@app.cell
def _(X_neo_rfe, run_classification, y_neo):
    run_classification(X_neo_rfe, y_neo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Значимость признаков
    """)
    return


@app.cell
def _(X_taxi, plt, tree_taxi):
    plt.barh(width=tree_taxi.feature_importances_, y=X_taxi.columns)
    return


@app.cell
def _(X_neo, plt, tree_neo):
    plt.barh(width=tree_neo.feature_importances_, y=X_neo.columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA
    """)
    return


@app.cell
def _(StandardScaler, X_taxi, pd):
    scaler_taxi = StandardScaler()

    X_taxi_scaled = scaler_taxi.fit_transform(X_taxi)
    X_taxi_scaled = pd.DataFrame(X_taxi_scaled, columns=X_taxi.columns)
    return (X_taxi_scaled,)


@app.cell
def _(PCA, X_taxi_scaled):
    pca_taxi = PCA(n_components=2)

    pca_taxi.fit(X_taxi_scaled)

    X_taxi_pca = pca_taxi.transform(X_taxi_scaled)
    return (X_taxi_pca,)


@app.cell
def _(X_taxi_pca, run_regression, y_taxi):
    run_regression(X_taxi_pca, y_taxi)
    return


@app.cell
def _(X_taxi_pca, plt, y_taxi):
    plt.scatter(X_taxi_pca[:, 0], X_taxi_pca[:, 1], c=y_taxi, alpha=0.7)
    return


@app.cell
def _(X_taxi_pca, plt, y_taxi):
    plt.xlim(-35, 28)
    plt.ylim(-18, 24)
    plt.scatter(X_taxi_pca[:, 0], X_taxi_pca[:, 1], c=y_taxi, alpha=0.7)
    return


@app.cell
def _(PCA, X_taxi_scaled, y_taxi):
    pca_taxi_3 = PCA(n_components=3)

    X_taxi_pca_3 = pca_taxi_3.fit_transform(X_taxi_scaled, y_taxi)
    return (X_taxi_pca_3,)


@app.cell
def _(X_taxi_pca_3, run_regression, y_taxi):
    run_regression(X_taxi_pca_3, y_taxi)
    return


@app.cell
def _(X_taxi_pca_3, plt, y_taxi):
    fig_taxi_pca_3 = plt.figure()
    ax_taxi_pca_3 = fig_taxi_pca_3.add_subplot(projection='3d')
    ax_taxi_pca_3.scatter(X_taxi_pca_3[:, 0], X_taxi_pca_3[:, 1], X_taxi_pca_3[:, 2], c=y_taxi, alpha=0.7)
    plt.xlim(-35, 28)
    plt.ylim(-18, 25)
    ax_taxi_pca_3.set_zlim(-5, 5)
    plt.show()
    return


@app.cell
def _(StandardScaler, X_neo, pd):
    scaler_neo = StandardScaler()

    X_neo_scaled = scaler_neo.fit_transform(X_neo)
    X_neo_scaled = pd.DataFrame(X_neo_scaled, columns=X_neo.columns)
    return (X_neo_scaled,)


@app.cell
def _(PCA, X_neo_scaled):
    pca_neo = PCA(n_components=2)

    pca_neo.fit(X_neo_scaled)

    X_neo_pca = pca_neo.transform(X_neo_scaled)
    return (X_neo_pca,)


@app.cell
def _(X_neo_pca, run_classification, y_neo):
    run_classification(X_neo_pca, y_neo)
    return


@app.cell
def _(X_neo_pca, plt, y_neo):
    plt.scatter(X_neo_pca[:, 0], X_neo_pca[:, 1], c=y_neo, alpha=0.7)
    return


@app.cell
def _(X_neo_pca, plt, y_neo):
    plt.xlim(-5, 25)
    plt.ylim(-5, 11)
    plt.scatter(X_neo_pca[:, 0], X_neo_pca[:, 1], c=y_neo, alpha=0.7)
    return


@app.cell
def _(PCA, X_neo_scaled, y_neo):
    pca_neo_3 = PCA(n_components=3)

    X_neo_pca_3 = pca_neo_3.fit_transform(X_neo_scaled, y_neo)
    return (X_neo_pca_3,)


@app.cell
def _(X_neo_pca_3, run_classification, y_neo):
    run_classification(X_neo_pca_3, y_neo)
    return


@app.cell
def _(X_neo_pca_3, plt, y_neo):
    fig_neo_pca_3 = plt.figure()
    ax_neo_pca_3 = fig_neo_pca_3.add_subplot(projection='3d')
    ax_neo_pca_3.scatter(X_neo_pca_3[:, 0], X_neo_pca_3[:, 1], X_neo_pca_3[:, 2], c=y_neo, alpha=0.7)
    plt.xlim(-5, 25)
    plt.ylim(-5, 11)
    ax_neo_pca_3.set_zlim(-5, 5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TSNE
    """)
    return


@app.cell
def _(SEED, TSNE, np, plt):
    def plot_tsne_random_sample(X, y, sample_size=10000, title=None, func=None):
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(len(X), sample_size, replace=False)

        if hasattr(X, "iloc"):
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X[sample_idx]

        if hasattr(y, "iloc"):
            y_sample = y.iloc[sample_idx]
        else:
            y_sample = y[sample_idx]

        tsne = TSNE(n_components=2, random_state=SEED)

        X_tsne = tsne.fit_transform(X_sample)

        plt.figure(figsize=(10,7))
        plt.scatter(
            X_tsne[:,0],
            X_tsne[:,1],
            c=y_sample,
            cmap="tab10",
            s=5
        )

        if title:
            plt.title(title)

        plt.show()

        return func(X_tsne, y_sample)

    return (plot_tsne_random_sample,)


@app.cell
def _(X_taxi_scaled, plot_tsne_random_sample, run_regression, y_taxi):
    plot_tsne_random_sample(X_taxi_scaled, y_taxi, func=run_regression)
    return


@app.cell
def _(X_neo_scaled, plot_tsne_random_sample, run_classification, y_neo):
    plot_tsne_random_sample(X_neo_scaled, y_neo, func=run_classification)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Isomap
    """)
    return


@app.cell
def _(Isomap, X_taxi_scaled):
    isomap_taxi = Isomap(n_components=2)
    X_taxi_isomap = isomap_taxi.fit_transform(X_taxi_scaled[:10000])
    return (X_taxi_isomap,)


@app.cell
def _(X_taxi_isomap, plt, y_taxi):
    plt.scatter(X_taxi_isomap[:,0], X_taxi_isomap[:,1], c=y_taxi[:10000], alpha=0.7)
    return


@app.cell
def _(X_taxi_isomap, run_regression, y_taxi):
    run_regression(X_taxi_isomap, y_taxi[:10000])
    return


@app.cell
def _(Isomap, X_neo_scaled):
    isomap_neo = Isomap(n_components=2)
    X_neo_isomap = isomap_neo.fit_transform(X_neo_scaled[:10000])
    return (X_neo_isomap,)


@app.cell
def _(X_neo_isomap, plt, y_neo):
    plt.scatter(X_neo_isomap[:,0], X_neo_isomap[:,1], c=y_neo[:10000], alpha=0.7)
    return


@app.cell
def _(X_neo_isomap, run_classification, y_neo):
    run_classification(X_neo_isomap, y_neo[:10000])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UMAP
    """)
    return


@app.cell(hide_code=True)
def _(SEED, np, plt, umap):
    def plot_umap_random_sample(X, y, sample_size=10000, title=None, func=None):
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(len(X), sample_size, replace=False)

        if hasattr(X, "iloc"):
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X[sample_idx]

        if hasattr(y, "iloc"):
            y_sample = y.iloc[sample_idx]
        else:
            y_sample = y[sample_idx]

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=SEED
        )

        embedding = reducer.fit_transform(X_sample)

        plt.figure(figsize=(10,7))
        plt.scatter(
            embedding[:,0],
            embedding[:,1],
            c=y_sample,
            cmap="tab10",
            s=5
        )

        if title:
            plt.title(title)

        plt.show()

        return func(embedding, y_sample)

    return (plot_umap_random_sample,)


@app.cell
def _(X_taxi_scaled, plot_umap_random_sample, run_regression, y_taxi):
    plot_umap_random_sample(X_taxi_scaled, y_taxi, sample_size=10000, func=run_regression)
    return


@app.cell
def _(X_neo_scaled, plot_umap_random_sample, run_classification, y_neo):
    plot_umap_random_sample(X_neo_scaled, y_neo, sample_size=10000, func=run_classification)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KernelPCA
    """)
    return


@app.cell
def _(KernelPCA, SEED, np, plt, run_classification, run_regression):
    def plot_kernelpca_random_sample(
        X, y, sample_size=None, kernel='rbf', 
        task='classification', title=None
    ):
        if sample_size is None or sample_size >= len(X):
            X_sample = X
            y_sample = y
        else:
            rng = np.random.default_rng(SEED)
            sample_idx = rng.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_idx] if hasattr(X, "iloc") else X[sample_idx]
            y_sample = y.iloc[sample_idx] if hasattr(y, "iloc") else y[sample_idx]

        kpca = KernelPCA(n_components=2, kernel=kernel, random_state=SEED)
        X_kpca = kpca.fit_transform(X_sample)

        plt.figure(figsize=(10,7))
        plt.scatter(
            X_kpca[:, 0],
            X_kpca[:, 1],
            c=y_sample if task=='classification' else 'blue',
            cmap='tab10' if task=='classification' else None,
            s=5,
            alpha=0.7
        )
        if title:
            plt.title(f"KernelPCA ({kernel}) - {title}")
        plt.show()

        if task == 'classification':
            result = run_classification(X_kpca, y_sample)
        elif task == 'regression':
            result = run_regression(X_kpca, y_sample)
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        return result

    return (plot_kernelpca_random_sample,)


@app.cell
def _(X_taxi_scaled, plot_kernelpca_random_sample, y_taxi):
    plot_kernelpca_random_sample(
        X_taxi_scaled, y_taxi,
        sample_size=20000,
        kernel='poly',
        task='regression',
        title='Taxi KernelPCA poly'
    )
    return


@app.cell
def _(X_taxi_scaled, plot_kernelpca_random_sample, y_taxi):
    plot_kernelpca_random_sample(
        X_taxi_scaled, y_taxi,
        sample_size=20000,
        kernel='rbf',
        task='regression',
        title='Taxi KernelPCA rbf'
    )
    return


@app.cell
def _(X_taxi_scaled, plot_kernelpca_random_sample, y_taxi):
    plot_kernelpca_random_sample(
        X_taxi_scaled, y_taxi,
        sample_size=20000,
        kernel='sigmoid',
        task='regression',
        title='Taxi KernelPCA sigmoid'
    )
    return


@app.cell
def _(X_neo_scaled, plot_kernelpca_random_sample, y_neo):
    plot_kernelpca_random_sample(
        X_neo_scaled, y_neo,
        sample_size=20000,
        kernel='poly',
        task='classification',
        title='NEO KernelPCA poly'
    )
    return


@app.cell
def _(X_neo_scaled, plot_kernelpca_random_sample, y_neo):
    plot_kernelpca_random_sample(
        X_neo_scaled, y_neo,
        sample_size=20000,
        kernel='rbf',
        task='classification',
        title='NEO KernelPCA rbf'
    )
    return


@app.cell
def _(X_neo_scaled, plot_kernelpca_random_sample, y_neo):
    plot_kernelpca_random_sample(
        X_neo_scaled, y_neo,
        sample_size=20000,
        kernel='sigmoid',
        task='classification',
        title='NEO KernelPCA sigmoid'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MyPCA
    """)
    return


@app.cell
def _(np):
    def my_pca(X, n_components=2):
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        principal_components = eigenvectors[:, idx[:n_components]]

        X_reduced = X_centered @ principal_components
        return np.real(X_reduced)

    return (my_pca,)


@app.cell
def _(X_neo_scaled, my_pca):
    X_neo_my_pca = my_pca(X_neo_scaled, n_components=2)
    return (X_neo_my_pca,)


@app.cell
def _(X_neo_my_pca, plt, y_neo):
    plt.scatter(X_neo_my_pca[:, 0], X_neo_my_pca[:, 1], c=y_neo, alpha=0.7)
    return


@app.cell
def _(X_neo_pca, plt, y_neo):
    plt.scatter(X_neo_pca[:, 0], X_neo_pca[:, 1], c=y_neo, alpha=0.7)
    return


if __name__ == "__main__":
    app.run()
