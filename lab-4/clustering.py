import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification, make_blobs
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, homogeneity_score, adjusted_rand_score
    from scipy.cluster import hierarchy
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, AffinityPropagation
    from sklearn.mixture import GaussianMixture
    from scipy.cluster import hierarchy
    import warnings
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import resample

    return (
        AffinityPropagation,
        AgglomerativeClustering,
        DBSCAN,
        GaussianMixture,
        KMeans,
        PCA,
        StandardScaler,
        adjusted_rand_score,
        calinski_harabasz_score,
        hierarchy,
        homogeneity_score,
        make_blobs,
        make_classification,
        mo,
        np,
        pd,
        plt,
        resample,
        silhouette_score,
        warnings,
    )


@app.cell
def _(np, warnings):
    warnings.filterwarnings('ignore')

    SEED = 42
    np.random.seed(SEED)
    return (SEED,)


@app.cell(hide_code=True)
def _(PCA, plt):
    def plot_clusters(X, y):
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X

        plt.figure(figsize=(10, 5))
        scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='plasma', s=50, alpha=0.7, edgecolor='k')
        plt.title("Cluster Visualization")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter, label='Cluster Label')
        plt.show()

    return (plot_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datasets
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### make_classification
    """)
    return


@app.cell
def _(SEED, make_classification):
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.5,
        flip_y=0.02,
        random_state=SEED
    )
    return X_class, y_class


@app.cell
def _(X_class, plot_clusters, y_class):
    plot_clusters(X_class, y_class)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### make_blobs
    """)
    return


@app.cell
def _(SEED, make_blobs):
    X_blobs, y_blobs = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=3,
        random_state=SEED
    )
    return X_blobs, y_blobs


@app.cell
def _(X_blobs, plot_clusters, y_blobs):
    plot_clusters(X_blobs, y_blobs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### NEO dataset
    """)
    return


@app.cell
def _(pd):
    df_neo_raw = pd.read_csv('./data/processed/planets.csv')
    return (df_neo_raw,)


@app.cell
def _(df_neo_raw):
    X_neo_raw = df_neo_raw.drop(columns=['hazardous'])
    y_neo = df_neo_raw['hazardous']
    return X_neo_raw, y_neo


@app.cell
def _(X_neo_raw, plot_clusters, y_neo):
    plot_clusters(X_neo_raw, y_neo)
    return


@app.cell
def _(StandardScaler, X_neo_raw):
    scaler = StandardScaler()

    X_neo = scaler.fit_transform(X_neo_raw)
    return (X_neo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Алгоритмы на синтетических данных
    """)
    return


@app.cell
def _(KMeans, SEED, plt):
    def show_kmeans_cluster_centers(X, y):
        kmeans = KMeans(3, n_init='auto', random_state=SEED).fit(X)
        plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='plasma', alpha=0.7, edgecolors='k')
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='red', edgecolors='k')
        plt.show()

    return (show_kmeans_cluster_centers,)


@app.cell
def _(X_class, show_kmeans_cluster_centers, y_class):
    show_kmeans_cluster_centers(X_class, y_class)
    return


@app.cell
def _(X_blobs, show_kmeans_cluster_centers, y_blobs):
    show_kmeans_cluster_centers(X_blobs, y_blobs)
    return


@app.cell
def _(KMeans, SEED, plt):
    def display_elbow_plot(X):
        inertias = []
        cluster_range = range(1, 9)
        for i in cluster_range:
            km = KMeans(n_clusters=i, random_state=SEED)
            km.fit(X)
            inertias.append(km.inertia_)

        plt.plot(cluster_range, inertias, marker='o')
        plt.title("Метод локтя")
        plt.xlabel("Количество кластеров")
        plt.ylabel("Суммарная внутрикластерная дисперсия")
        plt.xticks(cluster_range)
        plt.show()

    return (display_elbow_plot,)


@app.cell
def _(X_class, display_elbow_plot):
    display_elbow_plot(X_class)
    return


@app.cell
def _(X_blobs, display_elbow_plot):
    display_elbow_plot(X_blobs)
    return


@app.cell
def _(KMeans, SEED, np, plt, silhouette_score):
    def display_silhouette_plot(X, sample_size=5000):
        if X.shape[0] > sample_size:
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        silhouettes = []
        cluster_range = range(2, 16)
        for i in cluster_range:
            km = KMeans(n_clusters=i, random_state=SEED)
            labels = km.fit_predict(X_sample)
            score = silhouette_score(X_sample, labels)
            silhouettes.append(score)

        plt.plot(cluster_range, silhouettes, marker='o')
        plt.title("Анализ силуэта")
        plt.xlabel("Количество кластеров")
        plt.ylabel("Средняя оценка силуэта")
        plt.xticks(cluster_range)
        plt.show()

    return (display_silhouette_plot,)


@app.cell
def _(X_class, display_silhouette_plot):
    display_silhouette_plot(X_class)
    return


@app.cell
def _(X_blobs, display_silhouette_plot):
    display_silhouette_plot(X_blobs)
    return


@app.cell
def _(
    adjusted_rand_score,
    calinski_harabasz_score,
    homogeneity_score,
    mo,
    np,
    pd,
    plt,
    silhouette_score,
):
    def plot_clusters_comparison(X, y_true, labels, zoom=True):
        from sklearn.decomposition import PCA
        import numpy as np

        X_pca = PCA(n_components=2).fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if zoom:
            x_min, x_max = np.percentile(X_pca[:, 0], [1, 99])
            y_min, y_max = np.percentile(X_pca[:, 1], [1, 99])
        else:
            x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
            y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()

        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='plasma', s=50, alpha=0.7, edgecolor='k')
        axes[0].set_title("True Labels")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_xlim(x_min - 1, x_max + 1)
        axes[0].set_ylim(y_min - 1, y_max + 1)

        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', s=50, alpha=0.7, edgecolor='k')
        axes[1].set_title("Predicted Clusters")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_xlim(x_min - 1, x_max + 1)
        axes[1].set_ylim(y_min - 1, y_max + 1)

        plt.tight_layout()
        return fig

    def display_clustering_metrics(X, y_true, labels, df_original=None, silhouette_sample=5000):
        fig = plot_clusters_comparison(X, y_true, labels)

        if (X.shape[0] > silhouette_sample):
            idx = np.random.choice(X.shape[0], silhouette_sample, replace=False)
            X_sil = X[idx]
            labels_sil = labels[idx]
        else:
            X_sil = X
            labels_sil = labels


        df_metrics = pd.DataFrame(
            {
                f"Средний индекс силуэта (до {silhouette_sample})": [silhouette_score(X_sil, labels_sil)],
                "Индекс Калински-Харабаша": [round(calinski_harabasz_score(X, labels), 2)],
                "Уровень однородности": [homogeneity_score(y_true, labels)],
                "ARI": [adjusted_rand_score(y_true, labels)],
            }
        )

        if df_original is not None:
            df_clusters = df_original.copy()
        else:
            df_clusters = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

        df_clusters['Cluster'] = labels

        cluster_stats = df_clusters.groupby('Cluster').agg(['mean', 'median', 'std'])

        return mo.vstack(
            [
                fig,
                mo.md("Метрики"),
                mo.ui.table(df_metrics.round(4)),
                mo.md("Характеристики кластеров"),
                mo.ui.table(cluster_stats.round(4))

            ]
        )

    return (display_clustering_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KMeans
    """)
    return


@app.cell
def _(KMeans, SEED, display_clustering_metrics):
    def run_kmeans(X, y, **kwargs):
        km = KMeans(random_state=SEED, **kwargs)
        labels = km.fit_predict(X)

        return display_clustering_metrics(X, y, labels)

    return (run_kmeans,)


@app.cell
def _(X_class, run_kmeans, y_class):
    run_kmeans(X_class, y_class, n_clusters=3)
    return


@app.cell
def _(X_blobs, run_kmeans, y_blobs):
    run_kmeans(X_blobs, y_blobs, n_clusters=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Иерархическая кластеризация
    """)
    return


@app.cell
def _(hierarchy, plt):
    def display_hierarchy_plot(X):
        clusters = hierarchy.linkage(X, method="ward")

        plt.figure(figsize=(8, 6))
        dendrogram = hierarchy.dendrogram(clusters)
        plt.axhline(25, color='red', linestyle='--')
        plt.axhline(15, color='crimson')
        plt.show()

    return (display_hierarchy_plot,)


@app.cell
def _(X_class, display_hierarchy_plot):
    display_hierarchy_plot(X_class)
    return


@app.cell
def _(AgglomerativeClustering, display_clustering_metrics):
    def run_hier_clustering(X, y, **kwargs):
        hc = AgglomerativeClustering(linkage='ward', **kwargs)
        hc_clusters = hc.fit_predict(X)

        return display_clustering_metrics(X, y, hc_clusters)

    return (run_hier_clustering,)


@app.cell
def _(X_class, run_hier_clustering, y_class):
    run_hier_clustering(X_class, y_class, n_clusters=3)
    return


@app.cell
def _(X_blobs, display_hierarchy_plot):
    display_hierarchy_plot(X_blobs)
    return


@app.cell
def _(X_blobs, run_hier_clustering, y_blobs):
    run_hier_clustering(X_blobs, y_blobs, n_clusters=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DBSCAN
    """)
    return


@app.cell
def _(DBSCAN, StandardScaler, display_clustering_metrics):
    def run_dbscan(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dbscan = DBSCAN(min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)

        return display_clustering_metrics(X_scaled, y, clusters)

    return (run_dbscan,)


@app.cell
def _(X_class, run_dbscan, y_class):
    run_dbscan(X_class, y_class)
    return


@app.cell
def _(X_blobs, run_dbscan, y_blobs):
    run_dbscan(X_blobs, y_blobs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EM-алгоритм
    """)
    return


@app.cell
def _(GaussianMixture, display_clustering_metrics):
    def run_em_clustering(X, y, **kwargs):
        gm = GaussianMixture(**kwargs)

        clusters = gm.fit_predict(X)

        return display_clustering_metrics(X, y, clusters)

    return (run_em_clustering,)


@app.cell
def _(X_class, run_em_clustering, y_class):
    run_em_clustering(X_class, y_class, n_components=3)
    return


@app.cell
def _(X_blobs, run_em_clustering, y_blobs):
    run_em_clustering(X_blobs, y_blobs, n_components=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Affinity Propagation
    """)
    return


@app.cell
def _(AffinityPropagation, SEED, display_clustering_metrics):
    def run_affinity_propagation(X, y):
        ap = AffinityPropagation(random_state=SEED)

        clusters = ap.fit_predict(X)

        return display_clustering_metrics(X, y, clusters)

    return (run_affinity_propagation,)


@app.cell
def _(X_class, run_affinity_propagation, y_class):
    run_affinity_propagation(X_class, y_class)
    return


@app.cell
def _(X_blobs, run_affinity_propagation, y_blobs):
    run_affinity_propagation(X_blobs, y_blobs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Алгоритмы на данных для классификации (NEO)
    """)
    return


@app.cell
def _(X_neo, display_elbow_plot):
    display_elbow_plot(X_neo)
    return


@app.cell
def _(X_neo, display_silhouette_plot):
    display_silhouette_plot(X_neo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KMeans
    """)
    return


@app.cell
def _(X_neo, run_kmeans, y_neo):
    run_kmeans(X_neo, y_neo, n_clusters=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Иерархическая кластеризация
    """)
    return


@app.cell
def _(AgglomerativeClustering, SEED, display_clustering_metrics, resample):
    def run_hier_clustering_sampled(X, y, n_samples=5000, **kwargs):
        if len(X) > n_samples:
            X_sample, y_sample = resample(X, y, n_samples=n_samples, random_state=SEED)
        else:
            X_sample, y_sample = X, y

        hc = AgglomerativeClustering(linkage='ward', **kwargs)
        clusters = hc.fit_predict(X_sample)

        return display_clustering_metrics(X_sample, y_sample, clusters)


    return (run_hier_clustering_sampled,)


@app.cell
def _(X_neo, run_hier_clustering_sampled, y_neo):
    run_hier_clustering_sampled(X_neo, y_neo, n_clusters=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DBSCAN
    """)
    return


@app.cell
def _(X_neo, run_dbscan, y_neo):
    run_dbscan(X_neo[:1000], y_neo[:1000])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EM-алгоритм
    """)
    return


@app.cell
def _(X_neo, run_em_clustering, y_neo):
    run_em_clustering(X_neo, y_neo, n_components=3, max_iter=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Affinity Propagation
    """)
    return


@app.cell
def _(AffinityPropagation, SEED, display_clustering_metrics, resample):
    def run_affinity_propagation_sampled(X, y, n_samples=5000):

        if len(X) > n_samples:
            X_sample, y_sample = resample(X, y, n_samples=n_samples, random_state=SEED)
        else:
            X_sample, y_sample = X, y

        ap = AffinityPropagation(random_state=SEED)
        clusters = ap.fit_predict(X_sample)

        return display_clustering_metrics(X_sample, y_sample, clusters)

    return (run_affinity_propagation_sampled,)


@app.cell
def _(X_neo, run_affinity_propagation_sampled, y_neo):
    run_affinity_propagation_sampled(X_neo, y_neo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MyKMeans
    """)
    return


@app.cell
def _(SEED, np):
    class MyKMeans:
        def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=SEED):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.tol = tol
            self.random_state = random_state
            self.centroids = None
            self.labels_ = None
            self.inertia_ = None

        def fit(self, X):
            np.random.seed(self.random_state)

            n_samples, n_features = X.shape

            random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[random_idx]

            for iteration in range(self.max_iter):
                distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                                          for k in range(self.n_clusters)])

                if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                    break

                self.centroids = new_centroids

            self.labels_ = labels
            self.inertia_ = np.sum((X - self.centroids[labels])**2)
            return self

        def predict(self, X):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            return np.argmin(distances, axis=1)

        def fit_predict(self, X):
            self.fit(X)
            return self.labels_

    return (MyKMeans,)


@app.cell
def _(MyKMeans, display_clustering_metrics):
    def run_my_kmeans(X, y):
        myKMeans = MyKMeans(random_state=10)

        clusters = myKMeans.fit_predict(X)

        print(myKMeans.inertia_)

        return display_clustering_metrics(X, y, clusters)

    return (run_my_kmeans,)


@app.cell
def _(X_class, run_my_kmeans, y_class):
    run_my_kmeans(X_class, y_class)
    return


@app.cell
def _(X_blobs, run_my_kmeans, y_blobs):
    run_my_kmeans(X_blobs, y_blobs)
    return


if __name__ == "__main__":
    app.run()
