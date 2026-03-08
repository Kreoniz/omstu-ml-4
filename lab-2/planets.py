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
    import seaborn as sns
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

    import warnings
    warnings.filterwarnings("ignore")
    return (
        SMOTE,
        accuracy_score,
        confusion_matrix,
        f1_score,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        sns,
        train_test_split,
    )


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    f1_score,
    mo,
    pd,
    plt,
    precision_score,
    recall_score,
    roc_auc_score,
    sns,
):
    def plot_distributions(y_train, y_pred_train, y_test, y_pred_test):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.histplot(y_train, label='Истинные', color='blue', kde=False, bins=20, alpha=0.4, ax=axes[0], stat="density")
        sns.histplot(y_pred_train, label='Предсказанные', color='blue', kde=False, bins=20, alpha=0.7, linestyle='--', ax=axes[0], stat="density")
        axes[0].set_title('Train: Истинные vs Предсказанные')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        sns.histplot(y_test, label='Истинные', color='green', kde=False, bins=20, alpha=0.4, ax=axes[1], stat="density")
        sns.histplot(y_pred_test, label='Предсказанные', color='green', kde=False, bins=20, alpha=0.7, linestyle='--', ax=axes[1], stat="density")
        axes[1].set_title('Test: Истинные vs Предсказанные')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Распределения: Истинные vs Предсказанные', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

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

    def get_report(y_train, y_pred_train, y_test, y_pred_test):
        fig = plot_distributions(y_train, y_pred_train, y_test, y_pred_test)

        metrics_train = calculate_classification_metrics(y_train, y_pred_train)
        metrics_test = calculate_classification_metrics(y_test, y_pred_test)

        cm_train = confusion_matrix_df(metrics_train["Confusion Matrix"].flatten())
        cm_test = confusion_matrix_df(metrics_test["Confusion Matrix"].flatten())

        metrics_df_train = pd.DataFrame([metrics_train['metrics']]).round(5)

        metrics_df_test = pd.DataFrame([metrics_test['metrics']]).round(5)

        return mo.vstack([
            fig,
            mo.md("#### Метрики модели (Train vs Test)"),
            mo.md("**Train**"),
            metrics_df_train,
            mo.md("Confusion matrix"),
            cm_train,
            mo.md("**Test**"),
            metrics_df_test,
            mo.md("Confusion matrix"),
            cm_test
        ])

    return (get_report,)


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
    return X, transformed_X, transformed_y


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
def _(X, transformed_X, transformed_y):
    from sklearn.model_selection import KFold

    kFold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kFold.split(X):
        X_kf_train, X_kf_test = transformed_X.iloc[train_index], transformed_X.iloc[test_index]
        y_kf_train, y_kf_test = transformed_y.iloc[train_index], transformed_y.iloc[test_index]
    return KFold, X_kf_test, X_kf_train, kFold, y_kf_test, y_kf_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Алгоритмы классификации
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LogisticRegression
    """)
    return


@app.cell
def _(
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    from sklearn.linear_model import LogisticRegression

    def run_simple_logistic_regression(X_train, y_train, X_test, y_test):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)

        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_simple_logistic_regression(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return (LogisticRegression,)


@app.cell
def _(KFold, LogisticRegression, get_report, transformed_X, transformed_y):
    def run_simple_logistic_regression_kfold(X, y):
        kFold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_reports = []

        for train_index, test_index in kFold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            lr = LogisticRegression(max_iter=1000)

            lr.fit(X_train, y_train)
        
            y_pred_train = lr.predict(X_train)
            y_pred_test = lr.predict(X_test)

            fold_report = get_report(y_train, y_pred_train, y_test, y_pred_test)
            fold_reports.append(fold_report)

        return fold_reports

    run_simple_logistic_regression_kfold(transformed_X, transformed_y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LogisticRegression. GridSearchCV
    """)
    return


@app.cell
def _(
    LogisticRegression,
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    from sklearn.model_selection import GridSearchCV

    def run_grid_search_logistic_regression(X_train, y_train, X_test, y_test):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
        }

        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring='f1_micro')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        print('Лучшие гипермараметры:', grid_search.best_params_)
        print(f'Лучшая оценка f1_micro: {grid_search.best_score_:.4f}')

        predictions_train = best_model.predict(X_train)
        predictions_test = best_model.predict(X_test)

        return get_report(y_train, predictions_train, y_test, predictions_test)

    run_grid_search_logistic_regression(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return (GridSearchCV,)


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    X_holdout_test,
    X_holdout_train,
    get_report,
    kFold,
    y_holdout_test,
    y_holdout_train,
):
    def run_grid_search_logistic_regression_kfold(X_train, y_train, X_test, y_test):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
        }

        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=kFold, scoring='f1_micro')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        print('Лучшие гипермараметры:', grid_search.best_params_)
        print(f'Лучшая оценка f1_micro: {grid_search.best_score_:.4f}')

        predictions_train = best_model.predict(X_train)
        predictions_test = best_model.predict(X_test)

        return get_report(y_train, predictions_train, y_test, predictions_test)

    run_grid_search_logistic_regression_kfold(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LogisticRegression. RandomizedSearchCV
    """)
    return


@app.cell
def _(
    LogisticRegression,
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    from sklearn.model_selection import RandomizedSearchCV

    def run_randomized_search_logistic_regression(X_train, y_train, X_test, y_test):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
        }

        random_search = RandomizedSearchCV(LogisticRegression(), param_grid, cv=3, scoring='f1_micro')
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_

        print('Лучшие гипермараметры:', random_search.best_params_)
        print(f'Лучшая оценка f1_micro: {random_search.best_score_:.4f}')

        predictions_train = best_model.predict(X_train)
        predictions_test = best_model.predict(X_test)

        return get_report(y_train, predictions_train, y_test, predictions_test)

    run_randomized_search_logistic_regression(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell
def _(
    LogisticRegression,
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    import optuna
    from sklearn.model_selection import cross_val_score

    def run_optuna_logistic_regression(X_train, y_train, X_test, y_test):
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 10, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            }

            model = LogisticRegression(**params)

            score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_micro').mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_optuna_params = study.best_params

        print('Лучшие гипермараметры:', best_optuna_params)
        print(f'Лучшая оценка f1_micro: {study.best_value:.4f}')

        final_model = LogisticRegression(**best_optuna_params)
        final_model.fit(X_train, y_train)

        predictions_train = final_model.predict(X_train)
        predictions_test = final_model.predict(X_test)

        return get_report(y_train, predictions_train, y_test, predictions_test)

    run_optuna_logistic_regression(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN
    """)
    return


@app.cell
def _(
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    from sklearn.neighbors import KNeighborsClassifier

    def run_simple_knn(X_train, y_train, X_test, y_test):
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_simple_knn(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return (KNeighborsClassifier,)


@app.cell
def _(KFold, KNeighborsClassifier, get_report, transformed_X, transformed_y):
    def run_simple_knn_kfold(X, y):
        kFold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_reports = []

        for train_index, test_index in kFold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            classifier = KNeighborsClassifier()
        
            classifier.fit(X_train, y_train)
    
            y_pred_train = classifier.predict(X_train)
            y_pred_test = classifier.predict(X_test)
    
            fold_report = get_report(y_train, y_pred_train, y_test, y_pred_test)
            fold_reports.append(fold_report)
    
        return fold_reports


        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_simple_knn_kfold(transformed_X, transformed_y)
    return


@app.cell
def _(
    KNeighborsClassifier,
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    def run_minkowski_knn(X_train, y_train, X_test, y_test):
        classifier = KNeighborsClassifier(metric='minkowski', p=1)
        classifier.fit(X_holdout_train, y_holdout_train)
        predictions = classifier.predict(X_holdout_test)

        y_pred_train = classifier.predict(X_holdout_train)
        y_pred_test = classifier.predict(X_holdout_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_minkowski_knn(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell
def _(
    KNeighborsClassifier,
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    def run_brute_knn(X_train, y_train, X_test, y_test):
        classifier = KNeighborsClassifier(algorithm="brute")
        classifier.fit(X_holdout_train, y_holdout_train)
        predictions = classifier.predict(X_holdout_test)

        y_pred_train = classifier.predict(X_holdout_train)
        y_pred_test = classifier.predict(X_holdout_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_brute_knn(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Naive Bayes. GaussianNB
    """)
    return


@app.cell
def _(
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    from sklearn.naive_bayes import GaussianNB

    def run_gaussian_nb(X_train, y_train, X_test, y_test):
        classifier = GaussianNB()

        classifier.fit(X_train, y_train)

        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_gaussian_nb(X_holdout_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Naive Bayes. MultinomialNB
    """)
    return


@app.cell
def _(X_kf_test, X_kf_train, get_report, y_kf_test, y_kf_train):
    from sklearn.naive_bayes import MultinomialNB

    def run_multinomial_nb(X_train, y_train, X_test, y_test):
        classifier = MultinomialNB()
        classifier.fit(X_train, y_train)
    
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
    
        return get_report(y_train, y_pred_train, y_test, y_pred_test)
    
    run_multinomial_nb(X_kf_train, y_kf_train, X_kf_test, y_kf_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVM
    """)
    return


@app.cell
def _(X_holdout_test, X_k_train, get_report, y_holdout_test, y_holdout_train):
    from sklearn import svm

    def run_simple_svm(X_train, y_train, X_test, y_test):
        clf = svm.LinearSVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    run_simple_svm(X_k_train, y_holdout_train, X_holdout_test, y_holdout_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Кастомный KNN классификатор
    """)
    return


@app.cell
def _(np, pd):
    from collections import Counter

    class MyKNNClassifier:
        def __init__(self, n_neighbors=5, metric='minkowski', p=2):
            self.n_neighbors = n_neighbors
            self.metric = metric
            self.p = p
            self.X = None
            self.y = None

        def fit(self, X, y):
            self.X = X
            self.y = y

            return self

        def _calculate_distance(self, x1, x2):
            x1, x2 = np.array(x1), np.array(x2)

            match self.metric:
                case 'manhattan':
                    return np.sum(np.abs(x1 - x2))
                case 'minkowski':
                    return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
                case _:
                    print(f"ERROR: no such metric found: {self.metric}")
                    return None

        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.values

            predictions = [self._predict(x) for x in X]
            return np.array(predictions)

        def _predict(self, x):
            distances = [self._calculate_distance(x_train, x) for x_train in self.X.values]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = [self.y.iloc[i] for i in k_indices]

            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]

        def _get_neighbors(self, x):
            distance_between = [(idx, self._calculate_distance(x, self.X.iloc[idx])) for idx in range(len(self.X))]
            sorted_distance_between = sorted(distance_between, key=lambda x: x[1])
            idx_neighbors = [idx for idx, distance in sorted_distance_between]
            return idx_neighbors[:self.n_neighbors]

        def _get_most_frequent_neighbors_class(self, idx_neighbors):
            labels = [self.y.iloc[idx] for idx in idx_neighbors]
            frequency_dict = {label: labels.count(label) for label in set(labels)}
            sorted_frequency_dict = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
            return sorted_frequency_dict[0][0]

    return (MyKNNClassifier,)


@app.cell
def _(
    MyKNNClassifier,
    X_holdout_test,
    X_holdout_train,
    get_report,
    y_holdout_test,
    y_holdout_train,
):
    def run_my_knn_classifier(X_train, y_train, X_test, y_test):
        classifier = MyKNNClassifier()

        classifier.fit(X_train, y_train)

        predictions_train = classifier.predict(X_train)
        predictions_test = classifier.predict(X_test)

        return get_report(y_train, predictions_train, y_test, predictions_test)

    run_my_knn_classifier(X_holdout_train.sample(1000), y_holdout_train.sample(1000), X_holdout_test.sample(1000), y_holdout_test.sample(1000))    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
