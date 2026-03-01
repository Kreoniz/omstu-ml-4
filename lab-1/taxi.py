import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Регрессия. Такси
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score

    from scipy.stats import loguniform, uniform

    import warnings
    warnings.filterwarnings('ignore')
    return (
        ColumnTransformer,
        ElasticNet,
        GridSearchCV,
        Lasso,
        LinearRegression,
        MedianPruner,
        OneHotEncoder,
        Pipeline,
        PolynomialFeatures,
        RandomizedSearchCV,
        Ridge,
        StandardScaler,
        TPESampler,
        loguniform,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        mo,
        np,
        optuna,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
        uniform,
    )


@app.cell(hide_code=True)
def _(
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mo,
    np,
    pd,
    plt,
    r2_score,
    sns,
):
    def calculate_metrics(y_true, y_pred):
        """Вычисляет метрики через sklearn"""
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "R^2": r2_score(y_true, y_pred)
        }

    def calculate_manual_metrics(y_true, y_pred):
        """Вычисляет метрики вручную (для сравнения)"""
        n = y_true.size
        mse = (1 / n) * np.sum((y_true - y_pred)**2)
        return {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": (1 / n) * np.sum(np.abs(y_true - y_pred)),
            "MAPE": (1 / n) * np.sum(np.abs(y_true - y_pred) / np.abs(y_true) * 100),
            "R^2": 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
        }

    def get_metrics_report(y_train, y_pred_train, y_test, y_pred_test):
        """Возвращает DataFrame с метриками для таблицы"""
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["Train", "Test"],
            columns=["MSE", "RMSE", "MAE", "MAPE", "R^2"]
        )
        return df.round(5)

    def get_manual_metrics_comparison(y_test, y_pred_test):
        """Сравнение sklearn vs ручные метрики"""
        sklearn_metrics = calculate_metrics(y_test, y_pred_test)
        manual_metrics = calculate_manual_metrics(y_test, y_pred_test)

        df = pd.DataFrame({
            "Metric": ["MSE", "RMSE", "MAE", "MAPE", "R^2"],
            "sklearn": [sklearn_metrics[k] for k in ["MSE", "RMSE", "MAE", "MAPE", "R^2"]],
            "manual": [manual_metrics[k] for k in ["MSE", "RMSE", "MAE", "MAPE", "R^2"]],
        })
        return df.round(5)

    def plot_distributions(y_train, y_pred_train, y_test, y_pred_test):
        """Возвращает Figure (без plt.show!)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.kdeplot(y_train, label='Истинные', fill=True, color='blue', alpha=0.4, ax=axes[0])
        sns.kdeplot(y_pred_train, label='Предсказанные', fill=True, color='blue', alpha=0.7, linestyle='--', ax=axes[0])
        axes[0].set_title('Train')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        sns.kdeplot(y_test, label='Истинные', fill=True, color='green', alpha=0.4, ax=axes[1])
        sns.kdeplot(y_pred_test, label='Предсказанные', fill=True, color='green', alpha=0.7, linestyle='--', ax=axes[1])
        axes[1].set_title('Test')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Распределения: Истинные vs Предсказанные', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

    def get_report(y_train, y_pred_train, y_test, y_pred_test):
        """
        Возвращает графики и метрики в Marimo.
        Работает корректно при вызове из других функций.
        """
        fig = plot_distributions(y_train, y_pred_train, y_test, y_pred_test)
        metrics_df = get_metrics_report(y_train, y_pred_train, y_test, y_pred_test)
        manual_comparison_df = get_manual_metrics_comparison(y_test, y_pred_test)

        return mo.vstack([
            fig,
            mo.md("### Метрики модели (sklearn)"),
            metrics_df,
            mo.md("### Сравнение с ручными метриками (Test)"),
            manual_comparison_df
        ])

    return (get_report,)


@app.cell
def _(ElasticNet, GridSearchCV, get_report, np):
    def gridSearchCV(model, X_train, X_test, y_train, y_test):

        if type(model) == ElasticNet:
            param_grid = {
                'alpha': np.arange(0.1, 1, 0.1),
                'max_iter': np.arange(1000, 10001, 1000),
                'l1_ratio': np.arange(0.1, 0.9, 0.1)
        }
        else:
            param_grid = {
                'alpha': np.arange(0.1, 1, 0.1),
                'max_iter': np.arange(1000, 10001, 1000),
            }

        grid_model = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)

        grid_model.fit(X_train, y_train)

        print(grid_model.best_params_)
        print(grid_model.best_score_)

        y_pred_train = grid_model.predict(X_train)
        y_pred_test = grid_model.predict(X_test)

        print(grid_model.best_estimator_.coef_)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)


    return (gridSearchCV,)


@app.cell
def _(ElasticNet, RandomizedSearchCV, get_report, loguniform, np, uniform):
    def randomizedSearchCV(model, X_train, X_test, y_train, y_test):

        if type(model) == ElasticNet:
            param_grid = {
                'alpha': loguniform(1e-4, 1e2),
                'max_iter': np.arange(1000, 10001, 100),
                'l1_ratio': uniform(0.1, 0.9)
            }
        else:
            param_grid = {
                'alpha': loguniform(1e-4, 1e2),
                'max_iter': np.arange(1000, 10001, 100),
            }

        randomized_model = RandomizedSearchCV(model, param_grid, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1)

        randomized_model.fit(X_train, y_train)

        print(randomized_model.best_params_)
        print(randomized_model.best_score_)

        y_pred_train = randomized_model.predict(X_train)
        y_pred_test = randomized_model.predict(X_test)

        print(randomized_model.best_estimator_.coef_)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)


    return (randomizedSearchCV,)


@app.cell
def _(
    ElasticNet,
    MedianPruner,
    TPESampler,
    get_report,
    mean_squared_error,
    optuna,
):
    def optunaSearch(model_class, X_train, X_test, y_train, y_test):

        def objective(trial):

            alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)
            max_iter = trial.suggest_int('max_iter', 1000, 10000)


            if model_class == ElasticNet:
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.99)
                model = model_class(
                    alpha=alpha,
                    max_iter=max_iter,
                    l1_ratio=l1_ratio
                )
            else:
                model = model_class(
                    alpha=alpha,
                    max_iter=max_iter
                )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            return mse  

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(),
            pruner=MedianPruner(),
        )


        study.optimize(objective, n_trials=10, n_jobs=-1)

        print(study.best_params)

        best_params = study.best_params
        best_model = model_class(alpha=best_params['alpha'], max_iter=best_params['max_iter']) if model_class != ElasticNet else \
                                     model_class(alpha=best_params['alpha'], max_iter=best_params['max_iter'], l1_ratio=best_params['l1_ratio'])
        best_model.fit(X_train, y_train)

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)


    return (optunaSearch,)


@app.cell
def _(get_report):
    def predictingWithoutRegularization(model, X_train, y_train, X_test, y_test):
        reg = model.fit(X_train, y_train)

        y_pred_test = reg.predict(X_test)
        y_pred_train = reg.predict(X_train)

        print(reg.coef_)

        return get_report(y_train, y_pred_train, y_test, y_pred_test)

    return (predictingWithoutRegularization,)


@app.cell
def _(gridSearchCV, optunaSearch, randomizedSearchCV):
    def predictingWithRegularization(model, X_train, y_train, X_test, y_test):
        reports = {}

        print('GridSearchCV')
        reports['GridSearchCV'] = gridSearchCV(model, X_train, X_test, y_train, y_test)

        print('RandomizedSearchCV')
        reports['RandomizedSearchCV'] = randomizedSearchCV(model, X_train, X_test, y_train, y_test)

        print('Optuna')
        model_class = model.__class__
        reports['Optuna'] = optunaSearch(model_class, X_train, X_test, y_train, y_test)

        return reports

    return (predictingWithRegularization,)


@app.cell
def _(pd):
    df = pd.read_csv('./data/processed/taxi.csv')
    return (df,)


@app.cell
def _(ColumnTransformer, OneHotEncoder, StandardScaler, df, train_test_split):
    target = 'trip_duration'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['vendor_id'])

    cat_cols = ['hour', 'month', 'weekday', 'vendor_id']
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Простая линейная регрессия
    """)
    return


@app.cell
def _(
    LinearRegression,
    X_test,
    X_train,
    predictingWithoutRegularization,
    y_test,
    y_train,
):
    predictingWithoutRegularization(LinearRegression(), X_train, y_train, X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Линейная регрессия с L1-регуляризацией
    """)
    return


@app.cell
def _(Lasso, X_test, X_train, predictingWithRegularization, y_test, y_train):
    lasso_reports = predictingWithRegularization(Lasso(), X_train, y_train, X_test, y_test)
    return (lasso_reports,)


@app.cell
def _(lasso_reports):
    lasso_reports['GridSearchCV']
    return


@app.cell
def _(lasso_reports):
    lasso_reports['RandomizedSearchCV']
    return


@app.cell
def _(lasso_reports):
    lasso_reports['Optuna']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Линейная регрессия с L2-регуляризацией
    """)
    return


@app.cell
def _(Ridge, X_test, X_train, predictingWithRegularization, y_test, y_train):
    ridge_reports = predictingWithRegularization(Ridge(), X_train, y_train, X_test, y_test)
    return (ridge_reports,)


@app.cell
def _(ridge_reports):
    ridge_reports['GridSearchCV']
    return


@app.cell
def _(ridge_reports):
    ridge_reports['RandomizedSearchCV']
    return


@app.cell
def _(ridge_reports):
    ridge_reports['Optuna']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Линейная регрессия с двумя регуляризаторами
    """)
    return


@app.cell
def _(
    ElasticNet,
    X_test,
    X_train,
    predictingWithRegularization,
    y_test,
    y_train,
):
    elastic_reports = predictingWithRegularization(ElasticNet(), X_train, y_train, X_test, y_test)
    return (elastic_reports,)


@app.cell
def _(elastic_reports):
    elastic_reports['GridSearchCV']
    return


@app.cell
def _(elastic_reports):
    elastic_reports['RandomizedSearchCV']
    return


@app.cell
def _(elastic_reports):
    elastic_reports['Optuna']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Полиномиальная регрессия
    """)
    return


@app.cell
def _(
    LinearRegression,
    Pipeline,
    PolynomialFeatures,
    X_test,
    X_train,
    get_report,
    y_test,
    y_train,
):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),   
        ('linreg', LinearRegression())   
    ])         

    pipeline.fit(X_train, y_train)

    y_pred_train_pipeline = pipeline.predict(X_train)
    y_pred_test_pipeline = pipeline.predict(X_test)

    poly_report = get_report(y_train, y_pred_train_pipeline, y_test, y_pred_test_pipeline)
    poly_report
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Кастомная линейная регрессия с градиентным спуском
    """)
    return


@app.cell
def _(np):
    class MyLinearRegression:
        def __init__(self, learning_rate=0.001, n_iterations=1000, lamb=0.1):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.lamb = lamb
            self.coef_ = None # weights
            self.intercept_ = None # bias

        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.coef_ = np.random.normal(0, 0.01, size=n_features)
            self.intercept_ = 0

            for _ in range(self.n_iterations):
                y_pred = np.dot(X, self.coef_) + self.intercept_

                dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (2 / n_samples) * np.sum(y_pred - y)

                l2_reg = self.lamb * self.coef_

                self.coef_ -= self.learning_rate * (dw + 2 * l2_reg)
                self.intercept_ -= self.learning_rate * db

            return self

        def predict(self, X):
            return np.dot(X, self.coef_) + self.intercept_

    return (MyLinearRegression,)


@app.cell
def _(
    MyLinearRegression,
    X_test,
    X_train,
    predictingWithoutRegularization,
    y_test,
    y_train,
):
    predictingWithoutRegularization(MyLinearRegression(), X_train, y_train, X_test, y_test)
    return


@app.cell
def _(np, plt):
    def cost_function(x):
        return x**2

    def gradient(x):
        return 2 * x

    def gradient_descent(learning_rate, n_iterations, initial_x):
        x = initial_x
        path = [x]
    
        for _ in range(n_iterations):
            grad = gradient(x)
            x -= learning_rate * grad
            path.append(x)
    
        return path

    n_iterations = 200
    initial_x = 5

    learning_rates = [0.9, 0.01]

    path_large_lr = gradient_descent(learning_rate=learning_rates[0], n_iterations=n_iterations, initial_x=initial_x)
    path_small_lr = gradient_descent(learning_rate=learning_rates[1], n_iterations=n_iterations, initial_x=initial_x)

    x_values = np.linspace(-6, 6, 400)
    y_values = cost_function(x_values)

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, y_values, label='$y = x^2$', color='green', linewidth=2)

    plt.plot(path_large_lr, cost_function(np.array(path_large_lr)), label='Large Learning Rate (0.9)', color='red', marker='o', markersize=6)

    plt.plot(path_small_lr, cost_function(np.array(path_small_lr)), label='Small Learning Rate (0.01)', color='blue', marker='o', markersize=6)

    plt.scatter(0, 0, color='black', zorder=5)
    plt.text(0.2, 2, 'Optimal point (0, 0)', color='black', fontsize=12)

    plt.xlabel('x')
    plt.ylabel('Cost (Loss)')
    plt.title('Gradient Descent on a Parabola: Large vs Small Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Вывод

    Модель полиномиальной регрессии показывает себя лучше всего - что указывает на полиномиальную зависимость в данных. Модели линейной регрессии с регуляризацией и без справились одинаково плохо
    """)
    return


if __name__ == "__main__":
    app.run()
