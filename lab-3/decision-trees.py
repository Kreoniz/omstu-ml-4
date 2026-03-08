import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
    from sklearn.ensemble import BaggingRegressor, BaggingClassifier, GradientBoostingRegressor, GradientBoostingClassifier, StackingRegressor, StackingClassifier
    from scipy.stats import randint

    return (
        BaggingClassifier,
        DecisionTreeRegressor,
        GridSearchCV,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        mo,
        pd,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision trees
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Подготовка данных
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Такси (DecisionTreeRegressor)
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
def _(pd):
    df_taxi_raw = pd.read_csv('./data/processed/taxi.csv')
    df_taxi_raw
    return (df_taxi_raw,)


@app.cell
def _(df_taxi_raw):
    X_taxi = df_taxi_raw.drop('trip_duration', axis=1)
    y_taxi = df_taxi_raw['trip_duration']
    return X_taxi, y_taxi


@app.cell
def _(X_taxi, train_test_split, y_taxi):
    X_taxi_train, X_taxi_test, y_taxi_train, y_taxi_test = train_test_split(X_taxi, y_taxi, test_size=0.2, random_state=42, stratify=X_taxi['vendor_id'])
    return X_taxi_test, X_taxi_train, y_taxi_test, y_taxi_train


@app.cell
def _(DecisionTreeRegressor, X_taxi_train, y_taxi_train):
    tree_taxi = DecisionTreeRegressor(max_depth=3)
    tree_taxi.fit(X_taxi_train, y_taxi_train)
    return (tree_taxi,)


@app.cell
def _(X_taxi, df_taxi_raw, tree_taxi, y_taxi):
    from sklearn import tree
    from matplotlib import pyplot as plt

    text_representation = tree.export_text(tree_taxi)
    fig = plt.figure(figsize=(25,20))
    plt.figure(figsize=(25,20))
    tree.plot_tree(tree_taxi, feature_names=X_taxi.columns.tolist(), class_names=df_taxi_raw[y_taxi.name].unique().astype(str), filled=True)
    return


@app.cell
def _(X_taxi_test, X_taxi_train, tree_taxi):
    y_taxi_train_pred = tree_taxi.predict(X_taxi_train)
    y_taxi_pred = tree_taxi.predict(X_taxi_test)
    return y_taxi_pred, y_taxi_train_pred


@app.cell
def _(
    get_regression_report,
    y_taxi_pred,
    y_taxi_test,
    y_taxi_train,
    y_taxi_train_pred,
):
    get_regression_report(y_taxi_train, y_taxi_train_pred, y_taxi_test, y_taxi_pred)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    GridSearchCV,
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    y_taxi_test,
    y_taxi_train,
):
    def run_decision_tree_regressor_gridsearch_cv(X_train, y_train, X_test, y_test):
        param_grid = {
        'max_depth': [5, 10, 20, 50],  
        'min_samples_split': [2, 4, 6],      
        'min_samples_leaf': [1, 2, 3],        
        'criterion': ['squared_error'],                    
        'splitter': ['best'],                            
        'max_features': ['sqrt', 'log2', None]                  
    }

        rtree = DecisionTreeRegressor()
    
        grid_search = GridSearchCV(
            estimator=rtree,
            param_grid=param_grid,
            cv=3,                      
            n_jobs=-1
        )
    
        grid_search.fit(X_train, y_train)
    
        print("Лучшие гиперпараметры:", grid_search.best_params_)
    
        best_regressor = grid_search.best_estimator_
        train_predictions = best_regressor.predict(X_train)
        predictions = best_regressor.predict(X_test)
        score = best_regressor.score(X_test, y_test)
        print(f"Оценка R^2 на тестовых данных: {score:.2f}")

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_decision_tree_regressor_gridsearch_cv(X_taxi_train, y_taxi_train, X_taxi_test, y_taxi_test)
    return


@app.cell
def _(
    BaggingClassifier,
    GridSearchCV,
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    y_taxi_test,
    y_taxi_train,
):
    def run_bagging_decision_tree_regressor_gridsearch_cv(X_train, y_train, X_test, y_test):
        param_grid = {
            'n_estimators': [3, 6, 9],    
            'max_samples': [0.5, 0.7, 0.9, 1.0],        
            'bootstrap': [True, False],                   
        }
    
        bagging = BaggingClassifier(random_state=42)
    
        grid_search = GridSearchCV(
            bagging,
            param_grid=param_grid,
            cv=3,                      
            n_jobs=-1
        )
    
        grid_search.fit(X_train, y_train)
    
        print("Лучшие гиперпараметры:", grid_search.best_params_)
    
        best_regressor = grid_search.best_estimator_
        train_predictions = best_regressor.predict(X_train)
        predictions = best_regressor.predict(X_test)
        score = best_regressor.score(X_test, y_test)
        print(f"Оценка R^2 на тестовых данных: {score:.2f}")

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_bagging_decision_tree_regressor_gridsearch_cv(X_taxi_train, y_taxi_train, X_taxi_test, y_taxi_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Астероиды (DecisionTreeClassifier)
    """)
    return


@app.cell
def _(pd):
    df_neo_raw = pd.read_csv('./data/processed/planets.csv')
    df_neo_raw
    return (df_neo_raw,)


@app.cell
def _(df_neo_raw):
    X_neo = df_neo_raw.drop('hazardous', axis=1)
    y_neo = df_neo_raw['hazardous']
    return X_neo, y_neo


@app.cell
def _(X_neo, train_test_split, y_neo):
    X_neo_train, X_neo_test, y_neo_train, y_neo_test = train_test_split(X_neo, y_neo, test_size=0.2, random_state=42, stratify=['hazardous'])
    return


if __name__ == "__main__":
    app.run()
