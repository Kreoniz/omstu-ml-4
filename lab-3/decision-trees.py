import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.ensemble import BaggingRegressor, BaggingClassifier, GradientBoostingRegressor, GradientBoostingClassifier, StackingRegressor, StackingClassifier
    from scipy.stats import randint
    from catboost import CatBoostRegressor, CatBoostClassifier
    import xgboost
    import lightgbm
    from pycaret import regression as pr
    from pycaret import classification as pc
    import optuna

    return (
        BaggingClassifier,
        BaggingRegressor,
        CatBoostClassifier,
        CatBoostRegressor,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        GridSearchCV,
        StackingClassifier,
        StackingRegressor,
        accuracy_score,
        confusion_matrix,
        f1_score,
        lightgbm,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        mo,
        np,
        pc,
        pd,
        pr,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        root_mean_squared_error,
        train_test_split,
        xgboost,
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
    ## Такси (DecisionTreeRegressor)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### DecisionTreeRegressor. GridSearchCV
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### BaggingRegressor. DecisionTreeRegressor. GridSearchCV
    """)
    return


@app.cell
def _(
    BaggingRegressor,
    DecisionTreeRegressor,
    GridSearchCV,
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    y_taxi_test,
    y_taxi_train,
):
    def run_bagging_decision_tree_regressor_gridsearch_cv(X_train, y_train, X_test, y_test):
        param_grid = {
            'n_estimators': [5, 15],    
            'max_samples': [0.7, 0.85],        
            'bootstrap': [True, False],                   
        }

        base_estimator = DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42

        )

        bagging = BaggingRegressor(estimator=base_estimator, random_state=42)

        grid_search = GridSearchCV(
            bagging,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='r2'
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
    #### GradientBoostingRegressor. DecisionTreeRegressor
    """)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    y_taxi_test,
    y_taxi_train,
):
    def run_gradient_boosting_decision_tree_regressor(X_train, y_train, X_test, y_test):
        gbr = GradientBoostingRegressor()

        gbr.fit(X_train, y_train)

        train_predictions = gbr.predict(X_train)
        predictions = gbr.predict(X_test)

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_gradient_boosting_decision_tree_regressor(X_taxi_train, y_taxi_train, X_taxi_test, y_taxi_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### StackingRegressor. DecisionTreeRegressor
    """)
    return


@app.cell
def _(
    StackingRegressor,
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    y_taxi_test,
    y_taxi_train,
):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor

    def run_stacking_regressor(X_train, y_train, X_test, y_test):
        base_models = [
            ('lr', LinearRegression()),
            ('svr_lin', SVR(kernel='linear', C=10)),
            ('ridge', Ridge()),
        ]

        final_estimator = RandomForestRegressor() 

        stackingRegressor = StackingRegressor(
            estimators=base_models,
            final_estimator=final_estimator,
            n_jobs=-1
        )

        stackingRegressor.fit(X_train, y_train)

        train_predictions = stackingRegressor.predict(X_train)
        predictions = stackingRegressor.predict(X_test)

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_stacking_regressor(X_taxi_train[:10000], y_taxi_train[:10000], X_taxi_test[:10000], y_taxi_test[:10000])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CatBoostRegressor
    """)
    return


@app.cell
def _(
    CatBoostRegressor,
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    y_taxi_test,
    y_taxi_train,
):
    def run_catboost_regressor(X_train, y_train, X_test, y_test):
        cat_model = CatBoostRegressor(verbose=0)
        cat_model.fit(X_train, y_train)

        train_predictions = cat_model.predict(X_train)
        predictions = cat_model.predict(X_test)

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_catboost_regressor(X_taxi_train, y_taxi_train, X_taxi_test, y_taxi_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### XGBRegressor
    """)
    return


@app.cell
def _(
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    xgboost,
    y_taxi_test,
    y_taxi_train,
):
    def run_xgboost_regressor(X_train, y_train, X_test, y_test):
        xgb_model = xgboost.XGBRegressor()
        xgb_model.fit(X_train, y_train)

        train_predictions = xgb_model.predict(X_train)
        predictions = xgb_model.predict(X_test)

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_xgboost_regressor(X_taxi_train, y_taxi_train, X_taxi_test, y_taxi_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### LGBMRegressor
    """)
    return


@app.cell
def _(
    X_taxi_test,
    X_taxi_train,
    get_regression_report,
    lightgbm,
    y_taxi_test,
    y_taxi_train,
):
    def run_lightgbm_regressor(X_train, y_train, X_test, y_test):
        lgb_model = lightgbm.LGBMRegressor(objective='regression', metric='rmse')
        lgb_model.fit(X_train, y_train)

        train_predictions = lgb_model.predict(X_train)
        predictions = lgb_model.predict(X_test)

        return get_regression_report(y_train, train_predictions, y_test, predictions)

    run_lightgbm_regressor(X_taxi_train, y_taxi_train, X_taxi_test, y_taxi_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pycaret Regression
    """)
    return


@app.cell
def _(df_taxi_raw, pr):
    def run_pycaret_regression(data):
        pr.setup(data=data, target='trip_duration', train_size=0.8, preprocess=True)

        print(pr.models())

        dt = pr.create_model(estimator='dt')
        print('dt:')
        print(dt)
        tuned_dt = pr.tune_model(dt, fold=5)
        print('tuned_dt:')
        print(tuned_dt)


        gbr = pr.create_model(estimator='gbr')
        print('gbr:')
        print(gbr)
        tuned_gbr = pr.tune_model(gbr, fold=5)
        print('tuned_gbr:')
        print(tuned_gbr)


        lgbm = pr.create_model(estimator='lightgbm', force_col_wise=True)
        print('lgbm:')
        print(lgbm)
        tuned_lgbm = pr.tune_model(lgbm, fold=5)
        print('tuned_lgbm:')
        print(tuned_lgbm)

    run_pycaret_regression(df_taxi_raw.sample(50000))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Астероиды (DecisionTreeClassifier)
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
def _(pd):
    df_neo_raw = pd.read_csv('./data/processed/planets.csv')
    df_neo_raw
    return (df_neo_raw,)


@app.cell
def _(df_neo_raw):
    X_neo = df_neo_raw.drop('hazardous', axis=1)
    y_neo = df_neo_raw['hazardous']

    y_neo.value_counts()
    return X_neo, y_neo


@app.cell
def _(X_neo, df_neo_raw, train_test_split, y_neo):
    X_neo_train, X_neo_test, y_neo_train, y_neo_test = train_test_split(X_neo, y_neo, test_size=0.2, random_state=42, stratify=df_neo_raw['hazardous'])
    return X_neo_test, X_neo_train, y_neo_test, y_neo_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### DecisionTreeClassifier. GridSearchCV
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    GridSearchCV,
    X_neo_test,
    X_neo_train,
    get_classification_report,
    y_neo_test,
    y_neo_train,
):
    def run_decision_tree_classifier_gridsearch_cv(X_train, y_train, X_test, y_test):
        param_grid = {
            'max_depth': [10, 20, 30],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'criterion': ['gini'],
            'splitter': ['best']
    }

        ctree = DecisionTreeClassifier()

        grid_search = GridSearchCV(
            estimator=ctree,
            param_grid=param_grid,
            cv=3,                      
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("Лучшие гиперпараметры:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        train_predictions = best_model.predict(X_train)
        predictions = best_model.predict(X_test)
        test_accuracy = best_model.score(X_test, y_test)
        print(f"Точность на тестовых данных: {test_accuracy:.3f}")

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_decision_tree_classifier_gridsearch_cv(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### BaggingClassifier. DecisionTreeClassifier, GridSearchCV
    """)
    return


@app.cell
def _(
    BaggingClassifier,
    DecisionTreeClassifier,
    GridSearchCV,
    X_neo_test,
    X_neo_train,
    get_classification_report,
    y_neo_test,
    y_neo_train,
):
    def run_bagging_decision_tree_classifier_gridsearch_cv(X_train, y_train, X_test, y_test):
        param_grid = {
            'n_estimators': [5, 15],    
            'max_samples': [0.7, 0.85],        
            'bootstrap': [True, False],                   
        }

        base_estimator = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

        bagging = BaggingClassifier(estimator=base_estimator, random_state=42)

        grid_search = GridSearchCV(
            bagging,
            param_grid=param_grid,
            cv=3,                      
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("Лучшие гиперпараметры:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        train_predictions = best_model.predict(X_train)
        predictions = best_model.predict(X_test)
        test_accuracy = best_model.score(X_test, y_test)
        print(f"Точность на тестовых данных: {test_accuracy:.3f}")

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_bagging_decision_tree_classifier_gridsearch_cv(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### GradientBoostingClassifier. DecisionTreeClassifier
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    X_neo_test,
    X_neo_train,
    get_classification_report,
    y_neo_test,
    y_neo_train,
):
    def run_gradient_boosting_decision_tree_classifier(X_train, y_train, X_test, y_test):
        gbr = GradientBoostingClassifier()

        gbr.fit(X_train, y_train)

        train_predictions = gbr.predict(X_train)
        predictions = gbr.predict(X_test)

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_gradient_boosting_decision_tree_classifier(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### StackingClassifier. DecisionTreeClassifier
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    StackingClassifier,
    X_neo_test,
    X_neo_train,
    get_classification_report,
    y_neo_test,
    y_neo_train,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    def run_stacking_classifier(X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        base_models = [
            ('knn1', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ]

        final_estimator = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)

        stackingClassifier = StackingClassifier(
            estimators=base_models,
            cv=3,
            final_estimator=final_estimator,
            n_jobs=-1
        )

        stackingClassifier.fit(X_train_scaled, y_train)

        train_predictions = stackingClassifier.predict(X_train_scaled)
        predictions = stackingClassifier.predict(X_test_scaled)

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_stacking_classifier(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CatBoostClassifier
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    X_neo_test,
    X_neo_train,
    get_classification_report,
    y_neo_test,
    y_neo_train,
):
    def run_catboost_classifier(X_train, y_train, X_test, y_test):
        cat_model = CatBoostClassifier(verbose=0)
        cat_model.fit(X_train, y_train)

        train_predictions = cat_model.predict(X_train)
        predictions = cat_model.predict(X_test)

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_catboost_classifier(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### XGBClassifier
    """)
    return


@app.cell
def _(
    X_neo_test,
    X_neo_train,
    get_classification_report,
    xgboost,
    y_neo_test,
    y_neo_train,
):
    def run_xgboost_classifier(X_train, y_train, X_test, y_test):
        xgb_model = xgboost.XGBClassifier()
        xgb_model.fit(X_train, y_train)

        train_predictions = xgb_model.predict(X_train)
        predictions = xgb_model.predict(X_test)

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_xgboost_classifier(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### LGBMClassifier
    """)
    return


@app.cell
def _(
    X_neo_test,
    X_neo_train,
    get_classification_report,
    lightgbm,
    y_neo_test,
    y_neo_train,
):
    def run_lightgbm_classifier(X_train, y_train, X_test, y_test):
        lgb_model = lightgbm.LGBMClassifier(objective='binary', metric='binary_logloss')
        lgb_model.fit(X_train, y_train)

        train_predictions = lgb_model.predict(X_train)
        predictions = lgb_model.predict(X_test)

        return get_classification_report(y_train, train_predictions, y_test, predictions)

    run_lightgbm_classifier(X_neo_train, y_neo_train, X_neo_test, y_neo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pycaret Classification
    """)
    return


@app.cell
def _(df_neo_raw, pc):
    def run_pycaret_classification(data):
        pc.setup(data=data, target='hazardous', train_size=0.8, preprocess=True)

        print(pc.models())

        dt = pc.create_model(estimator='dt')
        print('dt:')
        print(dt)
        tuned_dt = pc.tune_model(dt, fold=3)
        print('tuned_dt:')
        print(tuned_dt)


        gbc = pc.create_model(estimator='gbc')
        print('gbc:')
        print(gbc)
        tuned_gbc = pc.tune_model(gbc, fold=3)
        print('tuned_gbc:')
        print(tuned_gbc)


        lgbm = pc.create_model(estimator='lightgbm', force_col_wise=True)
        print('lgbm:')
        print(lgbm)
        tuned_lgbm = pc.tune_model(lgbm, fold=3)
        print('tuned_lgbm:')
        print(tuned_lgbm)

    run_pycaret_classification(df_neo_raw)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MyDecisionTreeCART
    """)
    return


@app.cell(hide_code=True)
def _(np, pd):
    from copy import deepcopy

    class MyDecisionTreeCART:
        def __init__(self, max_depth=10, min_samples=2, ccp_alpha=0.0, regression=False):
            self.max_depth = max_depth
            self.min_samples = min_samples
            self.ccp_alpha = ccp_alpha
            self.regression = regression
            self.tree = None
            self._y_type = None
            self._num_all_samples = None

        def _set_df_type(self, X, y, dtype):
            X = X.astype(dtype)
            y = y.astype(dtype) if self.regression else y
            self._y_dtype = y.dtype

            return X, y

        @staticmethod
        def _purity(y):
            unique_classes = np.unique(y)

            return unique_classes.size == 1

        @staticmethod
        def _is_leaf_node(node):
            return not isinstance(node, dict)

        def _leaf_node(self, y):
            class_index = 0

            return np.mean(y) if self.regression else y.mode()[class_index]

        def _split_df(self, X, y, feature, threshold):
            feature_values = X[feature]
            left_indexes = X[feature_values <= threshold].index
            right_indexes = X[feature_values > threshold].index
            sizes = np.array([left_indexes.size, right_indexes.size])

            return self._leaf_node(y) if any(sizes == 0) else left_indexes, right_indexes

        @staticmethod
        def _gini_impurity(y):
            _, counts_classes = np.unique(y, return_counts=True)
            squared_probabilities = np.square(counts_classes / y.size)
            gini_impurity = 1 - sum(squared_probabilities)

            return gini_impurity

        @staticmethod
        def _mse(y):
            mse = np.mean((y - y.mean()) ** 2)

            return mse

        @staticmethod
        def _cost_function(left_df, right_df, method):
            total_df_size = left_df.size + right_df.size
            p_left_df = left_df.size / total_df_size
            p_right_df = right_df.size / total_df_size
            J_left = method(left_df)
            J_right = method(right_df)
            J = p_left_df*J_left + p_right_df*J_right

            return J

        def _node_error_rate(self, y, method):
            if self._num_all_samples is None:
                self._num_all_samples = y.size
            current_num_samples = y.size

            return current_num_samples / self._num_all_samples * method(y)

        def _best_split(self, X, y):
            features = X.columns
            min_cost_function = np.inf
            best_feature, best_threshold = None, None
            method = self._mse if self.regression else self._gini_impurity

            for feature in features:
                unique_feature_values = np.unique(X[feature])

                for i in range(1, len(unique_feature_values)):
                    current_value = unique_feature_values[i]
                    previous_value = unique_feature_values[i-1]
                    threshold = (current_value + previous_value) / 2
                    left_indexes, right_indexes = self._split_df(X, y, feature, threshold)
                    left_labels, right_labels = y.loc[left_indexes], y.loc[right_indexes]
                    current_J = self._cost_function(left_labels, right_labels, method)

                    if current_J <= min_cost_function:
                        min_cost_function = current_J
                        best_feature = feature
                        best_threshold = threshold

            return best_feature, best_threshold

        def _stopping_conditions(self, y, depth, n_samples):
            return self._purity(y), depth == self.max_depth, n_samples < self.min_samples

        def _grow_tree(self, X, y, depth=0):
            current_num_samples = y.size
            X, y = self._set_df_type(X, y, np.float64)
            method = self._mse if self.regression else self._gini_impurity

            if any(self._stopping_conditions(y, depth, current_num_samples)):
                RTi = self._node_error_rate(y, method)
                leaf_node = f'{self._leaf_node(y)} | error_rate {RTi}'
                return leaf_node

            Rt = self._node_error_rate(y, method)
            best_feature, best_threshold = self._best_split(X, y)
            decision_node = f'{best_feature} <= {best_threshold} | ' \
                            f'as_leaf {self._leaf_node(y)} error_rate {Rt}'

            left_indexes, right_indexes = self._split_df(X, y, best_feature, best_threshold)
            left_X, right_X = X.loc[left_indexes], X.loc[right_indexes]
            left_labels, right_labels = y.loc[left_indexes], y.loc[right_indexes]

            tree = {decision_node: []}
            left_subtree = self._grow_tree(left_X, left_labels, depth+1)
            right_subtree = self._grow_tree(right_X, right_labels, depth+1)

            if left_subtree == right_subtree:
                tree = left_subtree
            else:
                tree[decision_node].extend([left_subtree, right_subtree])

            return tree

        def _tree_error_rate_info(self, tree, error_rates_list):
            if self._is_leaf_node(tree):
                *_, leaf_error_rate = tree.split()
                error_rates_list.append(np.float64(leaf_error_rate))
            else:
                decision_node = next(iter(tree))
                left_subtree, right_subtree = tree[decision_node]
                self._tree_error_rate_info(left_subtree, error_rates_list)
                self._tree_error_rate_info(right_subtree, error_rates_list)

            RT = sum(error_rates_list)
            num_leaf_nodes = len(error_rates_list)

            return RT, num_leaf_nodes

        @staticmethod
        def _ccp_alpha_eff(decision_node_Rt, leaf_nodes_RTt, num_leafs):

            return (decision_node_Rt - leaf_nodes_RTt) / (num_leafs - 1)

        def _find_weakest_node(self, tree, weakest_node_info):
            if self._is_leaf_node(tree):
                return tree

            decision_node = next(iter(tree))
            left_subtree, right_subtree = tree[decision_node]
            *_, decision_node_error_rate = decision_node.split()

            Rt = np.float64(decision_node_error_rate)
            RTt, num_leaf_nodes = self._tree_error_rate_info(tree, [])
            ccp_alpha = self._ccp_alpha_eff(Rt, RTt, num_leaf_nodes)
            decision_node_index, min_ccp_alpha_index = 0, 1

            if ccp_alpha <= weakest_node_info[min_ccp_alpha_index]:
                weakest_node_info[decision_node_index] = decision_node
                weakest_node_info[min_ccp_alpha_index] = ccp_alpha

            self._find_weakest_node(left_subtree, weakest_node_info)
            self._find_weakest_node(right_subtree, weakest_node_info)

            return weakest_node_info

        def _prune_tree(self, tree, weakest_node):
            if self._is_leaf_node(tree):
                return tree

            decision_node = next(iter(tree))
            left_subtree, right_subtree = tree[decision_node]
            left_subtree_index, right_subtree_index = 0, 1
            _, leaf_node = weakest_node.split('as_leaf ')

            if weakest_node is decision_node:
                tree = weakest_node
            if weakest_node in left_subtree:
                tree[decision_node][left_subtree_index] = leaf_node
            if weakest_node in right_subtree:
                tree[decision_node][right_subtree_index] = leaf_node

            self._prune_tree(left_subtree, weakest_node)
            self._prune_tree(right_subtree, weakest_node)

            return tree

        def cost_complexity_pruning_path(self, X: pd.DataFrame, y: pd.Series):
            tree = self._grow_tree(X, y)
            tree_error_rate, _ = self._tree_error_rate_info(tree, [])
            error_rates = [tree_error_rate]
            ccp_alpha_list = [0.0]

            while not self._is_leaf_node(tree):
                initial_node = [None, np.inf]
                weakest_node, ccp_alpha = self._find_weakest_node(tree, initial_node)
                tree = self._prune_tree(tree, weakest_node)
                tree_error_rate, _ = self._tree_error_rate_info(tree, [])

                error_rates.append(tree_error_rate)
                ccp_alpha_list.append(ccp_alpha)

            return np.array(ccp_alpha_list), np.array(error_rates)

        def _ccp_tree_error_rate(self, tree_error_rate, num_leaf_nodes):

            return tree_error_rate + self.ccp_alpha*num_leaf_nodes

        def _optimal_tree(self, X, y):
            tree = self._grow_tree(X, y)
            min_RT_alpha, final_tree = np.inf, None

            while not self._is_leaf_node(tree):
                RT, num_leaf_nodes = self._tree_error_rate_info(tree, [])
                current_RT_alpha = self._ccp_tree_error_rate(RT, num_leaf_nodes)

                if current_RT_alpha <= min_RT_alpha:
                    min_RT_alpha = current_RT_alpha
                    final_tree = deepcopy(tree)

                initial_node = [None, np.inf]
                weakest_node, _ = self._find_weakest_node(tree, initial_node)
                tree = self._prune_tree(tree, weakest_node)

            return final_tree

        def fit(self, X: pd.DataFrame, y: pd.Series):
            self.tree = self._optimal_tree(X, y)

        def _traverse_tree(self, sample, tree):
            if self._is_leaf_node(tree):
                leaf, *_ = tree.split()
                return leaf

            decision_node = next(iter(tree))
            left_node, right_node = tree[decision_node]
            feature, other = decision_node.split(' <=')
            threshold, *_ = other.split()
            feature_value = sample[feature]

            if np.float64(feature_value) <= np.float64(threshold):
                next_node = self._traverse_tree(sample, left_node)
            else:
                next_node = self._traverse_tree(sample, right_node)

            return next_node

        def predict(self, samples: pd.DataFrame):
            results = samples.apply(self._traverse_tree, args=(self.tree,), axis=1)

            return np.array(results.astype(self._y_dtype))

    return (MyDecisionTreeCART,)


@app.cell
def _(MyDecisionTreeCART, get_regression_report):
    def run_my_regression_cart(X_train, y_train, X_test, y_test, max_depth=100):
        model = MyDecisionTreeCART(regression=True, max_depth=max_depth)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)    

        return get_regression_report(y_train, y_pred_train, y_test, y_pred)

    return (run_my_regression_cart,)


@app.cell
def _(
    X_taxi_test,
    X_taxi_train,
    run_my_regression_cart,
    y_taxi_test,
    y_taxi_train,
):
    run_my_regression_cart(X_taxi_train[:5000], y_taxi_train[:5000], X_taxi_test[:5000], y_taxi_test[:5000], max_depth=5)
    return


@app.cell
def _(
    X_taxi_test,
    X_taxi_train,
    run_my_regression_cart,
    y_taxi_test,
    y_taxi_train,
):
    run_my_regression_cart(X_taxi_train[:5000], y_taxi_train[:5000], X_taxi_test[:5000], y_taxi_test[:5000], max_depth=10)
    return


@app.cell
def _(MyDecisionTreeCART, get_classification_report):
    def run_my_classification_cart(X_train, y_train, X_test, y_test, max_depth=100):
        model = MyDecisionTreeCART(max_depth=max_depth)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        return get_classification_report(y_train, y_pred_train, y_test, y_pred)

    return (run_my_classification_cart,)


@app.cell
def _(
    X_neo_test,
    X_neo_train,
    run_my_classification_cart,
    y_neo_test,
    y_neo_train,
):
    run_my_classification_cart(X_neo_train[:5000], y_neo_train[:5000], X_neo_test[:5000], y_neo_test[:5000], max_depth=5)
    return


app._unparsable_cell(
    r"""
        run_my_classification_cart(X_neo_train[:5000], y_neo_train[:5000], X_neo_test[:5000], y_neo_test[:5000], max_depth=10)
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
