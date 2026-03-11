import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    import joblib

    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
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
        LogisticRegression,
        RandomForestClassifier,
        RandomUnderSampler,
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
        precision_recall_curve,
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
    return X, y


@app.cell
def _(SEED, X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)
    X_train.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler, X_test, X_train, pd):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_sc = pd.DataFrame(scaler.transform(X_test))
    return


@app.cell
def _(RandomUnderSampler, SEED, X_train, y_train):
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=SEED)

    X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

    X_train_undersampled.shape
    return


@app.cell
def _(SEED, SMOTEENN, X_train, y_train):
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=SEED)

    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    X_train_resampled.shape
    return


@app.cell
def _(np, precision_recall_curve):
    def find_best_threshold(y_true, y_proba, metric="f1"):
        """
        Подбирает порог, максимизирующий F1.
        Это КЛЮЧЕВОЕ улучшение — дефолтный 0.5 плох при дисбалансе.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        print(f"  Лучший порог: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
        return best_threshold


    return


@app.cell
def _():
    # def evaluate_model(model, X_train, y_train, X_test, y_test, use_threshold=True):
    #     y_proba_train = model.predict_proba(X_train)[:, 1]
    #     y_proba_test = model.predict_proba(X_test)[:, 1]

    #     if use_threshold:
    #         threshold = find_best_threshold(y_train, y_proba_train)
    #     else:
    #         threshold = 0.5

    #     y_pred_train = (y_proba_train >= threshold).astype(int)
    #     y_pred_test = (y_proba_test >= threshold).astype(int)

    #     return get_classification_report(y_train, y_pred_train, y_test, y_pred_test)
    return


@app.cell
def _(get_classification_report):
    def evaluate_model(model, X_train, y_train, X_test, y_test, use_threshold=True):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        return get_classification_report(y_train, y_pred_train, y_test, y_pred_test)

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
    X_test,
    X_train,
    evaluate_model,
    y_test,
    y_train,
):
    def wrapper():
        best_classic = DecisionTreeClassifier(
            min_samples_split=2,
            min_samples_leaf=75,
            class_weight='balanced',
            min_impurity_decrease=0.02,
            random_state=SEED
        )

        best_classic.fit(X_train, y_train)

        return evaluate_model(best_classic, X_train, y_train, X_test, y_test)
    wrapper()
    return


@app.cell
def _(DecisionTreeClassifier, SEED, X_train, y_train):
    best_classic = DecisionTreeClassifier(
        min_samples_split=10,
        min_samples_leaf=75,
        class_weight='balanced',
        random_state=SEED
    )

    best_classic.fit(X_train, y_train)
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
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight("balanced", y_train)

    best_boosting = GradientBoostingClassifier(
        n_estimators=2000,
        learning_rate=0.025,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=SEED,
    )

    best_boosting.fit(X_train, y_train, sample_weight=sample_weights)
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
        iterations=1000,
        depth=5,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        l2_leaf_reg=5,
        random_seed=SEED,
        verbose=0,
    )

    best_catboost.fit(X_train, y_train)
    return (best_catboost,)


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
    base_tree = DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=SEED,
    )
    best_bagging = BaggingClassifier(
        estimator=base_tree,
        n_estimators=300,
        max_samples=0.8,
        random_state=SEED,
        n_jobs=-1,
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
    CatBoostClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SEED,
    StackingClassifier,
    X_train,
    y_train,
):
    base_stacking = [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            random_state=SEED
        )),
        ("cb", CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=SEED, verbose=0
        )),
    ]

    best_stacking = StackingClassifier(
        estimators=base_stacking,
        final_estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        ),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    best_stacking.fit(X_train, y_train)
    return (best_stacking,)


@app.cell
def _(X_test, best_stacking, y_test):
    preds = best_stacking.predict(X_test)
    print(f"Predicted hazardous: {(preds==1).sum()}")
    print(f"Predicted safe:      {(preds==0).sum()}")
    print(f"Actual hazardous:    {(y_test==1).sum()}")
    print(f"Actual safe:         {(y_test==0).sum()}")
    return


@app.cell
def _(X_test, X_train, best_stacking, evaluate_model, y_test, y_train):
    evaluate_model(best_stacking, X_train, y_train, X_test, y_test, use_threshold=False)
    return


@app.cell
def _(best_stacking, joblib):
    joblib.dump(best_stacking, './models/best_stacking_model.joblib')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RESEARCH
    """)
    return


@app.cell
def _():
    def run_research():
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, precision_recall_curve, f1_score
        )
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier,
            BaggingClassifier, StackingClassifier
        )
        from sklearn.linear_model import LogisticRegression
        from catboost import CatBoostClassifier
        import joblib, os, warnings
        warnings.filterwarnings("ignore")

        SEED = 42

        # ══════════════════════════════════════════════════════════════════
        # 1. ДИАГНОСТИКА — смотрим что в данных
        # ══════════════════════════════════════════════════════════════════
        df = pd.read_csv("data/processed/planets.csv")

        print("Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nDtypes:\n", df.dtypes)
        print("\nHead:\n", df.head())
        print("\nDescribe:\n", df.describe())
        print("\nNulls:\n", df.isnull().sum())
        print("\nUnique values per column:")
        for col in df.columns:
            print(f"  {col}: {df[col].nunique()}")

        # Целевая переменная
        target_col = None
        for col in ("hazardous", "is_hazardous"):
            if col in df.columns:
                target_col = col
                break
        print(f"\nTarget: {target_col}")
        print(f"Value counts:\n{df[target_col].value_counts()}")

        # Корреляции — ищем мусор
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        print("\n\nКорреляция с target:")
        print(corr[target_col].sort_values(ascending=False))
        print("\n\nВысокие попарные корреляции (>0.9):")
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.9:
                    print(f"  {corr.columns[i]} <-> {corr.columns[j]}: {corr.iloc[i,j]:.4f}")

    run_research()
    return


@app.cell(hide_code=True)
def _():
    def run_improved():
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, precision_recall_curve, f1_score
        )
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier,
            BaggingClassifier, StackingClassifier
        )
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils.class_weight import compute_sample_weight
        from catboost import CatBoostClassifier
        import joblib, os, warnings
        warnings.filterwarnings("ignore")

        SEED = 42

        # ══════════════════════════════════════════════════════════════════
        # 1. ЗАГРУЗКА + FEATURE ENGINEERING (только полезные)
        # ══════════════════════════════════════════════════════════════════
        df = pd.read_csv("data/processed/planets.csv")
        target_col = "hazardous"

        # Среднее вместо двух коррелированных
        df["est_diameter_mean"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2
        df = df.drop(columns=["est_diameter_min", "est_diameter_max"])

        # Только фичи с реальной корреляцией > 0.05 или физическим смыслом
        df["log_diameter"] = np.log1p(df["est_diameter_mean"])
        df["log_velocity"] = np.log1p(df["relative_velocity"])
        df["magnitude_sq"] = df["absolute_magnitude"] ** 2
        df["diameter_per_mag"] = df["est_diameter_mean"] / (df["absolute_magnitude"] + 1)
        df["size_x_velocity"] = df["est_diameter_mean"] * df["relative_velocity"]
        df["kinetic_threat"] = (
            df["est_diameter_mean"] * df["relative_velocity"]
            / (df["miss_distance"] + 1)
        )

        # НЕ добавляем: velocity_per_distance (corr=-0.02), mag_per_distance (-0.03),
        # size_per_distance (importance=0), log_distance (corr=0.04)

        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
        print(f"Признаки ({len(feature_cols)}): {feature_cols}")

        X = df[feature_cols]
        y = df[target_col].astype(int)

        # ══════════════════════════════════════════════════════════════════
        # 2. РАЗБИЕНИЕ: train / val / test
        #    val — для подбора порога (НЕ train!)
        # ══════════════════════════════════════════════════════════════════
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.15, random_state=SEED, stratify=y_trainval
        )

        print(f"Train: {X_train.shape}")
        print(f"Val:   {X_val.shape}  (для порога)")
        print(f"Test:  {X_test.shape}  (финальная оценка)")
        print(f"Train balance: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")


        # ══════════════════════════════════════════════════════════════════
        # 3. УТИЛИТЫ
        # ══════════════════════════════════════════════════════════════════
        def find_best_threshold(y_true, y_proba):
            """Подбор порога по F1."""
            pr, re, th = precision_recall_curve(y_true, y_proba)
            f1s = 2 * pr * re / (pr + re + 1e-10)
            idx = np.argmax(f1s)
            return th[idx] if idx < len(th) else 0.5


        def evaluate(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te):
            """
            Порог подбирается на VALIDATION, оценка на TEST.
            """
            prob_tr = model.predict_proba(X_tr)[:, 1]
            prob_val = model.predict_proba(X_v)[:, 1]
            prob_te = model.predict_proba(X_te)[:, 1]

            # Порог на validation — честная оценка
            thr = find_best_threshold(y_v, prob_val)

            pred_tr = (prob_tr >= thr).astype(int)
            pred_te = (prob_te >= thr).astype(int)

            f1_tr = f1_score(y_tr, pred_tr)
            f1_te = f1_score(y_te, pred_te)
            auc = roc_auc_score(y_te, prob_te)

            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"  Threshold (val): {thr:.4f}")
            print(f"  Train F1: {f1_tr:.4f}")
            print(f"  Test  F1: {f1_te:.4f}  (gap: {f1_tr-f1_te:+.4f})")
            print(f"  Test AUC: {auc:.4f}")
            print(f"{'='*60}")
            print(classification_report(y_te, pred_te, target_names=["Safe", "Hazardous"]))
            print("Confusion Matrix:")
            print(confusion_matrix(y_te, pred_te))
            return thr


        # ══════════════════════════════════════════════════════════════════
        # 4. МОДЕЛИ — уменьшена сложность для борьбы с overfit
        # ══════════════════════════════════════════════════════════════════

        # ── Classic: RF — жёстко ограничиваем ───────────────────────────
        print("\n▶ CLASSIC")
        best_classic = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,              # ← было 20 → 8
            min_samples_leaf=20,      # ← было 3 → 20
            min_samples_split=40,     # ← добавили
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )
        best_classic.fit(X_train, y_train)
        thr_classic = evaluate("Classic (RF)", best_classic,
                                X_train, y_train, X_val, y_val, X_test, y_test)

        # ── Boosting: GB ─────────────────────────────────────────────────
        print("\n▶ BOOSTING")
        sw = compute_sample_weight("balanced", y_train)

        best_boosting = GradientBoostingClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=3,              # ← было 4 → 3 (слабые learners = меньше overfit)
            min_samples_leaf=30,      # ← было 15 → 30
            subsample=0.7,            # ← было 0.8 → 0.7
            max_features="sqrt",
            random_state=SEED,
        )
        best_boosting.fit(X_train, y_train, sample_weight=sw)
        thr_boosting = evaluate("Boosting (GB)", best_boosting,
                                 X_train, y_train, X_val, y_val, X_test, y_test)

        # ── CatBoost ─────────────────────────────────────────────────────
        print("\n▶ CATBOOST")
        best_catboost = CatBoostClassifier(
            iterations=2000,
            depth=4,                  # ← было 6 → 4
            learning_rate=0.03,
            auto_class_weights="Balanced",
            l2_leaf_reg=5,            # ← было 3 → 5 (сильнее регуляризация)
            min_data_in_leaf=30,      # ← добавили
            random_seed=SEED,
            verbose=200,
            early_stopping_rounds=100,
        )
        best_catboost.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=200)
        thr_catboost = evaluate("CatBoost", best_catboost,
                                 X_train, y_train, X_val, y_val, X_test, y_test)

        # ── Bagging — мелкие деревья ─────────────────────────────────────
        print("\n▶ BAGGING")
        base_dt = DecisionTreeClassifier(
            max_depth=6,              # ← было 15 → 6
            class_weight="balanced",
            min_samples_leaf=20,      # ← было 3 → 20
            random_state=SEED,
        )
        best_bagging = BaggingClassifier(
            estimator=base_dt,
            n_estimators=500,
            max_samples=0.6,          # ← было 0.7 → 0.6 (больше разнообразие)
            max_features=0.7,         # ← было 0.8 → 0.7
            random_state=SEED,
            n_jobs=-1,
        )
        best_bagging.fit(X_train, y_train)
        thr_bagging = evaluate("Bagging", best_bagging,
                                X_train, y_train, X_val, y_val, X_test, y_test)

        # ── Stacking ─────────────────────────────────────────────────────
        print("\n▶ STACKING")
        base_models = [
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.03,
                max_depth=3, min_samples_leaf=30, subsample=0.7,
                random_state=SEED
            )),
            ("cb", CatBoostClassifier(
                iterations=500, depth=4, learning_rate=0.05,
                auto_class_weights="Balanced", l2_leaf_reg=5,
                random_seed=SEED, verbose=0
            )),
        ]

        best_stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEED
            ),
            cv=5,
            n_jobs=-1,
        )
        best_stacking.fit(X_train, y_train)
        thr_stacking = evaluate("Stacking", best_stacking,
                                 X_train, y_train, X_val, y_val, X_test, y_test)


        # ══════════════════════════════════════════════════════════════════
        # 5. ИТОГОВОЕ ОБУЧЕНИЕ лучшей модели на train+val
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("  ФИНАЛЬНОЕ ОБУЧЕНИЕ лучших моделей на train+val")
        print("="*60)

        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])

        # Переобучаем все на полных данных
        best_classic.fit(X_full, y_full)
        sw_full = compute_sample_weight("balanced", y_full)
        best_boosting.fit(X_full, y_full, sample_weight=sw_full)
        best_catboost.fit(X_full, y_full)
        best_bagging.fit(X_full, y_full)
        best_stacking.fit(X_full, y_full)

        # Финальная оценка на test
        print("\n--- ФИНАЛЬНЫЕ МЕТРИКИ (test) ---")
        for name, model, thr in [
            ("Classic",  best_classic,  thr_classic),
            ("Boosting", best_boosting, thr_boosting),
            ("CatBoost", best_catboost, thr_catboost),
            ("Bagging",  best_bagging,  thr_bagging),
            ("Stacking", best_stacking, thr_stacking),
        ]:
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= thr).astype(int)
            f1 = f1_score(y_test, pred)
            auc = roc_auc_score(y_test, prob)
            print(f"  {name:10s}: F1={f1:.4f}  AUC={auc:.4f}  thr={thr:.4f}")


        # ══════════════════════════════════════════════════════════════════
        # 6. СОХРАНЕНИЕ
        # ══════════════════════════════════════════════════════════════════
        os.makedirs("models", exist_ok=True)

        joblib.dump(best_classic,  "models/best_classic_model.joblib")
        joblib.dump(best_boosting, "models/best_boosting_model.joblib")
        joblib.dump(best_catboost, "models/best_catboost_model.joblib")
        joblib.dump(best_bagging,  "models/best_bagging_model.joblib")
        joblib.dump(best_stacking, "models/best_stacking_model.joblib")
        joblib.dump(feature_cols,  "models/feature_names.joblib")

        stats = {col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
        } for col in feature_cols}
        joblib.dump(stats, "models/feature_stats.joblib")

        thresholds = {
            "classic": thr_classic,
            "boosting": thr_boosting,
            "catboost": thr_catboost,
            "bagging": thr_bagging,
            "stacking": thr_stacking,
        }
        joblib.dump(thresholds, "models/thresholds.joblib")

        print(f"\n✅ Saved")
        print(f"Features: {feature_cols}")
        print(f"Thresholds: {thresholds}")

    # run_improved()
    return


@app.cell(hide_code=True)
def _():
    def run_nasa():
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, precision_recall_curve, f1_score
        )
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier,
            BaggingClassifier, StackingClassifier
        )
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils.class_weight import compute_sample_weight
        from catboost import CatBoostClassifier
        import joblib, os, warnings
        warnings.filterwarnings("ignore")

        SEED = 42

        # ══════════════════════════════════════════════════════════════════
        # 1. ЗАГРУЗКА + DOMAIN KNOWLEDGE FEATURE ENGINEERING
        # ══════════════════════════════════════════════════════════════════
        # ФАКТ: NASA определяет PHA (Potentially Hazardous Asteroid) по:
        #   1. MOID ≤ 0.05 AU (≈ 7.48 млн км)
        #   2. H ≤ 22.0 (absolute_magnitude)
        #
        # miss_distance ≠ MOID, но мы можем создать proxy-признаки,
        # имитирующие логику NASA.
        # ══════════════════════════════════════════════════════════════════

        df = pd.read_csv("data/processed/planets.csv")
        target_col = "hazardous"

        print(f"Исходные: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")

        # ── Базовая обработка ────────────────────────────────────────────
        df["est_diameter_mean"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2
        df = df.drop(columns=["est_diameter_min", "est_diameter_max"])

        # ══════════════════════════════════════════════════════════════════
        # 2. FEATURE ENGINEERING — основан на физике
        # ══════════════════════════════════════════════════════════════════

        # --- Логарифмы (убираем скос распределений) ---
        df["log_diameter"] = np.log1p(df["est_diameter_mean"])
        df["log_velocity"] = np.log1p(df["relative_velocity"])
        df["log_distance"] = np.log1p(df["miss_distance"])

        # --- NASA PHA proxy: H ≤ 22 → большой объект ---
        df["is_large_H"] = (df["absolute_magnitude"] <= 22.0).astype(int)

        # --- NASA PHA proxy: miss_distance как proxy для MOID ---
        # 0.05 AU ≈ 7,479,894 км
        AU_005 = 7_479_894
        df["is_close"] = (df["miss_distance"] <= AU_005).astype(int)

        # --- Комбинация: имитация правила NASA ---
        df["nasa_rule_proxy"] = df["is_large_H"] * df["is_close"]

        # --- Квантили расстояния (нелинейные пороги) ---
        df["distance_quantile"] = pd.qcut(
            df["miss_distance"], q=10, labels=False, duplicates="drop"
        )

        # --- Квантили magnitude ---
        df["magnitude_quantile"] = pd.qcut(
            df["absolute_magnitude"], q=10, labels=False, duplicates="drop"
        )

        # --- Нелинейные трансформации ---
        df["magnitude_sq"] = df["absolute_magnitude"] ** 2
        df["magnitude_cube"] = df["absolute_magnitude"] ** 3
        df["diameter_sq"] = df["est_diameter_mean"] ** 2
        df["inv_magnitude"] = 1.0 / (df["absolute_magnitude"] + 1e-10)
        df["inv_distance"] = 1.0 / (df["miss_distance"] + 1)

        # --- Взаимодействия (физически осмысленные) ---
        df["diameter_per_mag"] = df["est_diameter_mean"] / (df["absolute_magnitude"] + 1)
        df["size_x_velocity"] = df["est_diameter_mean"] * df["relative_velocity"]

        # "Кинетическая энергия" ∝ mass × v² ∝ diameter³ × velocity²
        df["kinetic_energy_proxy"] = df["est_diameter_mean"] ** 3 * df["relative_velocity"] ** 2

        # Угроза = размер × скорость / расстояние
        df["kinetic_threat"] = (
            df["est_diameter_mean"] * df["relative_velocity"] / (df["miss_distance"] + 1)
        )

        # Gravitational focusing: чем ближе, тем сильнее притяжение → v растёт
        df["grav_focus"] = df["relative_velocity"] / (df["miss_distance"] ** 0.5 + 1)

        # Torino-like scale proxy: size × closeness
        df["torino_proxy"] = df["est_diameter_mean"] / (df["miss_distance"] / AU_005 + 0.1)

        # --- Бинарные пороги на основе EDA ---
        diameter_median = df["est_diameter_mean"].median()
        velocity_median = df["relative_velocity"].median()
        df["is_big"] = (df["est_diameter_mean"] > diameter_median).astype(int)
        df["is_fast"] = (df["relative_velocity"] > velocity_median).astype(int)
        df["big_and_fast"] = df["is_big"] * df["is_fast"]
        df["big_and_close"] = df["is_big"] * df["is_close"]

        # ── Итоговые признаки ────────────────────────────────────────────
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

        # Проверяем корреляции
        corr = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        print(f"\nПризнаки ({len(feature_cols)}):")
        print(corr.sort_values(ascending=False).to_string())

        # Убираем фичи с |corr| < 0.01
        low_corr = corr[abs(corr) < 0.01].index.tolist()
        if low_corr:
            print(f"\nУбираем шумовые (|corr|<0.01): {low_corr}")
            feature_cols = [c for c in feature_cols if c not in low_corr]
            print(f"Осталось: {len(feature_cols)}")

        # ══════════════════════════════════════════════════════════════════
        # 3. РАЗБИЕНИЕ
        # ══════════════════════════════════════════════════════════════════
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col].astype(int)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.15, random_state=SEED, stratify=y_trainval
        )

        print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Balance: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")

        # ══════════════════════════════════════════════════════════════════
        # 4. SMOTE — генерация синтетических примеров меньшинства
        # ══════════════════════════════════════════════════════════════════
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=SEED, k_neighbors=5)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            print(f"\nSMOTE: {X_train.shape[0]} → {X_train_sm.shape[0]}")
            print(f"After SMOTE: 0={(y_train_sm==0).sum()}, 1={(y_train_sm==1).sum()}")
            HAS_SMOTE = True
        except ImportError:
            print("\n⚠️ pip install imbalanced-learn  для SMOTE")
            X_train_sm, y_train_sm = X_train, y_train
            HAS_SMOTE = False


        # ══════════════════════════════════════════════════════════════════
        # 5. УТИЛИТЫ
        # ══════════════════════════════════════════════════════════════════
        def find_best_threshold(y_true, y_proba):
            pr, re, th = precision_recall_curve(y_true, y_proba)
            f1s = 2 * pr * re / (pr + re + 1e-10)
            idx = np.argmax(f1s)
            return th[idx] if idx < len(th) else 0.5


        def evaluate(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te):
            prob_tr = model.predict_proba(X_tr)[:, 1]
            prob_val = model.predict_proba(X_v)[:, 1]
            prob_te = model.predict_proba(X_te)[:, 1]

            thr = find_best_threshold(y_v, prob_val)

            pred_tr = (prob_tr >= thr).astype(int)
            pred_te = (prob_te >= thr).astype(int)

            f1_tr = f1_score(y_tr, pred_tr)
            f1_te = f1_score(y_te, pred_te)
            auc = roc_auc_score(y_te, prob_te)

            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"  Threshold (val): {thr:.4f}")
            print(f"  Train F1: {f1_tr:.4f}")
            print(f"  Test  F1: {f1_te:.4f}  (gap: {f1_tr-f1_te:+.4f})")
            print(f"  Test AUC: {auc:.4f}")
            print(f"{'='*60}")
            print(classification_report(y_te, pred_te, target_names=["Safe", "Hazardous"]))
            cm = confusion_matrix(y_te, pred_te)
            print(f"CM:\n{cm}")

            # Дополнительно: Precision@Recall=0.8
            pr, re, _ = precision_recall_curve(y_te, prob_te)
            idx_80 = np.argmin(np.abs(re - 0.8))
            print(f"  Precision@Recall=80%: {pr[idx_80]:.4f}")

            return thr


        # ══════════════════════════════════════════════════════════════════
        # 6. МОДЕЛИ
        # ══════════════════════════════════════════════════════════════════

        # ── Classic: RF на SMOTE-данных ──────────────────────────────────
        print("\n▶ CLASSIC (RF + SMOTE)")
        best_classic = RandomForestClassifier(
            n_estimators=700,
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=25,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )
        best_classic.fit(X_train_sm, y_train_sm)
        thr_classic = evaluate("Classic (RF+SMOTE)", best_classic,
                                X_train, y_train, X_val, y_val, X_test, y_test)

        # Feature importance
        imp = pd.Series(best_classic.feature_importances_, index=feature_cols)
        print("\nTop-10 Feature Importance:")
        print(imp.sort_values(ascending=False).head(10).to_string())

        # ── Boosting: GB с sample_weight ─────────────────────────────────
        print("\n▶ BOOSTING (GB + sample_weight)")
        sw = compute_sample_weight("balanced", y_train)

        best_boosting = GradientBoostingClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=4,
            min_samples_leaf=20,
            subsample=0.7,
            max_features="sqrt",
            random_state=SEED,
        )
        best_boosting.fit(X_train, y_train, sample_weight=sw)
        thr_boosting = evaluate("Boosting (GB)", best_boosting,
                                 X_train, y_train, X_val, y_val, X_test, y_test)

        # ── CatBoost ─────────────────────────────────────────────────────
        print("\n▶ CATBOOST")
        best_catboost = CatBoostClassifier(
            iterations=3000,
            depth=5,
            learning_rate=0.02,
            auto_class_weights="Balanced",
            l2_leaf_reg=5,
            min_data_in_leaf=20,
            random_seed=SEED,
            verbose=300,
            early_stopping_rounds=150,
        )
        best_catboost.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=300)
        thr_catboost = evaluate("CatBoost", best_catboost,
                                 X_train, y_train, X_val, y_val, X_test, y_test)

        # ── Bagging + SMOTE ──────────────────────────────────────────────
        print("\n▶ BAGGING (+ SMOTE)")
        base_dt = DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            min_samples_leaf=10,
            random_state=SEED,
        )
        best_bagging = BaggingClassifier(
            estimator=base_dt,
            n_estimators=500,
            max_samples=0.6,
            max_features=0.7,
            random_state=SEED,
            n_jobs=-1,
        )
        best_bagging.fit(X_train_sm, y_train_sm)
        thr_bagging = evaluate("Bagging (+SMOTE)", best_bagging,
                                X_train, y_train, X_val, y_val, X_test, y_test)

        # ── Stacking ─────────────────────────────────────────────────────
        print("\n▶ STACKING")
        base_models = [
            ("rf", RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=600, learning_rate=0.02,
                max_depth=4, min_samples_leaf=20, subsample=0.7,
                random_state=SEED
            )),
            ("cb", CatBoostClassifier(
                iterations=600, depth=5, learning_rate=0.03,
                auto_class_weights="Balanced", l2_leaf_reg=5,
                random_seed=SEED, verbose=0
            )),
        ]

        best_stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEED
            ),
            cv=5,
            n_jobs=-1,
        )
        best_stacking.fit(X_train, y_train)
        thr_stacking = evaluate("Stacking", best_stacking,
                                 X_train, y_train, X_val, y_val, X_test, y_test)


        # ══════════════════════════════════════════════════════════════════
        # 7. ФИНАЛЬНОЕ ОБУЧЕНИЕ на train+val
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("  ФИНАЛЬНОЕ ОБУЧЕНИЕ на train+val")
        print("="*60)

        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])

        if HAS_SMOTE:
            X_full_sm, y_full_sm = smote.fit_resample(X_full, y_full)
        else:
            X_full_sm, y_full_sm = X_full, y_full

        best_classic.fit(X_full_sm, y_full_sm)
        sw_full = compute_sample_weight("balanced", y_full)
        best_boosting.fit(X_full, y_full, sample_weight=sw_full)
        best_catboost.fit(X_full, y_full)
        best_bagging.fit(X_full_sm, y_full_sm)
        best_stacking.fit(X_full, y_full)

        print("\n--- ФИНАЛЬНЫЕ МЕТРИКИ (test) ---")
        for name, model, thr in [
            ("Classic",  best_classic,  thr_classic),
            ("Boosting", best_boosting, thr_boosting),
            ("CatBoost", best_catboost, thr_catboost),
            ("Bagging",  best_bagging,  thr_bagging),
            ("Stacking", best_stacking, thr_stacking),
        ]:
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= thr).astype(int)
            f1 = f1_score(y_test, pred)
            auc = roc_auc_score(y_test, prob)
            cm = confusion_matrix(y_test, pred)
            print(f"  {name:10s}: F1={f1:.4f}  AUC={auc:.4f}  thr={thr:.4f}")
            print(f"              CM: TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")


        # ══════════════════════════════════════════════════════════════════
        # 8. СОХРАНЕНИЕ
        # ══════════════════════════════════════════════════════════════════
        os.makedirs("models", exist_ok=True)

        for fname, model in [
            ("best_classic_model.joblib",  best_classic),
            ("best_boosting_model.joblib", best_boosting),
            ("best_catboost_model.joblib", best_catboost),
            ("best_bagging_model.joblib",  best_bagging),
            ("best_stacking_model.joblib", best_stacking),
        ]:
            joblib.dump(model, f"models/{fname}")

        joblib.dump(feature_cols, "models/feature_names.joblib")

        stats = {col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
        } for col in feature_cols}
        joblib.dump(stats, "models/feature_stats.joblib")

        thresholds = {
            "classic": thr_classic,
            "boosting": thr_boosting,
            "catboost": thr_catboost,
            "bagging": thr_bagging,
            "stacking": thr_stacking,
        }
        joblib.dump(thresholds, "models/thresholds.joblib")

        print(f"\n✅ Saved")
        print(f"Features ({len(feature_cols)}): {feature_cols}")

    # run_nasa()
    return


@app.cell
def _():
    def save_classes():
        import pandas as pd
    
        df = pd.read_csv("./data/processed/planets.csv")
    
        hazardous_1 = df[df["hazardous"] == 1]
        hazardous_0 = df[df["hazardous"] == 0]
    
        hazardous_1.to_csv("./data/processed/hazardous_1.csv", index=False)
        hazardous_0.to_csv("./data/processed/hazardous_0.csv", index=False)
    
        print("Файлы успешно сохранены!")

    save_classes()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
