def page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import os
    from sklearn.pipeline import Pipeline

    st.title("🤖 Предсказание модели ML")
    st.markdown(
        "### Классификация околоземных объектов (NEO)\n"
        "Определение потенциальной опасности астероида для Земли"
    )
    st.markdown("---")

    MODEL_FILES = {
        "DecisionTreeClassifier": "models/best_classic_model.joblib",
        "GradientBoostingClassifier": "models/best_boosting_model.joblib",
        "CatBoostClassifier": "models/best_catboost_model.joblib",
        "BaggingClassifier": "models/best_bagging_model.joblib",
        "StackingClassifier": "models/best_stacking_model.joblib",
    }

    @st.cache_data
    def load_processed():
        return pd.read_csv("data/processed/planets.csv")

    @st.cache_data
    def get_feature_info():
        df = load_processed()
        target_col = _find_target(df)
        if target_col is None:
            return None, None, None

        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
        ]
        X = df[feature_cols].fillna(df[feature_cols].median())

        stats = {}
        for col in feature_cols:
            stats[col] = {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean()),
                "median": float(X[col].median()),
            }
        return feature_cols, stats, target_col

    @st.cache_resource
    def load_model(path):
        model = joblib.load(path)
        needs_scaling = not isinstance(model, Pipeline)
        return model, needs_scaling

    @st.cache_resource
    def get_scaler(_feature_cols):
        from sklearn.preprocessing import StandardScaler

        df = load_processed()
        X = df[list(_feature_cols)].fillna(df[list(_feature_cols)].median())
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler

    available = {n: p for n, p in MODEL_FILES.items() if os.path.exists(p)}
    if not available:
        st.error("Не найдено моделей в `models/`.")
        return

    feature_cols, feature_stats, target_col = get_feature_info()
    if feature_cols is None:
        st.error("Целевая переменная не найдена в датасете.")
        return

    scaler = get_scaler(tuple(feature_cols))

    st.sidebar.markdown("### ⚙️ Модель")
    selected = st.sidebar.radio("Выберите модель:", list(available.keys()))
    model, needs_scaling = load_model(available[selected])
    st.info(f"Модель: **{selected}**")

    FEATURE_META = {
        "est_diameter_min": ("Мин. оценочный диаметр", "км"),
        "est_diameter_max": ("Макс. оценочный диаметр", "км"),
        "relative_velocity": ("Относительная скорость", "км/ч"),
        "miss_distance": ("Расстояние пролёта", "км"),
        "absolute_magnitude": ("Абсолютная звёздная величина", "H"),
    }

    input_mode = st.radio(
        "Способ ввода данных:",
        ["📝 Ручной ввод", "📁 Загрузка CSV"],
        horizontal=True,
    )

    if input_mode == "📝 Ручной ввод":
        st.subheader("Характеристики объекта")

        input_vals = {}
        cols = st.columns(2)

        for i, feat in enumerate(feature_cols):
            meta = FEATURE_META.get(feat, (feat, ""))
            label = f"{meta[0]}, {meta[1]}" if meta[1] else meta[0]
            s = feature_stats[feat]

            with cols[i % 2]:
                input_vals[feat] = st.number_input(
                    label,
                    min_value=0.0,
                    max_value=s["max"] * 3,
                    value=s["median"],
                    step=max(s["median"] / 100, 0.001),
                    format="%.6f" if s["max"] < 1 else "%.2f",
                    help=f"Диапазон: {s['min']:.4f} — {s['max']:.4f}",
                )

        errors = []
        if "est_diameter_min" in input_vals and "est_diameter_max" in input_vals:
            if input_vals["est_diameter_min"] > input_vals["est_diameter_max"]:
                errors.append("Мин. диаметр > макс. диаметр.")

        for e in errors:
            st.warning(f"⚠️ {e}")

        if st.button("🔮 Предсказать", type="primary", use_container_width=True):
            if errors:
                st.error("Исправьте ошибки.")
            else:
                inp = pd.DataFrame([input_vals])[feature_cols]
                X = scaler.transform(inp) if needs_scaling else inp
                pred = model.predict(X)[0]

                st.markdown("---")
                if int(pred) == 1:
                    st.error("## ⚠️ Потенциально опасный объект")
                else:
                    st.success("## ✅ Объект безопасен")

    else:
        st.subheader("Загрузите CSV-файл")
        st.markdown(f"Столбцы: **`{'`**, **`'.join(feature_cols)}`**")

        uploaded = st.file_uploader("CSV-файл", type=["csv"])

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Ошибка: {e}")
                return

            st.dataframe(df_in.head(), use_container_width=True)

            missing = [c for c in feature_cols if c not in df_in.columns]
            if missing:
                st.error(f"Отсутствуют: {', '.join(missing)}")
                return

            if st.button("🔮 Предсказать", type="primary", use_container_width=True):
                X = df_in[feature_cols].copy()
                X = X.apply(pd.to_numeric, errors="coerce")
                nans = X.isna().any(axis=1).sum()
                if nans:
                    st.warning(f"{nans} строк с пропусками — заполнены медианой.")
                    X = X.fillna(X.median())

                X_transformed = scaler.transform(X) if needs_scaling else X
                preds = model.predict(X_transformed)

                result = df_in.copy()
                result["Результат"] = [
                    "⚠️ Опасный" if p == 1 else "✅ Безопасный" for p in preds
                ]

                st.dataframe(result, use_container_width=True)

                n = len(preds)
                n_haz = int((preds == 1).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Всего", n)
                c2.metric("⚠️ Опасных", n_haz)
                c3.metric("✅ Безопасных", n - n_haz)

                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Скачать результаты",
                    data=csv,
                    file_name="neo_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def _find_target(df):
    for col in ("hazardous", "is_hazardous", "is_potentially_hazardous_asteroid"):
        if col in df.columns:
            return col
    return None
    if col in df.columns:
        return col
    return None
    if col in df.columns:
        return col
    return None
    if col in df.columns:
        return col
    return None
