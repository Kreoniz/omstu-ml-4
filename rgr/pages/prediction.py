def page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import os

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

    FEATURE_COLS = [
        "est_diameter_min",
        "est_diameter_max",
        "relative_velocity",
        "miss_distance",
        "absolute_magnitude",
    ]

    FEATURE_META = {
        "est_diameter_min": {
            "label": "Мин. оценочный диаметр",
            "unit": "км",
            "default": 0.1,
            "step": 0.01,
            "format": "%.4f",
        },
        "est_diameter_max": {
            "label": "Макс. оценочный диаметр",
            "unit": "км",
            "default": 0.2,
            "step": 0.01,
            "format": "%.4f",
        },
        "relative_velocity": {
            "label": "Относительная скорость",
            "unit": "км/ч",
            "default": 50000.0,
            "step": 1000.0,
            "format": "%.1f",
        },
        "miss_distance": {
            "label": "Расстояние пролёта",
            "unit": "км",
            "default": 30000000.0,
            "step": 1000000.0,
            "format": "%.0f",
        },
        "absolute_magnitude": {
            "label": "Абсолютная звёздная величина",
            "unit": "H",
            "default": 22.0,
            "step": 0.1,
            "format": "%.2f",
        },
    }

    def load_model(path):
        return joblib.load(path)

    def get_stats():
        df = pd.read_csv("data/processed/planets.csv")
        stats = {}
        for col in FEATURE_COLS:
            stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
            }
        return stats

    available = {n: p for n, p in MODEL_FILES.items() if os.path.exists(p)}
    if not available:
        st.error("Не найдено моделей в `models/`.")
        return

    stats = get_stats()

    st.sidebar.markdown("### ⚙️ Модель")
    selected = st.sidebar.radio("Выберите модель:", list(available.keys()))
    model = load_model(available[selected])
    st.info(f"Модель: **{selected}**")

    input_mode = st.radio(
        "Способ ввода данных:",
        ["📝 Ручной ввод", "📁 Загрузка CSV"],
        horizontal=True,
    )

    # ══════════════════════════════════════════════════════════════
    #  РУЧНОЙ ВВОД — УЛУЧШЕННЫЙ
    # ══════════════════════════════════════════════════════════════
    if input_mode == "📝 Ручной ввод":
        st.subheader("Характеристики объекта")

        # 1. Создаём 5 полей ввода в 2 колонки
        input_vals = {}
        cols = st.columns(2)

        for i, feat in enumerate(FEATURE_COLS):
            meta = FEATURE_META[feat]
            s = stats[feat]

            with cols[i % 2]:
                # 2. Свободный ввод — без min/max ограничений
                value = st.number_input(
                    f"{meta['label']}, {meta['unit']}",
                    value=meta["default"],  # фиксированный дефолт
                    step=meta["step"],  # фиксированный шаг
                    format=meta["format"],  # фиксированный формат
                    help=(
                        f"Мин в данных: {s['min']:.4f} | "
                        f"Макс в данных: {s['max']:.4f} | "
                        f"Среднее: {s['mean']:.4f}"
                    ),
                    key=f"manual_{feat}",  # уникальный ключ
                )
                input_vals[feat] = value

        # 3. Валидация — только предупреждения, не блокируем
        warnings_list = []
        if input_vals["est_diameter_min"] > input_vals["est_diameter_max"]:
            warnings_list.append("⚠️ Мин. диаметр больше макс. диаметра.")
        if input_vals["est_diameter_min"] < 0 or input_vals["est_diameter_max"] < 0:
            warnings_list.append("⚠️ Диаметр не может быть отрицательным.")
        if input_vals["relative_velocity"] < 0:
            warnings_list.append("⚠️ Скорость не может быть отрицательной.")
        if input_vals["miss_distance"] < 0:
            warnings_list.append("⚠️ Расстояние не может быть отрицательным.")
        if input_vals["absolute_magnitude"] <= 0:
            warnings_list.append("⚠️ Абсолютная звёздная величина должна быть > 0.")

        for w in warnings_list:
            st.warning(w)

        # 4. Кнопка предсказания
        if st.button("🔮 Предсказать", type="primary", use_container_width=True):
            # Создаём DataFrame с введёнными значениями
            X = pd.DataFrame([input_vals])[FEATURE_COLS]

            # Предсказание
            pred = model.predict(X)[0]

            # Результат
            st.markdown("---")
            if int(pred) == 1:
                st.error("## ⚠️ Потенциально опасный объект")
                st.markdown("Модель классифицировала объект как **опасный** для Земли.")
            else:
                st.success("## ✅ Объект безопасен")
                st.markdown(
                    "Модель классифицировала объект как **не представляющий угрозы**."
                )

            # Дополнительно: показываем введённые значения
            with st.expander("📋 Введённые значения"):
                for feat in FEATURE_COLS:
                    meta = FEATURE_META[feat]
                    st.write(
                        f"**{meta['label']}**: {input_vals[feat]:.6f} {meta['unit']}"
                    )

    # ══════════════════════════════════════════════════════════════
    #  CSV
    # ══════════════════════════════════════════════════════════════
    else:
        st.subheader("Загрузите CSV-файл")
        st.markdown(f"Столбцы: **`{'`**, **`'.join(FEATURE_COLS)}`**")

        uploaded = st.file_uploader("CSV-файл", type=["csv"])

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Ошибка: {e}")
                return

            st.dataframe(df_in.head(), use_container_width=True)

            missing = [c for c in FEATURE_COLS if c not in df_in.columns]
            if missing:
                st.error(f"Отсутствуют: {', '.join(missing)}")
                return

            # Галочка: проверить качество модели
            target_col = _find_target(df_in)
            evaluate_mode = False
            if target_col:
                evaluate_mode = st.checkbox(
                    f"📊 Оценить качество модели (столбец `{target_col}` найден)",
                    value=False,
                )
            if st.button("🔮 Предсказать", type="primary", use_container_width=True):
                X = df_in[FEATURE_COLS].copy()
                X = X.apply(pd.to_numeric, errors="coerce")
                nans = X.isna().any(axis=1).sum()
                if nans:
                    st.warning(f"{nans} строк с пропусками — заполнены медианой.")
                    X = X.fillna(X.median())
                preds = model.predict(X)

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

                # Оценка качества
                if evaluate_mode and target_col:
                    _show_evaluation(df_in[target_col], preds)

                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Скачать результаты",
                    data=csv,
                    file_name="neo_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def _show_evaluation(y_true, preds):
    """Показывает метрики и confusion matrix."""
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    st.markdown("---")
    st.subheader("📊 Оценка качества модели")

    y = y_true.astype(int).values
    p = np.array(preds).astype(int)

    acc = accuracy_score(y, p)
    prec = precision_score(y, p, zero_division=0)
    rec = recall_score(y, p, zero_division=0)
    f1 = f1_score(y, p, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-Score", f"{f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y, p)

    # ПРАВИЛЬНЫЙ ПОРЯДОК: [TN, FP, FN, TP]
    tn, fp, fn, tp = cm.ravel()

    # Создаём таблицу для отображения
    st.markdown("### Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Числа")
        st.markdown(f"""
        | | Реально Safe | Реально Hazardous |
        |---|---|---|
        | **Предсказано Safe** | **{tn:,}** (TN) | **{fn:,}** (FN) |
        | **Предсказано Hazardous** | **{fp:,}** (FP) | **{tp:,}** (TP) |
        """)

    with col2:
        st.markdown("#### Проценты")
        total = tn + fp + fn + tp
        st.markdown(f"""
        | | Реально Safe | Реально Hazardous |
        |---|---|---|
        | **Предсказано Safe** | **{tn / total * 100:.1f}%** | **{fn / total * 100:.1f}%** |
        | **Предсказано Hazardous** | **{fp / total * 100:.1f}%** | **{tp / total * 100:.1f}%** |
        """)

    # Визуализация
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Safe (0)", "Hazardous (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Предсказание", fontsize=10)
    ax.set_ylabel("Реальность", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=13,
            )

    fig.tight_layout()

    pad = 0.175
    _, col_center, _ = st.columns([pad, 1 - 2 * pad, pad])
    with col_center:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Интерпретация
    st.markdown("### 📈 Интерпретация")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "TP (True Positive)", f"{tp:,}", f"{tp / (tp + fn) * 100:.1f}% от опасных"
        )
        st.caption("Правильно предсказанные опасные")

    with col2:
        st.metric(
            "TN (True Negative)",
            f"{tn:,}",
            f"{tn / (tn + fp) * 100:.1f}% от безопасных",
        )
        st.caption("Правильно предсказанные безопасные")

    with col3:
        st.metric(
            "FP (False Positive)",
            f"{fp:,}",
            f"{fp / (fp + tp) * 100:.1f}% от предсказанных опасных",
        )
        st.caption("Ложные тревоги")

    with col4:
        st.metric(
            "FN (False Negative)",
            f"{fn:,}",
            f"{fn / (fn + tp) * 100:.1f}% от реальных опасных",
        )
        st.caption("Пропущенные опасные")

    # Ключевые метрики
    st.markdown("### 🎯 Ключевые показатели")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Точность обнаружения опасных:**
        - **Recall (полнота)**: {rec:.4f}
        - **Пропущено опасных**: {fn:,} из {tp + fn:,} ({fn / (tp + fn) * 100:.1f}%)
        """)

    with col2:
        st.markdown(f"""
        **Качество предсказаний опасных:**
        - **Precision (точность)**: {prec:.4f}
        - **Ложные тревоги**: {fp:,} из {fp + tp:,} ({fp / (fp + tp) * 100:.1f}%)
        """)


def _find_target(df):
    for col in ("hazardous", "is_hazardous", "is_potentially_hazardous_asteroid"):
        if col in df.columns:
            return col
    return None
    return None
    return None
    return None
