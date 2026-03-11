def page():
    import streamlit as st
    import pandas as pd

    st.title("📊 Информация о наборе данных")

    st.header("🌍 Предметная область")
    st.markdown(
        """
        **Near Earth Objects (NEO)** — околоземные объекты, такие как астероиды
        и кометы, орбиты которых проходят вблизи орбиты Земли. NASA ведёт
        постоянный мониторинг этих объектов для оценки потенциальной угрозы
        столкновения с нашей планетой.

        Набор данных содержит характеристики наблюдений NEO: оценочный диаметр,
        скорость, расстояние пролёта и **метку опасности** (целевая переменная
        для задачи бинарной классификации).
        """
    )

    FEATURE_DESCRIPTIONS = {
        "id": "Уникальный идентификатор записи",
        "neo_id": "Идентификатор околоземного объекта в базе NASA",
        "name": "Название / обозначение объекта",
        "est_diameter_min": "Минимальный оценочный диаметр объекта, км",
        "est_diameter_max": "Максимальный оценочный диаметр объекта, км",
        "relative_velocity": "Относительная скорость объекта, км/ч",
        "miss_distance": "Расстояние пролёта мимо Земли, км",
        "orbiting_body": "Тело, вокруг которого движется объект (обычно Earth)",
        "sentry_object": "Отслеживается ли объект системой Sentry (True/False)",
        "absolute_magnitude": "Абсолютная звёздная величина (H) — мера яркости",
        "hazardous": "Потенциально опасный объект (True (1) / False (0)) — целевая переменная",
    }

    tab_neo, tab_planets = st.tabs(
        ["🗃️ NEO Dataset (raw)", "📊 NEO Dataset (processed)"]
    )

    with tab_neo:
        try:
            df = pd.read_csv("data/raw/neo_task.csv")
            st.success(
                f"Загружено **{df.shape[0]}** строк × **{df.shape[1]}** столбцов"
            )

            st.subheader("Первые записи")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("📋 Описание признаков")
            desc_rows = []
            for col in df.columns:
                desc_rows.append(
                    {
                        "Признак": col,
                        "Тип": str(df[col].dtype),
                        "Описание": FEATURE_DESCRIPTIONS.get(col, "—"),
                        "Пропуски": int(df[col].isna().sum()),
                        "Уникальных": int(df[col].nunique()),
                    }
                )
            st.dataframe(
                pd.DataFrame(desc_rows), use_container_width=True, hide_index=True
            )

            st.subheader("📐 Статистические характеристики")
            st.dataframe(
                df.describe().T.style.format("{:.4f}"), use_container_width=True
            )

            target_col = _find_target(df)
            if target_col:
                st.subheader("🎯 Баланс классов целевой переменной")
                vc = df[target_col].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Не опасных (0 / False)", int(vc.get(False, vc.get(0, 0)))
                    )
                with col2:
                    st.metric("Опасных (1 / True)", int(vc.get(True, vc.get(1, 0))))

        except FileNotFoundError:
            st.error("Файл `data/raw/neo_task.csv` не найден.")

    with tab_planets:
        try:
            df_p = pd.read_csv("data/processed/planets.csv")
            st.success(
                f"Загружено **{df_p.shape[0]}** строк × **{df_p.shape[1]}** столбцов"
            )
            st.dataframe(df_p.head(10), use_container_width=True)
            st.subheader("Статистические характеристики")
            st.dataframe(
                df_p.describe().T.style.format("{:.4f}"), use_container_width=True
            )
        except FileNotFoundError:
            st.error("Файл `data/processed/planets.csv` не найден.")

    st.markdown("---")
    st.header("🔧 Особенности предобработки данных")
    st.markdown(
        """
        | Этап | Описание |
        |------|----------|
        | 1. Обработка пропусков | удаление пропусков |
        | 2. Удаление идентификаторов | Столбцы `id`, `name` не несут предсказательной силы |
        | 3. Кодирование целевой переменной | `hazardous` → 0 / 1 |
        | 4. Масштабирование | `StandardScaler` для приведения признаков к единому масштабу |
        """
    )

    st.header("🔍 EDA — ключевые наблюдения")
    st.markdown(
        """
        - Классы **несбалансированы**: безопасных объектов значительно больше.
        - `est_diameter_min` и `est_diameter_max` **сильно коррелированы** (≈ 1.0) →
          можно оставить один или создать среднее.
        - `absolute_magnitude` обратно связана с размером объекта.
        - Опасные объекты в среднем **крупнее** и имеют **меньшую абсолютную
          звёздную величину**.
        - Распределения скорости и расстояния **правосторонне скошены**.
        """
    )


def _find_target(df):
    for col in ("hazardous", "is_hazardous", "is_potentially_hazardous_asteroid"):
        if col in df.columns:
            return col
    return None
