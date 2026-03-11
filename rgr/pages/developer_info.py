def page():
    import streamlit as st

    st.title("👨‍💻 Информация о разработчике моделей ML")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        photo_url = "https://avatars.githubusercontent.com/u/77620531"
        st.image(photo_url, caption="Фото GitHub", width=250)

    with col2:
        FULL_NAME = st.secrets.get("FULL_NAME", "Не указано")
        GROUP = st.secrets.get("GROUP", "Не указано")

        st.markdown(f"### {FULL_NAME}")
        st.markdown(f"**Учебная группа:** {GROUP}")
        st.markdown(
            "**Тема РГР:** Разработка Web-приложения (дашборда) "
            "для инференса моделей ML и анализа данных"
        )
        st.markdown("---")
        st.markdown(
            """
            **Краткое описание проекта:**

            Данное Web-приложение представляет собой интерактивный дашборд
            для анализа данных о **Near Earth Objects (NEO)** — околоземных
            объектах (астероидах), отслеживаемых NASA.

            Приложение позволяет:
            - 📊 Изучить структуру и статистику набора данных
            - 📈 Визуализировать зависимости между признаками
            - 🤖 Получить предсказание модели ML о потенциальной опасности объекта
            """
        )
